# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

**CrossyBot** — agent d'apprentissage par renforcement entraîné à jouer à un clone de Crossy Road.
Algorithme : PPO (Proximal Policy Optimization) implémenté en PyTorch pur, sans bibliothèque RL externe.
Deux versions du jeu : **Python** (Arcade, entraînement) et **Web** (Three.js, démo visuelle).
Suivi des expériences via WandB.

## Commands

```bash
# Installer les dépendances Python
pip install gymnasium numpy torch arcade wandb

# Jouer manuellement (Python/Arcade)
python play.py

# Faire jouer l'agent entraîné (Python)
python play.py --agent training/models/crossybot.pt

# Mode debug : observation step par step
python play.py --debug
python play.py --debug --agent training/models/crossybot.pt

# Lancer l'entraînement
python train.py
python train.py --updates 3000 --envs 16 --steps 512 --ent-coef 0.05
python train.py --load training/models/crossybot.pt --updates 1000
python train.py --no-wandb

# Exporter le modèle entraîné pour la version web
python export_model.py
python export_model.py --model training/models/crossybot.pt --output web/assets/crossybot.json

# Version web (Three.js)
cd web && npm install && npm run dev

# WandB
wandb login --relogin
wandb status
```

No test suite or linter is configured yet.

## Architecture

```
CrosyRoad/
├── train.py                        # Point d'entrée CLI entraînement (argparse)
├── play.py                         # Version jouable via Arcade (flèches + R)
│                                   # --agent : fait jouer le modèle
│                                   # --debug : affiche le vecteur d'obs brut, step par step
├── export_model.py                 # Exporte les poids PyTorch → JSON pour la version web
├── training/
│   ├── env/
│   │   ├── crossy_env.py           # Environnement Gymnasium (CrossyEnv)
│   │   ├── lane.py                 # Logique d'une ligne (Lane, LaneType)
│   │   └── generation_config.py    # Tables de génération procédurale (Table / Prob)
│   └── agent/
│       ├── network.py              # Architecture ActorCritic (LaneEncoder + MLP)
│       ├── rollout.py              # Buffer de collecte + calcul GAE
│       ├── ppo.py                  # Mise à jour PPO (loss clippée)
│       └── trainer.py              # Boucle d'entraînement + logging WandB
└── web/                            # Version Three.js (demo visuelle + mode IA)
    ├── index.html
    ├── package.json                # Vite + Three.js
    ├── vite.config.js
    ├── assets/                     # Modèles GLB + crossybot.json (poids exportés)
    └── src/
        ├── main.js                 # Boucle d'animation, inputs, scroll, mode IA
        ├── ai/
        │   └── agent.js            # Inférence JS pure (charge crossybot.json)
        ├── game/
        │   ├── constants.js
        │   ├── env.js              # CrossyEnv JS — physique identique au Python
        │   ├── generationConfig.js # Génération procédurale (miroir de generation_config.py)
        │   └── lane.js             # Lane JS — mêmes propriétés .type/.speed/.obstacleAt()
        └── renderer/
            ├── scene.js            # Scène Three.js, lumières, ombres
            ├── gameView.js         # Rendu 3D, animation de saut, caméra
            └── modelLoader.js      # Chargement des modèles GLB
```

## Environment (`CrossyEnv`)

- **Grille** : 13 colonnes × hauteur infinie. Colonnes 0-1 et 11-12 = murs (mort immédiate).
  Zone jouable : colonnes 2–10 (9 colonnes, PLAYABLE_MIN=2, PLAYABLE_MAX=10).
- **Observation** : vecteur `float32` de taille `OBS_SIZE = 57`
  - `OBS_LANES = 5` lanes autour du joueur (1 derrière + joueur + 3 devant)
  - Pour chaque lane : `[type, speed, occ_c2..occ_c10]` — 11 valeurs
  - `occ_x` = taux d'occupation de la case x (0.0–1.0, via `obstacle_at`)
  - 2 valeurs finales : `player_x` normalisé dans la zone jouable, position verticale normalisée (`/ GRID_H` avec `GRID_H=13`)
- **Actions** : `Discrete(5)` — 0 rester, 1 avancer, 2 reculer, 3 gauche, 4 droite
- **Récompenses** :
  - `+1` par point de score gagné (nouveau maximum de ligne atteint)
  - `-0.1` par step sans progression
  - `-1` à la mort → `terminated = True`
- **Épisode** : tronqué à 1000 steps
- **Rythme** : `STEPS_PER_SEC = 3.0` (rythme humain, ~333 ms par action)
- **Défilement automatique** : `SCROLL_SPEED = 0.5` rangées/sec → `SCROLL_STEP = 1/6` rangée/step.
  La caméra avance en permanence ; si le joueur est rattrapé il meurt.

## Types de lanes

| Type  | Comportement |
|-------|-------------|
| SAFE  | Ligne de départ, aucun obstacle |
| GRASS | Arbres statiques, bloquants |
| ROAD  | Voitures (flux continu, vitesse variable) — mort par collision |
| WATER | Bûches (flux continu) — mort si pas sur une bûche |
| LILY  | Nénuphars statiques — mort si pas sur un nénuphar |

## Agent PPO

### Réseau (`network.py`)
```
obs (57,)
  ├── lanes (5, 11) → LaneEncoder partagé (11→64→32) → (5, 32) → flatten (160,)
  └── player_pos (2,)
        └── concat (162,) → Linear(256) → Linear(128)
              ├── PolicyHead → logits (5,) → Categorical
              └── ValueHead  → scalaire V(s)
```
- ~90K paramètres
- Initialisation orthogonale sur toutes les couches
- `act()` : sans gradient (collecte), `evaluate()` : avec gradient (update)

### Buffer (`rollout.py`)
- Stocke `n_steps × n_envs` transitions : `(obs, action, log_prob, reward, done, value)`
- `compute_returns()` : GAE en parcours arrière (gamma=0.99, lambda=0.95)
- `get_batches()` : mini-batchs mélangés avec normalisation des avantages par mini-batch

### PPO (`ppo.py`)
- Loss = `L_policy + 0.5 * L_value - 0.05 * L_entropy`
- `L_policy` : surrogate clippé (clip_range=0.2)
- Gradient clipping : `max_grad_norm=0.5`
- Optimizer : Adam (lr=3e-4, eps=1e-5)

### Hyperparamètres par défaut
| Paramètre | Valeur |
|-----------|--------|
| n_envs | 16 |
| n_steps | 512 |
| n_epochs | 4 |
| batch_size | 256 |
| lr | 3e-4 |
| gamma | 0.99 |
| gae_lambda | 0.95 |
| clip_range | 0.2 |
| ent_coef | 0.05 |

## Version web (Three.js)

La version web est une démo visuelle 3D. Sa physique est identique à la version Python pour que le modèle entraîné soit réutilisable directement.

- **Bundler** : Vite 5 — `npm run dev` pour démarrer, `npm run build` pour la prod
- **Rendu** : Three.js, caméra orthographique, ombres PCF, modèles GLB
- **Mode IA** : bouton en haut à droite, charge `web/assets/crossybot.json`
- **Inférence** : JS pur dans `web/src/ai/agent.js` — reconstruit le réseau manuellement, < 1 ms par décision, aucune dépendance externe
- **Défilement automatique** : identique au Python, `SCROLL_SPEED = 0.5` rangées/sec
- **Caméra** : suit le joueur latéralement avec amortissement, Z suit `max(scrollRow + LOOK_BEHIND, playerZ)`

### Flux pour déployer un nouveau modèle dans le web
```bash
python train.py                          # entraîner
python export_model.py                   # exporter les poids → web/assets/crossybot.json
cd web && npm run dev                    # tester
```

## Conventions importantes

- Les positions des obstacles sont des **floats absolus** (pas de modulo GRID_W)
- `player_log_slot` : slot entier sur la bûche, indépendant de la dérive flottante
- `camera_start_row` ne recule jamais ; `deque_start_row` suit la mémoire
- `_carry_player()` est appelé **avant** `_update_obstacles()` dans `step()`
- `_advance_scroll()` est appelé dans `step()` — PAS dans `_apply_action()`. En mode play.py humain/agent, le scroll est géré manuellement dans `on_update(dt)` via `_scroll_row` flottant.
- L'observation (`OBS_LANES=5`) est plus petite que la fenêtre visuelle (`GRID_H=13`)
- L'observation ne contient que les colonnes jouables (c2..c10), pas les murs
- La normalisation Y de l'obs utilise `GRID_H=13` (Python) — le JS agent.js utilise `PYTHON_GRID_H=13` pour rester compatible avec le modèle entraîné
- Entraînement sur **CPU uniquement** (réseau trop petit pour bénéficier du GPU)
- `obstacle_at(x)` retourne un **float** 0.0–1.0 (taux d'occupation), pas un bool
- `Ctrl+C` pendant l'entraînement sauvegarde le modèle proprement avant de quitter

## WandB — métriques à surveiller

| Métrique | Comportement attendu |
|---|---|
| `perfs/reward_mean` | Doit augmenter progressivement |
| `losses/entropy` | Ne doit pas tomber à 0 (exploration maintenue) |
| `ppo/clip_fraction` | Doit rester entre 0.05 et 0.30 |
| `losses/value` | Doit diminuer (critic apprend V(s)) |
