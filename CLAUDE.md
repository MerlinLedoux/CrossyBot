# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

**CrossyBot** — agent d'apprentissage par renforcement entraîné à jouer à un clone de Crossy Road.
Algorithme : PPO (Proximal Policy Optimization) implémenté en PyTorch pur, entraîné sur des environnements Gymnasium parallèles.

## Commands

```bash
# Installer les dépendances
pip install gymnasium numpy torch arcade

# Lancer l'entraînement
python train.py
python train.py --updates 2000 --envs 32 --steps 1024

# Reprendre un checkpoint
python train.py --load training/models/crossybot.pt --updates 500

# Jouer manuellement
python play.py
```

No test suite or linter is configured yet.

## Architecture

```
CrosyRoad/
├── train.py                        # Point d'entrée CLI (argparse)
├── play.py                         # Version jouable via Arcade (flèches + R)
└── training/
    ├── env/
    │   ├── crossy_env.py           # Environnement Gymnasium (CrossyEnv)
    │   ├── lane.py                 # Logique d'une ligne (Lane, LaneType)
    │   └── generation_config.py    # Table de génération procédurale
    └── agent/
        ├── network.py              # Architecture ActorCritic (LaneEncoder + Conv1D)
        ├── rollout.py              # Buffer de collecte + calcul GAE
        ├── ppo.py                  # Mise à jour PPO (loss clippée)
        └── trainer.py              # Boucle d'entraînement complète
```

## Environment (`CrossyEnv`)

- **Grille** : 13 colonnes × hauteur infinie. Colonnes 0-1 et 11-12 = murs (mort immédiate).
  Zone jouable : colonnes 2–10 (9 colonnes).
- **Observation** : vecteur `float32` de taille `13 * 15 + 2 = 197`
  - Pour chacune des 13 lanes visibles : `[type, speed, occ_0..occ_12]` (15 valeurs)
  - `occ_x` = taux d'occupation de la case x (0.0–1.0, via `obstacle_at`)
  - 2 valeurs finales : `player_x / GRID_W`, position verticale normalisée
- **Actions** : `Discrete(5)` — 0 rester, 1 avancer, 2 reculer, 3 gauche, 4 droite
- **Récompenses** :
  - `+1` par point de score gagné (nouveau maximum de ligne atteint)
  - `-0.1` par step sans progression
  - `-10` à la mort → `terminated = True`
- **Épisode** : tronqué à 1000 steps
- **Rythme** : `STEPS_PER_SEC = 3.0` (rythme humain, ~333 ms par action)

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
obs (197,)
  ├── lanes (13, 15) → LaneEncoder partagé (15→64→32) → (13, 32)
  │                 → SpatialConv Conv1D×2 (32→64→64) → flatten (832,)
  └── player_pos (2,)
        └── concat (834,) → Linear(256) → Linear(128)
              ├── PolicyHead → logits (5,) → Categorical
              └── ValueHead  → scalaire V(s)
```
- Initialisation orthogonale sur toutes les couches
- `act()` : sans gradient (collecte), `evaluate()` : avec gradient (update)

### Buffer (`rollout.py`)
- Stocke `n_steps × n_envs` transitions : `(obs, action, log_prob, reward, done, value)`
- `compute_returns()` : GAE en parcours arrière (gamma=0.99, lambda=0.95)
- `get_batches()` : mini-batchs mélangés avec normalisation des avantages

### PPO (`ppo.py`)
- Loss = `L_policy + 0.5 * L_value - 0.01 * L_entropy`
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

## Conventions importantes

- Les positions des obstacles sont des **floats absolus** (pas de modulo GRID_W)
- `player_log_slot` : slot entier sur la bûche, indépendant de la dérive flottante
- `camera_start_row` ne recule jamais ; `deque_start_row` suit la mémoire
- `_carry_player()` est appelé **avant** `_update_obstacles()` dans `step()`
- Entraînement sur **CPU uniquement** (réseau trop petit pour bénéficier du GPU)
