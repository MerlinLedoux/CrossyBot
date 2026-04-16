# CrossyBot

Agent d'apprentissage par renforcement entraîné à jouer à un clone de Crossy Road.
Implémentation PPO complète en **PyTorch pur**, sans bibliothèque RL externe.

---

## Installation

```bash
pip install gymnasium numpy torch arcade
```

---

## Utilisation

### Jouer manuellement

```bash
python play.py
```

Contrôles : flèches directionnelles | `R` pour rejouer

### Lancer l'entraînement

```bash
python train.py
python train.py --updates 2000 --envs 32 --steps 1024
python train.py --load training/models/crossybot.pt --updates 500
```

---

## Architecture

### Environnement

Grille de 13 colonnes × hauteur infinie. Le joueur avance vers le haut en évitant les obstacles.

| Lane | Règle |
|------|-------|
| Herbe | Arbres statiques et bloquants |
| Route | Voitures en flux continu — mort par collision |
| Eau | Bûches en flux continu — mort si dans l'eau |
| Nénuphars | Mort si pas sur un nénuphar |

**Observation** : vecteur de 197 valeurs décrivant les 13 lanes visibles + position du joueur.  
**Actions** : 5 — rester, avancer, reculer, gauche, droite.  
**Récompenses** : +1 par score, -0.1 par step sans progression, -10 à la mort.

### Réseau de neurones

```
LaneEncoder (MLP partagé par lane)
      ↓
SpatialConv (Conv1D entre lanes)
      ↓
Tronc partagé Actor-Critic (MLP)
      ├── PolicyHead → distribution sur les actions
      └── ValueHead  → estimation de V(s)
```

### Algorithme

PPO (Proximal Policy Optimization) avec :
- GAE pour le calcul des avantages
- Clipping du surrogate (ε = 0.2)
- Bonus d'entropie pour l'exploration
- Initialisation orthogonale
- Gradient clipping

---

## Structure du projet

```
CrosyRoad/
├── train.py
├── play.py
└── training/
    ├── env/
    │   ├── crossy_env.py
    │   ├── lane.py
    │   └── generation_config.py
    └── agent/
        ├── network.py
        ├── rollout.py
        ├── ppo.py
        └── trainer.py
```
