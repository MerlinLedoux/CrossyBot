# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

**CrossyBot** — a reinforcement learning agent trained to play a Crossy Road clone. The agent is trained via PPO (Proximal Policy Optimization) using parallelized Gymnasium environments.

## Commands

```bash
# Install dependencies
pip install gymnasium numpy stable-baselines3

# Run training
python train.py
```

No test suite or linter is configured yet.

## Architecture

```
CrosyRoad/
├── train.py                   # Entry point — currently empty, wires Trainer and starts training
└── training/
    ├── env/
    │   └── crossy_env.py      # Custom Gymnasium environment (CrossyEnv)
    ├── agent/
    │   └── trainer.py         # PPO trainer using AsyncVectorEnv
    ├── models/                # Saved model checkpoints (empty)
    └── visual/                # Rendering / visualization utilities (empty)
```

### Environment (`CrossyEnv`)

- **Observation space**: flat `float32` array of shape `(GRID_H * (GRID_W + 2),)` = `(110,)`, values in `[-1.0, 1.0]`
  - For each of the 10 visible lanes: `lane.type`, `lane.speed`, then 9 obstacle flags
  - Last 2 values: normalized `player_x` and normalized score
- **Action space**: `Discrete(5)` — likely up/down/left/right/wait
- **Episode length**: truncated at 2000 steps
- **Termination**: collision with an obstacle

### Trainer

- Runs `n_envs=16` parallel environments via `AsyncVectorEnv`
- PPO agent is a stub (`PPO(...)`) — needs a concrete implementation (e.g. `stable_baselines3.PPO` or a custom implementation)
- `train.py` is empty and must import and call `Trainer`

### Key implementation gaps

- `_generate_lanes()`, `_apply_action()`, `_update_obstacles()`, `_check_collision()`, `_compute_reward()`, `get_visible_lanes()` are referenced in `crossy_env.py` but not yet implemented
- `render()` is a stub
- `PPO(...)` in `trainer.py` needs a real agent class
- `train.py` entry point needs to be written
