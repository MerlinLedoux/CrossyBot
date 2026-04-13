import gymnasium as gym
from gymnasium import spaces
import numpy as np 

GRID_W = 9   # colonnes visibles
GRID_H = 10   # lignes visibles devant/derrière

class CrossyEnv(gym.Env):
    
    def __init__(self):
        super().__init__()
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(GRID_H * (GRID_W + 2),), dtype=np.float32)
        self.action_space = spaces.Discrete(5)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.player_x = GRID_W // 2     # centre
        self.player_row = 0             # ligne de départ
        self.steps = 0
        self.last_row = 0

        # génère le terrain initial
        self.lanes = self._generate_lanes()

        obs = self._get_observation()
        return obs, {}

    def step(self, action):
        self.steps += 1

        # 1. déplace le joueur
        self._apply_action(action)

        # 2. fait avancer les obstacles
        self._update_obstacles()

        # 3. vérifie collision
        terminated = self._check_collision()

        # 4. calcule la récompense
        reward = self._compute_reward(terminated)

        # 5. truncation (épisode trop long)
        truncated = self.steps >= 2000

        obs = self._get_observation()
        return obs, reward, terminated, truncated, {}
    
    def _get_observation(self):
        obs = []
        for lane in self.get_visible_lanes():  # 9 lignes
            obs.append(lane.type)              # scalaire
            obs.append(lane.speed)             # scalaire
            for x in range(GRID_W):            # 7 cellules
                obs.append(1.0 if lane.has_obstacle_at(x) else 0.0)
        obs.append(self.player_x / GRID_W)    # position normalisée
        obs.append(min(self.score / 100, 1.0)) # score normalisé
        return np.array(obs, dtype=np.float32)

    def render(self):
        ...

