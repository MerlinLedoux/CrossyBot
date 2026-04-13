import gymnasium as gym
from gymnasium import spaces
import numpy as np
from .lane import Lane, LaneType, GRID_W, MAX_SPEED

GRID_H = 10   # lignes visibles devant/derrière le joueur
LOOK_BEHIND = 1  # lignes visibles derrière le joueur (le reste est devant)

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
        self.score    = 0               # max row atteint
        self.last_row = 0               # dernière ligne récompensée
        self._last_was_dangerous = False  # pour forcer une herbe entre deux sections dangereuses

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
        for lane in self.get_visible_lanes():  
            obs.append(lane.type)             
            obs.append(lane.speed)             
            for x in range(GRID_W):            
                obs.append(1.0 if lane.has_obstacle_at(x) else 0.0)
        obs.append(self.player_x / GRID_W)   
        obs.append(min(self.score / 100, 1.0)) 
        return np.array(obs, dtype=np.float32)

    def _generate_lanes(self) -> list:
        """Génère les lanes initiales avec un buffer d'avance."""
        lanes = [Lane(LaneType.SAFE, 0.0, self.np_random)]
        while len(lanes) < GRID_H + 10:
            lanes.extend(self._generate_section(len(lanes)))
        return lanes

    def _generate_section(self, start_row: int) -> list:
        """
        Génère une section homogène de lanes.
        Alterne automatiquement entre sections sûres (herbe) et dangereuses
        (route / eau) pour éviter deux zones mortelles consécutives.
        La difficulté (vitesse) augmente avec la distance.
        """
        rng = self.np_random
        difficulty = min(start_row / 100.0, 1.0)

        # Toujours intercaler une herbe après une section dangereuse
        if self._last_was_dangerous:
            self._last_was_dangerous = False
            n = int(rng.integers(1, 3))  # 1–2 lanes d'herbe
            return [Lane(LaneType.GRASS, 0.0, rng) for _ in range(n)]

        roll = float(rng.random())

        if roll < 0.25:  # 25 % → herbe
            n = int(rng.integers(1, 3))
            return [Lane(LaneType.GRASS, 0.0, rng) for _ in range(n)]

        elif roll < 0.60:  # 35 % → route
            self._last_was_dangerous = True
            n = int(rng.integers(1, 4))  # 1–3 voies
            speed = float(rng.uniform(MAX_SPEED * 0.2, MAX_SPEED * (0.3 + 0.7 * difficulty)))
            direction = float(rng.choice([-1.0, 1.0]))
            return [Lane(LaneType.ROAD, speed * direction, rng) for _ in range(n)]

        else:  # 40 % → eau (bûches ou nénuphars)
            self._last_was_dangerous = True
            n = int(rng.integers(1, 4))  # 1–3 lanes d'eau
            if float(rng.random()) < 0.3:  # 30 % nénuphars (statiques)
                return [Lane(LaneType.LILY, 0.0, rng) for _ in range(n)]
            else:  # bûches mobiles
                speed = float(rng.uniform(MAX_SPEED * 0.1, MAX_SPEED * (0.2 + 0.6 * difficulty)))
                direction = float(rng.choice([-1.0, 1.0]))
                return [Lane(LaneType.WATER, speed * direction, rng) for _ in range(n)]

    def get_visible_lanes(self) -> list:
        """Retourne les GRID_H lanes visibles autour du joueur.
        Génère de nouvelles lanes à la volée si le joueur avance."""
        start = max(0, self.player_row - LOOK_BEHIND)
        while len(self.lanes) < start + GRID_H:
            self.lanes.extend(self._generate_section(len(self.lanes)))
        return self.lanes[start: start + GRID_H]

    # action → (delta_x, delta_row)
    _ACTION_DELTAS = {
        0: ( 0,  0),  # rester
        1: ( 0,  1),  # avancer
        2: ( 0, -1),  # reculer
        3: (-1,  0),  # gauche
        4: ( 1,  0),  # droite
    }

    def _apply_action(self, action: int):
        dx, drow = self._ACTION_DELTAS[action]
        new_x   = max(0, min(GRID_W - 1, self.player_x + dx))
        new_row = max(0, self.player_row + drow)

        # Bloquer si la destination est un obstacle GRASS (case inaccessible)
        target = self.lanes[new_row] if new_row < len(self.lanes) else None
        if target is not None and target.lane_type == LaneType.GRASS:
            if target.has_obstacle_at(new_x):
                return  # mouvement annulé

        self.player_x   = new_x
        self.player_row = new_row
        self.score      = max(self.score, self.player_row)

    def _update_obstacles(self):
        for lane in self.lanes:
            lane.update()

    def _check_collision(self) -> bool:
        lane = self.lanes[self.player_row]

        if lane.lane_type == LaneType.ROAD:
            return lane.has_obstacle_at(self.player_x)          # voiture → mort

        if lane.lane_type in (LaneType.WATER, LaneType.LILY):
            return not lane.has_obstacle_at(self.player_x)      # pas de support → mort

        return False  # SAFE / GRASS : jamais de mort directe

    def _compute_reward(self, terminated: bool) -> float:
        if terminated:
            return -1.0

        if self.player_row > self.last_row:
            reward = float(self.player_row - self.last_row)
            self.last_row = self.player_row
            return reward

        return 0.0

    def render(self):
        ...

