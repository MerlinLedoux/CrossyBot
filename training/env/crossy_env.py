import gymnasium as gym
from gymnasium import spaces
import numpy as np
from .lane import Lane, LaneType, GRID_W, MAX_SPEED

GRID_H      = 10
LOOK_BEHIND = 1

_ACTION_DELTAS = {
    0: ( 0,  0),   # rester
    1: ( 0,  1),   # avancer
    2: ( 0, -1),   # reculer
    3: (-1,  0),   # gauche
    4: ( 1,  0),   # droite
}

class CrossyEnv(gym.Env):

    def __init__(self):
        super().__init__()
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0,
            shape=(GRID_H * (GRID_W + 2),),
            dtype=np.float32,
        )
        self.action_space = spaces.Discrete(5)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # player_x est un float : centre du sprite en unités de case
        # Au repos, toujours à x_entier + 0.5 (centre de case)
        self.player_x   = float(GRID_W // 2) + 0.5
        self.player_row = 0
        self.camera_row = 0
        self.steps      = 0
        self.score      = 0

        self.lanes = self._generate_lanes()

        return self._get_observation(), {}

    def step(self, action):
        self.steps += 1

        self._apply_action(action)
        self._update_obstacles()
        self._carry_player()          # transporte le joueur avec la bûche (RL)

        truncated = self.steps >= 2000
        return self._get_observation(), 0.0, False, truncated, {}

    # --- observation ---------------------------------------------------------

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

    # --- terrain -------------------------------------------------------------

    def _generate_lanes(self) -> list:
        lanes = [Lane(LaneType.SAFE, 0.0, self.np_random)]
        while len(lanes) < GRID_H + 10:
            lanes.extend(self._new_section())
        return lanes

    def _new_section(self) -> list:
        rng     = self.np_random
        section = []

        for _ in range(int(rng.integers(1, 3))):
            section.append(Lane(LaneType.GRASS, 0.0, rng))

        roll = float(rng.random())
        if roll < 0.40:       # 40 % → route
            speed     = float(rng.uniform(MAX_SPEED * 0.3, MAX_SPEED))
            direction = float(rng.choice([-1.0, 1.0]))
            for _ in range(int(rng.integers(1, 3))):
                section.append(Lane(LaneType.ROAD, speed * direction, rng))
        elif roll < 0.70:     # 30 % → bûches (1–2 lanes, même vitesse)
            speed     = float(rng.uniform(MAX_SPEED * 0.2, MAX_SPEED))
            direction = float(rng.choice([-1.0, 1.0]))
            for _ in range(int(rng.integers(1, 3))):
                section.append(Lane(LaneType.WATER, speed * direction, rng))
        else:                 # 30 % → nénuphars (1 seule lane)
            section.append(Lane(LaneType.LILY, 0.0, rng))

        return section

    def get_visible_lanes(self) -> list:
        while len(self.lanes) < self.camera_row + GRID_H:
            self.lanes.extend(self._new_section())
        return self.lanes[self.camera_row: self.camera_row + GRID_H]

    # --- actions -------------------------------------------------------------

    def _apply_action(self, action: int):
        dx, drow = _ACTION_DELTAS[action]

        if dx == 0 and drow == 0:
            return   # rester : pas de snap, la bûche continue de porter le joueur

        lane = self.lanes[self.player_row]

        # Mouvement horizontal sur une bûche : grille de la bûche, pas du jeu
        if dx != 0 and drow == 0 and lane.lane_type == LaneType.WATER:
            log = lane.get_log_at(self.player_x)
            if log is not None:
                log_start, log_width = log
                current_slot = int((self.player_x - log_start) % GRID_W)
                new_slot     = current_slot + dx
                if 0 <= new_slot < log_width:
                    self.player_x = (log_start + new_slot + 0.5) % GRID_W
                    return
                # Slot hors bûche : chute dans l'eau, pas de mouvement

        # Mouvement normal sur la grille du jeu
        current_cell_x = int(self.player_x)
        new_cell_x     = max(0, min(GRID_W - 1, current_cell_x + dx))
        new_row        = max(self.camera_row, self.player_row + drow)

        if new_row < len(self.lanes):
            target = self.lanes[new_row]
            if target.lane_type == LaneType.GRASS and target.has_obstacle_at(new_cell_x):
                return

        # Snap au centre du slot cible
        target_x = new_cell_x + 0.5
        if drow != 0 and new_row < len(self.lanes):
            target_lane = self.lanes[new_row]
            if target_lane.lane_type == LaneType.WATER:
                log = target_lane.get_log_at(target_x)
                if log is not None:
                    log_start, log_width = log
                    slot = int((target_x - log_start) % GRID_W)
                    slot = max(0, min(log_width - 1, slot))
                    target_x = (log_start + slot + 0.5) % GRID_W
        self.player_x   = target_x
        self.player_row = new_row
        self.score      = max(self.score, self.player_row)
        self.camera_row = max(self.camera_row, self.player_row - LOOK_BEHIND)

    def _update_obstacles(self):
        for lane in self.lanes:
            lane.update()

    def _carry_player(self):
        """Déplace le joueur avec la bûche sur laquelle il se trouve (entraînement RL)."""
        lane = self.lanes[self.player_row]
        if lane.lane_type == LaneType.WATER and lane.is_on_log(self.player_x):
            self.player_x += lane._speed
            # Tombe dans l'eau si poussé hors de la grille
            self.player_x = max(-0.5, min(GRID_W - 0.5, self.player_x))

    # -------------------------------------------------------------------------

    def render(self):
        ...
