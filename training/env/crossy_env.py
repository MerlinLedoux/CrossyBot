from collections import deque
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from .lane import Lane, LaneType, GRID_W
from .generation_config import CONFIG

GRID_H       = 13   # lignes visibles : 2 derrière + joueur + 10 devant
LOOK_BEHIND  = 2
LOOK_AHEAD   = 10
PLAYABLE_MIN = 2              # première colonne accessible au joueur
PLAYABLE_MAX = GRID_W - 3     # dernière colonne accessible (= 10)
MIN_LANES    = 16             # seuil déclenchant la génération d'un module
MAX_LANES    = 25             # taille cible à l'initialisation

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
            shape=(GRID_H * (GRID_W + 2) + 2,),   # 13*15+2 = 197
            dtype=np.float32,
        )
        self.action_space = spaces.Discrete(5)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.player_x         = float(GRID_W // 2) + 0.5   # centre = 6.5
        self.player_row       = 0
        self.deque_start_row  = 0
        self.camera_start_row = 0   # ne recule jamais
        self.steps            = 0
        self.score            = 0
        self._last_was_unsafe = False

        self.lanes = deque()
        self._init_lanes()

        return self._get_observation(), {}

    def step(self, action):
        self.steps += 1

        self._apply_action(action)
        self._trim_lanes()
        self._ensure_lanes()
        self._update_obstacles()
        self._carry_player()

        truncated = self.steps >= 2000
        return self._get_observation(), 0.0, False, truncated, {}

    # --- propriété utilitaire ------------------------------------------------

    @property
    def player_lane(self) -> Lane:
        return self.lanes[self.player_row - self.deque_start_row]

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

    def _init_lanes(self) -> None:
        """Remplit la deque jusqu'à MAX_LANES au démarrage."""
        self.lanes.append(Lane(LaneType.SAFE, 0.0, self.np_random))
        while len(self.lanes) < MAX_LANES:
            for lane in self._new_section():
                self.lanes.append(lane)
        while len(self.lanes) > MAX_LANES:
            self.lanes.pop()

    def _new_section(self) -> list:
        rng = self.np_random
        s = self.score
        section = []

        if not self._last_was_unsafe :
            if float(rng.random()) < 0.5:
                section.extend(self._make_road_group(s, rng))
            else:
                section.extend(self._make_river_group(s, rng))
            self._last_was_unsafe = True

        else :
            skip_grass = (float(rng.random()) < CONFIG.unsafe_prob.at(s))
            if not skip_grass:
                n_grass     = CONFIG.grass_lines.sample(s, rng)
                grass_lanes = [Lane(LaneType.GRASS, 0.0, rng, score=s) for _ in range(n_grass)]
                self._validate_grass_group(grass_lanes, rng)
                section.extend(grass_lanes)
                self._last_was_unsafe = False


            else :
                if float(rng.random()) < 0.5:
                    section.extend(self._make_road_group(s, rng))
                else:
                    section.extend(self._make_river_group(s, rng))
                self._last_was_unsafe = True

        return section

    def _make_road_group(self, score: float, rng) -> list:
        n     = CONFIG.road_riv_group_lines.sample(score, rng)
        lanes = []
        for _ in range(n):
            speed_cps = float(CONFIG.car_speed.sample(score, rng))
            direction = float(rng.choice([-1.0, 1.0]))
            lanes.append(Lane(LaneType.ROAD, speed_cps * direction, rng, score=score))
        return lanes

    def _make_river_group(self, score: float, rng) -> list:
        n       = CONFIG.road_riv_group_lines.sample(score, rng)
        water_p = CONFIG.water_prob.at(score)

        lanes: list[Lane]            = []
        last_water_dirs: list[float] = []

        for _ in range(n):
            if float(rng.random()) < water_p:
                speed_cps = float(CONFIG.log_speed.sample(score, rng))
                if (len(last_water_dirs) >= 2
                        and last_water_dirs[-1] == last_water_dirs[-2]):
                    direction = -last_water_dirs[-1]
                else:
                    direction = float(rng.choice([-1.0, 1.0]))
                last_water_dirs.append(direction)
                lanes.append(Lane(LaneType.WATER, speed_cps * direction, rng,
                                  score=score))
            else:
                last_water_dirs = []
                lanes.append(Lane(LaneType.LILY, 0.0, rng, score=score))

        lily_run: list[Lane] = []
        for lane in lanes:
            if lane.lane_type == LaneType.LILY:
                lily_run.append(lane)
            else:
                if len(lily_run) > 1:
                    self._validate_lily_connectivity(lily_run)
                lily_run = []
        if len(lily_run) > 1:
            self._validate_lily_connectivity(lily_run)

        return lanes

    # --- validation herbe (BFS sur colonnes jouables uniquement) -------------

    def _validate_grass_group(self, lanes: list, rng) -> None:
        """Garantit un chemin libre dans la zone jouable (colonnes 2–10).
        Si le BFS échoue, creuse une colonne aléatoire jouable (5 tentatives)."""
        if len(lanes) <= 1:
            return
        for _ in range(5):
            if self._bfs_grass(lanes):
                return
            col = int(rng.integers(PLAYABLE_MIN, PLAYABLE_MAX + 1))
            for lane in lanes:
                new_pos, new_wid = [], []
                for p, w in zip(lane._positions, lane._widths):
                    if int(p % GRID_W) != col:
                        new_pos.append(p)
                        new_wid.append(w)
                lane._positions, lane._widths = new_pos, new_wid

    def _bfs_grass(self, lanes: list) -> bool:
        """True s'il existe un chemin 4-connexe de la rangée 0 à n-1
        en restant dans la zone jouable (colonnes PLAYABLE_MIN..PLAYABLE_MAX)."""
        n     = len(lanes)
        queue = [(0, x) for x in range(PLAYABLE_MIN, PLAYABLE_MAX + 1)
                 if not lanes[0].has_obstacle_at(x)]
        if not queue:
            return False
        visited = set(queue)
        while queue:
            row, col = queue.pop(0)
            if row == n - 1:
                return True
            for dr, dc in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                nr = row + dr
                nc = col + dc
                if nc < PLAYABLE_MIN or nc > PLAYABLE_MAX:
                    continue    # mur latéral
                if 0 <= nr < n and (nr, nc) not in visited:
                    if not lanes[nr].has_obstacle_at(nc):
                        visited.add((nr, nc))
                        queue.append((nr, nc))
        return False

    # --- validation nénuphars (connectivité verticale, zone jouable) ---------

    def _validate_lily_connectivity(self, lily_lanes: list) -> None:
        """Pour chaque nénuphar de la ligne i, assure qu'il en existe un
        accessible (±1 colonne) sur la ligne i+1 dans la zone jouable."""
        for i in range(len(lily_lanes) - 1):
            a, b   = lily_lanes[i], lily_lanes[i + 1]
            b_cols = {int(p) for p, _ in b.iter_obstacles()
                      if PLAYABLE_MIN <= int(p) <= PLAYABLE_MAX}
            for pos, _ in a.iter_obstacles():
                col = int(pos)
                if col < PLAYABLE_MIN or col > PLAYABLE_MAX:
                    continue
                if not ({col - 1, col, col + 1} & b_cols):
                    target = max(PLAYABLE_MIN, min(PLAYABLE_MAX, col))
                    b._positions.append(float(target))
                    b._widths.append(1)
                    b_cols.add(target)

    def get_visible_lanes(self) -> list:
        start_idx = self.camera_start_row - self.deque_start_row
        lanes_list = list(self.lanes)
        return lanes_list[start_idx: start_idx + GRID_H]

    # --- buffer management ---------------------------------------------------

    def _trim_lanes(self) -> None:
        """Supprime les lignes sorties derrière la caméra (qui ne recule jamais)."""
        while self.deque_start_row < self.camera_start_row:
            self.lanes.popleft()
            self.deque_start_row += 1

    def _ensure_lanes(self) -> None:
        """Génère un nouveau module dès que la deque passe sous MIN_LANES."""
        while len(self.lanes) < MIN_LANES:
            for lane in self._new_section():
                self.lanes.append(lane)

    # --- actions -------------------------------------------------------------

    def _apply_action(self, action: int):
        dx, drow = _ACTION_DELTAS[action]

        if dx == 0 and drow == 0:
            return

        off  = self.deque_start_row
        lane = self.lanes[self.player_row - off]

        # Mouvement horizontal sur une bûche : grille relative à la bûche
        if dx != 0 and drow == 0 and lane.lane_type == LaneType.WATER:
            log = lane.get_log_at(self.player_x)
            if log is not None:
                log_start, log_width = log
                current_slot = int((self.player_x - log_start) % GRID_W)
                new_slot     = current_slot + dx
                if 0 <= new_slot < log_width:
                    self.player_x = (log_start + new_slot + 0.5) % GRID_W
                    return
                # Slot hors bûche : pas de déplacement

        # Mouvement normal sur la grille du jeu (limité à la zone jouable)
        current_cell_x = int(self.player_x)
        new_cell_x     = max(PLAYABLE_MIN, min(PLAYABLE_MAX, current_cell_x + dx))
        new_row        = max(self.camera_start_row, self.player_row + drow)
        new_idx        = new_row - off

        if new_idx < len(self.lanes):
            target = self.lanes[new_idx]
            if target.lane_type == LaneType.GRASS and target.has_obstacle_at(new_cell_x):
                return

        # Snap au centre du slot (bûche) ou de la case cible
        target_x = new_cell_x + 0.5
        if drow != 0 and new_idx < len(self.lanes):
            target_lane = self.lanes[new_idx]
            if target_lane.lane_type == LaneType.WATER:
                log = target_lane.get_log_at(target_x)
                if log is not None:
                    log_start, log_width = log
                    slot     = int((target_x - log_start) % GRID_W)
                    slot     = max(0, min(log_width - 1, slot))
                    target_x = (log_start + slot + 0.5) % GRID_W

        self.player_x         = target_x
        self.player_row       = new_row
        self.score            = max(self.score, self.player_row)
        self.camera_start_row = max(self.camera_start_row, self.player_row - LOOK_BEHIND)

    def _update_obstacles(self):
        """Met à jour uniquement les lignes visibles + marge devant."""
        player_idx = self.player_row - self.deque_start_row
        lo = max(0, player_idx - LOOK_BEHIND)
        hi = min(len(self.lanes), player_idx + LOOK_AHEAD + 5)
        for i in range(lo, hi):
            self.lanes[i].update()

    def _carry_player(self):
        """Déplace le joueur avec la bûche sur laquelle il se trouve."""
        lane = self.lanes[self.player_row - self.deque_start_row]
        if lane.lane_type == LaneType.WATER and lane.is_on_log(self.player_x):
            self.player_x += lane._speed
            self.player_x = max(-0.5, min(GRID_W - 0.5, self.player_x))

    # -------------------------------------------------------------------------

    def render(self):
        ...
