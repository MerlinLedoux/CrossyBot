import gymnasium as gym
from gymnasium import spaces
import numpy as np
from .lane import Lane, LaneType, GRID_W
from .generation_config import CONFIG

GRID_H      = 10
LOOK_BEHIND = 1
BUFFER_BEHIND = LOOK_BEHIND + 2   # lignes conservées derrière la caméra

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

        self.player_x         = float(GRID_W // 2) + 0.5
        self.player_row       = 0
        self.camera_row       = 0
        self.steps            = 0
        self.score            = 0
        self._last_was_unsafe = False
        self._lane_offset     = 0      # nombre de lignes supprimées en bas du buffer

        self.lanes = self._generate_lanes()

        return self._get_observation(), {}

    def step(self, action):
        self.steps += 1

        self._apply_action(action)
        self._update_obstacles()
        self._carry_player()

        self._trim_lanes()

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
        s       = self.score
        section = []

        # herbe : toujours ajoutée, sauf si on enchaîne deux zones dangereuses ---
        # unsafe_prob = probabilité de sauter le tampon d'herbe après une zone dangereuse
        skip_grass = (self._last_was_unsafe
                      and float(rng.random()) < CONFIG.unsafe_prob.at(s))
        if not skip_grass:
            n_grass     = CONFIG.grass_lines.sample(s, rng)
            grass_lanes = [Lane(LaneType.GRASS, 0.0, rng, score=s)
                           for _ in range(n_grass)]
            self._validate_grass_group(grass_lanes, rng)
            section.extend(grass_lanes)

        # --- groupe route ou rivière (50/50) ---
        if float(rng.random()) < 0.5:
            section.extend(self._make_road_group(s, rng))
        else:
            section.extend(self._make_river_group(s, rng))

        self._last_was_unsafe = True
        return section

    def _make_road_group(self, score: float, rng) -> list:
        """Groupe de lignes de route. Chaque ligne a sa propre vitesse et direction."""
        n     = CONFIG.road_riv_group_lines.sample(score, rng)
        lanes = []
        for _ in range(n):
            speed_cps = float(CONFIG.car_speed.sample(score, rng))
            direction = float(rng.choice([-1.0, 1.0]))
            lanes.append(Lane(LaneType.ROAD, speed_cps * direction, rng, score=score))
        return lanes

    def _make_river_group(self, score: float, rng) -> list:
        """Groupe de lignes de rivière (bûches ou nénuphars, choix par ligne).
        Chaque ligne WATER a sa propre vitesse et direction.
        Interdit : 3 lignes WATER consécutives dans la même direction."""
        n       = CONFIG.road_riv_group_lines.sample(score, rng)
        water_p = CONFIG.water_prob.at(score)

        lanes: list[Lane]            = []
        last_water_dirs: list[float] = []

        for _ in range(n):
            if float(rng.random()) < water_p:
                speed_cps = float(CONFIG.log_speed.sample(score, rng))
                # Forcer un changement si les 2 dernières WATER étaient dans le même sens
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

        # Connectivité verticale entre lignes de nénuphars consécutives
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

    # --- validation herbe (BFS) ----------------------------------------------

    def _validate_grass_group(self, lanes: list, rng) -> None:
        """Garantit un chemin libre de bas en haut dans le groupe d'herbe.
        Si le BFS échoue, creuse une colonne aléatoire (jusqu'à 5 tentatives)."""
        if len(lanes) <= 1:
            return
        for _ in range(5):
            if self._bfs_grass(lanes):
                return
            col = int(rng.integers(0, GRID_W))
            for lane in lanes:
                new_pos, new_wid = [], []
                for p, w in zip(lane._positions, lane._widths):
                    if int(p % GRID_W) != col:
                        new_pos.append(p)
                        new_wid.append(w)
                lane._positions, lane._widths = new_pos, new_wid

    def _bfs_grass(self, lanes: list) -> bool:
        """True s'il existe un chemin 4-connexe de la rangée 0 à la rangée n-1."""
        n     = len(lanes)
        queue = [(0, x) for x in range(GRID_W) if not lanes[0].has_obstacle_at(x)]
        if not queue:
            return False
        visited = set(queue)
        while queue:
            row, col = queue.pop(0)
            if row == n - 1:
                return True
            for dr, dc in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                nr = row + dr
                nc = (col + dc) % GRID_W        # wrap horizontal
                if 0 <= nr < n and (nr, nc) not in visited:
                    if not lanes[nr].has_obstacle_at(nc):
                        visited.add((nr, nc))
                        queue.append((nr, nc))
        return False

    # --- validation nénuphars (connectivité verticale) -----------------------

    def _validate_lily_connectivity(self, lily_lanes: list) -> None:
        """Pour chaque nénuphar de la ligne i, assure qu'il en existe un
        accessible (±1 colonne) sur la ligne i+1."""
        for i in range(len(lily_lanes) - 1):
            a, b   = lily_lanes[i], lily_lanes[i + 1]
            b_cols = {int(p) for p, _ in b.iter_obstacles()}
            for pos, _ in a.iter_obstacles():
                col = int(pos)
                if not ({col - 1, col, col + 1} & b_cols):
                    target = max(0, min(GRID_W - 1, col))
                    b._positions.append(float(target))
                    b._widths.append(1)
                    b_cols.add(target)

    def get_visible_lanes(self) -> list:
        end = self.camera_row + GRID_H
        while self._lane_offset + len(self.lanes) < end:
            self.lanes.extend(self._new_section())
        lo = self.camera_row - self._lane_offset
        return self.lanes[lo: lo + GRID_H]

    def _trim_lanes(self) -> None:
        """Supprime les lignes trop loin derrière la caméra."""
        keep_from = max(0, self.camera_row - BUFFER_BEHIND)
        n_drop    = keep_from - self._lane_offset
        if n_drop > 0:
            del self.lanes[:n_drop]
            self._lane_offset += n_drop

    # --- actions -------------------------------------------------------------

    def _apply_action(self, action: int):
        dx, drow = _ACTION_DELTAS[action]

        if dx == 0 and drow == 0:
            return   # rester : la bûche continue de porter le joueur

        off  = self._lane_offset
        lane = self.lanes[self.player_row - off]

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
                # Slot hors bûche : chute dans l'eau, pas de déplacement

        # Mouvement normal sur la grille du jeu
        current_cell_x = int(self.player_x)
        new_cell_x     = max(0, min(GRID_W - 1, current_cell_x + dx))
        new_row        = max(self.camera_row, self.player_row + drow)
        new_idx        = new_row - off

        if new_idx < len(self.lanes):
            target = self.lanes[new_idx]
            if target.lane_type == LaneType.GRASS and target.has_obstacle_at(new_cell_x):
                return

        # Snap au centre du slot cible (bûche) ou de la case cible
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

        self.player_x   = target_x
        self.player_row = new_row
        self.score      = max(self.score, self.player_row)
        self.camera_row = max(self.camera_row, self.player_row - LOOK_BEHIND)

    def _update_obstacles(self):
        """Met à jour uniquement les lignes visibles + marge devant."""
        lo = max(0, self.camera_row - self._lane_offset)
        hi = min(len(self.lanes), lo + GRID_H + 5)
        for lane in self.lanes[lo:hi]:
            lane.update()

    def _carry_player(self):
        """Déplace le joueur avec la bûche sur laquelle il se trouve (entraînement RL)."""
        lane = self.lanes[self.player_row - self._lane_offset]
        if lane.lane_type == LaneType.WATER and lane.is_on_log(self.player_x):
            self.player_x += lane._speed
            self.player_x = max(-0.5, min(GRID_W - 0.5, self.player_x))

    # -------------------------------------------------------------------------

    def render(self):
        ...
