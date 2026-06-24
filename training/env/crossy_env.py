from collections import deque
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from .lane import Lane, LaneType, GRID_W, PLAYABLE_MIN, PLAYABLE_MAX
from .generation_config import CONFIG

GRID_H       = 13   # lignes visibles pour l'affichage (play.py)
LOOK_BEHIND  = 2    # recul max du joueur / caméra
LOOK_AHEAD   = 10
MIN_LANES    = 16
MAX_LANES    = 25

SCROLL_SPEED = 0.5   # rangées/seconde (identique à la version web)
SCROLL_STEP  = SCROLL_SPEED / 3.0   # avancement par step RL (STEPS_PER_SEC = 3.0)

# ── fenêtre d'observation de l'agent (plus petite que la fenêtre visuelle) ──
OBS_LOOK_BEHIND  = 1                              # 1 ligne derrière le joueur
OBS_LOOK_AHEAD   = 3                              # 3 lignes devant
OBS_LANES        = OBS_LOOK_BEHIND + 1 + OBS_LOOK_AHEAD   # = 5
OBS_PLAYABLE_W   = PLAYABLE_MAX - PLAYABLE_MIN + 1         # = 9 colonnes jouables
OBS_SIZE         = OBS_LANES * (OBS_PLAYABLE_W + 2) + 2   # = 5*11+2 = 57

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
            shape=(OBS_SIZE,),   # 5*11+2 = 57
            dtype=np.float32,
        )
        self.action_space = spaces.Discrete(5)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.player_x = float(GRID_W // 2) + 0.5   # centre = 6.5
        self.player_y = 2
        self.player_row = 0
        self.player_log_slot = None   # slot entier sur la bûche courante, None sinon
        self.deque_start_row = 0
        self.camera_start_row = 0   # ne recule jamais
        self.scroll_row = -3.0      # démarre 3 lignes en avance pour donner un buffer initial
        self.steps = 0
        self.score = 0
        self._last_was_unsafe = False

        self.lanes = deque()
        self._init_lanes()

        return self._get_observation(), {}

    def step(self, action):
        self.steps += 1
        prev_score = self.score

        self._apply_action(action)
        self._advance_scroll()
        self._trim_lanes()
        self._ensure_lanes()
        self._carry_player()
        self._update_obstacles()

        terminated = self._is_dead()
        truncated  = self.steps >= 1000

        if terminated:
            reward = -1.0
        elif self.score > prev_score:
            reward = float(self.score - prev_score)
        else:
            reward = -0.1

        return self._get_observation(), reward, terminated, truncated, {}

    # --- propriété utilitaire ------------------------------------------------

    @property
    def player_lane(self) -> Lane:
        return self.lanes[self.player_row - self.deque_start_row]

    # --- observation ---------------------------------------------------------

    def _get_observation(self):
        """
        Retourne le vecteur d'observation de taille OBS_SIZE = 57.

        Contenu :
          Pour chacune des OBS_LANES = 5 lanes autour du joueur
          (OBS_LOOK_BEHIND=1 derrière, joueur, OBS_LOOK_AHEAD=3 devant) :
            - type  (1 valeur)
            - speed (1 valeur)
            - occupation des OBS_PLAYABLE_W=9 colonnes jouables (colonnes 2..10)
          Puis : player_x normalisé, player_y normalisé
        """
        obs = []
        for lane in self._get_agent_lanes():
            obs.append(lane.type)
            obs.append(lane.speed)
            for x in range(PLAYABLE_MIN, PLAYABLE_MAX + 1):   # colonnes 2..10
                obs.append(lane.obstacle_at(x))
        # Position du joueur normalisée dans la zone jouable
        obs.append((self.player_x - PLAYABLE_MIN) / OBS_PLAYABLE_W)
        obs.append((self.player_row - self.camera_start_row) / GRID_H)
        return np.array(obs, dtype=np.float32)

    def _get_agent_lanes(self) -> list:
        """Retourne exactement OBS_LANES lanes centrées sur le joueur."""
        lanes_list  = list(self.lanes)
        player_idx  = self.player_row - self.deque_start_row
        start       = max(0, player_idx - OBS_LOOK_BEHIND)
        end         = start + OBS_LANES
        # Si on dépasse la fin de la deque, on recadre par la gauche
        if end > len(lanes_list):
            end   = len(lanes_list)
            start = max(0, end - OBS_LANES)
        return lanes_list[start:end]

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

        if self.lanes and section:
            self._validate_transition(self.lanes[-1], section[0])
        return section

    def _make_road_group(self, score: float, rng) -> list:
        n = CONFIG.road_riv_group_lines.sample(score, rng)
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
        """Garantit qu'au moins une colonne est partagée entre chaque paire
        de lignes de nénuphars consécutives. Ajoute au plus 1 nénuphar."""
        for i in range(len(lily_lanes) - 1):
            a, b   = lily_lanes[i], lily_lanes[i + 1]
            b_cols = {int(p) for p, _ in b.iter_obstacles()
                      if PLAYABLE_MIN <= int(p) <= PLAYABLE_MAX}
            connected = any(
                PLAYABLE_MIN <= int(pos) <= PLAYABLE_MAX and int(pos) in b_cols
                for pos, _ in a.iter_obstacles()
            )
            if not connected:
                for pos, _ in a.iter_obstacles():
                    col = int(pos)
                    if PLAYABLE_MIN <= col <= PLAYABLE_MAX:
                        b._positions.append(float(col))
                        b._widths.append(1)
                        break

    def _validate_transition(self, prev_lane, next_lane) -> None:
        """Garantit qu'une transition LILY→GRASS ou GRASS→LILY est franchissable."""
        is_grass = lambda t: t == LaneType.GRASS
        is_lily  = lambda t: t == LaneType.LILY

        if is_lily(prev_lane.lane_type) and is_grass(next_lane.lane_type):
            # Supprimer les arbres à toutes les positions des nénuphars
            for x in range(PLAYABLE_MIN, PLAYABLE_MAX + 1):
                if prev_lane.has_obstacle_at(x):
                    for idx, p in enumerate(next_lane._positions):
                        if int(p) == x:
                            next_lane._positions.pop(idx)
                            next_lane._widths.pop(idx)
                            break

        elif is_grass(prev_lane.lane_type) and is_lily(next_lane.lane_type):
            # S'assurer qu'au moins une colonne libre de l'herbe a un nénuphar
            for x in range(PLAYABLE_MIN, PLAYABLE_MAX + 1):
                if not prev_lane.has_obstacle_at(x) and next_lane.has_obstacle_at(x):
                    return
            for x in range(PLAYABLE_MIN, PLAYABLE_MAX + 1):
                if not prev_lane.has_obstacle_at(x):
                    next_lane._positions.append(float(x))
                    next_lane._widths.append(1)
                    return

    def get_visible_lanes(self) -> list:
        start_idx = self.camera_start_row - self.deque_start_row
        lanes_list = list(self.lanes)
        return lanes_list[start_idx: start_idx + GRID_H]

    # --- buffer management ---------------------------------------------------

    def _advance_scroll(self) -> None:
        """Avance le défilement automatique. Si le joueur est en avance,
        scroll_row se synchronise sur camera_start_row pour ne jamais le freiner."""
        self.scroll_row = max(self.scroll_row, float(self.camera_start_row))
        self.scroll_row += SCROLL_STEP
        new_cam = int(self.scroll_row)
        if new_cam > self.camera_start_row:
            self.camera_start_row = new_cam

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

        # Mouvement horizontal sur une ligne d'eau : déplacement d'un slot entier.
        # player_log_slot est l'entier exact, indépendant de la dérive du log.
        if dx != 0 and drow == 0 and lane.lane_type == LaneType.WATER:
            if self.player_log_slot is not None:
                log = lane.get_log_at(self.player_x)
                if log is not None:
                    log_start, log_width = log
                    new_slot = self.player_log_slot + dx
                    new_x    = log_start + new_slot + 0.5
                    if 0 <= new_slot < log_width:
                        # Toujours sur le même log
                        self.player_log_slot = new_slot
                        self.player_x        = new_x
                    else:
                        # Hors du log courant : vérifier s'il y a un log adjacent
                        adj_log = lane.get_log_at(new_x)
                        if adj_log is not None:
                            adj_start, adj_width = adj_log
                            adj_slot             = max(0, min(adj_width - 1,
                                                       int(new_x - adj_start)))
                            self.player_log_slot = adj_slot
                            self.player_x        = adj_start + adj_slot + 0.5
                        else:
                            # Pas de log : le joueur tombe à l'eau → mort
                            self.player_x       = new_x
                            self.player_log_slot = None
            return   # pas de mouvement vertical sur l'eau

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
        target_x   = new_cell_x + 0.5
        target_slot = None
        if drow != 0 and new_idx < len(self.lanes):
            target_lane = self.lanes[new_idx]
            if target_lane.lane_type == LaneType.WATER:
                log = target_lane.get_log_at(target_x)
                if log is not None:
                    log_start, log_width = log
                    slot        = max(0, min(log_width - 1, int(target_x - log_start)))
                    target_slot = slot
                    target_x    = log_start + slot + 0.5

        self.player_x         = target_x
        self.player_row       = new_row
        self.player_log_slot  = target_slot
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
        """Avance le joueur avec sa bûche en maintenant son slot exact.
        Appelé AVANT _update_obstacles : le log est encore à sa position actuelle.
        On prédit la position post-mouvement = log_start + speed + slot + 0.5."""
        if self.player_log_slot is None:
            return
        lane = self.lanes[self.player_row - self.deque_start_row]
        if lane.lane_type != LaneType.WATER:
            return
        log = lane.get_log_at(self.player_x)
        if log is None:
            return
        log_start, _ = log
        self.player_x = log_start + lane._speed + self.player_log_slot + 0.5
        self.player_x = max(-0.5, min(GRID_W - 0.5, self.player_x))

    # --- détection de mort ---------------------------------------------------

    def _is_dead(self) -> bool:
        if self.player_row < self.camera_start_row:
            return True
        if not (PLAYABLE_MIN <= self.player_x < PLAYABLE_MAX + 1):
            return True
        lane = self.player_lane
        if lane.lane_type == LaneType.ROAD:
            return lane.overlaps_cell(int(self.player_x), hitbox=0.5)
        if lane.lane_type == LaneType.LILY:
            return not lane.has_obstacle_at(int(self.player_x))
        if lane.lane_type == LaneType.WATER:
            return not lane.is_on_log(self.player_x)
        return False

    # -------------------------------------------------------------------------

    def render(self):
        ...
