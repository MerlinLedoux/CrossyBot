import numpy as np
from enum import IntEnum

GRID_W        = 9
MAX_SPEED     = 0.5   # déplacement max en cellules/step (RL)
CELLS_PER_SEC = 3.0   # vitesse visuelle maximale en cases/seconde

class LaneType(IntEnum):
    SAFE  = 0
    GRASS = 1
    ROAD  = 2
    WATER = 3
    LILY  = 4

_TYPE_TO_OBS = {
    LaneType.SAFE:  -1.00,
    LaneType.GRASS: -0.50,
    LaneType.ROAD:   0.00,
    LaneType.WATER:  0.50,
    LaneType.LILY:   1.00,
}

class Lane:
    """
    speed_cps : vitesse signée en cases/seconde (visuel).
                Positif = droite, négatif = gauche.
    score     : score courant du joueur.
    """

    def __init__(self, lane_type: LaneType, speed_cps: float, rng: np.random.Generator, score: float = 0):
        from .generation_config import CONFIG

        self.lane_type = lane_type

        # Conversion cases/seconde → déplacement par step RL
        self._speed = float(np.clip(
            (speed_cps / CELLS_PER_SEC) * MAX_SPEED,
            -MAX_SPEED, MAX_SPEED,
        ))

        self._positions: list[float] = []
        self._widths:    list[int]   = []

        # --- HERBE ---
        if lane_type == LaneType.GRASS:
            n_trees = min(CONFIG.tree_count.sample(score, rng), GRID_W)
            if n_trees > 0:
                xs = sorted(rng.choice(GRID_W, size=n_trees, replace=False).tolist())
                for x in xs:
                    self._positions.append(float(x))
                    self._widths.append(1)

        # --- NÉNUPHARS ---
        elif lane_type == LaneType.LILY:
            n_pads = min(CONFIG.lily_count.sample(score, rng), GRID_W)
            if n_pads > 0:
                xs = sorted(rng.choice(GRID_W, size=n_pads, replace=False).tolist())
                for x in xs:
                    self._positions.append(float(x))
                    self._widths.append(1)

        # --- BÛCHES (largeur variable par bûche) ---
        elif lane_type == LaneType.WATER:
            min_gap = int(min(CONFIG.log_space._dist(score).keys()))
            # Estimer n_logs depuis la largeur moyenne et le gap minimum
            log_dist = CONFIG.log_size._dist(score)
            avg_w    = sum(k * v for k, v in log_dist.items()) / sum(log_dist.values())
            n_max    = max(1, int(GRID_W // (avg_w + max(min_gap, 1))))
            n_logs   = int(rng.integers(1, n_max + 1))
            # Echantillonner les largeurs, réduire si elles dépassent la grille
            log_widths = [CONFIG.log_size.sample(score, rng) for _ in range(n_logs)]
            while sum(log_widths) >= GRID_W and len(log_widths) > 1:
                log_widths.pop()
            n_logs = len(log_widths)
            # Distribution des gaps : total = GRID_W - sum(widths), chaque gap >= min_gap
            total_gap = float(GRID_W - sum(log_widths))
            residual  = max(0.0, total_gap - n_logs * min_gap)
            raw  = [float(CONFIG.log_space.sample(score, rng)) for _ in range(n_logs)]
            s    = sum(raw) if sum(raw) > 0 else 1.0
            gaps = [float(min_gap) + r / s * residual for r in raw]
            # Placement avec offset circulaire entier
            offset = int(rng.integers(0, GRID_W))
            x = 0.0
            for lw, g in zip(log_widths, gaps):
                self._positions.append(float((x + offset) % GRID_W))
                self._widths.append(lw)
                x += lw + g

        # --- ROUTE (même largeur pour toutes les voitures de la ligne) ---
        elif lane_type == LaneType.ROAD:
            car_width = CONFIG.car_size.sample(score, rng)
            min_gap   = int(min(CONFIG.car_space._dist(score).keys()))
            # Nombre de voitures qui tiennent en circulaire avec gap minimum
            n_max  = max(1, GRID_W // (car_width + max(min_gap, 1)))
            n_cars = int(rng.integers(1, n_max + 1))
            # Distribution des gaps : total = GRID_W - n*car_width, chaque gap >= min_gap
            total_gap = float(GRID_W - n_cars * car_width)
            residual  = max(0.0, total_gap - n_cars * min_gap)
            raw  = [float(CONFIG.car_space.sample(score, rng)) for _ in range(n_cars)]
            s    = sum(raw) if sum(raw) > 0 else 1.0
            gaps = [float(min_gap) + r / s * residual for r in raw]
            # Placement avec offset circulaire entier
            offset = int(rng.integers(0, GRID_W))
            x = 0.0
            for g in gaps:
                self._positions.append(float((x + offset) % GRID_W))
                self._widths.append(car_width)
                x += car_width + g

    # --- propriétés observation RL ----------------------------------------

    @property
    def type(self) -> float:
        return _TYPE_TO_OBS[self.lane_type]

    @property
    def speed(self) -> float:
        return self._speed / MAX_SPEED

    # --- tests de collision ------------------------------------------------

    def has_obstacle_at(self, x: int) -> bool:
        for pos, width in zip(self._positions, self._widths):
            for i in range(width):
                if int((pos + i) % GRID_W) == x:
                    return True
        return False

    def get_log_at(self, x_float: float):
        """Retourne (log_start_effectif, width) du log sous le joueur, ou None."""
        hb       = 0.25
        hb_start = x_float - hb
        hb_end   = x_float + hb
        for pos, width in zip(self._positions, self._widths):
            start = pos % GRID_W
            end   = start + width
            if end <= GRID_W:
                if start < hb_end and end > hb_start:
                    return (start, width)
            else:
                if start < hb_end and GRID_W > hb_start:
                    return (start, width)
                if 0 < hb_end and end - GRID_W > hb_start:
                    return (start - GRID_W, width)
        return None

    def is_on_log(self, x_float: float) -> bool:
        hb       = 0.25
        hb_start = x_float - hb
        hb_end   = x_float + hb
        for pos, width in zip(self._positions, self._widths):
            start = pos % GRID_W
            end   = start + width
            if end <= GRID_W:
                if start < hb_end and end > hb_start:
                    return True
            else:
                if start < hb_end and GRID_W > hb_start:
                    return True
                if 0 < hb_end and end - GRID_W > hb_start:
                    return True
        return False

    def overlaps_cell(self, x: int, hitbox: float = 0.5) -> bool:
        margin   = (1.0 - hitbox) / 2.0
        hb_start = x + margin
        hb_end   = x + 1.0 - margin
        for pos, width in zip(self._positions, self._widths):
            start = pos % GRID_W
            end   = start + width
            if end <= GRID_W:
                if start < hb_end and end > hb_start:
                    return True
            else:
                if start < hb_end:
                    return True
                if end - GRID_W > hb_start:
                    return True
        return False

    def iter_obstacles(self):
        yield from zip(self._positions, self._widths)

    # --- mise à jour des positions -----------------------------------------

    def update(self):
        """Avance les obstacles d'un step RL."""
        self._positions = [(p + self._speed) % GRID_W for p in self._positions]

    def update_visual(self, dt: float):
        """Avance les obstacles en temps réel pour play.py.
        Utilise directement _speed qui encode déjà le ratio cases/seconde."""
        if not self._positions or self._speed == 0.0:
            return
        # _speed = (speed_cps / CELLS_PER_SEC) * MAX_SPEED
        # → delta = (_speed / MAX_SPEED) * CELLS_PER_SEC * dt = speed_cps * dt
        delta = (self._speed / MAX_SPEED) * CELLS_PER_SEC * dt
        self._positions = [(p + delta) % GRID_W for p in self._positions]
