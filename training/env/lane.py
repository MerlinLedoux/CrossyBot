import numpy as np
from enum import IntEnum

GRID_W    = 9
MAX_SPEED = 0.5  # déplacement max en cellules/step

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
    def __init__(self, lane_type: LaneType, speed: float, rng: np.random.Generator,
                 max_cars: int = None,
                 log_width: int = None,
                 log_coverage_min: float = 0.40,
                 tree_probability: float = 0.25,
                 lily_count: tuple = (2, 3)):
        self.lane_type   = lane_type
        self._speed      = float(np.clip(speed, -MAX_SPEED, MAX_SPEED))
        self._positions: list[float] = []
        self._widths:    list[int]   = []

        if lane_type == LaneType.GRASS:
            for x in range(GRID_W):
                if rng.random() < tree_probability:
                    self._positions.append(float(x))
                    self._widths.append(1)

        elif lane_type == LaneType.LILY:
            n_pads   = int(rng.integers(lily_count[0], lily_count[1] + 1))
            shuffled = list(rng.permutation(GRID_W))
            positions: list[int] = []
            for x in shuffled:
                if len(positions) == n_pads:
                    break
                if all(abs(x - p) > 1 for p in positions):
                    positions.append(int(x))
            # Compléter sans contrainte si pas assez de cases disponibles (cas rare)
            for x in shuffled:
                if len(positions) >= n_pads:
                    break
                if int(x) not in positions:
                    positions.append(int(x))
            for x in sorted(positions):
                self._positions.append(float(x))
                self._widths.append(1)

        elif lane_type == LaneType.WATER:
            # Toutes les bûches d'une ligne ont la même largeur (cohérence visuelle)
            lw = log_width if log_width is not None else int(rng.integers(2, 5))  # 2, 3 ou 4

            # Couverture minimale configurable
            min_logs = max(1, int(np.ceil(log_coverage_min * GRID_W / lw)))
            # Max bûches tenant avec gap min 1 (grille circulaire)
            max_logs = min(3, max(1, GRID_W // (lw + 1)))
            if min_logs > max_logs:
                min_logs = max_logs
            n_logs = int(rng.integers(min_logs, max_logs + 1))

            total_gap = float(GRID_W - n_logs * lw)
            extra     = max(0.0, total_gap - float(n_logs))  # au-delà du gap min=1
            weights   = rng.random(n_logs)
            weights   = weights / weights.sum() * extra
            gaps      = [1.0 + float(w) for w in weights]

            x = float(rng.uniform(0, GRID_W))
            for i in range(n_logs):
                self._positions.append(x % GRID_W)
                self._widths.append(lw)
                x += lw + gaps[i]

        elif lane_type == LaneType.ROAD:
            car_width    = int(rng.integers(1, 4))             # 1, 2 ou 3 cases
            # Gap min 2 cases entre voitures
            internal_max = max(1, GRID_W // (car_width + 2))
            if max_cars is not None:
                internal_max = min(internal_max, max_cars)
            n_cars = int(rng.integers(1, internal_max + 1))

            total_gap = float(GRID_W - n_cars * car_width)
            extra     = max(0.0, total_gap - n_cars * 2.0)
            weights   = rng.random(n_cars)
            weights   = weights / weights.sum() * extra
            gaps      = [2.0 + float(w) for w in weights]

            x = float(rng.uniform(0, GRID_W))
            for i in range(n_cars):
                self._positions.append(x % GRID_W)
                self._widths.append(car_width)
                x += car_width + gaps[i]

    @property
    def type(self) -> float:
        return _TYPE_TO_OBS[self.lane_type]

    @property
    def speed(self) -> float:
        return self._speed / MAX_SPEED

    def has_obstacle_at(self, x: int) -> bool:
        for pos, width in zip(self._positions, self._widths):
            for i in range(width):
                if int((pos + i) % GRID_W) == x:
                    return True
        return False

    def get_log_at(self, x_float: float):
        """Retourne (log_start_effectif, width) du log dont la hitbox du joueur chevauche.
        Le start effectif peut être négatif si le log wrappe et le joueur est dans la
        partie wrappée (ex: log à pos=8.5, joueur à x=0.3 → start_effectif = -0.5).
        Retourne None si aucun log trouvé."""
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
                # Segment 1 : [start, GRID_W) — start effectif = start (positif)
                if start < hb_end and GRID_W > hb_start:
                    return (start, width)
                # Segment 2 : [0, end-GRID_W) — start effectif = start - GRID_W (négatif)
                if 0 < hb_end and end - GRID_W > hb_start:
                    return (start - GRID_W, width)
        return None

    def is_on_log(self, x_float: float) -> bool:
        """True si la hitbox du joueur (centrée en x_float) chevauche une bûche."""
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
        """True si un obstacle chevauche la hitbox du joueur dans la cellule x."""
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

    def update(self):
        """Avance les obstacles d'un step (utilisé par l'entraînement RL)."""
        self._positions = [(p + self._speed) % GRID_W for p in self._positions]

    def update_visual(self, dt: float, cells_per_second: float = 3.0):
        """Avance les obstacles en temps réel pour le rendu visuel (play.py)."""
        if not self._positions or self._speed == 0.0:
            return
        delta = (self._speed / MAX_SPEED) * cells_per_second * dt
        self._positions = [(p + delta) % GRID_W for p in self._positions]
