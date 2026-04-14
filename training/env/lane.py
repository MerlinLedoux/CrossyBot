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
    def __init__(self, lane_type: LaneType, speed: float, rng: np.random.Generator):
        self.lane_type   = lane_type
        self._speed      = float(np.clip(speed, -MAX_SPEED, MAX_SPEED))
        self._positions: list[float] = []
        self._widths:    list[int]   = []

        if lane_type == LaneType.GRASS:
            # Chaque cellule a ~25 % de chance d'avoir un arbre
            for x in range(GRID_W):
                if rng.random() < 0.25:
                    self._positions.append(float(x))
                    self._widths.append(1)

        elif lane_type == LaneType.LILY:
            # ~50 % des cellules ont un nénuphar, centrés sur la case
            for x in range(GRID_W):
                if rng.random() < 0.5:
                    self._positions.append(float(x))
                    self._widths.append(1)

        elif lane_type == LaneType.WATER:
            # Bûches de taille 2 ou 3 (variable par bûche), gap min 1 cellule
            log_widths: list[int] = []
            n_logs = int(rng.integers(1, 4))
            for _ in range(n_logs):
                log_widths.append(int(rng.integers(2, 4)))
            # Ajuster si trop de bûches pour tenir dans GRID_W avec gap min 1
            while n_logs > 1 and sum(log_widths) + n_logs > GRID_W:
                n_logs -= 1
                log_widths = log_widths[:n_logs]
            total_log = sum(log_widths)
            extra     = float(GRID_W - total_log - n_logs)   # espace au-delà du gap min
            weights   = rng.random(n_logs)
            weights   = weights / weights.sum() * extra
            gaps      = [1.0 + float(w) for w in weights]    # gap >= 1 cellule
            x = float(rng.uniform(0, GRID_W))
            for i in range(n_logs):
                self._positions.append(x % GRID_W)
                self._widths.append(log_widths[i])
                x += log_widths[i] + gaps[i]

        elif lane_type == LaneType.ROAD:
            car_width = int(rng.integers(1, 4))             # 1, 2 ou 3 cellules
            # Avec un gap min de 2, chaque voiture occupe car_width + 2 cellules
            max_cars  = max(1, GRID_W // (car_width + 2))
            n_cars    = int(rng.integers(1, max_cars + 1))

            # Distribuer l'espace résiduel aléatoirement entre les gaps
            # pour que l'espacement ne soit pas uniforme
            total_gap = float(GRID_W - n_cars * car_width)
            extra     = total_gap - n_cars * 2.0            # espace en plus du minimum
            weights   = rng.random(n_cars)
            weights   = weights / weights.sum() * extra
            gaps      = [2.0 + float(w) for w in weights]  # chaque gap >= 2.0

            x = float(rng.uniform(0, GRID_W))               # position de départ aléatoire
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
        """True si la hitbox du joueur (centrée en x_float) chevauche une bûche.
        Tient compte du wrap-around et de la position flottante du joueur."""
        hb      = 0.25   # demi-largeur hitbox (hitbox = 0.5 case)
        hb_start = x_float - hb
        hb_end   = x_float + hb
        for pos, width in zip(self._positions, self._widths):
            start = pos % GRID_W
            end   = start + width
            if end <= GRID_W:
                if start < hb_end and end > hb_start:
                    return True
            else:
                if start < hb_end and GRID_W > hb_start:   # segment [start, GRID_W)
                    return True
                if 0 < hb_end and end - GRID_W > hb_start: # segment [0, end-GRID_W)
                    return True
        return False

    def overlaps_cell(self, x: int, hitbox: float = 0.5) -> bool:
        """True si un obstacle chevauche la hitbox du joueur dans la cellule x.
        hitbox : fraction de la case occupée par le joueur (centrée).
        Avec hitbox=0.5 : zone de collision = [x+0.25, x+0.75]."""
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
                # Déborde : [start, GRID_W) et [0, end - GRID_W)
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
        """Avance les obstacles en temps réel pour le rendu visuel (play.py).
        cells_per_second : vitesse visuelle à vitesse normalisée maximale."""
        if not self._positions or self._speed == 0.0:
            return
        delta = (self._speed / MAX_SPEED) * cells_per_second * dt
        self._positions = [(p + delta) % GRID_W for p in self._positions]
