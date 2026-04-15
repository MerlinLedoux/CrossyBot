import numpy as np
from enum import IntEnum
from .generation_config import CONFIG

GRID_W        = 13
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

    WATER et ROAD utilisent un flux continu d'objets indépendants :
    chaque objet a sa propre largeur (log_size / car_size) et chaque
    espacement est tiré indépendamment (log_space / car_space).
    Les objets entrent d'un côté de l'écran et sortent de l'autre
    sans jamais se répéter.
    """

    def __init__(self, lane_type: LaneType, speed_cps: float,
                 rng: np.random.Generator, score: float = 0):
        self.lane_type = lane_type

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

        # --- BÛCHES et VOITURES : flux continu ---
        elif lane_type in (LaneType.WATER, LaneType.ROAD):
            self._rng   = rng
            self._score = score
            self._generate_stream()

    # --- génération du flux ---------------------------------------------------

    def _sample_gap(self) -> float:
        if self.lane_type == LaneType.WATER:
            return float(CONFIG.log_space.sample(self._score, self._rng))
        return float(CONFIG.car_space.sample(self._score, self._rng))

    def _sample_width(self) -> int:
        if self.lane_type == LaneType.WATER:
            return CONFIG.log_size.sample(self._score, self._rng)
        return CONFIG.car_size.sample(self._score, self._rng)

    def _generate_stream(self) -> None:
        """Remplit le flux initial : objets de -GRID_W à 2*GRID_W."""
        x = -float(GRID_W) + self._sample_gap()
        while x < 2 * GRID_W:
            width = self._sample_width()
            self._positions.append(float(x))
            self._widths.append(width)
            x += width + self._sample_gap()

    def _trim_and_fill(self, moving_right: bool) -> None:
        """Supprime les objets sortis de l'écran et génère les suivants
        côté entrant pour maintenir un buffer hors-écran d'un GRID_W."""
        if moving_right:
            # sortis à droite
            while self._positions and self._positions[-1] >= GRID_W:
                self._positions.pop()
                self._widths.pop()
            # entrent par la gauche
            while not self._positions or self._positions[0] > -GRID_W:
                gap   = self._sample_gap()
                width = self._sample_width()
                left  = (self._positions[0] - gap - width) if self._positions \
                        else -float(GRID_W)
                self._positions.insert(0, float(left))
                self._widths.insert(0, width)
        else:
            # sortis à gauche
            while self._positions and self._positions[0] + self._widths[0] <= 0:
                self._positions.pop(0)
                self._widths.pop(0)
            # entrent par la droite
            while not self._positions or \
                    self._positions[-1] + self._widths[-1] < 2 * GRID_W:
                gap   = self._sample_gap()
                width = self._sample_width()
                right = (self._positions[-1] + self._widths[-1]) if self._positions \
                        else float(GRID_W)
                self._positions.append(float(right + gap))
                self._widths.append(width)

    # --- propriétés observation RL -------------------------------------------

    @property
    def type(self) -> float:
        return _TYPE_TO_OBS[self.lane_type]

    @property
    def speed(self) -> float:
        return self._speed / MAX_SPEED

    # --- tests de collision --------------------------------------------------
    # Les positions sont désormais absolues (pas de % GRID_W).
    # Un objet à pos avec largeur width occupe [pos, pos+width).

    def has_obstacle_at(self, x: int) -> bool:
        for pos, width in zip(self._positions, self._widths):
            if pos < x + 1 and pos + width > x:
                return True
        return False

    def get_log_at(self, x_float: float):
        """Retourne (pos, width) du log sous le joueur, ou None."""
        hb = 0.25
        for pos, width in zip(self._positions, self._widths):
            if pos < x_float + hb and pos + width > x_float - hb:
                return (pos, width)
        return None

    def is_on_log(self, x_float: float) -> bool:
        hb = 0.25
        for pos, width in zip(self._positions, self._widths):
            if pos < x_float + hb and pos + width > x_float - hb:
                return True
        return False

    def overlaps_cell(self, x: int, hitbox: float = 0.5) -> bool:
        margin   = (1.0 - hitbox) / 2.0
        hb_start = x + margin
        hb_end   = x + 1.0 - margin
        for pos, width in zip(self._positions, self._widths):
            if pos < hb_end and pos + width > hb_start:
                return True
        return False

    def iter_obstacles(self):
        yield from zip(self._positions, self._widths)

    # --- mise à jour des positions -------------------------------------------

    def update(self):
        """Avance les obstacles d'un step RL."""
        self._positions = [p + self._speed for p in self._positions]
        if self.lane_type in (LaneType.WATER, LaneType.ROAD):
            self._trim_and_fill(self._speed > 0)

    def update_visual(self, dt: float):
        """Avance les obstacles en temps réel pour play.py."""
        if not self._positions or self._speed == 0.0:
            return
        delta = (self._speed / MAX_SPEED) * CELLS_PER_SEC * dt
        self._positions = [p + delta for p in self._positions]
        if self.lane_type in (LaneType.WATER, LaneType.ROAD):
            self._trim_and_fill(delta > 0)
