import numpy as np
from enum import IntEnum

GRID_W = 9
MAX_SPEED = 0.5  # déplacement max en cellules/step

class LaneType(IntEnum):
    SAFE  = 0  # zone de départ, toujours sûre
    GRASS = 1  # herbe, obstacles statiques non-létaux
    ROAD  = 2  # voitures mobiles
    WATER = 3  # bûches mobiles (le joueur doit être sur une bûche)
    LILY  = 4  # nénuphars statiques de taille 1 (le joueur doit être sur un nénuphar)

_TYPE_TO_OBS = {
    LaneType.SAFE:  -1.00,
    LaneType.GRASS: -0.50,
    LaneType.ROAD:   0.00,
    LaneType.WATER:  0.50,
    LaneType.LILY:   1.00,
}

class Lane:
    def __init__(self, lane_type: LaneType, speed: float, rng: np.random.Generator):
        """
        lane_type : type de la lane
        speed     : déplacement en cellules/step (négatif = vers la gauche)
        rng       : numpy Generator pour la reproductibilité
        """
        self.lane_type = lane_type
        self._speed = float(np.clip(speed, -MAX_SPEED, MAX_SPEED))
        self._positions: list[float] = []  # position x du bord gauche de chaque obstacle
        self._widths: list[int] = []       # largeur en cellules de chaque obstacle

        if lane_type in (LaneType.ROAD, LaneType.WATER, LaneType.LILY, LaneType.GRASS):
            self._generate_obstacles(rng)

    @property
    def type(self) -> float:
        return _TYPE_TO_OBS[self.lane_type]

    @property
    def obstacle_is_lethal(self) -> bool:
        """
        ROAD        : un obstacle (voiture) tue le joueur.
        WATER / LILY: l'absence d'obstacle (pas de bûche / nénuphar) tue le joueur.
        GRASS       : un obstacle bloque le déplacement mais ne tue pas.
        """
        return self.lane_type == LaneType.ROAD

    @property
    def speed(self) -> float:
        """Vitesse normalisée dans [-1, 1]."""
        return self._speed / MAX_SPEED

    # --- logique de jeu ---

    def has_obstacle_at(self, x: int) -> bool:
        """True si un obstacle occupe la colonne x (gère le wrap autour de la grille)."""
        for pos, width in zip(self._positions, self._widths):
            for i in range(width):
                if int((pos + i) % GRID_W) == x:
                    return True
        return False

    def update(self):
        """Avance tous les obstacles d'un step."""
        self._positions = [(p + self._speed) % GRID_W for p in self._positions]

    # --- génération interne ---

    def _generate_obstacles(self, rng: np.random.Generator):
        if self.lane_type == LaneType.ROAD:
            n = int(rng.integers(1, 4))              # 1–3 voitures
            car_width = int(rng.integers(1, 3))      # largeur identique pour toutes les voitures de la lane
            widths = [car_width] * n
        elif self.lane_type == LaneType.WATER:
            n = int(rng.integers(2, 4))              # 2–3 bûches
            widths = [int(rng.integers(1, 5)) for _ in range(n)]  # largeur individuelle 1–4
        elif self.lane_type == LaneType.LILY:
            n = int(rng.integers(2, 6))              # 2–5 nénuphars
            widths = [1] * n                         # toujours taille 1
        else:  # GRASS
            n = int(rng.integers(0, 4))              # 0–3 obstacles statiques (arbres, rochers)
            widths = [1] * n                         # largeur 1 cellule

        if n == 0:
            return

        # Positions réparties régulièrement + légère variation aléatoire
        spacing = GRID_W / n
        for i in range(n):
            x = (spacing * i + rng.uniform(0, spacing * 0.5)) % GRID_W
            self._positions.append(float(x))
            self._widths.append(widths[i])
