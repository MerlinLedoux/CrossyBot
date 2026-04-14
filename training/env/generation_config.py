"""
generation_config.py — Paramètres de génération procédurale de CrossyBot.

Deux outils disponibles :

  Table(rows...)  — distribution discrète de probabilités par tranche de score.
      Chaque ligne : (score_min, score_max, {valeur: poids, ...})
      Les poids sont normalisés automatiquement (pas besoin qu'ils somment à 100).
      score_max = None signifie « jusqu'à l'infini ».

  Range(rows...)  — plage continue uniforme par tranche de score.
      Chaque ligne : (score_min, score_max, (min_val, max_val))
      .sample(score, rng) retourne un float aléatoire dans la plage.

  Prob(rows...)   — probabilité scalaire (float entre 0 et 1) par tranche de score.
      Chaque ligne : (score_min, score_max, probabilite)
      .at(score) retourne simplement la probabilité correspondante.

Exemples rapides :
    Table((0, 50, {1: 60, 2: 40}), (50, None, {1: 10, 2: 20, 3: 30, 4: 40}))
    Range((0, 100, (0.3, 0.6)),    (100, None, (0.5, 1.0)))
    Prob ((0,  50, 0.45),          (50,  None, 0.55))
"""

from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Outils
# ---------------------------------------------------------------------------

class Table:
    """Distribution de probabilités discrètes par tranche de score."""

    def __init__(self, *rows):
        """
        rows : (score_min, score_max, {valeur: poids})
               score_max = None → s'applique jusqu'à l'infini
        """
        self._rows = rows

    def _dist(self, score: float) -> dict:
        for smin, smax, dist in self._rows:
            if smin <= score and (smax is None or score < smax):
                return dist
        return self._rows[-1][2]   # fallback : dernière ligne

    def sample(self, score: float, rng) -> int:
        """Tire une valeur selon la distribution correspondant au score."""
        dist   = self._dist(score)
        values = list(dist.keys())
        total  = sum(dist.values())
        r      = float(rng.random()) * total
        cumul  = 0.0
        for v, w in zip(values, dist.values()):
            cumul += w
            if r <= cumul:
                return v
        return values[-1]


class Range:
    """Plage continue (min, max) avec échantillonnage uniforme, par tranche de score."""

    def __init__(self, *rows):
        """
        rows : (score_min, score_max, (min_val, max_val))
               score_max = None → jusqu'à l'infini
        """
        self._rows = rows

    def _get(self, score: float) -> tuple:
        for smin, smax, rng_vals in self._rows:
            if smin <= score and (smax is None or score < smax):
                return rng_vals
        return self._rows[-1][2]

    def sample(self, score: float, rng) -> float:
        """Retourne un float aléatoire dans la plage correspondant au score."""
        lo, hi = self._get(score)
        return float(rng.uniform(lo, hi))

    def at(self, score: float) -> tuple:
        """Retourne la plage (min, max) sans tirer de valeur."""
        return self._get(score)


class Prob:
    """Probabilité scalaire fixe (float 0–1) par tranche de score."""

    def __init__(self, *rows):
        """
        rows : (score_min, score_max, probabilite)
               score_max = None → jusqu'à l'infini
        """
        self._rows = rows

    def at(self, score: float) -> float:
        for smin, smax, val in self._rows:
            if smin <= score and (smax is None or score < smax):
                return val
        return self._rows[-1][2]


# ---------------------------------------------------------------------------
# Configuration — modifiez uniquement ce bloc
# ---------------------------------------------------------------------------

@dataclass
class WorldConfig:

    # =========================================================================
    # STRUCTURE DES SECTIONS
    # =========================================================================

    # Nombre de lignes d'herbe entre deux groupes
    grass_lines: Table = field(default_factory=lambda: Table(
        (0, None, {1: 60, 2: 40}),
    ))

    # Nombre de lignes consécutives dans un groupe de ROUTE
    road_group_lines: Table = field(default_factory=lambda: Table(
        (  0,  50,  {1: 60, 2: 40}),
        ( 50, 150,  {1: 20, 2: 30, 3: 30, 4: 20}),
        (150, None, {1: 10, 2: 15, 3: 20, 4: 20, 5: 20, 6: 15}),
    ))

    # Nombre de lignes consécutives dans un groupe de RIVIÈRE
    river_group_lines: Table = field(default_factory=lambda: Table(
        (  0,  50,  {1: 60, 2: 40}),
        ( 50, 150,  {1: 20, 2: 30, 3: 30, 4: 20}),
        (150, None, {1: 10, 2: 15, 3: 20, 4: 25, 5: 30}),
    ))

    # Probabilité que le prochain groupe soit une ROUTE (sinon rivière)
    road_prob: Prob = field(default_factory=lambda: Prob(
        (  0, 150, 0.45),
        (150, None, 0.50),
    ))

    # =========================================================================
    # ROUTE — voitures
    # =========================================================================

    # Vitesse des voitures (fraction de MAX_SPEED)
    car_speed: Range = field(default_factory=lambda: Range(
        (  0,  50,  (0.30, 0.55)),
        ( 50, 150,  (0.35, 0.70)),
        (150, 300,  (0.45, 0.85)),
        (300, None, (0.55, 1.00)),
    ))

    # Nombre de voitures sur la ligne
    car_count: Table = field(default_factory=lambda: Table(
        (  0,  50,  {1: 50, 2: 35, 3: 15}),
        ( 50, 150,  {1: 25, 2: 35, 3: 30, 4: 10}),
        (150, None, {1: 10, 2: 25, 3: 35, 4: 30}),
    ))

    # Écart minimum garanti entre deux voitures (en cases)
    car_min_gap: int = 2

    # =========================================================================
    # RIVIÈRE — composition de chaque ligne
    # =========================================================================

    # Probabilité qu'une ligne de rivière soit des BÛCHES (sinon nénuphars)
    water_prob: Prob = field(default_factory=lambda: Prob(
        (  0, 100, 0.60),
        (100, None, 0.70),
    ))

    # =========================================================================
    # BÛCHES
    # =========================================================================

    # Vitesse des bûches (fraction de MAX_SPEED)
    log_speed: Range = field(default_factory=lambda: Range(
        (  0,  50,  (0.20, 0.45)),
        ( 50, 150,  (0.25, 0.60)),
        (150, 300,  (0.35, 0.75)),
        (300, None, (0.45, 0.90)),
    ))

    # Largeur des bûches (en cases) — toutes les bûches d'une ligne ont la même largeur
    log_width: Table = field(default_factory=lambda: Table(
        (  0, 100, {2: 50, 3: 35, 4: 15}),
        (100, None, {2: 30, 3: 40, 4: 30}),
    ))

    # Couverture minimale de la ligne par des bûches (fraction 0–1)
    log_coverage_min: float = 0.40

    # =========================================================================
    # NÉNUPHARS
    # =========================================================================

    # Nombre de nénuphars par ligne
    lily_count: Table = field(default_factory=lambda: Table(
        (0, None, {2: 50, 3: 50}),
    ))

    # =========================================================================
    # HERBE
    # =========================================================================

    # Probabilité d'arbre par case d'herbe
    tree_probability: float = 0.25


CONFIG = WorldConfig()
