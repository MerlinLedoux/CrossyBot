"""
generation_config.py — Paramètres de génération procédurale de CrossyBot. 
La génération de l'environement est très importante pour reproduire le jeux.
Si la vitesse/densité des vehicules et des buches n'est pas bonne la dinamique du jeu n'est pas reproduite. 

Deux outils disponibles :

  Table(rows...)  — distribution discrète de probabilités par tranche de score.
      Chaque ligne : (score_min, score_max, {valeur: poids, ...})
      Les poids sont normalisés automatiquement (pas besoin qu'ils somment à 100).
      score_max = None signifie « jusqu'à l'infini ».

  Prob(rows...)   — probabilité scalaire (float entre 0 et 1) par tranche de score.
      Chaque ligne : (score_min, score_max, probabilite)
      .at(score) retourne simplement la probabilité correspondante.

Exemples rapides :
    Table((0, 50, {1: 60, 2: 40}), (50, None, {1: 10, 2: 20, 3: 30, 4: 40}))
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


# Configuration
@dataclass
class WorldConfig:

    # ==================================================================================================================================================
    # STRUCTURE DES SECTIONS

    # Taille des sections d'herbe
    grass_lines: Table = field(default_factory=lambda: Table(
        (0, None, {1: 40, 2: 30, 3: 20, 4: 10}),
    ))

    # Taille des sections de route et d'eau
    road_riv_group_lines: Table = field(default_factory=lambda: Table(
        (0, 50, {1: 20, 2: 40, 3: 20, 4: 20}),
        (50, 100, {1: 10, 2: 15, 3: 25, 4: 25, 5: 15, 6: 10}),
        (100, 200, {1: 5, 2: 10, 3: 15, 4: 20, 5: 20, 6: 15, 7: 10, 8: 5}),
        (200, None, {3: 5, 4: 10, 5: 15, 6: 20, 7: 20, 8: 15, 9: 10, 10: 5}),
    ))

    # Probabilité qu'aprés une section unsafe la section suivante soit unsafe
    # Les section unsafe sont les section de route et d'eau a partir de 50 de score on peut avoir route puis eau puis route à nouveau.
    unsafe_prob: Prob = field(default_factory=lambda: Prob(
        (0, 50, 0.0),
        (50, 100, 0.25),
        (100, 200, 0.50),
        (200, None, 0.75),
    ))

    # ==================================================================================================================================================
    # ROUTE — voitures
    # Vitesse des voitures
    car_speed: Table = field(default_factory=lambda: Table(
        (0, 50, {0.25: 10, 0.50: 10, 0.75: 20, 1.00: 30, 1.25: 20, 1.50: 10}),
        (50, 100, {0.25: 10, 0.50: 10, 0.75: 15, 1.00: 15, 1.25: 15, 1.50: 15, 1.75: 10, 2.00: 10}),
        (100, 200, {0.25: 5, 0.50: 5, 0.75: 10, 1.00: 15, 1.25: 20, 1.50: 20, 1.75: 15, 2.00: 10}),
        (200, None, {0.25: 5, 0.50: 5, 0.75: 5, 1.00: 10, 1.25: 15, 1.50: 20, 1.75: 25, 2.00: 15}),
    ))

    # Taille des véhicules en fonction de la distance
    car_size: Table = field(default_factory=lambda: Table(
        (0, 100, {1: 15, 2: 70, 3: 15}),
        (100, None, {1: 15, 2: 60, 3: 25}),
    ))

    # Distance entre deux véhicule en fonction de la distance
    # Pour l'instant c'est indépendant de la vitesse il pourai peut être que dans le jeu d'origine les deux sont liée
    car_space: Table = field(default_factory=lambda: Table(
        (0, 50, {3: 10, 4: 20, 5: 30, 6: 25, 7: 10, 8: 5}),
        (50, 100, {2: 5, 3: 10, 4: 20, 5: 30, 6: 20, 7: 10, 8: 5}),
        (100, 200, {2: 5, 3: 15, 4: 30, 5: 25, 6: 20, 7: 5}),
        (200, None, {2: 15, 3: 20, 4: 30, 5: 20, 6: 15}),
    ))

    # ==================================================================================================================================================
    # RIVIÈRE — composition de chaque ligne
    # Probabilité qu'une ligne de rivière soit des BÛCHES (sinon nénuphars)
    water_prob: Prob = field(default_factory=lambda: Prob(
        (  0, None, 0.90),
    ))

    # ==================================================================================================================================================
    # BÛCHES
    # Vitesse des bûches (Cases par seconde) (La speed est commune a toute la ligne)
    log_speed: Table = field(default_factory=lambda: Table(
        (0, 50, {0.25: 10, 0.50: 10, 0.75: 20, 1.00: 30, 1.25: 20, 1.50: 10}),
        (50, 100, {0.25: 10, 0.50: 10, 0.75: 15, 1.00: 15, 1.25: 15, 1.50: 15, 1.75: 10, 2.00: 10}),
        (100, 200, {0.25: 5, 0.50: 5, 0.75: 10, 1.00: 15, 1.25: 20, 1.50: 20, 1.75: 15, 2.00: 10}),
        (200, None, {0.25: 5, 0.50: 5, 0.75: 5, 1.00: 10, 1.25: 15, 1.50: 20, 1.75: 25, 2.00: 15}),
    ))

    # Ecart entre deux buche, je laisse la meme peut importe la distance car la diminution de la taille moyenne des buches avec la distance
    # remplie déjà le rôle de diminuer la densité avec la distance
    log_space: Table = field(default_factory=lambda: Table(
        (  0, None, {0: 10, 1: 15, 2: 30, 3: 30, 4: 15}),
    ))

    # Taille des bûches (en cases) - Contrairement aux véhicule pour les route toutes les buches d'une ligne n'ont pas la même taille
    log_size: Table = field(default_factory=lambda: Table(
        (0, 100, {2: 70, 3: 15, 4: 15}),
        (100, 200, {1: 10, 2: 40, 3: 30, 4: 20}),
        (200, None, {1: 20, 2: 40, 3: 30, 4: 10}),
    ))

    # ==================================================================================================================================================
    # NÉNUPHARS
    # Nombre de nénuphars par ligne
    lily_count: Table = field(default_factory=lambda: Table(
        (0, None, {1: 10, 2: 40, 3: 30, 4: 20}),
    ))

    # ==================================================================================================================================================
    # HERBE
    # Nombre d'arbre par ligne
    tree_count: Table = field(default_factory=lambda: Table(
        (0, None, {0: 10, 1: 10, 2: 30, 3: 30, 4: 20}),
    ))

CONFIG = WorldConfig()
