"""
play.py — version jouable de CrossyBot avec Arcade.
Contrôles : flèches directionnelles | R pour rejouer

Lancement :
    python play.py
"""
import arcade
from training.env.crossy_env import CrossyEnv, GRID_H, LOOK_BEHIND
from training.env.lane import LaneType, GRID_W

CELL     = 64     # taille d'une cellule en pixels
UI_H     = 52     # hauteur de la barre de score en bas
WIN_W    = GRID_W * CELL          # 576 px
WIN_H    = GRID_H * CELL          # 640 px
OBS_TICK = 0.12   # secondes entre chaque avance des obstacles

# --------------------------------------------------------------------------- #
#  Palettes                                                                    #
# --------------------------------------------------------------------------- #

_LANE_BG = {
    LaneType.SAFE:  (110, 200,  75),
    LaneType.GRASS: ( 55, 130,  30),
    LaneType.ROAD:  ( 50,  50,  50),
    LaneType.WATER: ( 35, 100, 200),
    LaneType.LILY:  ( 25,  75, 165),
}

_OBS_COLOR = {
    LaneType.ROAD:  (210,  50,  30),   # voiture  → rouge
    LaneType.WATER: (139,  90,  40),   # bûche    → marron
    LaneType.LILY:  ( 55, 175,  65),   # nénuphar → vert
    LaneType.GRASS: ( 30,  90,  20),   # arbre    → vert foncé
}

# --------------------------------------------------------------------------- #

class CrossyGame(arcade.Window):

    def __init__(self):
        super().__init__(WIN_W, WIN_H + UI_H, "CrossyBot")
        arcade.set_background_color((20, 20, 20))
        self.env = CrossyEnv()
        self._reset()

    # --- état ----------------------------------------------------------------

    def _reset(self):
        self.env.reset()
        self._timer = 0.0
        self._dead  = False

    # --- boucle principale ---------------------------------------------------

    def on_update(self, dt: float):
        if self._dead:
            return
        self._timer += dt
        if self._timer >= OBS_TICK:
            self._timer = 0.0
            self.env._update_obstacles()
            if self.env._check_collision():
                self._dead = True

    def on_key_press(self, key, mod):
        if self._dead:
            if key == arcade.key.R:
                self._reset()
            return
        action = {
            arcade.key.UP:    1,
            arcade.key.DOWN:  2,
            arcade.key.LEFT:  3,
            arcade.key.RIGHT: 4,
        }.get(key)
        if action is not None:
            self.env._apply_action(action)
            if self.env._check_collision():
                self._dead = True

    # --- dessin --------------------------------------------------------------

    def on_draw(self):
        self.clear()

        visible = self.env.get_visible_lanes()
        start   = max(0, self.env.player_row - LOOK_BEHIND)

        for i, lane in enumerate(visible):
            lane_y = UI_H + i * CELL
            self._draw_lane_bg(lane, lane_y)
            for x in range(GRID_W):
                if lane.has_obstacle_at(x):
                    self._draw_obstacle(lane.lane_type, x, lane_y)

        p_vis = self.env.player_row - start
        self._draw_player(self.env.player_x, p_vis)
        self._draw_ui()

        if self._dead:
            self._draw_gameover()

    # --- helpers de dessin ---------------------------------------------------

    def _draw_lane_bg(self, lane, y: int):
        arcade.draw_rect_filled(arcade.XYWH(WIN_W / 2, y + CELL / 2, WIN_W, CELL), _LANE_BG[lane.lane_type])
        arcade.draw_line(0, y, WIN_W, y, (0, 0, 0), 1)
        if lane.lane_type == LaneType.ROAD:
            arcade.draw_line(0, y + CELL // 2, WIN_W, y + CELL // 2, (200, 180, 30), 1)

    def _draw_obstacle(self, lt: LaneType, x: int, lane_y: int):
        cx = x * CELL + CELL / 2
        cy = lane_y + CELL / 2
        c  = _OBS_COLOR.get(lt, arcade.color.WHITE)

        if lt == LaneType.ROAD:
            arcade.draw_rect_filled(arcade.XYWH(cx, cy, CELL - 8, CELL - 22), c)
            arcade.draw_rect_filled(arcade.XYWH(cx, cy + 4, CELL - 18, CELL - 36), (150, 200, 240))

        elif lt == LaneType.WATER:
            arcade.draw_rect_filled(arcade.XYWH(cx, cy, CELL - 4, CELL - 34), c)

        elif lt == LaneType.LILY:
            arcade.draw_circle_filled(cx, cy, CELL // 2 - 10, c)
            arcade.draw_circle_outline(cx, cy, CELL // 2 - 10, (30, 120, 40), 2)

        elif lt == LaneType.GRASS:
            arcade.draw_rect_filled(arcade.XYWH(cx, cy - 8, 10, CELL // 2 - 4), (100, 60, 20))
            arcade.draw_circle_filled(cx, cy + 10, 18, c)

    def _draw_player(self, grid_x: int, vis_row: int):
        cx = grid_x * CELL + CELL / 2
        cy = UI_H + vis_row * CELL + CELL / 2
        r  = CELL // 2 - 10
        arcade.draw_circle_filled(cx, cy, r, (255, 230, 50))
        arcade.draw_circle_outline(cx, cy, r, (0, 0, 0), 2)
        arcade.draw_circle_filled(cx - 7, cy + 5, 4, (30, 30, 30))
        arcade.draw_circle_filled(cx + 7, cy + 5, 4, (30, 30, 30))

    def _draw_ui(self):
        arcade.draw_rect_filled(arcade.XYWH(WIN_W / 2, UI_H / 2, WIN_W, UI_H), (25, 25, 25))
        arcade.draw_text(f"Score : {self.env.score}", 14, 14, arcade.color.WHITE, 22, bold=True)
        arcade.draw_text("↑ ↓ ← →  pour jouer",
                         WIN_W - 10, 14, arcade.color.LIGHT_GRAY, 14, anchor_x="right")

    def _draw_gameover(self):
        cx, cy = WIN_W / 2, (WIN_H + UI_H) / 2
        arcade.draw_rect_filled(arcade.XYWH(cx, cy, WIN_W, WIN_H + UI_H), (0, 0, 0, 170))
        arcade.draw_text("GAME OVER", cx, cy + 30,
                         arcade.color.RED, 46, anchor_x="center", bold=True)
        arcade.draw_text(f"Score final : {self.env.score}", cx, cy - 20,
                         arcade.color.WHITE, 22, anchor_x="center")
        arcade.draw_text("[ R ] pour rejouer", cx, cy - 60,
                         arcade.color.LIGHT_GRAY, 18, anchor_x="center")


# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    game = CrossyGame()
    arcade.run()
