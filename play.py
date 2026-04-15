"""
play.py — version jouable de CrossyBot avec Arcade.
Contrôles : flèches directionnelles | R pour rejouer

Lancement :
    python play.py
"""
import arcade
from training.env.crossy_env import CrossyEnv, GRID_H, PLAYABLE_MIN
from training.env.lane import LaneType, GRID_W, MAX_SPEED, CELLS_PER_SEC

CELL  = 64
UI_H  = 52
WIN_W = GRID_W * CELL
WIN_H = GRID_H * CELL

_LANE_BG = {
    LaneType.SAFE:  (110, 200,  75),
    LaneType.GRASS: ( 55, 130,  30),
    LaneType.ROAD:  ( 50,  50,  50),
    LaneType.WATER: ( 35, 100, 200),
    LaneType.LILY:  ( 25,  75, 165),
}

_KEY_TO_ACTION = {
    arcade.key.UP:    1,
    arcade.key.DOWN:  2,
    arcade.key.LEFT:  3,
    arcade.key.RIGHT: 4,
}


class CrossyGame(arcade.Window):

    def __init__(self):
        super().__init__(WIN_W, WIN_H + UI_H, "CrossyBot")
        arcade.set_background_color((20, 20, 20))
        self.env = CrossyEnv()
        self._reset()

    def _reset(self):
        self.env.reset()
        self._dead = False

    # --- boucle --------------------------------------------------------------

    def on_update(self, dt: float):
        if self._dead:
            return

        lane = self.env.player_lane

        # 1. Vérifier si le joueur est sur une bûche AVANT que les obstacles bougent
        on_log = (lane.lane_type == LaneType.WATER and
                  lane.is_on_log(self.env.player_x))

        # 2. Avancer les obstacles visuellement
        for l in self.env.get_visible_lanes():
            l.update_visual(dt)

        # 3. Transporter le joueur avec la bûche (même delta que update_visual)
        if on_log:
            delta = (lane._speed / MAX_SPEED) * CELLS_PER_SEC * dt
            self.env.player_x += delta
            if not (0 <= self.env.player_x < GRID_W):
                self._dead = True
                return

        # 4. Vérifier collision
        self._check_collision()
        self.env._trim_lanes()
        self.env._ensure_lanes()

    # --- entrées -------------------------------------------------------------

    def on_key_press(self, key, mod):
        if key == arcade.key.R:
            self._reset()
            return
        if self._dead:
            return
        action = _KEY_TO_ACTION.get(key)
        if action is not None:
            self.env._apply_action(action)
            self.env._trim_lanes()
            self.env._ensure_lanes()
            self._check_collision()

    # --- collision -----------------------------------------------------------

    def _check_collision(self):
        lane = self.env.player_lane
        if lane.lane_type == LaneType.ROAD:
            if lane.overlaps_cell(int(self.env.player_x), hitbox=0.5):
                self._dead = True
        elif lane.lane_type == LaneType.LILY:
            if not lane.has_obstacle_at(int(self.env.player_x)):
                self._dead = True
        elif lane.lane_type == LaneType.WATER:
            if not lane.is_on_log(self.env.player_x):
                self._dead = True

    # --- dessin --------------------------------------------------------------

    def on_draw(self):
        self.clear()

        visible = self.env.get_visible_lanes()

        for i, lane in enumerate(visible):
            lane_y = UI_H + i * CELL
            self._draw_lane_bg(lane, lane_y)

            if lane.lane_type == LaneType.GRASS:
                for pos, _ in lane.iter_obstacles():
                    self._draw_tree(pos, lane_y)
            elif lane.lane_type == LaneType.ROAD:
                for pos, width in lane.iter_obstacles():
                    self._draw_car(pos, width, lane_y)
            elif lane.lane_type == LaneType.WATER:
                for pos, width in lane.iter_obstacles():
                    self._draw_log(pos, width, lane_y)
            elif lane.lane_type == LaneType.LILY:
                for pos, _ in lane.iter_obstacles():
                    self._draw_lily(pos, lane_y)

        p_vis = self.env.player_row - self.env.camera_start_row
        self._draw_player(self.env.player_x, p_vis)
        self._draw_ui()

        if self._dead:
            self._draw_gameover()

    # --- helpers de dessin ---------------------------------------------------

    def _draw_lane_bg(self, lane, y: int):
        arcade.draw_rect_filled(
            arcade.XYWH(WIN_W / 2, y + CELL / 2, WIN_W, CELL),
            _LANE_BG[lane.lane_type],
        )
        arcade.draw_line(0, y, WIN_W, y, (0, 0, 0), 1)
        if lane.lane_type == LaneType.ROAD:
            arcade.draw_line(0, y + CELL // 2, WIN_W, y + CELL // 2, (180, 160, 20), 1)
        # Colonnes mur inaccessibles (légèrement assombries)
        wall_w = PLAYABLE_MIN * CELL
        arcade.draw_rect_filled(
            arcade.XYWH(wall_w / 2, y + CELL / 2, wall_w, CELL), (0, 0, 0, 80),
        )
        arcade.draw_rect_filled(
            arcade.XYWH(WIN_W - wall_w / 2, y + CELL / 2, wall_w, CELL), (0, 0, 0, 80),
        )

    def _draw_tree(self, pos: float, lane_y: int):
        cx = pos * CELL + CELL / 2
        cy = lane_y + CELL / 2
        arcade.draw_rect_filled(arcade.XYWH(cx, cy - 8, 10, CELL // 2 - 4), (100, 60, 20))
        arcade.draw_circle_filled(cx, cy + 8, 20, (30, 90, 20))

    def _draw_car(self, pos: float, width: int, lane_y: int):
        px = pos * CELL
        pw = width * CELL
        cy = lane_y + CELL / 2
        self._draw_car_segment(px, pw, cy)

    def _draw_car_segment(self, px: float, pw: float, cy: float):
        arcade.draw_rect_filled(
            arcade.XYWH(px + pw / 2, cy, pw - 4, CELL - 20),
            (210, 40, 40),
        )

    def _draw_log(self, pos: float, width: int, lane_y: int):
        px = pos * CELL
        pw = width * CELL
        cy = lane_y + CELL / 2
        self._draw_log_segment(px, pw, cy)

    def _draw_log_segment(self, px: float, pw: float, cy: float):
        arcade.draw_rect_filled(
            arcade.XYWH(px + pw / 2, cy, pw - 4, CELL - 24),
            (139, 90, 40),
        )

    def _draw_lily(self, pos: float, lane_y: int):
        cx = pos * CELL + CELL / 2
        cy = lane_y + CELL / 2
        arcade.draw_circle_filled(cx, cy, CELL // 2 - 8, (50, 160, 60))
        arcade.draw_circle_outline(cx, cy, CELL // 2 - 8, (30, 110, 40), 3)

    def _draw_player(self, player_x: float, vis_row: int):
        # player_x est le centre du sprite en unités de case
        cx = player_x * CELL
        cy = UI_H + vis_row * CELL + CELL / 2
        r  = CELL // 4
        arcade.draw_circle_filled(cx, cy, r, (255, 230, 50))
        arcade.draw_circle_outline(cx, cy, r, (0, 0, 0), 2)
        arcade.draw_circle_filled(cx - r // 3, cy + r // 4, r // 4, (30, 30, 30))
        arcade.draw_circle_filled(cx + r // 3, cy + r // 4, r // 4, (30, 30, 30))

    def _draw_ui(self):
        arcade.draw_rect_filled(
            arcade.XYWH(WIN_W / 2, UI_H / 2, WIN_W, UI_H),
            (25, 25, 25),
        )
        arcade.draw_text(f"Score : {self.env.score}", 14, 14,
                         arcade.color.WHITE, 22, bold=True)

    def _draw_gameover(self):
        cx, cy = WIN_W / 2, (WIN_H + UI_H) / 2
        arcade.draw_rect_filled(arcade.XYWH(cx, cy, WIN_W, WIN_H + UI_H), (0, 0, 0, 170))
        arcade.draw_text("GAME OVER", cx, cy + 30,
                         arcade.color.RED, 46, anchor_x="center", bold=True)
        arcade.draw_text(f"Score final : {self.env.score}", cx, cy - 20,
                         arcade.color.WHITE, 22, anchor_x="center")
        arcade.draw_text("[ R ] pour rejouer", cx, cy - 60,
                         arcade.color.LIGHT_GRAY, 18, anchor_x="center")


if __name__ == "__main__":
    game = CrossyGame()
    arcade.run()
