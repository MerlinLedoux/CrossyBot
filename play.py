"""
play.py — version jouable de CrossyBot avec Arcade.

Mode humain  : python play.py
Mode agent   : python play.py --agent training/models/crossybot.pt
Mode debug   : python play.py --debug
               python play.py --debug --agent training/models/crossybot.pt

Contrôles communs     : R = rejouer | A = basculer agent/humain
Mode debug seulement  : ESPACE = avancer d'un step RL
                        Flèches = choisir l'action manuellement (step par step)
"""
import argparse
import torch
import numpy as np
import arcade
from training.env.crossy_env import CrossyEnv, GRID_H, OBS_LANES, OBS_PLAYABLE_W, SCROLL_SPEED
from training.env.lane import (LaneType, GRID_W, MAX_SPEED, CELLS_PER_SEC,
                                PLAYABLE_MIN, PLAYABLE_MAX)
from training.agent.network import ActorCritic

# ── dimensions fenêtre de jeu ────────────────────────────────────────────────
CELL  = 64
UI_H  = 52
WIN_W = GRID_W * CELL          # 832
WIN_H = GRID_H * CELL          # 832

# ── dimensions panneau de debug (affiché à droite du jeu) ───────────────────
DBG_W        = 660             # largeur du panneau
DBG_MARGIN   = 10              # marge intérieure
DBG_ROW_H    = 18              # hauteur d'une ligne de lane dans le panneau
DBG_FONT     = 10              # taille de police

AGENT_ACTION_INTERVAL = 1.0 / 3.0

_LANE_BG = {
    LaneType.SAFE:  (110, 200,  75),
    LaneType.GRASS: ( 55, 130,  30),
    LaneType.ROAD:  ( 50,  50,  50),
    LaneType.WATER: ( 35, 100, 200),
    LaneType.LILY:  ( 25,  75, 165),
}

_TYPE_LABEL = {
    LaneType.SAFE:  "SAFE ",
    LaneType.GRASS: "GRASS",
    LaneType.ROAD:  "ROAD ",
    LaneType.WATER: "WATER",
    LaneType.LILY:  "LILY ",
}

_KEY_TO_ACTION = {
    arcade.key.UP:    1,
    arcade.key.DOWN:  2,
    arcade.key.LEFT:  3,
    arcade.key.RIGHT: 4,
}

_ACTION_LABEL = {0: "·", 1: "↑", 2: "↓", 3: "←", 4: "→"}


class CrossyGame(arcade.Window):

    def __init__(self, agent_path: str = None, debug: bool = False):
        total_w = WIN_W + DBG_W if debug else WIN_W
        super().__init__(total_w, WIN_H + UI_H, "CrossyBot — Debug" if debug else "CrossyBot")
        arcade.set_background_color((20, 20, 20))

        self.debug      = debug
        self.env        = CrossyEnv()
        self.network    = None
        self.agent_mode = agent_path is not None
        self._last_action_label = "·"
        self._agent_timer       = 0.0

        # En mode debug : on attend une touche avant d'avancer
        self._paused        = debug      # True = on attend ESPACE ou flèche
        self._pending_action = None      # action choisie par flèche en mode debug

        if agent_path is not None:
            self.network = ActorCritic()
            ckpt = torch.load(agent_path, map_location="cpu")
            self.network.load_state_dict(ckpt["network_state"])
            self.network.eval()
            print(f"Modèle chargé : {agent_path}  "
                  f"(steps: {ckpt.get('total_steps', '?'):,})")

        self._reset()

    # ── reset ────────────────────────────────────────────────────────────────

    def _reset(self):
        obs, _ = self.env.reset()
        self._obs        = obs
        self._dead       = False
        self._agent_timer = 0.0
        self._last_action_label = "·"
        self._paused     = self.debug
        self._scroll_row = -3.0

    # ── boucle temps réel ────────────────────────────────────────────────────

    def on_update(self, dt: float):
        if self._dead or self.debug:
            # En mode debug : les obstacles ne bougent PAS entre deux steps.
            # Le monde est figé jusqu'à ce que l'utilisateur appuie sur ESPACE.
            return

        lane   = self.env.player_lane
        on_log = (lane.lane_type == LaneType.WATER and
                  lane.is_on_log(self.env.player_x))

        for l in self.env.get_visible_lanes():
            l.update_visual(dt)

        if on_log:
            delta = (lane._speed / MAX_SPEED) * CELLS_PER_SEC * dt
            self.env.player_x += delta
            if not (PLAYABLE_MIN <= self.env.player_x < PLAYABLE_MAX + 1):
                self._dead = True
                return

        if self.agent_mode and self.network is not None:
            self._agent_timer += dt
            if self._agent_timer >= AGENT_ACTION_INTERVAL:
                self._agent_timer -= AGENT_ACTION_INTERVAL
                self._agent_step()

        self._scroll_row = max(self._scroll_row, float(self.env.camera_start_row))
        self._scroll_row += SCROLL_SPEED * dt
        new_cam = int(self._scroll_row)
        if new_cam > self.env.camera_start_row:
            self.env.camera_start_row = new_cam

        self._check_collision()
        if not self._dead and self.env.player_row < self.env.camera_start_row:
            self._dead = True

        self.env._trim_lanes()
        self.env._ensure_lanes()

    # ── entrées ──────────────────────────────────────────────────────────────

    def on_key_press(self, key, mod):
        if key == arcade.key.R:
            self._reset()
            return

        if key == arcade.key.A and self.network is not None:
            self.agent_mode = not self.agent_mode
            print(f"Mode : {'AGENT' if self.agent_mode else 'HUMAIN'}")
            return

        # ── contrôles mode debug ──────────────────────────────────────────
        if self.debug:
            if key == arcade.key.SPACE:
                # Avancer d'un step : l'agent choisit (ou action=0 si humain)
                if self.agent_mode and self.network is not None:
                    self._agent_step_debug()
                else:
                    self._manual_step_debug(0)   # action : rester
            elif key in _KEY_TO_ACTION:
                # Flèche = choisir l'action et avancer d'un step
                self._manual_step_debug(_KEY_TO_ACTION[key])
            return

        # ── contrôles mode normal ─────────────────────────────────────────
        if self._dead or self.agent_mode:
            return
        action = _KEY_TO_ACTION.get(key)
        if action is not None:
            self.env._apply_action(action)
            self.env._trim_lanes()
            self.env._ensure_lanes()
            self._check_collision()

    # ── steps debug (un step RL à la fois) ───────────────────────────────────

    def _agent_step_debug(self):
        """Mode debug + agent : le réseau choisit l'action, on avance d'un step."""
        obs_t  = torch.tensor(self._obs, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            action, _, _ = self.network.act(obs_t)
        self._apply_debug_step(int(action.item()))

    def _manual_step_debug(self, action: int):
        """Mode debug + humain : on avance d'un step avec l'action donnée."""
        self._apply_debug_step(action)

    def _apply_debug_step(self, action: int):
        """Applique un step RL complet et met à jour l'observation."""
        if self._dead:
            return
        self._last_action_label = _ACTION_LABEL[action]
        obs, reward, terminated, truncated, _ = self.env.step(action)
        self._obs  = obs
        self._dead = terminated or truncated
        print(f"Action: {_ACTION_LABEL[action]}  "
              f"Reward: {reward:+.1f}  "
              f"Score: {self.env.score}  "
              f"{'MORT' if terminated else ''}")

    # ── step agent mode normal ────────────────────────────────────────────────

    def _agent_step(self):
        obs_t  = torch.tensor(self._obs, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            action, _, _ = self.network.act(obs_t)
        action_int = int(action.item())
        self._last_action_label = _ACTION_LABEL[action_int]
        self.env._apply_action(action_int)
        self.env._trim_lanes()
        self.env._ensure_lanes()
        self._check_collision()
        self._obs = self.env._get_observation()

    # ── collision ─────────────────────────────────────────────────────────────

    def _check_collision(self):
        if not (PLAYABLE_MIN <= self.env.player_x < PLAYABLE_MAX + 1):
            self._dead = True
            return
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

    # ── dessin ────────────────────────────────────────────────────────────────

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

        if self.debug:
            self._draw_debug_panel()

        if self._dead:
            self._draw_gameover()

    # ── panneau de debug ──────────────────────────────────────────────────────

    def _draw_debug_panel(self):
        """
        Affiche les flottants bruts du vecteur d'observation exactement
        tels qu'ils sont passés au réseau de neurones.

        Format par lane :
          [idx]  type_val  spd_val  | occ[0] occ[1] ... occ[12]

        Exemple :
          [ 0]  +0.00  +0.33  | 0.00  0.00  0.00  1.00  0.75  0.00 ...
        """
        obs   = self._obs
        px    = WIN_W + DBG_MARGIN
        p_vis = self.env.player_row - self.env.camera_start_row

        # ── fond ──────────────────────────────────────────────────────────
        arcade.draw_rect_filled(
            arcade.XYWH(WIN_W + DBG_W / 2, (WIN_H + UI_H) / 2, DBG_W, WIN_H + UI_H),
            (25, 25, 25),
        )

        # ── en-tête ───────────────────────────────────────────────────────
        top = WIN_H + UI_H - DBG_MARGIN
        arcade.draw_text("VECTEUR D'OBSERVATION BRUT", px, top - 14,
                         (220, 220, 220), 12, bold=True)
        arcade.draw_text("ESPACE = step (rester)  |  Flèches = step + action",
                         px, top - 28, (130, 130, 130), DBG_FONT)

        # ── header colonnes ───────────────────────────────────────────────
        # L'observation contient seulement les 9 colonnes jouables (c2..c10)
        header_y = top - 46
        X_IDX   = px
        X_TYPE  = px + 30
        X_SPD   = px + 78
        X_SEP   = px + 126
        X_OCC   = px + 136   # début des 9 valeurs d'occupation
        OCC_W   = 46         # largeur par colonne (plus large car moins de colonnes)

        arcade.draw_text("idx", X_IDX,  header_y, (150, 150, 150), DBG_FONT)
        arcade.draw_text("type", X_TYPE, header_y, (150, 150, 150), DBG_FONT)
        arcade.draw_text("speed", X_SPD, header_y, (150, 150, 150), DBG_FONT)
        for ci, c in enumerate(range(PLAYABLE_MIN, PLAYABLE_MAX + 1)):
            arcade.draw_text(f"c{c}", X_OCC + ci * OCC_W, header_y,
                             (200, 200, 80), DBG_FONT)

        arcade.draw_line(px, header_y - 2, WIN_W + DBG_W - DBG_MARGIN,
                         header_y - 2, (60, 60, 60), 1)

        # ── une ligne par lane observée (OBS_LANES = 5) ───────────────────
        # obs[i * 11 : i * 11 + 11] = [type, speed, occ_c2..occ_c10]
        obs_lane_feat = OBS_PLAYABLE_W + 2   # = 11
        for i in range(OBS_LANES):
            base   = i * obs_lane_feat
            v_type = float(obs[base])
            v_spd  = float(obs[base + 1])
            v_occ  = [float(obs[base + 2 + ci]) for ci in range(OBS_PLAYABLE_W)]

            row_y  = header_y - 14 - i * DBG_ROW_H

            # La lane du joueur est toujours à l'index OBS_LOOK_BEHIND = 1
            is_player_lane = (i == 1)
            if is_player_lane:
                arcade.draw_rect_filled(
                    arcade.XYWH(WIN_W + DBG_W / 2, row_y + DBG_ROW_H / 2 - 2,
                                DBG_W, DBG_ROW_H),
                    (60, 55, 15),
                )

            rel     = i - 1   # 1 = index du joueur dans la fenêtre obs
            idx_str = f"{rel:+d}" if rel != 0 else " 0"
            row_col = (255, 230, 80) if is_player_lane else (170, 170, 170)

            arcade.draw_text(f"[{idx_str}]", X_IDX, row_y, row_col, DBG_FONT)
            arcade.draw_text(f"{v_type:+.2f}", X_TYPE, row_y,
                             (180, 220, 255), DBG_FONT)

            spd_col = (100, 220, 100) if v_spd > 0 else \
                      (220, 100, 100) if v_spd < 0 else (150, 150, 150)
            arcade.draw_text(f"{v_spd:+.2f}", X_SPD, row_y, spd_col, DBG_FONT)
            arcade.draw_text("|", X_SEP, row_y, (80, 80, 80), DBG_FONT)

            # occ[c2..c10] — 9 valeurs brutes
            player_col_idx = int(self.env.player_x) - PLAYABLE_MIN
            for ci in range(OBS_PLAYABLE_W):
                val = v_occ[ci]
                if val == 0.0:
                    txt_col = (80, 80, 80)
                elif ci == player_col_idx:
                    txt_col = (80, 255, 80)
                else:
                    txt_col = (255, 200, 80)
                arcade.draw_text(f"{val:.2f}", X_OCC + ci * OCC_W, row_y,
                                 txt_col, DBG_FONT)

        # ── séparateur ────────────────────────────────────────────────────
        sep_y = header_y - 14 - OBS_LANES * DBG_ROW_H - 4
        arcade.draw_line(px, sep_y, WIN_W + DBG_W - DBG_MARGIN,
                         sep_y, (60, 60, 60), 1)

        # ── 2 derniers flottants : player_x_norm et player_y_norm ─────────
        v_px = float(obs[-2])
        v_py = float(obs[-1])
        arcade.draw_text(
            f"obs[-2]  player_x_norm = {v_px:.4f}"
            f"   (x réel = {self.env.player_x:.3f})",
            px, sep_y - 14, (160, 255, 160), DBG_FONT,
        )
        arcade.draw_text(
            f"obs[-1]  player_y_norm = {v_py:.4f}"
            f"   (row={self.env.player_row}  cam={self.env.camera_start_row})",
            px, sep_y - 28, (160, 255, 160), DBG_FONT,
        )
        arcade.draw_text(
            f"score={self.env.score}   steps={self.env.steps}"
            f"   action={self._last_action_label}",
            px, sep_y - 46, (200, 200, 120), DBG_FONT,
        )

        # ── probabilités d'action (si réseau chargé) ──────────────────────
        if self.network is not None:
            obs_t = torch.tensor(self._obs, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                logits = self.network.policy_head(
                    self.network._encode(obs_t)
                )                               # (1, 5)
            probs = torch.softmax(logits, dim=-1).squeeze(0)  # (5,)

            # Étiquettes et couleurs
            action_labels = ["·", "↑", "↓", "←", "→"]
            prob_y = sep_y - 66
            arcade.draw_text("Probabilités d'action :", px, prob_y,
                             (150, 150, 150), DBG_FONT)

            prob_y -= 16
            x_cursor = px
            for act_i, (label, p) in enumerate(zip(action_labels, probs.tolist())):
                # Barre de fond gris
                bar_max_w = 90
                bar_w = max(1, int(p * bar_max_w))
                bar_h = 13
                bar_x = x_cursor + bar_max_w / 2
                arcade.draw_rect_filled(
                    arcade.XYWH(bar_x, prob_y + bar_h / 2, bar_max_w, bar_h),
                    (40, 40, 40),
                )
                # Barre de remplissage — bleu si action la plus probable
                max_p = max(probs.tolist())
                fill_col = (80, 200, 80) if p == max_p else (80, 120, 200)
                arcade.draw_rect_filled(
                    arcade.XYWH(x_cursor + bar_w / 2, prob_y + bar_h / 2,
                                bar_w, bar_h),
                    fill_col,
                )
                # Texte : étiquette + pourcentage
                txt_col = (255, 255, 100) if p == max_p else (200, 200, 200)
                arcade.draw_text(f"{label} {p*100:4.1f}%",
                                 x_cursor + 2, prob_y + 1, txt_col, DBG_FONT)
                x_cursor += bar_max_w + 8

    # ── helpers de dessin (jeu) ───────────────────────────────────────────────

    def _draw_lane_bg(self, lane, y: int):
        arcade.draw_rect_filled(
            arcade.XYWH(WIN_W / 2, y + CELL / 2, WIN_W, CELL),
            _LANE_BG[lane.lane_type],
        )
        arcade.draw_line(0, y, WIN_W, y, (0, 0, 0), 1)
        if lane.lane_type == LaneType.ROAD:
            arcade.draw_line(0, y + CELL // 2, WIN_W, y + CELL // 2, (180, 160, 20), 1)
        wall_w = PLAYABLE_MIN * CELL
        arcade.draw_rect_filled(
            arcade.XYWH(wall_w / 2, y + CELL / 2, wall_w, CELL), (0, 0, 0, 80),
        )
        arcade.draw_rect_filled(
            arcade.XYWH(WIN_W - wall_w / 2, y + CELL / 2, wall_w, CELL), (0, 0, 0, 80),
        )

    def _draw_tree(self, pos, lane_y):
        cx = pos * CELL + CELL / 2
        cy = lane_y + CELL / 2
        arcade.draw_rect_filled(arcade.XYWH(cx, cy - 8, 10, CELL // 2 - 4), (100, 60, 20))
        arcade.draw_circle_filled(cx, cy + 8, 20, (30, 90, 20))

    def _draw_car(self, pos, width, lane_y):
        arcade.draw_rect_filled(
            arcade.XYWH(pos * CELL + width * CELL / 2, lane_y + CELL / 2,
                        width * CELL - 4, CELL - 20), (210, 40, 40))

    def _draw_log(self, pos, width, lane_y):
        arcade.draw_rect_filled(
            arcade.XYWH(pos * CELL + width * CELL / 2, lane_y + CELL / 2,
                        width * CELL - 4, CELL - 24), (139, 90, 40))

    def _draw_lily(self, pos, lane_y):
        cx = pos * CELL + CELL / 2
        cy = lane_y + CELL / 2
        arcade.draw_circle_filled(cx, cy, CELL // 2 - 8, (50, 160, 60))
        arcade.draw_circle_outline(cx, cy, CELL // 2 - 8, (30, 110, 40), 3)

    def _draw_player(self, player_x, vis_row):
        cx = player_x * CELL
        cy = UI_H + vis_row * CELL + CELL / 2
        r  = CELL // 4
        arcade.draw_circle_filled(cx, cy, r, (255, 230, 50))
        arcade.draw_circle_outline(cx, cy, r, (0, 0, 0), 2)
        arcade.draw_circle_filled(cx - r // 3, cy + r // 4, r // 4, (30, 30, 30))
        arcade.draw_circle_filled(cx + r // 3, cy + r // 4, r // 4, (30, 30, 30))

    def _draw_ui(self):
        arcade.draw_rect_filled(
            arcade.XYWH(WIN_W / 2, UI_H / 2, WIN_W, UI_H), (25, 25, 25))
        arcade.draw_text(f"Score : {self.env.score}", 14, 14,
                         arcade.color.WHITE, 22, bold=True)
        if self.network is not None:
            mode  = "AGENT" if self.agent_mode else "HUMAIN"
            color = (100, 220, 100) if self.agent_mode else (200, 200, 200)
            arcade.draw_text(f"{mode}  {self._last_action_label}",
                             WIN_W - 150, 14, color, 20, bold=True)

    def _draw_gameover(self):
        cx, cy = WIN_W / 2, (WIN_H + UI_H) / 2
        arcade.draw_rect_filled(
            arcade.XYWH(cx, cy, WIN_W, WIN_H + UI_H), (0, 0, 0, 170))
        arcade.draw_text("GAME OVER", cx, cy + 30,
                         arcade.color.RED, 46, anchor_x="center", bold=True)
        arcade.draw_text(f"Score final : {self.env.score}", cx, cy - 20,
                         arcade.color.WHITE, 22, anchor_x="center")
        arcade.draw_text("[ R ] pour rejouer", cx, cy - 60,
                         arcade.color.LIGHT_GRAY, 18, anchor_x="center")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent", type=str, default=None,
                        help="Chemin vers le checkpoint")
    parser.add_argument("--debug", action="store_true",
                        help="Mode debug : affiche l'observation, step par step")
    args = parser.parse_args()

    game = CrossyGame(agent_path=args.agent, debug=args.debug)
    arcade.run()
