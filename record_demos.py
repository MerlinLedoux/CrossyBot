"""
record_demos.py — Génère les GIF de démonstration pour le README.

Usage :
    python record_demos.py                          # génère tous les GIFs
    python record_demos.py --only agent             # GIF agent uniquement
    python record_demos.py --only debug             # GIF debug uniquement
    python record_demos.py --model training/models/crossybot.pt
    python record_demos.py --duration 15 --fps 10
"""
import argparse
import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from training.env.crossy_env import (
    CrossyEnv, GRID_H, SCROLL_SPEED, OBS_LANES, OBS_PLAYABLE_W,
    OBS_LOOK_BEHIND,
)
from training.env.lane import (
    LaneType, GRID_W, PLAYABLE_MIN, PLAYABLE_MAX,
    MAX_SPEED, CELLS_PER_SEC,
)
from training.agent.network import ActorCritic

# ── Rendu ────────────────────────────────────────────────────────────────────
CELL = 48
UI_H = 44
SS   = 2       # supersampling (rendu à 2× puis downscale pour antialiasing)

LANE_BG = {
    LaneType.SAFE:  (110, 200, 75),
    LaneType.GRASS: ( 55, 130, 30),
    LaneType.ROAD:  ( 50,  50,  50),
    LaneType.WATER: ( 35, 100, 200),
    LaneType.LILY:  ( 25,  75, 165),
}

WALL_OVERLAY = (0, 0, 0)
ACTION_LABEL = {0: "·", 1: "↑", 2: "↓", 3: "←", 4: "→"}

# ── Police ───────────────────────────────────────────────────────────────────

def _load_fonts(scale=1):
    s = scale
    try:
        return {
            "big":   ImageFont.truetype("consola.ttf", 22 * s),
            "med":   ImageFont.truetype("consola.ttf", 16 * s),
            "small": ImageFont.truetype("consola.ttf", 12 * s),
            "tiny":  ImageFont.truetype("consola.ttf", 10 * s),
        }
    except OSError:
        try:
            return {
                "big":   ImageFont.truetype("arial.ttf", 22 * s),
                "med":   ImageFont.truetype("arial.ttf", 16 * s),
                "small": ImageFont.truetype("arial.ttf", 12 * s),
                "tiny":  ImageFont.truetype("arial.ttf", 10 * s),
            }
        except OSError:
            f = ImageFont.load_default()
            return {"big": f, "med": f, "small": f, "tiny": f}


# ── Rendu d'une frame ────────────────────────────────────────────────────────

def render_frame(env, last_action=0, dead=False, fonts=None,
                 show_debug=False, network=None, obs=None):
    """Rend l'état du jeu en PIL Image (coordonnées : y=0 en haut)."""
    C  = CELL * SS
    uh = UI_H * SS

    game_w = GRID_W * C
    game_h = GRID_H * C + uh

    dbg_w = 520 * SS if show_debug else 0
    img_w = game_w + dbg_w
    img_h = game_h

    img  = Image.new("RGB", (img_w, img_h), (20, 20, 20))
    draw = ImageDraw.Draw(img)

    visible = env.get_visible_lanes()

    # ── Fond de lanes ────────────────────────────────────────────────────────
    for i, lane in enumerate(visible):
        # PIL : lane 0 (derrière) en bas, lane GRID_H-1 (devant) en haut
        ly = uh + (GRID_H - 1 - i) * C  # top of lane strip

        draw.rectangle([0, ly, game_w, ly + C], fill=LANE_BG[lane.lane_type])

        # Ligne de séparation
        draw.line([(0, ly + C), (game_w, ly + C)], fill=(0, 0, 0), width=max(1, SS))

        # Marquage route
        if lane.lane_type == LaneType.ROAD:
            y_mid = ly + C // 2
            dash_len = 16 * SS
            gap_len  = 12 * SS
            x = 0
            while x < game_w:
                draw.line([(x, y_mid), (min(x + dash_len, game_w), y_mid)],
                          fill=(180, 160, 20), width=max(1, SS))
                x += dash_len + gap_len

        # Murs latéraux
        wall_w = PLAYABLE_MIN * C
        draw.rectangle([0, ly, wall_w, ly + C], fill=(15, 15, 15))
        draw.rectangle([game_w - wall_w, ly, game_w, ly + C], fill=(15, 15, 15))

        # ── Obstacles ────────────────────────────────────────────────────────
        if lane.lane_type == LaneType.GRASS:
            for pos, _ in lane.iter_obstacles():
                cx = int(pos * C + C // 2)
                cy = ly + C // 2
                # Tronc
                tw = 5 * SS
                th = C // 3
                draw.rectangle([cx - tw, cy, cx + tw, cy + th], fill=(100, 60, 20))
                # Feuillage
                r = 15 * SS
                draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=(30, 110, 25))
                r2 = 11 * SS
                draw.ellipse([cx - r2 - 4*SS, cy - r2 + 2*SS,
                              cx + r2 - 4*SS, cy + r2 + 2*SS], fill=(40, 125, 30))

        elif lane.lane_type == LaneType.ROAD:
            for pos, width in lane.iter_obstacles():
                x1 = int(pos * C) + 2 * SS
                x2 = int((pos + width) * C) - 2 * SS
                y1 = ly + 8 * SS
                y2 = ly + C - 8 * SS
                if x2 < -C or x1 > game_w + C:
                    continue
                # Carrosserie
                draw.rectangle([x1, y1, x2, y2], fill=(200, 45, 45))
                # Toit
                inset = max(3 * SS, int((x2 - x1) * 0.15))
                draw.rectangle([x1 + inset, y1 + 3*SS, x2 - inset, y2 - 3*SS],
                               fill=(170, 35, 35))

        elif lane.lane_type == LaneType.WATER:
            for pos, width in lane.iter_obstacles():
                x1 = int(pos * C) + 2 * SS
                x2 = int((pos + width) * C) - 2 * SS
                y1 = ly + 10 * SS
                y2 = ly + C - 10 * SS
                if x2 < -C or x1 > game_w + C:
                    continue
                draw.rectangle([x1, y1, x2, y2], fill=(139, 90, 40))
                # Stries du bois
                for sx in range(x1 + 6*SS, x2 - 2*SS, 10*SS):
                    draw.line([(sx, y1 + 2*SS), (sx, y2 - 2*SS)],
                              fill=(120, 75, 30), width=SS)

        elif lane.lane_type == LaneType.LILY:
            for pos, _ in lane.iter_obstacles():
                cx = int(pos * C + C // 2)
                cy = ly + C // 2
                r = C // 2 - 6 * SS
                draw.ellipse([cx - r, cy - r, cx + r, cy + r],
                             fill=(50, 160, 60), outline=(30, 110, 40), width=2*SS)

    # ── Joueur ───────────────────────────────────────────────────────────────
    p_vis = env.player_row - env.camera_start_row
    px = int(env.player_x * C)
    py = uh + (GRID_H - 1 - p_vis) * C + C // 2
    r = C // 4

    # Ombre
    draw.ellipse([px - r, py + r//2, px + r, py + r], fill=(0, 0, 0, 60))
    # Corps
    draw.ellipse([px - r, py - r, px + r, py + r],
                 fill=(255, 230, 50), outline=(0, 0, 0), width=2*SS)
    # Yeux
    er = max(2, r // 4)
    draw.ellipse([px - r//3 - er, py - r//4 - er,
                  px - r//3 + er, py - r//4 + er], fill=(30, 30, 30))
    draw.ellipse([px + r//3 - er, py - r//4 - er,
                  px + r//3 + er, py - r//4 + er], fill=(30, 30, 30))

    # ── Barre d'interface ────────────────────────────────────────────────────
    draw.rectangle([0, 0, game_w, uh], fill=(25, 25, 25))
    if fonts:
        draw.text((14 * SS, 10 * SS), f"Score : {env.score}",
                  fill=(255, 255, 255), font=fonts["big"])
        draw.text((game_w - 160 * SS, 12 * SS),
                  f"AGENT  {ACTION_LABEL[last_action]}",
                  fill=(100, 220, 100), font=fonts["med"])

    # ── Game Over ────────────────────────────────────────────────────────────
    if dead and fonts:
        overlay = Image.new("RGBA", (game_w, game_h), (0, 0, 0, 160))
        img.paste(Image.alpha_composite(
            img.crop((0, 0, game_w, game_h)).convert("RGBA"), overlay
        ).convert("RGB"), (0, 0))
        draw = ImageDraw.Draw(img)
        cx = game_w // 2
        cy = game_h // 2
        draw.text((cx, cy - 30*SS), "GAME OVER",
                  fill=(255, 60, 60), font=fonts["big"], anchor="mm")
        draw.text((cx, cy + 20*SS), f"Score : {env.score}",
                  fill=(220, 220, 220), font=fonts["med"], anchor="mm")

    # ── Panneau debug ────────────────────────────────────────────────────────
    if show_debug and obs is not None and fonts:
        _draw_debug_panel(draw, obs, env, network, game_w, img_h, dbg_w, fonts, SS)

    # Downscale
    final_w = img_w // SS
    final_h = img_h // SS
    return img.resize((final_w, final_h), Image.LANCZOS)


def _draw_debug_panel(draw, obs, env, network, x0, h, pw, fonts, ss):
    """Dessine le panneau d'observation brute à droite du jeu."""
    draw.rectangle([x0, 0, x0 + pw, h], fill=(25, 25, 25))

    mx = x0 + 10 * ss
    ty = 12 * ss

    draw.text((mx, ty), "OBSERVATION BRUTE (57 floats)",
              fill=(220, 220, 220), font=fonts["small"])
    ty += 22 * ss

    lane_feat = OBS_PLAYABLE_W + 2  # 11

    # En-tête colonnes
    col_labels = [f"c{c}" for c in range(PLAYABLE_MIN, PLAYABLE_MAX + 1)]
    draw.text((mx, ty), " idx  type   spd   | " + " ".join(f"{l:>5s}" for l in col_labels),
              fill=(140, 140, 140), font=fonts["tiny"])
    ty += 16 * ss

    draw.line([(mx, ty), (x0 + pw - 10*ss, ty)], fill=(60, 60, 60), width=ss)
    ty += 6 * ss

    player_col_idx = int(env.player_x) - PLAYABLE_MIN

    for i in range(OBS_LANES):
        base = i * lane_feat
        v_type = float(obs[base])
        v_spd  = float(obs[base + 1])
        v_occ  = [float(obs[base + 2 + ci]) for ci in range(OBS_PLAYABLE_W)]

        is_player = (i == OBS_LOOK_BEHIND)
        if is_player:
            draw.rectangle([mx - 4*ss, ty - 2*ss, x0 + pw - 10*ss, ty + 14*ss],
                           fill=(55, 50, 15))

        rel = i - OBS_LOOK_BEHIND
        idx_str = f"{rel:+d}" if rel != 0 else " 0"
        col = (255, 230, 80) if is_player else (170, 170, 170)

        # Valeurs d'occupation avec couleurs
        occ_parts = []
        for ci, val in enumerate(v_occ):
            if val == 0.0:
                occ_parts.append(f" 0.00")
            else:
                occ_parts.append(f" {val:.2f}")

        line = f"[{idx_str}] {v_type:+.2f} {v_spd:+.2f}  |"
        draw.text((mx, ty), line, fill=col, font=fonts["tiny"])

        # Occupation colorée
        occ_x = mx + len(line) * 6 * ss
        for ci, val in enumerate(v_occ):
            if val == 0.0:
                oc = (70, 70, 70)
            elif ci == player_col_idx and is_player:
                oc = (80, 255, 80)
            else:
                oc = (255, 200, 80)
            draw.text((mx + (22 + ci * 6) * 6 * ss // 6 + 130*ss, ty),
                      f"{val:.2f}", fill=oc, font=fonts["tiny"])

        ty += 16 * ss

    # Séparateur
    ty += 4 * ss
    draw.line([(mx, ty), (x0 + pw - 10*ss, ty)], fill=(60, 60, 60), width=ss)
    ty += 8 * ss

    # Position joueur
    v_px = float(obs[-2])
    v_py = float(obs[-1])
    draw.text((mx, ty),
              f"player_x_norm = {v_px:.4f}  (x = {env.player_x:.2f})",
              fill=(160, 255, 160), font=fonts["tiny"])
    ty += 16 * ss
    draw.text((mx, ty),
              f"player_y_norm = {v_py:.4f}  (row={env.player_row} cam={env.camera_start_row})",
              fill=(160, 255, 160), font=fonts["tiny"])
    ty += 20 * ss

    # Probabilités d'action
    if network is not None:
        obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            logits = network.policy_head(network._encode(obs_t))
        probs = torch.softmax(logits, dim=-1).squeeze(0).tolist()

        draw.text((mx, ty), "Probabilites d'action :",
                  fill=(150, 150, 150), font=fonts["tiny"])
        ty += 18 * ss

        labels = ["stay", " up ", "down", "left", "rght"]
        max_p = max(probs)
        bar_max = 180 * ss

        for ai, (label, p) in enumerate(zip(labels, probs)):
            bar_w = max(2, int(p * bar_max))
            is_best = (p == max_p)

            # Barre de fond
            draw.rectangle([mx, ty, mx + bar_max, ty + 12*ss], fill=(40, 40, 40))
            # Barre remplie
            fill_c = (80, 200, 80) if is_best else (80, 120, 200)
            draw.rectangle([mx, ty, mx + bar_w, ty + 12*ss], fill=fill_c)
            # Label
            txt_c = (255, 255, 100) if is_best else (200, 200, 200)
            draw.text((mx + 4*ss, ty), f"{label} {p*100:5.1f}%",
                      fill=txt_c, font=fonts["tiny"])

            ty += 16 * ss


# ── Simulation ───────────────────────────────────────────────────────────────

def check_collision(env):
    if not (PLAYABLE_MIN <= env.player_x < PLAYABLE_MAX + 1):
        return True
    lane = env.player_lane
    if lane.lane_type == LaneType.ROAD:
        return lane.overlaps_cell(int(env.player_x), hitbox=0.5)
    if lane.lane_type == LaneType.LILY:
        return not lane.has_obstacle_at(int(env.player_x))
    if lane.lane_type == LaneType.WATER:
        return not lane.is_on_log(env.player_x)
    return False


def record_gif(model_path, output_path, duration=12, fps=10,
               show_debug=False):
    """Enregistre un GIF de l'agent jouant."""
    print(f"Enregistrement : {output_path}  ({duration}s @ {fps}fps"
          f"{' + debug' if show_debug else ''})")

    env = CrossyEnv()
    obs, _ = env.reset()

    network = ActorCritic()
    ckpt = torch.load(model_path, map_location="cpu", weights_only=True)
    network.load_state_dict(ckpt["network_state"])
    network.eval()

    fonts = _load_fonts(SS)
    frames = []

    scroll_row   = -3.0
    agent_timer  = 0.0
    dead         = False
    last_action  = 0
    dead_frames  = 0

    total_frames = fps * duration
    dt = 1.0 / fps
    agent_interval = 1.0 / 3.0

    for frame_idx in range(total_frames):
        # Reset si mort depuis quelques frames
        if dead:
            dead_frames += 1
            if dead_frames > fps * 2:  # 2 secondes de game over
                obs, _ = env.reset()
                scroll_row   = -3.0
                agent_timer  = 0.0
                dead         = False
                dead_frames  = 0
                last_action  = 0

        if not dead:
            # Mise à jour visuelle des obstacles (identique à play.py on_update)
            for lane in env.get_visible_lanes():
                lane.update_visual(dt)

            # Portage du joueur par la bûche
            lane = env.player_lane
            if (lane.lane_type == LaneType.WATER
                    and lane.is_on_log(env.player_x)):
                delta = (lane._speed / MAX_SPEED) * CELLS_PER_SEC * dt
                env.player_x += delta
                if not (PLAYABLE_MIN <= env.player_x < PLAYABLE_MAX + 1):
                    dead = True

            # Action de l'agent AVANT le scroll (même ordre que play.py)
            if not dead:
                agent_timer += dt
                if agent_timer >= agent_interval:
                    agent_timer -= agent_interval
                    obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
                    with torch.no_grad():
                        features = network._encode(obs_t)
                        logits = network.policy_head(features)
                        last_action = int(logits.argmax(dim=-1).item())
                    env._apply_action(last_action)
                    env._trim_lanes()
                    env._ensure_lanes()
                    if check_collision(env):
                        dead = True
                    obs = env._get_observation()

            # Défilement automatique (après l'action de l'agent)
            if not dead:
                scroll_row  = max(scroll_row, float(env.camera_start_row))
                scroll_row += SCROLL_SPEED * dt
                new_cam = int(scroll_row)
                if new_cam > env.camera_start_row:
                    env.camera_start_row = new_cam

                if check_collision(env):
                    dead = True
                if env.player_row < env.camera_start_row:
                    dead = True

                env._trim_lanes()
                env._ensure_lanes()

        frame = render_frame(
            env, last_action, dead, fonts,
            show_debug=show_debug, network=network if show_debug else None,
            obs=obs,
        )
        frames.append(frame)

        if (frame_idx + 1) % fps == 0:
            elapsed = frame_idx + 1
            print(f"  {elapsed}/{total_frames} frames  "
                  f"(score={env.score}, {'DEAD' if dead else 'alive'})")

    # Sauvegarde GIF
    print(f"  Optimisation du GIF...")
    ms_per_frame = int(1000 / fps)
    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=ms_per_frame,
        loop=0,
        optimize=True,
    )

    import os
    size_kb = os.path.getsize(output_path) / 1024
    print(f"  -> {output_path}  ({len(frames)} frames, {size_kb:.0f} KB)\n")


def main():
    parser = argparse.ArgumentParser(description="Génère les GIFs de démo")
    parser.add_argument("--model", default="training/models/crossybot.pt")
    parser.add_argument("--duration", type=int, default=12,
                        help="Durée en secondes")
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--only", choices=["agent", "debug"],
                        help="Générer un seul GIF")
    args = parser.parse_args()

    if args.only != "debug":
        record_gif(args.model, "docs/demo_agent.gif",
                   duration=args.duration, fps=args.fps,
                   show_debug=False)

    if args.only != "agent":
        record_gif(args.model, "docs/demo_debug.gif",
                   duration=args.duration, fps=args.fps,
                   show_debug=True)

    print("Terminé ! Les GIFs sont dans docs/")


if __name__ == "__main__":
    main()
