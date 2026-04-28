/**
 * agent.js — Inférence JS pure du réseau CrossyBot.
 *
 * Charge les poids exportés par export_model.py et reconstruit
 * le réseau (LaneEncoder + Trunk + PolicyHead) sans dépendance externe.
 * L'inférence est synchrone et prend < 1 ms (modèle ~90 K params).
 *
 * API publique :
 *   loadAgent(url?)  → Promise<void>   charge les poids depuis l'URL
 *   getAction(env)   → int (0–4)       retourne l'action argmax
 */

import { PLAYABLE_MIN, PLAYABLE_MAX } from '../game/constants.js';

// ── Constantes d'observation (identiques à crossy_env.py) ─────────────────────
const OBS_LOOK_BEHIND = 1;
const OBS_LOOK_AHEAD  = 3;
const OBS_LANES       = OBS_LOOK_BEHIND + 1 + OBS_LOOK_AHEAD;  // 5
const OBS_PLAYABLE_W  = PLAYABLE_MAX - PLAYABLE_MIN + 1;         // 9
const OBS_SIZE        = OBS_LANES * (OBS_PLAYABLE_W + 2) + 2;   // 57
const LANE_FEAT       = OBS_PLAYABLE_W + 2;                      // 11
const LANE_EMBED_DIM  = 32;
// Constantes issues de crossy_env.py — ne PAS utiliser les constantes du jeu web.
const PYTHON_GRID_H      = 13;   // GRID_H Python (vs 20 en web)
const PYTHON_LOOK_BEHIND = 2;    // LOOK_BEHIND Python (vs 5 en web) : max delta player/camera

// ── Réseau chargé (null jusqu'à loadAgent) ────────────────────────────────────
let _net = null;

// ── Chargement des poids ──────────────────────────────────────────────────────
export async function loadAgent(url = '/crossybot.json') {
  const resp = await fetch(url);
  if (!resp.ok) throw new Error(`Impossible de charger le modèle : ${url} (${resp.status})`);
  const data = await resp.json();

  _net = {
    enc1: _prepareLayer(data.lane_encoder_fc1),
    enc2: _prepareLayer(data.lane_encoder_fc2),
    tr1:  _prepareLayer(data.trunk_fc1),
    tr2:  _prepareLayer(data.trunk_fc2),
    pol:  _prepareLayer(data.policy_head),
  };
}

function _prepareLayer(d) {
  return {
    W:   new Float32Array(d.weight),
    b:   new Float32Array(d.bias),
    in:  d.in_features,
    out: d.out_features,
  };
}

// ── Primitives numériques ─────────────────────────────────────────────────────

function _linear(l, x) {
  const out = new Float32Array(l.out);
  for (let o = 0; o < l.out; o++) {
    let s = l.b[o];
    const off = o * l.in;
    for (let i = 0; i < l.in; i++) s += l.W[off + i] * x[i];
    out[o] = s;
  }
  return out;
}

function _relu(x) {
  const out = new Float32Array(x.length);
  for (let i = 0; i < x.length; i++) out[i] = x[i] > 0 ? x[i] : 0;
  return out;
}

function _argmax(x) {
  let best = 0;
  for (let i = 1; i < x.length; i++) if (x[i] > x[best]) best = i;
  return best;
}

// ── Passe avant ───────────────────────────────────────────────────────────────

function _infer(obs) {
  // LaneEncoder partagé appliqué indépendamment à chaque lane
  const laneEmbeds = new Float32Array(OBS_LANES * LANE_EMBED_DIM);  // (5, 32)
  for (let li = 0; li < OBS_LANES; li++) {
    const feat = obs.slice(li * LANE_FEAT, (li + 1) * LANE_FEAT);   // (11,)
    let e = _relu(_linear(_net.enc1, feat));                         // (64,)
    e     = _relu(_linear(_net.enc2, e));                            // (32,)
    laneEmbeds.set(e, li * LANE_EMBED_DIM);
  }

  // Concaténation avec player_pos (2 derniers éléments de obs)
  const trunkIn = new Float32Array(OBS_LANES * LANE_EMBED_DIM + 2); // (162,)
  trunkIn.set(laneEmbeds);
  trunkIn.set(obs.slice(OBS_LANES * LANE_FEAT), OBS_LANES * LANE_EMBED_DIM);

  // Trunk
  let h = _relu(_linear(_net.tr1, trunkIn));  // (256,)
  h     = _relu(_linear(_net.tr2, h));         // (128,)

  // Policy head → logits (5,)
  return _linear(_net.pol, h);
}

// ── Construction de l'observation ────────────────────────────────────────────

function _buildObs(env) {
  const playerIdx = env.playerRow - env.dequeStartRow;
  let start = Math.max(0, playerIdx - OBS_LOOK_BEHIND);
  let end   = start + OBS_LANES;
  if (end > env.lanes.length) {
    end   = env.lanes.length;
    start = Math.max(0, end - OBS_LANES);
  }
  const agentLanes = env.lanes.slice(start, end);

  const obs = new Float32Array(OBS_SIZE);
  let i = 0;
  for (const lane of agentLanes) {
    obs[i++] = lane.type;
    obs[i++] = lane.speed;
    for (let x = PLAYABLE_MIN; x <= PLAYABLE_MAX; x++) obs[i++] = lane.obstacleAt(x);
  }
  obs[i++] = (env.playerX - PLAYABLE_MIN) / OBS_PLAYABLE_W;
  // Plafonner le delta à PYTHON_LOOK_BEHIND=2 : en Python camera_start_row
  // est toujours à max 2 rangs derrière le joueur (LOOK_BEHIND=2), alors que
  // le jeu web utilise LOOK_BEHIND=5. Sans ce cap, le modèle reçoit des
  // valeurs hors distribution (~0.38 vs ~0.15 max vu à l'entraînement).
  obs[i]   = Math.min(env.playerRow - env.cameraStartRow, PYTHON_LOOK_BEHIND) / PYTHON_GRID_H;
  return obs;
}

// ── API publique ──────────────────────────────────────────────────────────────

/** Retourne l'action choisie par le réseau (0=stay 1=up 2=down 3=left 4=right). */
export function getAction(env) {
  if (!_net) throw new Error('loadAgent() doit être appelé avant getAction()');
  return _argmax(_infer(_buildObs(env)));
}
