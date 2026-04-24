import { Lane, LaneType } from './lane.js';
import { GRID_W, PLAYABLE_MIN, PLAYABLE_MAX, GRID_H, LOOK_BEHIND, MIN_LANES, MAX_LANES } from './constants.js';
import { CONFIG } from './generationConfig.js';

const ACTION_DELTAS = { 0: [0, 0], 1: [0, 1], 2: [0, -1], 3: [1, 0], 4: [-1, 0] };

const BLOCKING_ROWS = 3;   // lignes d'herbe bloquantes au départ

function makeBlockingGrass() {
  const lane = new Lane(LaneType.GRASS, 0, 0);
  lane._positions = [];
  lane._widths    = [];
  for (let x = -2; x < GRID_W + 2; x++) {
    lane._positions.push(x);
    lane._widths.push(1);
  }
  return lane;
}

export class CrossyEnv {
  constructor() { this.reset(); }

  reset() {
    this.playerX        = Math.floor(GRID_W / 2) + 0.5;  // 6.5
    this.playerRow      = 0;
    this.playerLogSlot  = null;
    this.dequeStartRow  = 0;
    this.cameraStartRow = 0;
    this.steps          = 0;
    this.score          = 0;
    this._startRow      = 0;
    this._lastWasUnsafe = false;
    this.dead           = false;
    this.lanes          = [];
    this._initLanes();   // fixe playerRow et _startRow
  }

  get playerLane() { return this.lanes[this.playerRow - this.dequeStartRow]; }

  getVisibleLanes() {
    const start = this.cameraStartRow - this.dequeStartRow;
    return this.lanes.slice(start, start + GRID_H);
  }

  // ── terrain ───────────────────────────────────────────────────────────────
  _initLanes() {
    // 3 lignes d'herbe entièrement bloquées → mur naturel derrière le joueur
    for (let i = 0; i < BLOCKING_ROWS; i++) this.lanes.push(makeBlockingGrass());
    // Ligne de départ sûre
    this.lanes.push(new Lane(LaneType.SAFE, 0, 0));
    // Le joueur démarre sur la ligne SAFE, le score est relatif à cette position
    this.playerRow  = BLOCKING_ROWS;
    this._startRow  = BLOCKING_ROWS;
    // Générer les lanes devant
    while (this.lanes.length < MAX_LANES) for (const l of this._newSection()) this.lanes.push(l);
    while (this.lanes.length > MAX_LANES) this.lanes.pop();
  }

  _newSection() {
    const s = this.score;
    const section = [];
    if (!this._lastWasUnsafe) {
      section.push(...(Math.random() < 0.5 ? this._makeRoadGroup(s) : this._makeRiverGroup(s)));
      this._lastWasUnsafe = true;
    } else {
      if (Math.random() < CONFIG.unsafe_prob.at(s)) {
        section.push(...(Math.random() < 0.5 ? this._makeRoadGroup(s) : this._makeRiverGroup(s)));
        this._lastWasUnsafe = true;
      } else {
        const n = CONFIG.grass_lines.sample(s);
        const grass = Array.from({ length: n }, () => new Lane(LaneType.GRASS, 0, s));
        this._validateGrassGroup(grass);
        section.push(...grass);
        this._lastWasUnsafe = false;
      }
    }
    return section;
  }

  _makeRoadGroup(score) {
    const n = CONFIG.road_riv_group_lines.sample(score);
    return Array.from({ length: n }, () => {
      const speed = CONFIG.car_speed.sample(score) * (Math.random() < 0.5 ? -1 : 1);
      return new Lane(LaneType.ROAD, speed, score);
    });
  }

  _makeRiverGroup(score) {
    const n = CONFIG.road_riv_group_lines.sample(score);
    const waterP = CONFIG.water_prob.at(score);
    const lanes = [];
    const lastDirs = [];

    for (let i = 0; i < n; i++) {
      if (Math.random() < waterP) {
        const spd = CONFIG.log_speed.sample(score);
        let dir;
        if (lastDirs.length >= 2 && lastDirs[lastDirs.length - 1] === lastDirs[lastDirs.length - 2])
          dir = -lastDirs[lastDirs.length - 1];
        else
          dir = Math.random() < 0.5 ? -1 : 1;
        lastDirs.push(dir);
        lanes.push(new Lane(LaneType.WATER, spd * dir, score));
      } else {
        lastDirs.length = 0;
        lanes.push(new Lane(LaneType.LILY, 0, score));
      }
    }

    let run = [];
    for (const lane of lanes) {
      if (lane.laneType === LaneType.LILY) { run.push(lane); }
      else { if (run.length > 1) this._validateLilyConnectivity(run); run = []; }
    }
    if (run.length > 1) this._validateLilyConnectivity(run);
    return lanes;
  }

  _validateGrassGroup(lanes) {
    if (lanes.length <= 1) return;
    for (let attempt = 0; attempt < 5; attempt++) {
      if (this._bfsGrass(lanes)) return;
      const col = PLAYABLE_MIN + Math.floor(Math.random() * (PLAYABLE_MAX - PLAYABLE_MIN + 1));
      for (const lane of lanes) {
        const newP = [], newW = [];
        for (let i = 0; i < lane._positions.length; i++) {
          if (Math.floor(lane._positions[i] % GRID_W) !== col) {
            newP.push(lane._positions[i]); newW.push(lane._widths[i]);
          }
        }
        lane._positions = newP; lane._widths = newW;
      }
    }
  }

  _bfsGrass(lanes) {
    const n = lanes.length;
    const queue = [];
    for (let x = PLAYABLE_MIN; x <= PLAYABLE_MAX; x++)
      if (!lanes[0].hasObstacleAt(x)) queue.push([0, x]);
    if (!queue.length) return false;
    const visited = new Set(queue.map(([r, c]) => `${r},${c}`));
    let head = 0;
    while (head < queue.length) {
      const [row, col] = queue[head++];
      if (row === n - 1) return true;
      for (const [dr, dc] of [[1,0],[-1,0],[0,1],[0,-1]]) {
        const nr = row + dr, nc = col + dc;
        if (nc < PLAYABLE_MIN || nc > PLAYABLE_MAX) continue;
        const key = `${nr},${nc}`;
        if (nr >= 0 && nr < n && !visited.has(key) && !lanes[nr].hasObstacleAt(nc)) {
          visited.add(key); queue.push([nr, nc]);
        }
      }
    }
    return false;
  }

  _validateLilyConnectivity(lilyLanes) {
    for (let i = 0; i < lilyLanes.length - 1; i++) {
      const a = lilyLanes[i], b = lilyLanes[i + 1];
      const bCols = new Set();
      for (const [pos] of b.iterObstacles()) {
        const col = Math.floor(pos);
        if (col >= PLAYABLE_MIN && col <= PLAYABLE_MAX) bCols.add(col);
      }
      for (const [pos] of a.iterObstacles()) {
        const col = Math.floor(pos);
        if (col < PLAYABLE_MIN || col > PLAYABLE_MAX) continue;
        if (!bCols.has(col)) { b._positions.push(col); b._widths.push(1); bCols.add(col); }
      }
    }
  }

  // ── buffer management ─────────────────────────────────────────────────────
  _trimLanes() {
    while (this.dequeStartRow < this.cameraStartRow) { this.lanes.shift(); this.dequeStartRow++; }
  }

  _ensureLanes() {
    while (this.lanes.length < MIN_LANES) for (const l of this._newSection()) this.lanes.push(l);
  }

  // ── actions ───────────────────────────────────────────────────────────────
  _applyAction(action) {
    const [dx, drow] = ACTION_DELTAS[action];
    if (dx === 0 && drow === 0) return;

    const off  = this.dequeStartRow;
    const lane = this.lanes[this.playerRow - off];

    // Mouvement horizontal sur eau : déplacement d'un slot entier
    if (dx !== 0 && drow === 0 && lane.laneType === LaneType.WATER) {
      if (this.playerLogSlot !== null) {
        const log = lane.getLogAt(this.playerX);
        if (log !== null) {
          const [logStart, logWidth] = log;
          const newSlot = this.playerLogSlot + dx;
          const newX    = logStart + newSlot + 0.5;
          if (newSlot >= 0 && newSlot < logWidth) {
            this.playerLogSlot = newSlot; this.playerX = newX;
          } else {
            const adj = lane.getLogAt(newX);
            if (adj !== null) {
              const [adjStart, adjWidth] = adj;
              const adjSlot = Math.max(0, Math.min(adjWidth - 1, Math.floor(newX - adjStart)));
              this.playerLogSlot = adjSlot; this.playerX = adjStart + adjSlot + 0.5;
            } else {
              this.playerX = newX; this.playerLogSlot = null;
            }
          }
        }
      }
      return;
    }

    // Mouvement normal
    const curCellX = Math.floor(this.playerX);
    const newCellX = Math.max(PLAYABLE_MIN, Math.min(PLAYABLE_MAX, curCellX + dx));
    const newRow   = Math.max(this.cameraStartRow, this.playerRow + drow);
    const newIdx   = newRow - off;

    if (newIdx < this.lanes.length) {
      const target = this.lanes[newIdx];
      if (target.laneType === LaneType.GRASS && target.hasObstacleAt(newCellX)) return;
    }

    let targetX = newCellX + 0.5, targetSlot = null;
    if (drow !== 0 && newIdx < this.lanes.length) {
      const targetLane = this.lanes[newIdx];
      if (targetLane.laneType === LaneType.WATER) {
        const log = targetLane.getLogAt(targetX);
        if (log !== null) {
          const [logStart, logWidth] = log;
          const slot = Math.max(0, Math.min(logWidth - 1, Math.floor(targetX - logStart)));
          targetSlot = slot; targetX = logStart + slot + 0.5;
        }
      }
    }

    this.playerX        = targetX;
    this.playerRow      = newRow;
    this.playerLogSlot  = targetSlot;
    this.score          = Math.max(this.score, this.playerRow - this._startRow);
    this.cameraStartRow = Math.max(this.cameraStartRow, this.playerRow - LOOK_BEHIND);
  }

  _updateObstacles() {
    const pi = this.playerRow - this.dequeStartRow;
    const lo = Math.max(0, pi - LOOK_BEHIND);
    const hi = Math.min(this.lanes.length, pi + 15);
    for (let i = lo; i < hi; i++) this.lanes[i].update();
  }

  _carryPlayer() {
    if (this.playerLogSlot === null) return;
    const lane = this.lanes[this.playerRow - this.dequeStartRow];
    if (lane.laneType !== LaneType.WATER) return;
    const log = lane.getLogAt(this.playerX);
    if (!log) return;
    const [logStart] = log;
    this.playerX = Math.max(-0.5, Math.min(GRID_W - 0.5,
      logStart + lane._speed + this.playerLogSlot + 0.5));
  }

  // ── mort ──────────────────────────────────────────────────────────────────
  isDead() {
    if (!(this.playerX >= PLAYABLE_MIN && this.playerX < PLAYABLE_MAX + 1)) return true;
    const lane = this.playerLane;
    if (lane.laneType === LaneType.ROAD)  return lane.overlapsPosition(this.playerX, 0.3);
    if (lane.laneType === LaneType.LILY)  return !lane.hasObstacleAt(Math.floor(this.playerX));
    if (lane.laneType === LaneType.WATER) return !lane.isOnLog(this.playerX);
    return false;
  }
}
