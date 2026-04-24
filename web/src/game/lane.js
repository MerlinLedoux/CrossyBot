import { GRID_W, PLAYABLE_MIN, PLAYABLE_MAX, PLAYABLE_W, CELLS_PER_SEC, MAX_SPEED } from './constants.js';
import { CONFIG } from './generationConfig.js';

export const LaneType = Object.freeze({ SAFE: 0, GRASS: 1, ROAD: 2, WATER: 3, LILY: 4 });

const TYPE_TO_OBS = {
  [LaneType.SAFE]:  -1.00,
  [LaneType.GRASS]: -0.50,
  [LaneType.ROAD]:   0.00,
  [LaneType.WATER]:  0.50,
  [LaneType.LILY]:   1.00,
};

export class Lane {
  constructor(laneType, speedCps, score = 0) {
    this.laneType   = laneType;
    this._speed     = Math.max(-MAX_SPEED, Math.min(MAX_SPEED, (speedCps / CELLS_PER_SEC) * MAX_SPEED));
    this._score     = score;
    this._positions = [];
    this._widths    = [];

    if (laneType === LaneType.GRASS) {
      // Colonnes hors zone jouable : toujours des arbres (-2 à 14)
      for (let x = -2; x < GRID_W + 2; x++) {
        if (x < PLAYABLE_MIN || x > PLAYABLE_MAX) {
          this._positions.push(x);
          this._widths.push(1);
        }
      }
      // Colonnes jouables : arbres aléatoires
      const n = Math.min(CONFIG.tree_count.sample(score), PLAYABLE_W);
      if (n > 0) {
        for (const s of _sampleNoReplace(PLAYABLE_W, n).sort((a, b) => a - b)) {
          this._positions.push(PLAYABLE_MIN + s);
          this._widths.push(1);
        }
      }
      this._positions.sort((a, b) => a - b);
    } else if (laneType === LaneType.LILY) {
      const n = Math.min(CONFIG.lily_count.sample(score), PLAYABLE_W);
      if (n > 0) {
        for (const s of _sampleNoReplace(PLAYABLE_W, n).sort((a, b) => a - b)) {
          this._positions.push(PLAYABLE_MIN + s);
          this._widths.push(1);
        }
      }
    } else if (laneType === LaneType.WATER || laneType === LaneType.ROAD) {
      this._generateStream();
    }
  }

  // ── propriétés observation RL ─────────────────────────────────────────────
  get type()  { return TYPE_TO_OBS[this.laneType]; }
  get speed() { return this._speed / MAX_SPEED; }

  // ── génération du flux ────────────────────────────────────────────────────
  _sampleGap()   { return this.laneType === LaneType.WATER ? CONFIG.log_space.sample(this._score) : CONFIG.car_space.sample(this._score); }
  _sampleWidth() { return this.laneType === LaneType.WATER ? CONFIG.log_size.sample(this._score)  : CONFIG.car_size.sample(this._score); }

  _generateStream() {
    let x = -GRID_W + this._sampleGap();
    while (x < 2 * GRID_W) {
      const w = this._sampleWidth();
      this._positions.push(x);
      this._widths.push(w);
      x += w + this._sampleGap();
    }
  }

  _trimAndFill(movingRight) {
    if (movingRight) {
      while (this._positions.length && this._positions[this._positions.length - 1] >= GRID_W) {
        this._positions.pop(); this._widths.pop();
      }
      while (!this._positions.length || this._positions[0] > -GRID_W) {
        const gap = this._sampleGap(), width = this._sampleWidth();
        const left = this._positions.length ? this._positions[0] - gap - width : -GRID_W;
        this._positions.unshift(left); this._widths.unshift(width);
      }
    } else {
      while (this._positions.length && this._positions[0] + this._widths[0] <= 0) {
        this._positions.shift(); this._widths.shift();
      }
      while (!this._positions.length ||
             this._positions[this._positions.length - 1] + this._widths[this._widths.length - 1] < 2 * GRID_W) {
        const gap = this._sampleGap(), width = this._sampleWidth();
        const n = this._positions.length;
        const right = n ? this._positions[n - 1] + this._widths[n - 1] : GRID_W;
        this._positions.push(right + gap); this._widths.push(width);
      }
    }
  }

  // ── tests de collision ────────────────────────────────────────────────────
  hasObstacleAt(x) {
    for (let i = 0; i < this._positions.length; i++) {
      if (this._positions[i] < x + 1 && this._positions[i] + this._widths[i] > x) return true;
    }
    return false;
  }

  obstacleAt(x) {
    let cov = 0;
    for (let i = 0; i < this._positions.length; i++) {
      const ov = Math.min(this._positions[i] + this._widths[i], x + 1) - Math.max(this._positions[i], x);
      if (ov > 0) cov += ov;
    }
    return Math.min(cov, 1.0);
  }

  getLogAt(xf) {
    const hb = 0.25;
    for (let i = 0; i < this._positions.length; i++) {
      const p = this._positions[i], w = this._widths[i];
      if (p < xf + hb && p + w > xf - hb) return [p, w];
    }
    return null;
  }

  isOnLog(xf) {
    const hb = 0.25;
    for (let i = 0; i < this._positions.length; i++) {
      if (this._positions[i] < xf + hb && this._positions[i] + this._widths[i] > xf - hb) return true;
    }
    return false;
  }

  overlapsCell(x, hitbox = 0.5) {
    const margin = (1 - hitbox) / 2;
    const hbStart = x + margin, hbEnd = x + 1 - margin;
    for (let i = 0; i < this._positions.length; i++) {
      if (this._positions[i] < hbEnd && this._positions[i] + this._widths[i] > hbStart) return true;
    }
    return false;
  }

  *iterObstacles() {
    for (let i = 0; i < this._positions.length; i++) yield [this._positions[i], this._widths[i]];
  }

  // ── mise à jour ───────────────────────────────────────────────────────────
  update() {
    this._positions = this._positions.map(p => p + this._speed);
    if (this.laneType === LaneType.WATER || this.laneType === LaneType.ROAD)
      this._trimAndFill(this._speed > 0);
  }

  updateVisual(dt) {
    if (!this._positions.length || this._speed === 0) return;
    const delta = (this._speed / MAX_SPEED) * CELLS_PER_SEC * dt;
    this._positions = this._positions.map(p => p + delta);
    if (this.laneType === LaneType.WATER || this.laneType === LaneType.ROAD)
      this._trimAndFill(delta > 0);
  }
}

function _sampleNoReplace(n, k) {
  const arr = Array.from({ length: n }, (_, i) => i);
  for (let i = arr.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [arr[i], arr[j]] = [arr[j], arr[i]];
  }
  return arr.slice(0, k);
}
