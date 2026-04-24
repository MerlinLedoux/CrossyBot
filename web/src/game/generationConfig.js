class Table {
  constructor(...rows) {
    this._rows = rows;
  }

  _dist(score) {
    for (const [smin, smax, dist] of this._rows) {
      if (smin <= score && (smax === null || score < smax)) return dist;
    }
    return this._rows[this._rows.length - 1][2];
  }

  sample(score) {
    const dist    = this._dist(score);
    const entries = Object.entries(dist);
    const total   = entries.reduce((s, [, w]) => s + w, 0);
    let r = Math.random() * total;
    for (const [v, w] of entries) {
      r -= w;
      if (r <= 0) return Number(v);
    }
    return Number(entries[entries.length - 1][0]);
  }
}

class Prob {
  constructor(...rows) {
    this._rows = rows;
  }

  at(score) {
    for (const [smin, smax, val] of this._rows) {
      if (smin <= score && (smax === null || score < smax)) return val;
    }
    return this._rows[this._rows.length - 1][2];
  }
}

export const CONFIG = {
  grass_lines: new Table(
    [0, null, { 1: 40, 2: 30, 3: 20, 4: 10 }],
  ),

  road_riv_group_lines: new Table(
    [0,   50,  { 1: 20, 2: 40, 3: 20, 4: 20 }],
    [50,  100, { 1: 10, 2: 15, 3: 25, 4: 25, 5: 15, 6: 10 }],
    [100, 200, { 1: 5,  2: 10, 3: 15, 4: 20, 5: 20, 6: 15, 7: 10, 8: 5 }],
    [200, null,{ 3: 5,  4: 10, 5: 15, 6: 20, 7: 20, 8: 15, 9: 10, 10: 5 }],
  ),

  unsafe_prob: new Prob(
    [0,   50,  0.00],
    [50,  100, 0.25],
    [100, 200, 0.50],
    [200, null,0.75],
  ),

  car_speed: new Table(
    [0,   50,  { 0.50: 15, 0.75: 20, 1.00: 30, 1.25: 20, 1.50: 15 }],
    [50,  100, { 0.50: 15, 0.75: 15, 1.00: 15, 1.25: 15, 1.50: 15, 1.75: 15, 2.00: 10 }],
    [100, 200, { 0.50: 10, 0.75: 10, 1.00: 15, 1.25: 20, 1.50: 20, 1.75: 15, 2.00: 10 }],
    [200, null,{ 0.50: 5,  0.75: 10, 1.00: 10, 1.25: 15, 1.50: 20, 1.75: 25, 2.00: 15 }],
  ),

  car_size: new Table(
    [0,   100, { 1: 15, 2: 70, 3: 15 }],
    [100, null,{ 1: 15, 2: 60, 3: 25 }],
  ),

  car_space: new Table(
    [0,   50,  { 3: 10, 4: 20, 5: 30, 6: 25, 7: 10, 8: 5 }],
    [50,  100, { 2: 5,  3: 10, 4: 20, 5: 30, 6: 20, 7: 10, 8: 5 }],
    [100, 200, { 2: 5,  3: 15, 4: 30, 5: 25, 6: 20, 7: 5 }],
    [200, null,{ 2: 15, 3: 20, 4: 30, 5: 20, 6: 15 }],
  ),

  water_prob: new Prob(
    [0, null, 0.90],
  ),

  log_speed: new Table(
    [0,   50,  { 0.50: 15, 0.75: 20, 1.00: 30, 1.25: 20, 1.50: 15 }],
    [50,  100, { 0.50: 15, 0.75: 15, 1.00: 15, 1.25: 15, 1.50: 15, 1.75: 15, 2.00: 10 }],
    [100, 200, { 0.50: 10, 0.75: 10, 1.00: 15, 1.25: 20, 1.50: 20, 1.75: 15, 2.00: 10 }],
    [200, null,{ 0.50: 5,  0.75: 10, 1.00: 10, 1.25: 15, 1.50: 20, 1.75: 25, 2.00: 15 }],
  ),

  log_space: new Table(
    [0, null, { 0: 10, 1: 15, 2: 30, 3: 30, 4: 15 }],
  ),

  log_size: new Table(
    [0,   100, { 2: 70, 3: 15, 4: 15 }],
    [100, 200, { 1: 10, 2: 40, 3: 30, 4: 20 }],
    [200, null,{ 1: 20, 2: 40, 3: 30, 4: 10 }],
  ),

  lily_count: new Table(
    [0, null, { 1: 10, 2: 40, 3: 30, 4: 20 }],
  ),

  tree_count: new Table(
    [0, null, { 0: 10, 1: 10, 2: 30, 3: 30, 4: 20 }],
  ),
};
