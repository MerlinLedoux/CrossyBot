export const GRID_W        = 13;
export const PLAYABLE_MIN  = 2;
export const PLAYABLE_MAX  = 10;
export const PLAYABLE_W    = PLAYABLE_MAX - PLAYABLE_MIN + 1; // 9

export const CELLS_PER_SEC = 3.0;
export const MAX_SPEED     = 1.0;  // cases/step

export const GRID_H        = 20;   // lignes visibles (+3 derrière, +4 devant)
export const LOOK_BEHIND   = 5;   // caméra recule jusqu'à 5 lignes derrière
export const LOOK_AHEAD    = 10;
export const MIN_LANES     = 28;
export const MAX_LANES     = 25;
