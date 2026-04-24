import { createScene }  from './renderer/scene.js';
import { loadModels }   from './renderer/modelLoader.js';
import { GameView }     from './renderer/gameView.js';
import { CrossyEnv }    from './game/env.js';
import { LaneType }     from './game/lane.js';
import { PLAYABLE_MIN, PLAYABLE_MAX, CELLS_PER_SEC, MAX_SPEED, GRID_W } from './game/constants.js';

const KEY_TO_ACTION = { ArrowUp: 1, ArrowDown: 2, ArrowLeft: 3, ArrowRight: 4 };

async function main() {
  const { renderer, scene, camera } = createScene();
  const models = await loadModels();

  const env  = new CrossyEnv();
  const view = new GameView(scene, camera, models);

  const scoreEl     = document.getElementById('score');
  const gameoverEl  = document.getElementById('gameover');
  const finalScoreEl = document.getElementById('final-score');

  let dead = false;

  // ── Détection de mort ──────────────────────────────────────────────────────
  function checkCollision() {
    if (!(env.playerX >= PLAYABLE_MIN && env.playerX < PLAYABLE_MAX + 1)) return true;
    const lane = env.playerLane;
    if (lane.laneType === LaneType.ROAD  && lane.overlapsCell(Math.floor(env.playerX), 0.5)) return true;
    if (lane.laneType === LaneType.LILY  && !lane.hasObstacleAt(Math.floor(env.playerX)))    return true;
    if (lane.laneType === LaneType.WATER && !lane.isOnLog(env.playerX))                      return true;
    return false;
  }

  function triggerGameOver() {
    if (dead) return;
    dead = true;
    gameoverEl.style.display   = 'block';
    finalScoreEl.textContent   = `Score final : ${env.score}`;
  }

  // ── Entrées clavier ────────────────────────────────────────────────────────
  document.addEventListener('keydown', e => {
    if (e.key === 'r' || e.key === 'R') {
      env.reset();
      dead = false;
      gameoverEl.style.display = 'none';
      scoreEl.textContent      = '0';
      return;
    }

    if (dead) return;

    const action = KEY_TO_ACTION[e.key];
    if (action === undefined) return;
    e.preventDefault();

    env._applyAction(action);
    env._trimLanes();
    env._ensureLanes();
    scoreEl.textContent = env.score;

    if (checkCollision()) triggerGameOver();
  });

  // ── Boucle d'animation ────────────────────────────────────────────────────
  let lastTime = performance.now();

  function animate(now) {
    requestAnimationFrame(animate);
    const dt = Math.min((now - lastTime) / 1000, 0.05);
    lastTime = now;

    if (!dead) {
      const lane  = env.playerLane;
      const onLog = lane.laneType === LaneType.WATER && lane.isOnLog(env.playerX);

      // Déplacement visuel continu des obstacles
      for (const l of env.getVisibleLanes()) l.updateVisual(dt);

      // Entraînement du joueur par la bûche (visuel)
      if (onLog) {
        const delta = (lane._speed / MAX_SPEED) * CELLS_PER_SEC * dt;
        env.playerX = Math.max(-0.5, Math.min(GRID_W - 0.5, env.playerX + delta));
      }

      // Vérification collision continue (voiture, eau, bord)
      if (checkCollision()) triggerGameOver();
    }

    view.update(env);
    renderer.render(scene, camera);
  }

  requestAnimationFrame(animate);
}

main().catch(err => {
  console.error('Erreur de chargement :', err);
  document.body.innerHTML = `<pre style="color:red;padding:20px">Erreur : ${err.message}</pre>`;
});
