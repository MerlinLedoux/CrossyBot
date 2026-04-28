import { createScene }  from './renderer/scene.js';
import { loadModels }   from './renderer/modelLoader.js';
import { GameView }     from './renderer/gameView.js';
import { CrossyEnv }    from './game/env.js';
import { LaneType }     from './game/lane.js';
import { PLAYABLE_MIN, PLAYABLE_MAX, CELLS_PER_SEC, MAX_SPEED, GRID_W } from './game/constants.js';
import { loadAgent, getAction } from './ai/agent.js';

const KEY_TO_ACTION = { ArrowUp: 1, ArrowDown: 2, ArrowLeft: 4, ArrowRight: 3 };

const SCROLL_SPEED = 0.5; // rangées par seconde

async function main() {
  const { renderer, scene, camera, sun } = createScene();
  const models = await loadModels();

  const env  = new CrossyEnv();
  const view = new GameView(scene, camera, models, sun);

  const scoreEl      = document.getElementById('score');
  const gameoverEl   = document.getElementById('gameover');
  const finalScoreEl = document.getElementById('final-score');
  const aiBtnEl      = document.getElementById('ai-btn');

  let dead      = false;
  let scrollRow = env.cameraStartRow;

  // ── Mode IA ────────────────────────────────────────────────────────────────
  let aiMode  = false;
  let aiTimer = 0;
  const AI_INTERVAL = 1 / 3;   // 3 actions/sec, identique à l'entraînement
  const ACTION_ROT  = { 1: Math.PI / 2, 2: -Math.PI / 2, 3: 0, 4: Math.PI };

  loadAgent('/crossybot.json').then(() => {
    aiBtnEl.disabled    = false;
    aiBtnEl.textContent = '🤖 Mode IA';
  }).catch(() => {
    aiBtnEl.textContent = '❌ Modèle absent';
    console.warn('Modèle IA introuvable — lancez : python export_model.py');
  });

  aiBtnEl.addEventListener('click', () => {
    aiMode  = !aiMode;
    aiTimer = 0;
    aiBtnEl.classList.toggle('active', aiMode);
    aiBtnEl.textContent = aiMode ? '👤 Mode Humain' : '🤖 Mode IA';
  });

  // ── Détection de mort ──────────────────────────────────────────────────────
  function checkCollision() {
    if (!(env.playerX >= PLAYABLE_MIN && env.playerX < PLAYABLE_MAX + 1)) return true;
    const lane = env.playerLane;
    if (lane.laneType === LaneType.ROAD  && lane.overlapsPosition(env.playerX, 0.1)) return true;
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
      dead      = false;
      scrollRow = env.cameraStartRow;
      aiTimer   = 0;
      gameoverEl.style.display = 'none';
      scoreEl.textContent      = '0';
      return;
    }

    if (dead || aiMode) return;

    const action = KEY_TO_ACTION[e.key];
    if (action === undefined) return;
    e.preventDefault();

    view.triggerHop();
    env._applyAction(action);
    env._trimLanes();
    env._ensureLanes();
    scoreEl.textContent = env.score;

    const ROT = { 1: Math.PI / 2, 2: -Math.PI / 2, 3: 0, 4: Math.PI };
    if (ROT[action] !== undefined) view.setPlayerRotation(ROT[action]);

    if (checkCollision()) triggerGameOver();
  });

  // ── Boucle d'animation ────────────────────────────────────────────────────
  let lastTime = performance.now();

  function animate(now) {
    requestAnimationFrame(animate);
    const dt = Math.min((now - lastTime) / 1000, 0.05);
    lastTime = now;

    if (!dead) {
      // ── Défilement automatique ─────────────────────────────────────────────
      // Si le joueur a avancé plus vite que le scroll, scrollRow le suit.
      scrollRow  = Math.max(scrollRow, env.cameraStartRow);
      scrollRow += SCROLL_SPEED * dt;
      env.cameraStartRow = Math.max(env.cameraStartRow, Math.floor(scrollRow));
      env._trimLanes();
      env._ensureLanes();

      // Mort si le joueur est repoussé hors de l'écran par le scroll
      if (env.playerRow < env.cameraStartRow) { triggerGameOver(); }

      const lane  = env.playerLane;
      const onLog = lane.laneType === LaneType.WATER && lane.isOnLog(env.playerX);

      // Déplacement visuel continu des obstacles
      for (const l of env.getVisibleLanes()) l.updateVisual(dt);

      // Entraînement du joueur par la bûche (visuel)
      if (onLog) {
        const delta = (lane._speed / MAX_SPEED) * CELLS_PER_SEC * dt;
        env.playerX = Math.max(-0.5, Math.min(GRID_W - 0.5, env.playerX + delta));
      }

      // ── Action de l'IA ────────────────────────────────────────────────────
      if (aiMode) {
        aiTimer += dt;
        if (aiTimer >= AI_INTERVAL) {
          aiTimer -= AI_INTERVAL;
          const action = getAction(env);
          if (action !== 0) {
            view.triggerHop();
            if (ACTION_ROT[action] !== undefined) view.setPlayerRotation(ACTION_ROT[action]);
          }
          env._applyAction(action);
          env._trimLanes();
          env._ensureLanes();
          scoreEl.textContent = env.score;
        }
      }

      // Vérification collision continue (voiture, eau, bord)
      if (checkCollision()) triggerGameOver();
    }

    view.update(env, dt, scrollRow);
    renderer.render(scene, camera);
  }

  requestAnimationFrame(animate);
}

main().catch(err => {
  console.error('Erreur de chargement :', err);
  document.body.innerHTML = `<pre style="color:red;padding:20px">Erreur : ${err.message}</pre>`;
});
