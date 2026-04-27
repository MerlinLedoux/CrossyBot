import * as THREE from 'three';
import { LaneType } from '../game/lane.js';
import { GRID_W, GRID_H, PLAYABLE_MIN, PLAYABLE_MAX, LOOK_BEHIND } from '../game/constants.js';

const RENDER_ROWS = GRID_H + 3;  // GRID_H + TRIM_BUFFER : pool de meshes couvrant les rows tampon

// ── couleurs de fond des lanes ────────────────────────────────────────────────
const LANE_COLORS = {
  [LaneType.SAFE]: 0x6ec84b,
  [LaneType.GRASS]: 0x8bb34f,
  [LaneType.ROAD]: 0x3d414e,
  [LaneType.WATER]: 0x67c4e9,
  [LaneType.LILY]: 0x67c4e9,
};
const GRASS_COLOR_ALT = 0x7aa345;

// ── hauteur Y de chaque type de lane ─────────────────────────────────────────
const LANE_Y = {
  [LaneType.SAFE]:  0.2,
  [LaneType.GRASS]: 0.2,
  [LaneType.ROAD]:  0.1,
  [LaneType.WATER]: 0,
  [LaneType.LILY]:  0,
};

// Hauteur supplémentaire du joueur quand il se tient sur un objet (bûche / nénuphar)
const PLAYER_Y_OFFSET = {
  [LaneType.WATER]: 0.15,
  [LaneType.LILY]:  0.08,
};

const HOP_HEIGHT   = 0.35;  // hauteur max de l'arc en unités monde
const HOP_DURATION = 0.13;  // durée du saut en secondes

// Épaisseur des dalles de terrain (assez grande pour ne jamais voir de vide)
const SLAB_THICKNESS = 0.5;

// ── géométrie des 3 dalles par lane ──────────────────────────────────────────
const EXTRA_COLS  = 3;
const SIDE_W      = PLAYABLE_MIN + EXTRA_COLS;                        // 5
const CENTER_W    = PLAYABLE_MAX - PLAYABLE_MIN + 1;                  // 9
const RIGHT_W     = GRID_W + EXTRA_COLS - PLAYABLE_MAX - 1;          // 5
const LEFT_CX     = (PLAYABLE_MIN - EXTRA_COLS) / 2;                 // -0.5
const CENTER_CX   = (PLAYABLE_MIN + PLAYABLE_MAX + 1) / 2;           // 6.5
const RIGHT_CX    = (PLAYABLE_MAX + 1 + GRID_W + EXTRA_COLS) / 2;   // 13.5

// Couleurs des murs (colonnes 0-1 et 11-12)
const WALL_COLOR = 0x1a1a1a;

// ── échelles des modèles — à ajuster en fonction de la taille Blender ─────────
const MODEL_SCALE = {
  player: 0.1,
  [LaneType.GRASS]: 0.1,
  [LaneType.ROAD]: 0.1,
  [LaneType.WATER]: 0.1,
  [LaneType.LILY]: 0.1,
};

// Taille de base (en cases) de chaque modèle tel qu'exporté depuis Blender.
// Sert à normaliser la mise à l'échelle sur la largeur réelle de l'obstacle.
const MODEL_BASE_WIDTH = {
  [LaneType.ROAD]: 2,   // modèle voiture = 2 cases
  [LaneType.WATER]: 3,   // modèle bûche   = 3 cases
};

export class GameView {
  constructor(scene, camera, models, sun) {
    this.scene = scene;
    this.camera = camera;
    this.models = models;
    this._sun = sun;

    // Groupes Three.js
    this.laneGroup = new THREE.Group();
    this.obstacleGroup = new THREE.Group();
    scene.add(this.laneGroup);
    scene.add(this.obstacleGroup);

    // ── Fond de lanes : 3 dalles par lane (gauche / centre / droite) ─────────
    const leftGeo   = new THREE.BoxGeometry(SIDE_W,   SLAB_THICKNESS, 1);
    const centerGeo = new THREE.BoxGeometry(CENTER_W, SLAB_THICKNESS, 1);
    const rightGeo  = new THREE.BoxGeometry(RIGHT_W,  SLAB_THICKNESS, 1);
    this._laneL = []; this._laneC = []; this._laneR = [];
    this._laneMatL = []; this._laneMatC = []; this._laneMatR = [];
    for (let i = 0; i < RENDER_ROWS; i++) {
      const mL = new THREE.MeshLambertMaterial({ color: 0x000000 });
      const mC = new THREE.MeshLambertMaterial({ color: LANE_COLORS[LaneType.SAFE] });
      const mR = new THREE.MeshLambertMaterial({ color: 0x000000 });
      const meshL = new THREE.Mesh(leftGeo,   mL);
      const meshC = new THREE.Mesh(centerGeo, mC);
      const meshR = new THREE.Mesh(rightGeo,  mR);
      meshL.position.x = LEFT_CX;
      meshC.position.x = CENTER_CX;
      meshR.position.x = RIGHT_CX;
      meshL.receiveShadow = meshC.receiveShadow = meshR.receiveShadow = true;
      this.laneGroup.add(meshL); this.laneGroup.add(meshC); this.laneGroup.add(meshR);
      this._laneL.push(meshL); this._laneC.push(meshC); this._laneR.push(meshR);
      this._laneMatL.push(mL); this._laneMatC.push(mC); this._laneMatR.push(mR);
    }

    // ── Murs latéraux (bandes permanentes gauche et droite) ───────────────────
    const wallGeoL = new THREE.PlaneGeometry(2, 200);
    const wallGeoR = new THREE.PlaneGeometry(2, 200);
    const wallMat = new THREE.MeshLambertMaterial({ color: WALL_COLOR });
    const wallL = new THREE.Mesh(wallGeoL, wallMat);
    const wallR = new THREE.Mesh(wallGeoR, wallMat);
    wallL.rotation.x = -Math.PI / 2;
    wallR.rotation.x = -Math.PI / 2;
    wallL.position.set(1, -0.01, 100);
    wallR.position.set(GRID_W - 1, -0.01, 100);
    wallL.receiveShadow = wallR.receiveShadow = true;
    scene.add(wallL); scene.add(wallR);
    this._wallL = wallL; this._wallR = wallR;




    // ── Marquages au sol entre lanes de route consécutives ───────────────────
    const roadW  = PLAYABLE_MAX - PLAYABLE_MIN + 1;
    const roadCx = (PLAYABLE_MIN + PLAYABLE_MAX + 1) / 2;
    const dashCanvas = document.createElement('canvas');
    dashCanvas.width = 128; dashCanvas.height = 8;
    const dctx = dashCanvas.getContext('2d');
    dctx.clearRect(0, 0, 128, 8);
    dctx.fillStyle = '#202125';
    dctx.fillRect(64, 0, 128, 8);
    const fullW = GRID_W + EXTRA_COLS * 2;
    const dashTex = new THREE.CanvasTexture(dashCanvas);
    dashTex.wrapS = THREE.RepeatWrapping;
    dashTex.repeat.set(fullW / 2, 1);
    const stripeGeo = new THREE.PlaneGeometry(fullW, 0.10);
    const stripeMat = new THREE.MeshBasicMaterial({ map: dashTex, transparent: true, alphaTest: 0.4 });
    this._stripes = [];
    for (let i = 0; i < RENDER_ROWS; i++) {
      const mesh = new THREE.Mesh(stripeGeo, stripeMat);
      mesh.rotation.x = -Math.PI / 2;
      mesh.position.x = GRID_W / 2;
      mesh.position.y = LANE_Y[LaneType.ROAD] + 0.003;
      mesh.visible = false;
      scene.add(mesh);
      this._stripes.push(mesh);
    }

    // ── Joueur ────────────────────────────────────────────────────────────────
    const s = MODEL_SCALE.player;
    this.playerMesh = models.player.clone();
    this.playerMesh.scale.setScalar(s);
    this.playerMesh.rotation.y = Math.PI/2;
    this.playerMesh.traverse(c => { if (c.isMesh) { c.castShadow = true; c.receiveShadow = true; } });
    scene.add(this.playerMesh);

    // Cible de caméra lissée
    this._camTarget = new THREE.Vector3();

    // Animation de saut
    this._playerVisualPos = new THREE.Vector3();
    this._playerAnimFrom  = new THREE.Vector3();
    this._playerAnimTo    = new THREE.Vector3();
    this._playerAnimT     = 1;
  }

  triggerHop() {
    this._playerAnimFrom.copy(this._playerVisualPos);
    this._playerAnimT = 0;
  }

  setPlayerRotation(y) {
    this.playerMesh.rotation.y = y;
  }

  update(env, dt, scrollRow = env.cameraStartRow) {
    const visLanes = env.getVisibleLanes();
    const camRow = env.dequeStartRow;   // = cameraStartRow - TRIM_BUFFER
    const playerRow = env.playerRow;
    const playerX = env.playerX;

    // ── Soleil suit le joueur ─────────────────────────────────────────────────
    this._sun.position.set(playerX + 10, 10, playerRow);
    this._sun.target.position.set(playerX, 0, playerRow);

    // Mise à jour des fonds de lanes (3 dalles par lane)
    for (let i = 0; i < RENDER_ROWS; i++) {
      const meshL = this._laneL[i], meshC = this._laneC[i], meshR = this._laneR[i];
      if (i < visLanes.length) {
        const lane   = visLanes[i];
        const rowAbs = camRow + i;
        const laneY  = LANE_Y[lane.laneType] ?? 0;
        const y      = laneY - SLAB_THICKNESS / 2;
        const z      = camRow + i;

        const centerColor = lane.laneType === LaneType.GRASS && rowAbs % 2 === 1
          ? GRASS_COLOR_ALT : LANE_COLORS[lane.laneType];
        const sideColor = new THREE.Color(centerColor).multiplyScalar(0.65);

        this._laneMatC[i].color.setHex(centerColor);
        this._laneMatL[i].color.copy(sideColor);
        this._laneMatR[i].color.copy(sideColor);

        meshL.position.set(LEFT_CX,   y, z);
        meshC.position.set(CENTER_CX, y, z);
        meshR.position.set(RIGHT_CX,  y, z);
        meshL.visible = meshC.visible = meshR.visible = true;
      } else {
        meshL.visible = meshC.visible = meshR.visible = false;
      }
    }


    // Mise à jour des murs
    this._wallL.position.z = playerRow;
    this._wallR.position.z = playerRow;


    // Marquages au sol
    let si = 0;
    for (let i = 0; i < visLanes.length - 1; i++) {
      if (visLanes[i].laneType === LaneType.ROAD && visLanes[i + 1].laneType === LaneType.ROAD) {
        this._stripes[si].position.z = camRow + i + 0.5;
        this._stripes[si].visible = true;
        si++;
      }
    }
    for (; si < this._stripes.length; si++) this._stripes[si].visible = false;

    // Reconstruction des obstacles
    // On vide le groupe et on le reconstruit chaque frame.
    // Avec GRID_H=13 lignes max, c'est acceptable en perf.
    this.obstacleGroup.clear();

    for (let i = 0; i < visLanes.length; i++) {
      const lane = visLanes[i];
      const rowAbs = camRow + i;
      const baseModel = this.models[lane.laneType];
      if (!baseModel) continue;

      const scale = MODEL_SCALE[lane.laneType] ?? 0.5;
      const movingLeft = lane._speed < 0;

      for (const [pos, width] of lane.iterObstacles()) {
        // Ne pas dessiner les obstacles hors de l'écran
        if (pos + width < -1 || pos > GRID_W + 1) continue;

        const mesh = baseModel.clone();
        mesh.scale.setScalar(scale);

        // Largeur proportionnelle pour voitures et bûches
        if (lane.laneType === LaneType.ROAD || lane.laneType === LaneType.WATER) {
          const baseW = MODEL_BASE_WIDTH[lane.laneType] ?? 1;
          mesh.scale.x = scale * (width / baseW);
        }

        // Retournement : voitures toujours dans le sens de déplacement
        if (lane.laneType === LaneType.ROAD) {
          mesh.rotation.y = movingLeft ? 0 : Math.PI;
        }
        if (lane.laneType === LaneType.LILY) {
          const t = performance.now() / 1000;
          const phase = ((rowAbs * 31 + Math.floor(pos)) * 2.399963) % (Math.PI * 2);
          mesh.rotation.y = phase + t * 0.25;
        }

        // Position centrée sur la case/obstacle
        const cx = pos + width / 2;
        mesh.position.set(cx, LANE_Y[lane.laneType] ?? 0, rowAbs);

        mesh.traverse(c => { if (c.isMesh) { c.castShadow = true; c.receiveShadow = true; } });
        this.obstacleGroup.add(mesh);
      }
    }

    // ── Animation de saut du joueur ───────────────────────────────────────────
    const laneType = env.playerLane.laneType;
    const targetY = (LANE_Y[laneType] ?? 0) + (PLAYER_Y_OFFSET[laneType] ?? 0);
    this._playerAnimTo.set(playerX, targetY, playerRow);

    if (this._playerAnimT < 1) {
      this._playerAnimT = Math.min(1, this._playerAnimT + dt / HOP_DURATION);
      const t = this._playerAnimT;
      this._playerVisualPos.lerpVectors(this._playerAnimFrom, this._playerAnimTo, t);
      this._playerVisualPos.y += Math.sin(t * Math.PI) * HOP_HEIGHT;
    } else {
      this._playerVisualPos.copy(this._playerAnimTo);
    }

    this.playerMesh.position.copy(this._playerVisualPos);

    // ── Caméra : avance avec le scroll, suivi latéral amorti ─────────────────
    // Z : scroll ou joueur (si en avance)
    const camTargetZ = Math.max(scrollRow + LOOK_BEHIND, this._playerVisualPos.z);
    // X : amorti vers le centre de la carte — évite de montrer le vide sur les côtés
    const mapCenterX = GRID_W / 2;
    const camTargetX = mapCenterX + (this._playerVisualPos.x - mapCenterX) * 0.2;
    this._camTarget.lerp(
      new THREE.Vector3(camTargetX, 0, camTargetZ),
      0.12,
    );

    this.camera.position.set(
      this._camTarget.x - 2.5,
      7,
      this._camTarget.z - 5,
    );
    this.camera.lookAt(this._camTarget.x, 0, this._camTarget.z + 2);
  }
}
