import * as THREE from 'three';
import { LaneType } from '../game/lane.js';
import { GRID_W, GRID_H } from '../game/constants.js';

// ── couleurs de fond des lanes ────────────────────────────────────────────────
const LANE_COLORS = {
  [LaneType.SAFE]:  0x6ec84b,
  [LaneType.GRASS]: 0x378020,
  [LaneType.ROAD]:  0x2e2e2e,
  [LaneType.WATER]: 0x1a6fc4,
  [LaneType.LILY]:  0x1458a0,
};

// Couleurs des murs (colonnes 0-1 et 11-12)
const WALL_COLOR = 0x1a1a1a;

// ── échelles des modèles — à ajuster en fonction de la taille Blender ─────────
const MODEL_SCALE = {
  player:           0.1,
  [LaneType.GRASS]: 0.1,
  [LaneType.ROAD]:  0.1,
  [LaneType.WATER]: 0.1,
  [LaneType.LILY]:  0.1,
};

// Taille de base (en cases) de chaque modèle tel qu'exporté depuis Blender.
// Sert à normaliser la mise à l'échelle sur la largeur réelle de l'obstacle.
const MODEL_BASE_WIDTH = {
  [LaneType.ROAD]:  2,   // modèle voiture = 2 cases
  [LaneType.WATER]: 3,   // modèle bûche   = 3 cases
};

export class GameView {
  constructor(scene, camera, models) {
    this.scene   = scene;
    this.camera  = camera;
    this.models  = models;

    // Groupes Three.js
    this.laneGroup     = new THREE.Group();
    this.obstacleGroup = new THREE.Group();
    scene.add(this.laneGroup);
    scene.add(this.obstacleGroup);

    // ── Fond de lanes : pool de GRID_H plans ─────────────────────────────────
    // EXTRA_COLS = 3 colonnes supplémentaires non jouables sur chaque côté.
    // Le plan s'étend de x = -3 à x = GRID_W + 3, centre inchangé (GRID_W/2).
    const EXTRA_COLS = 3;
    const planeGeo = new THREE.PlaneGeometry(GRID_W + EXTRA_COLS * 2, 1);
    this._laneMeshes    = [];
    this._laneMaterials = [];
    for (let i = 0; i < GRID_H; i++) {
      const mat  = new THREE.MeshLambertMaterial({ color: LANE_COLORS[LaneType.SAFE] });
      const mesh = new THREE.Mesh(planeGeo, mat);
      mesh.rotation.x = -Math.PI / 2;
      this.laneGroup.add(mesh);
      this._laneMeshes.push(mesh);
      this._laneMaterials.push(mat);
    }

    // ── Murs latéraux (bandes permanentes gauche et droite) ───────────────────
    const wallGeoL = new THREE.PlaneGeometry(2, 200);
    const wallGeoR = new THREE.PlaneGeometry(2, 200);
    const wallMat  = new THREE.MeshLambertMaterial({ color: WALL_COLOR });
    const wallL = new THREE.Mesh(wallGeoL, wallMat);
    const wallR = new THREE.Mesh(wallGeoR, wallMat);
    wallL.rotation.x = -Math.PI / 2;
    wallR.rotation.x = -Math.PI / 2;
    wallL.position.set(1,          -0.01, 100);
    wallR.position.set(GRID_W - 1, -0.01, 100);
    scene.add(wallL); scene.add(wallR);
    this._wallL = wallL; this._wallR = wallR;

    // ── Assombrissement des colonnes hors grille (-3 à 0 et 13 à 16) ─────────
    // Plan large centré sur la zone extra de chaque côté, suit le joueur en Z.
    const EXTRA = 5;
    const darkGeo = new THREE.PlaneGeometry(EXTRA, 60);
    const darkMat = new THREE.MeshBasicMaterial({
      color: 0x000000, transparent: true, opacity: 0.28, depthWrite: false,
    });
    this._darkL = new THREE.Mesh(darkGeo, darkMat);
    this._darkR = new THREE.Mesh(darkGeo, darkMat);
    this._darkL.rotation.x = -Math.PI / 2;
    this._darkR.rotation.x = -Math.PI / 2;
    this._darkL.position.set(-0.5,          0.002, 0);
    this._darkR.position.set(GRID_W + 0.5,  0.002, 0);
    scene.add(this._darkL); scene.add(this._darkR);

    // ── Joueur ────────────────────────────────────────────────────────────────
    const s = MODEL_SCALE.player;
    this.playerMesh = models.player.clone();
    this.playerMesh.scale.setScalar(s);
    this.playerMesh.traverse(c => { if (c.isMesh) { c.castShadow = true; c.receiveShadow = true; } });
    scene.add(this.playerMesh);

    // Cible de caméra lissée
    this._camTarget = new THREE.Vector3();
  }

  update(env) {
    const visLanes  = env.getVisibleLanes();
    const camRow    = env.cameraStartRow;
    const playerRow = env.playerRow;
    const playerX   = env.playerX;

    // ── Mise à jour des fonds de lanes ────────────────────────────────────────
    for (let i = 0; i < GRID_H; i++) {
      const mesh = this._laneMeshes[i];
      if (i < visLanes.length) {
        this._laneMaterials[i].color.setHex(LANE_COLORS[visLanes[i].laneType]);
        mesh.position.set(GRID_W / 2, 0, camRow + i);
        mesh.visible = true;
      } else {
        mesh.visible = false;
      }
    }

    // ── Mise à jour de la position des murs et des zones sombres ─────────────
    this._wallL.position.z  = playerRow;
    this._wallR.position.z  = playerRow;
    this._darkL.position.z  = playerRow;
    this._darkR.position.z  = playerRow;

    // ── Reconstruction des obstacles ─────────────────────────────────────────
    // On vide le groupe et on le reconstruit chaque frame.
    // Avec GRID_H=13 lignes max, c'est acceptable en perf.
    this.obstacleGroup.clear();

    for (let i = 0; i < visLanes.length; i++) {
      const lane    = visLanes[i];
      const rowAbs  = camRow + i;
      const baseModel = this.models[lane.laneType];
      if (!baseModel) continue;

      const scale      = MODEL_SCALE[lane.laneType] ?? 0.5;
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

        // Retournement pour les obstacles allant à gauche
        if (movingLeft) mesh.rotation.y = Math.PI;

        // Position centrée sur la case/obstacle
        const cx = pos + width / 2;
        mesh.position.set(cx, 0, rowAbs);

        mesh.traverse(c => { if (c.isMesh) { c.castShadow = true; c.receiveShadow = true; } });
        this.obstacleGroup.add(mesh);
      }
    }

    // ── Position du joueur ────────────────────────────────────────────────────
    this.playerMesh.position.set(playerX, 0, playerRow);

    // ── Caméra : suit le joueur avec lerp doux ────────────────────────────────
    const targetCamX = playerX;
    const targetCamZ = playerRow;
    this._camTarget.lerp(
      new THREE.Vector3(targetCamX, 0, targetCamZ),
      0.12,
    );

    this.camera.position.set(
      this._camTarget.x - 2.5,   // décalage droite
      7,                       // hauteur → montre le dessus des objets
      this._camTarget.z - 5,   // recul → montre les lignes devant
    );
    this.camera.lookAt(this._camTarget.x, 0, this._camTarget.z + 2);
  }
}
