import { GLTFLoader } from 'three/examples/jsm/loaders/GLTFLoader.js';
import { LaneType } from '../game/lane.js';

const loader = new GLTFLoader();

function loadGLB(url) {
  return new Promise((resolve, reject) => {
    loader.load(url, gltf => resolve(gltf.scene), undefined, reject);
  });
}

export async function loadModels() {
  const [
    player, lily,
    voiture1, voiture2Jaune, voiture2Rouge, voiture2Vert, voiture2Violet, camion3,
    buche1, buche2, buche3, buche4,
    tree1, tree2, tree3, tree4,
  ] = await Promise.all([
    loadGLB('/player.glb'),
    loadGLB('/lily.glb'),
    loadGLB('/voiture_1.glb'),
    loadGLB('/voiture_2_Jaune.glb'),
    loadGLB('/voiture_2_rouge.glb'),
    loadGLB('/voiture_2_vert_.glb'),
    loadGLB('/voiture_2_violet_.glb'),
    loadGLB('/camion_3.glb'),
    loadGLB('/buche_1.glb'),
    loadGLB('/buche_2.glb'),
    loadGLB('/buche_3.glb'),
    loadGLB('/buche_4.glb'),
    loadGLB('/tree_1.glb'),
    loadGLB('/tree_2.glb'),
    loadGLB('/tree_3.glb'),
    loadGLB('/tree_4.glb'),
  ]);

  return {
    player,
    [LaneType.LILY]: lily,
    // cars[1] = voiture_1, cars[2] = tableau des 4 couleurs, cars[3] = camion_3
    cars: [null, voiture1, [voiture2Jaune, voiture2Rouge, voiture2Vert, voiture2Violet], camion3],
    // logs[n] = buche_n (indices 1 à 4)
    logs: [null, buche1, buche2, buche3, buche4],
    treesPlayable: [tree1, tree2, tree3],   // zone jouable : 3 variantes
    treesWall:     [tree3, tree4],          // zone mur     : 2 variantes
  };
}
