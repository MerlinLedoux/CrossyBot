import { GLTFLoader } from 'three/examples/jsm/loaders/GLTFLoader.js';
import { LaneType } from '../game/lane.js';

const loader = new GLTFLoader();

function loadGLB(url) {
  return new Promise((resolve, reject) => {
    loader.load(url, gltf => resolve(gltf.scene), undefined, reject);
  });
}

export async function loadModels() {
  const [player, tree, car, log, lily] = await Promise.all([
    loadGLB('/player.glb'),
    loadGLB('/tree.glb'),
    loadGLB('/car.glb'),
    loadGLB('/log.glb'),
    loadGLB('/lily.glb'),
  ]);

  return {
    player,
    [LaneType.GRASS]: tree,
    [LaneType.ROAD]:  car,
    [LaneType.WATER]: log,
    [LaneType.LILY]:  lily,
  };
}
