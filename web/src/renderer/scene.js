import * as THREE from 'three';

export function createScene() {
  const renderer = new THREE.WebGLRenderer({ antialias: true });
  renderer.setSize(window.innerWidth, window.innerHeight);
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  renderer.shadowMap.enabled = true;
  renderer.shadowMap.type    = THREE.PCFSoftShadowMap;
  renderer.outputColorSpace   = THREE.SRGBColorSpace;
  document.body.appendChild(renderer.domElement);

  const scene = new THREE.Scene();
  scene.background = new THREE.Color(0x87ceeb);

  // Caméra orthographique — zoom = demi-hauteur en unités monde
  // Augmenter zoom → dézoom, diminuer → zoom avant
  const ZOOM   = 4;
  const aspect = window.innerWidth / window.innerHeight;
  const camera = new THREE.OrthographicCamera(
    -ZOOM * aspect,   // left
     ZOOM * aspect,   // right
     ZOOM,            // top
    -ZOOM,            // bottom
     0.1,             // near
     200,             // far
  );
  camera._zoom = ZOOM;   // on garde la valeur pour le resize

  // Éclairage
  const ambient = new THREE.AmbientLight(0xffffff, 1.0);
  scene.add(ambient);

  const sun = new THREE.DirectionalLight(0xfffce0, 2.0);
  sun.position.set(10, 10, 0);
  sun.castShadow = true;
  
  sun.shadow.mapSize.width  = 2048;
  sun.shadow.mapSize.height = 2048;
  sun.shadow.camera.near = 1;
  sun.shadow.camera.far  = 60;
  sun.shadow.camera.left   = -15;
  sun.shadow.camera.right  =  15;
  sun.shadow.camera.top    =  15;
  sun.shadow.camera.bottom = -15;
  sun.shadow.bias = -0.001;
  scene.add(sun.target);
  scene.add(sun);

  // Lumière de remplissage douce
  const fill = new THREE.DirectionalLight(0xb0d0ff, 0.5);
  fill.position.set(-5, 10, 10);
  scene.add(fill);

  window.addEventListener('resize', () => {
    const a = window.innerWidth / window.innerHeight;
    camera.left   = -camera._zoom * a;
    camera.right  =  camera._zoom * a;
    camera.top    =  camera._zoom;
    camera.bottom = -camera._zoom;
    camera.updateProjectionMatrix();
    renderer.setSize(window.innerWidth, window.innerHeight);
  });

  return { renderer, scene, camera, sun };
}
