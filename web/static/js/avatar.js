/**
 * avatar.js – Three.js + @pixiv/three-vrm avatar renderer.
 *
 * Loads /web/assets/avatar.vrm (served at /assets/avatar.vrm) and renders it
 * in the #avatar-canvas element.  Exports `applyExpression(name, intensity)`
 * so app.js can animate the avatar in response to LLM expression events.
 *
 * VRM Blend-shape names supported:
 *   happy, sad, angry, surprised, relaxed, neutral,
 *   blink, blinkLeft, blinkRight, aa, ih, ou, ee, oh
 */

import * as THREE from 'three';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { VRMLoaderPlugin, VRMUtils } from '@pixiv/three-vrm';

// ── Scene setup ───────────────────────────────────────────────────────────

const canvas   = document.getElementById('avatar-canvas');
const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: true });
renderer.setPixelRatio(window.devicePixelRatio);
renderer.outputColorSpace = THREE.SRGBColorSpace;

const scene  = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(30, 1, 0.1, 20);
camera.position.set(0, 1.4, 3.2);

// Ambient + directional light
scene.add(new THREE.AmbientLight(0xffffff, 0.6));
const dirLight = new THREE.DirectionalLight(0xffffff, 1.2);
dirLight.position.set(1, 2, 2);
scene.add(dirLight);

// ── VRM loader ────────────────────────────────────────────────────────────

const loader = new GLTFLoader();
loader.register(parser => new VRMLoaderPlugin(parser));

let vrm = null;

const statusEl = document.getElementById('avatar-status');

async function loadAvatar(url) {
  statusEl.textContent = 'Loading avatar…';
  try {
    const gltf = await loader.loadAsync(url);
    vrm = gltf.userData.vrm;
    VRMUtils.rotateVRM0(vrm);            // corrects older VRM 0.x axis
    scene.add(vrm.scene);
    statusEl.textContent = '';
  } catch (err) {
    console.warn('Avatar VRM not found, using placeholder.', err);
    _addPlaceholder();
    statusEl.textContent = '(placeholder – place avatar.vrm in web/assets/)';
  }
}

function _addPlaceholder() {
  const geo  = new THREE.CapsuleGeometry(0.18, 0.7, 8, 16);
  const mat  = new THREE.MeshToonMaterial({ color: 0xc084fc });
  const mesh = new THREE.Mesh(geo, mat);
  mesh.position.set(0, 1.0, 0);
  scene.add(mesh);
  // Simple head sphere
  const headGeo = new THREE.SphereGeometry(0.22, 16, 16);
  const head    = new THREE.Mesh(headGeo, mat);
  head.position.set(0, 1.7, 0);
  scene.add(head);
}

loadAvatar('/assets/avatar.vrm');

// ── Expression API ────────────────────────────────────────────────────────

/**
 * Apply a VRM expression blend-shape by name.
 * @param {string} name       – VRM expression key (e.g. 'happy', 'sad')
 * @param {number} intensity  – 0.0 – 1.0
 * @param {number} duration   – milliseconds to hold before fading back
 */
export function applyExpression(name, intensity = 1.0, duration = 2500) {
  if (!vrm) return;
  const expr = vrm.expressionManager;
  if (!expr) return;

  // Fade out current expressions
  expr.setValue('neutral', 0);
  ['happy', 'sad', 'angry', 'surprised', 'relaxed'].forEach(n => expr.setValue(n, 0));

  try {
    expr.setValue(name, intensity);
  } catch (_) {
    expr.setValue('neutral', 1);  // fallback
  }

  setTimeout(() => {
    if (!vrm) return;
    try { expr.setValue(name, 0); } catch (_) {}
    expr.setValue('neutral', 1);
  }, duration);
}

// ── Idle blink animation ──────────────────────────────────────────────────

let _blinkTimer = 0;
let _blinkInterval = _randomBlinkInterval();

function _randomBlinkInterval() { return 3000 + Math.random() * 4000; }

function _updateBlink(deltaMs) {
  _blinkTimer += deltaMs;
  if (_blinkTimer >= _blinkInterval) {
    _blinkTimer = 0;
    _blinkInterval = _randomBlinkInterval();
    if (vrm && vrm.expressionManager) {
      const expr = vrm.expressionManager;
      expr.setValue('blink', 1);
      setTimeout(() => {
        if (vrm && vrm.expressionManager) vrm.expressionManager.setValue('blink', 0);
      }, 120);
    }
  }
}

// ── Render loop ───────────────────────────────────────────────────────────

function _resize() {
  const w = canvas.clientWidth;
  const h = canvas.clientHeight;
  renderer.setSize(w, h, false);
  camera.aspect = w / h;
  camera.updateProjectionMatrix();
}

window.addEventListener('resize', _resize);
_resize();

const clock = new THREE.Clock();
(function animate() {
  requestAnimationFrame(animate);
  const delta = clock.getDelta();
  _updateBlink(delta * 1000);
  if (vrm) vrm.update(delta);
  renderer.render(scene, camera);
})();
