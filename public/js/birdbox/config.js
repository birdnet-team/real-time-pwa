/**
 * Paths for the BirdBox modules. The app is served under a path prefix
 * (window.PATH_PREFIX, set by the page layout); workers and dynamic imports
 * need absolute URLs.
 */
const prefix = window.PATH_PREFIX || '/';
const abs = path => new URL(prefix + path, location.href).href;

export const CONFIG = {
  models: abs('models/'),
  workerUrl: abs('js/birdbox/ort-worker.js'),
  ortModule: abs('vendor/onnxruntime-web/ort.wasm.min.mjs'),
  ortWasm: abs('vendor/onnxruntime-web/'),
};
