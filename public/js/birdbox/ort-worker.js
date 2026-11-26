/**
 * Inference worker: one single-threaded ONNX Runtime Web (WASM) session with
 * BirdNET V3.0, which scores the audio of BirdBox's boxes. The pool in
 * ort-pool.js runs it.
 * Ported from the BirdNET website (birdnet-team/birdnet-website, dev branch):
 * public/js/demo/ort-worker.js, box scoring only.
 *
 * Messages in:
 *   { type: 'init', ortModule, wasmPaths, model: ArrayBuffer, config }
 *   { type: 'v3box', id, audio: Float32Array, sampleRate, flo, fhi, cuts: Float32Array, min }
 * Messages out: { type: 'ready', ms } | { type: 'result', id, ... } | { type: 'error', id, error }
 */

import { isolateBox } from './region-cut.js';

let ort = null;
let session = null;
let config = {};

const post = (msg, transfer) => self.postMessage(msg, transfer || []);

async function init(msg) {
  const t0 = performance.now();
  ort = await import(msg.ortModule);
  ort.env.wasm.wasmPaths = msg.wasmPaths;
  ort.env.wasm.numThreads = 1;
  ort.env.wasm.proxy = false;
  ort.env.logLevel = 'error';
  config = msg.config || {};
  session = await ort.InferenceSession.create(new Uint8Array(msg.model), {
    executionProviders: ['wasm'],
    graphOptimizationLevel: 'all',
  });
  post({ type: 'ready', ms: performance.now() - t0 });
}

/**
 * The audio of one box: what is outside the box's band, and the regions `cuts` (the
 * sound of other boxes inside it), are replaced by background noise; then it is scored
 * at its own length (the model takes any). Returns the classes scoring at least `min`,
 * not the whole table.
 */
async function runV3Box(msg) {
  const clip = isolateBox(msg.audio, msg.sampleRate, msg.flo, msg.fhi, msg.cuts);
  const input = new ort.Tensor('float32', clip, [1, clip.length]);
  const out = await session.run({ [config.input]: input }, [config.output]);
  const scores = out[config.output].data, min = msg.min;
  let n = 0;
  for (let s = 0; s < scores.length; s++) if (scores[s] >= min) n++;
  const index = new Int32Array(n), score = new Float32Array(n);
  for (let s = 0, k = 0; s < scores.length; s++) if (scores[s] >= min) { index[k] = s; score[k++] = scores[s]; }
  return [{ index, score }, [index.buffer, score.buffer]];
}

const handlers = { v3box: runV3Box };

self.onmessage = async ({ data: msg }) => {
  if (msg.type === 'init') {
    try {
      await init(msg);
    } catch (err) {
      post({ type: 'error', error: String(err?.message || err) });
    }
    return;
  }
  try {
    const t0 = performance.now();
    const [result, transfer] = await handlers[msg.type](msg);
    post({ type: 'result', id: msg.id, ms: performance.now() - t0, ...result }, transfer);
  } catch (err) {
    post({ type: 'error', id: msg.id, error: String(err?.message || err) });
  }
};
