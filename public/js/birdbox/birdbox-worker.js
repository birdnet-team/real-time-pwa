/**
 * BirdBox worker: the model's two streaming graphs in single-threaded ONNX
 * Runtime Web (WASM) sessions (birdbox-stream.js), for the microphone.
 * Ported from the BirdNET website (birdnet-team/birdnet-website, dev branch):
 * public/js/demo/birdbox-worker.js, live part only (BirdNET Live has no files).
 *
 * Messages in:
 *   { type: 'init', ortModule, wasmPaths, spectrogram: ArrayBuffer, detect: ArrayBuffer, config }
 *   { type: 'live-start' }                         a recording starts
 *   { type: 'live-audio', pcm: Float32Array }      its next samples, 32 kHz
 *   { type: 'windows', id, from, count, conf }     its next windows
 * Messages out:
 *   { type: 'ready', ms }
 *   { type: 'result', id, ms, data, rows, next }   when the windows are done
 *   { type: 'error', id?, error }
 */
import { StreamGraph } from './birdbox-stream.js';

let graph = null;
let live = null;   // the recording in progress, with its spectrogram frames

const post = (msg, transfer) => self.postMessage(msg, transfer || []);

async function init(msg) {
  const t0 = performance.now();
  const ort = await import(msg.ortModule);
  ort.env.wasm.wasmPaths = msg.wasmPaths;
  ort.env.wasm.numThreads = 1;
  ort.env.wasm.proxy = false;
  ort.env.logLevel = 'error';
  graph = await StreamGraph.create(ort, msg, { executionProviders: ['wasm'], graphOptimizationLevel: 'all' }, msg.config);
  post({ type: 'ready', ms: performance.now() - t0 });
}

self.onmessage = async ({ data: msg }) => {
  // Between the awaits of a run: the run reads the samples there are when it starts
  if (msg.type === 'live-start') { live = graph.live(); return; }
  if (msg.type === 'live-audio') { live?.push(msg.pcm); return; }
  if (msg.type === 'init') {
    try {
      await init(msg);
    } catch (err) {
      post({ type: 'error', error: String(err?.message || err) });
    }
    return;
  }
  if (msg.type !== 'windows') return;
  try {
    const t0 = performance.now();
    if (!live) throw new Error('no recording in progress');
    const result = await live.windows(msg.conf, msg.from, msg.count);
    post({ type: 'result', id: msg.id, ms: performance.now() - t0, ...result }, [result.data.buffer]);
  } catch (err) {
    post({ type: 'error', id: msg.id, error: String(err?.message || err) });
  }
};
