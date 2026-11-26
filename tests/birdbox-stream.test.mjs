// BirdBox's live stream (public/js/birdbox/birdbox-stream.js) gives the rows of a whole-file run,
// bit for bit, although it lets go of the audio it has turned into spectrogram frames.
// Needs the models (npm run fetch-models); skipped without them.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { existsSync, readFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const ROOT = join(dirname(fileURLToPath(import.meta.url)), '..');
const MODELS = join(ROOT, '_models', 'dist');
const ORT = join(ROOT, 'node_modules', 'onnxruntime-web', 'dist');
const haveModels = existsSync(join(MODELS, 'manifest.json'));

const SR = 32000;
const CONF = 0.05;   // the stream's confidence (RAW_CONF in birdbox-service.js)

/** 75 s of noise with frequency sweeps and trills: 49 windows, exactly, so a whole-file run adds no padding. */
function syntheticSong(seconds = 75) {
  const x = new Float32Array(seconds * SR);
  let seed = 1;
  const rnd = () => ((seed = (Math.imul(seed, 1103515245) + 12345) >>> 0) / 4294967296);
  for (let i = 0; i < x.length; i++) x[i] = (rnd() - 0.5) * 0.01;
  for (let t = 0.7; t < seconds - 1; t += 1.3 + rnd() * 1.7) {
    const f0 = 2500 + rnd() * 4000, f1 = f0 + (rnd() - 0.3) * 3000, dur = 0.15 + rnd() * 0.5, notes = 1 + Math.floor(rnd() * 5);
    for (let k = 0; k < notes; k++) {
      const s0 = Math.round((t + k * dur * 1.2) * SR), n = Math.round(dur * SR);
      let phase = 0;
      for (let i = 0; i < n && s0 + i < x.length; i++) {
        const u = i / n;
        phase += 2 * Math.PI * (f0 + (f1 - f0) * u) / SR;
        x[s0 + i] += 0.3 * Math.sin(Math.PI * u) ** 2 * Math.sin(phase);
      }
    }
  }
  return x;
}

/** Rows as arrays of their first 8 columns: the 9th, the file end, differs by design. */
function rowsOf(data, rows) {
  const out = [];
  for (let r = 0; r < rows; r++) out.push(Array.from(data.subarray(r * 9, r * 9 + 8)));
  return out;
}

/** Rows by window (column 7, its start), then by value: the order within a run is not compared. */
const ordered = rows => rows.sort((a, b) => a[7] - b[7] || a.join().localeCompare(b.join()));

test('live() gives the rows of a whole-file run while dropping audio', { skip: !haveModels && 'models missing: run npm run fetch-models' }, async () => {
  const ort = await import(join(ORT, 'ort.wasm.min.mjs'));
  ort.env.wasm.wasmPaths = `${ORT}/`;
  ort.env.wasm.numThreads = 1;
  ort.env.logLevel = 'error';
  const { StreamGraph, readyWindows, windowCount, NO_END } = await import(join(ROOT, 'public', 'js', 'birdbox', 'birdbox-stream.js'));
  const m = JSON.parse(readFileSync(join(MODELS, 'manifest.json'), 'utf8')).birdbox;
  const models = () => ({ spectrogram: readFileSync(join(MODELS, m.spectrogram.path)), detect: readFileSync(join(MODELS, m.model.path)) });
  const config = { floorWindowSeconds: m.floorWindowSeconds, frameOrigin: m.frameOrigin };
  const options = { executionProviders: ['wasm'], graphOptimizationLevel: 'all' };
  const audio = syntheticSong();

  // Whole file, one window at a time
  const file = await StreamGraph.create(ort, models(), options, config);
  const fileRows = [];
  for await (const chunk of file.run(audio, CONF, 1)) fileRows.push(...rowsOf(chunk.data, chunk.rows));

  // Live, as the app feeds it: microphone-sized pieces, all windows that are ready in one run
  const graph = await StreamGraph.create(ort, models(), options, config);
  const live = graph.live();
  const liveRows = [];
  let next = 0;
  for (let i = 0; i < audio.length; i += 1365) {
    live.push(audio.slice(i, i + 1365));
    const ready = readyWindows(live.length);
    if (ready <= next) continue;
    const res = await live.windows(CONF, next, Math.min(ready - next, 8));
    for (let r = 0; r < res.rows; r++) assert.equal(res.data[r * 9 + 8], NO_END);
    liveRows.push(...rowsOf(res.data, res.rows));
    next = res.next;
  }

  assert.equal(next, windowCount(audio.length), 'every window was detected');
  assert.ok(fileRows.length > 50, `the audio gives rows to compare (${fileRows.length})`);
  assert.deepStrictEqual(ordered(liveRows), ordered(fileRows));
});
