#!/usr/bin/env node
/**
 * Fetch and prepare the ONNX models for BirdBox live detection.
 *
 * Ported from the BirdNET website (birdnet-team/birdnet-website, scripts/fetch-models.mjs
 * on the dev branch), reduced to what this app needs: the BirdBox Just-Bird model and
 * BirdNET+ V3.0, which names the boxes. The models are too large to keep in this
 * repository, so they are pulled from a pinned Zenodo record and a pinned BirdBox commit
 * at build time, verified by SHA-256, and written to _models/dist/ (gitignored).
 * Eleventy copies that folder to /models/ in the built site, next to the bundled
 * BirdNET V2.4 files. Downloads are cached in _models/upstream/, so repeated builds are
 * offline and fast; CI caches the whole _models folder.
 *
 * Derived files:
 *   v3/labels.json        species table for the V3.0 audio model
 *   v3/names/<lang>.json  localized common names, aligned with labels.json
 *   birdbox/<model>.spectrogram.onnx, birdbox/<model>.detect.onnx
 *                         the two streaming graphs stored in the BirdBox model:
 *                         audio to spectrogram frames, and frames to the boxes
 *                         of each 3 s window, which takes each window's noise
 *                         floor as an input (weights untouched)
 *   manifest.json         sizes and SHA-256 of everything the app loads
 *
 * Usage: node scripts/fetch-models.mjs [--force]
 */

import { execFile } from 'node:child_process';
import { createHash } from 'node:crypto';
import { createReadStream, existsSync, mkdirSync, readdirSync, readFileSync, renameSync, rmSync, statSync, writeFileSync } from 'node:fs';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const OUT = join(ROOT, '_models');
const UPSTREAM = join(OUT, 'upstream');
const DIST = join(OUT, 'dist');
const FORCE = process.argv.includes('--force');

// Pinned upstream sources, the same as the BirdNET website (dev). Bump the record or
// commit (and the expected hashes) to update.
// BirdNET+ V3.0 model files (version 3.0.1 of the record)
const ZENODO = { record: '24362980' };
const BIRDBOX = { repo: 'birdnet-team/BirdBox', commit: '24632a5560f6fd93959c2de157e94471be8ea0e7' };

const zenodo = name => `https://zenodo.org/api/records/${ZENODO.record}/files/${encodeURIComponent(name)}/content`;
const raw = (src, path) => `https://raw.githubusercontent.com/${src.repo}/${src.commit}/${path.split('/').map(encodeURIComponent).join('/')}`;

// The pruned FP16 audio model uses one of the model's three classifier heads: 69 MB instead of 166 MB
const V3_MODEL = 'BirdNET+_V3.0_Global_12K_FP16_pruned.onnx';
const V3_LABELS = 'BirdNET+_V3.0_Global_12K_Labels.csv';
const BIRDBOX_MODEL = 'Just-Bird_fp16.onnx';

const FILES = {
  v3Model: { url: zenodo(V3_MODEL), name: V3_MODEL, sha256: 'feadeccc84feb021aeb9c51ec5288f6f5bd7065fe3773e2d48a69ea8a26b412a' },
  v3Labels: { url: zenodo(V3_LABELS), name: V3_LABELS, sha256: '88d976477fac81127ac8ac4c215c37749d0a6537944112ea606d321317cb7640' },
  birdboxModel: { url: raw(BIRDBOX, `docs/models/${BIRDBOX_MODEL}`), name: BIRDBOX_MODEL, sha256: '34b5720cb4fb65d5201d9e9d7cecd3b670d4838552d2a84a6843964407f7e057' },
};

// Species groups (labels column taxon_group). The model's non-animal classes
// (gun, chainsaw, siren, ...) form one group, Sound.
const CLASS_ORDER = ['Aves', 'Amphibia', 'Insecta', 'Arachnida', 'Mammalia', 'Reptilia', 'Actinopterygii', 'Sound'];
const SOUND_GROUPS = new Set(['Anthropogenic', 'Geophony', 'Music']);

// Species name languages (labels column suffix). Columns with poor coverage are
// left out; missing names fall back to English.
const NAME_LANGS = ['en', 'de', 'es', 'fr', 'it', 'pt', 'pt_PT', 'nl', 'pl', 'ru', 'cs', 'sk', 'sv', 'da', 'no', 'fi',
  'uk', 'tr', 'ja', 'zh-CN', 'ca', 'bg', 'sr', 'hr', 'lt', 'et', 'cy', 'fa', 'sl', 'hu', 'ro', 'ar', 'ko', 'th', 'af'];
const MIN_NAME_COVERAGE = 0.4;

/* ------------------------------------------------------------------ */
/* Downloads                                                           */
/* ------------------------------------------------------------------ */

function sha256File(path) {
  return new Promise((res, rej) => {
    const h = createHash('sha256');
    createReadStream(path).on('data', d => h.update(d)).on('end', () => res(h.digest('hex'))).on('error', rej);
  });
}

const sha256 = buf => createHash('sha256').update(buf).digest('hex');

/**
 * The bytes at a URL. Zenodo's bot protection answers Node's own HTTP client with
 * 403 ("unusual traffic") on some networks while it serves curl, so a 403 is tried
 * once more with curl where there is one. The caller checks the SHA-256 either way.
 */
async function fetchBytes(url) {
  const res = await fetch(url);
  if (res.ok) return Buffer.from(await res.arrayBuffer());
  if (res.status !== 403) throw new Error(`HTTP ${res.status}`);
  await res.body?.cancel();
  return new Promise((resolve, reject) => {
    execFile('curl', ['--fail', '--silent', '--show-error', '--location', '--retry', '2', url],
      { encoding: 'buffer', maxBuffer: 1 << 30 },
      (err, stdout, stderr) => (err
        ? reject(new Error(err.code === 'ENOENT' ? 'HTTP 403, and no curl to try instead' : `HTTP 403, and curl: ${String(stderr).trim() || err.message}`))
        : resolve(stdout)));
  });
}

async function download(file) {
  const dest = join(UPSTREAM, file.name);
  const meta = dest + '.url';
  // Cached copies are only trusted for the same pinned URL (and hash, if known)
  if (!FORCE && existsSync(dest) && existsSync(meta) && readFileSync(meta, 'utf8') === file.url) {
    if (!file.sha256 || (await sha256File(dest)) === file.sha256) return dest;
  }
  for (let attempt = 1; ; attempt++) {
    try {
      process.stdout.write(`  downloading ${file.name} ... `);
      const buf = await fetchBytes(file.url);
      if (file.sha256 && sha256(buf) !== file.sha256) {
        throw new Error(`SHA-256 mismatch (got ${sha256(buf)}, expected ${file.sha256})`);
      }
      writeFileSync(dest + '.part', buf);
      renameSync(dest + '.part', dest);
      writeFileSync(meta, file.url);
      console.log(`${(buf.length / 1048576).toFixed(1)} MB`);
      return dest;
    } catch (err) {
      console.log(`failed: ${err.message}`);
      if (attempt >= 3) throw new Error(`Could not download ${file.url}: ${err.message}`);
      await new Promise(r => setTimeout(r, 2000 * attempt));
    }
  }
}

/* ------------------------------------------------------------------ */
/* CSV helpers                                                         */
/* ------------------------------------------------------------------ */

function parseCsv(text, delimiter = ',') {
  const rows = [];
  let row = [], field = '', quoted = false;
  for (let i = 0; i < text.length; i++) {
    const c = text[i];
    if (quoted) {
      if (c === '"') {
        if (text[i + 1] === '"') { field += '"'; i++; } else quoted = false;
      } else field += c;
    } else if (c === '"') quoted = true;
    else if (c === delimiter) { row.push(field); field = ''; }
    else if (c === '\n' || c === '\r') {
      if (c === '\r' && text[i + 1] === '\n') i++;
      row.push(field); field = '';
      if (row.length > 1 || row[0] !== '') rows.push(row);
      row = [];
    } else field += c;
  }
  if (field !== '' || row.length) { row.push(field); rows.push(row); }
  const [header, ...data] = rows;
  return data.map(r => Object.fromEntries(header.map((h, i) => [h.trim(), (r[i] ?? '').trim()])));
}

/* ------------------------------------------------------------------ */
/* Minimal protobuf reader/writer for ONNX ModelProto surgery          */
/* ------------------------------------------------------------------ */

function readVarint(buf, pos) {
  let result = 0n, shift = 0n;
  for (;;) {
    const b = buf[pos++];
    result |= BigInt(b & 0x7f) << shift;
    if (!(b & 0x80)) return [result, pos];
    shift += 7n;
  }
}

function encodeVarint(value) {
  let v = BigInt(value);
  const out = [];
  do {
    let b = Number(v & 0x7fn);
    v >>= 7n;
    if (v) b |= 0x80;
    out.push(b);
  } while (v);
  return Buffer.from(out);
}

/** Iterate the top-level fields of a protobuf message. */
function* fields(buf, start = 0, end = buf.length) {
  let pos = start;
  while (pos < end) {
    const tagStart = pos;
    let tag;
    [tag, pos] = readVarint(buf, pos);
    const field = Number(tag >> 3n), wire = Number(tag & 7n);
    let valueStart = pos, valueEnd, varint;
    if (wire === 0) { [varint, pos] = readVarint(buf, pos); valueEnd = pos; }
    else if (wire === 1) { pos += 8; valueEnd = pos; }
    else if (wire === 5) { pos += 4; valueEnd = pos; }
    else if (wire === 2) {
      let len;
      [len, pos] = readVarint(buf, pos);
      valueStart = pos;
      pos += Number(len);
      valueEnd = pos;
    } else throw new Error(`Unsupported protobuf wire type ${wire}`);
    yield { field, wire, tagStart, valueStart, valueEnd, end: pos, varint };
  }
}

const str = (buf, f) => buf.toString('utf8', f.valueStart, f.valueEnd);
const lenField = (field, payload) => Buffer.concat([encodeVarint((field << 3) | 2), encodeVarint(payload.length), payload]);
const varintField = (field, value) => Buffer.concat([encodeVarint(field << 3), encodeVarint(value)]);

function parseNode(buf, f) {
  const node = { inputs: [], outputs: [], op: '' };
  for (const g of fields(buf, f.valueStart, f.valueEnd)) {
    if (g.field === 1) node.inputs.push(str(buf, g));
    else if (g.field === 2) node.outputs.push(str(buf, g));
    else if (g.field === 4) node.op = str(buf, g);
  }
  return node;
}

function parseTensor(buf, f) {
  const t = { dims: [], dataType: 0, name: '', raw: null };
  for (const g of fields(buf, f.valueStart, f.valueEnd)) {
    if (g.field === 1) {
      if (g.wire === 0) t.dims.push(Number(g.varint));
      else for (let p = g.valueStart; p < g.valueEnd;) { const [v, n] = readVarint(buf, p); t.dims.push(Number(v)); p = n; }
    } else if (g.field === 2) t.dataType = Number(g.varint);
    else if (g.field === 8) t.name = str(buf, g);
    else if (g.field === 9) t.raw = buf.subarray(g.valueStart, g.valueEnd);
  }
  return t;
}

/** Nodes (with their bytes) and initializers of a GraphProto. */
function parseGraph(buf) {
  const nodes = [], inits = new Map();
  for (const f of fields(buf)) {
    if (f.field === 1) nodes.push({ ...parseNode(buf, f), record: buf.subarray(f.tagStart, f.end) });
    else if (f.field === 5) inits.set(parseTensor(buf, f).name, buf.subarray(f.tagStart, f.end));
  }
  return { nodes, inits, producers: new Map(nodes.flatMap(n => n.outputs.map(o => [o, n]))) };
}

/** The nodes `roots` depend on, and the tensors they read that no node produces (or that are in `stop`). */
function ancestors(graph, roots, stop = new Set()) {
  const nodes = new Set(), leaves = new Set(), seen = new Set(), todo = [...roots];
  while (todo.length) {
    const name = todo.pop();
    if (!name || seen.has(name)) continue;
    seen.add(name);
    const node = stop.has(name) ? null : graph.producers.get(name);
    if (!node) { leaves.add(name); continue; }
    nodes.add(node);
    todo.push(...node.inputs);
  }
  return { nodes, leaves };
}

/** The model's metadata_props as an object. */
function modelMetadata(model) {
  const out = {};
  for (const f of fields(model)) {
    if (f.field !== 14) continue;
    let key = null, value = '';
    for (const g of fields(model, f.valueStart, f.valueEnd)) {
      if (g.field === 1) key = str(model, g);
      else if (g.field === 2) value = str(model, g);
    }
    if (key != null) out[key] = value;
  }
  return out;
}

/**
 * stream_detect with each window's noise floor as an input, so the page computes it
 * (RollingFloor in birdbox-stream.js) and the network needs only the windows' own
 * frames. The same as floorCut in BirdBox's docs/javascripts/demo/onnx-graph.js: the
 * floor, (n_mels, windows, 1), is the interpolated percentile the graph takes from the
 * sorted frames of the floor window (TopK), and enters the image as the input of the
 * first Transpose after the TopK. The nodes only the floor needs go, and with them the
 * inputs only they read (sound). An Identity node gives the input its name, `floor`.
 *
 * @returns the function, cut; null if it has no single TopK leading to a Transpose
 */
function floorCut(fn) {
  const consumers = new Map();
  for (const n of fn.nodes) for (const t of n.inputs) (consumers.get(t) || consumers.set(t, []).get(t)).push(n);
  const topk = fn.nodes.filter(n => n.op === 'TopK');
  if (topk.length !== 1) return null;
  let cut = null;
  const seen = new Set(), queue = [topk[0]];
  while (queue.length && !cut) {
    for (const t of queue.shift().outputs) {
      for (const n of consumers.get(t) || []) {
        if (seen.has(n)) continue;
        seen.add(n);
        if (n.op === 'Transpose') { cut = n.inputs[0]; break; }
        queue.push(n);
      }
      if (cut) break;
    }
  }
  if (!cut) return null;
  const graph = { producers: new Map(fn.nodes.flatMap(n => n.outputs.map(o => [o, n]))) };
  const { nodes, leaves } = ancestors(graph, fn.outputs, new Set([cut]));
  if ([...nodes].some(n => n.op === 'TopK')) return null;   // the floor is still computed somewhere else
  const identity = Buffer.concat([lenField(1, Buffer.from('floor')), lenField(2, Buffer.from(cut)), lenField(4, Buffer.from('Identity'))]);
  return {
    ...fn,
    nodes: [{ op: 'Identity', inputs: ['floor'], outputs: [cut], record: lenField(1, identity) }, ...fn.nodes.filter(n => nodes.has(n))],
    inputs: [...fn.inputs.filter(t => leaves.has(t)), 'floor'],
  };
}

/**
 * The streaming graphs of a BirdBox model with a rolling noise floor, as two
 * models the demo runs in turn (birdbox-stream.js):
 *
 *   spectrogram(audio) -> db (n_mels, frames), sound (frames)
 *   detect(db, first_frame, times, conf, floor) -> raw, the boxes of the windows
 *       starting at `times`, before window fusion, with their floors from the page
 *       (floorCut)
 *
 * BirdBox's export stores them in the model as ONNX functions (domain
 * birdbox.stream). Their other inputs are tensors of the model: weights of its
 * segment Loop body, and the DFT kernel its graph builds. Those become
 * initializers and leading nodes here. A window's rows equal those of the whole
 * graph, so the demo can show boxes a few seconds of audio at a time, and while
 * a recording is still in progress.
 */
function streamGraphs(model) {
  const functions = new Map();
  for (const f of fields(model)) {
    if (f.field !== 25) continue;   // FunctionProto: name 1, input 4, output 5, node 7, domain 10
    const fn = { name: '', domain: '', inputs: [], outputs: [], nodes: [] };
    for (const g of fields(model, f.valueStart, f.valueEnd)) {
      if (g.field === 1) fn.name = str(model, g);
      else if (g.field === 10) fn.domain = str(model, g);
      else if (g.field === 4) fn.inputs.push(str(model, g));
      else if (g.field === 5) fn.outputs.push(str(model, g));
      else if (g.field === 7) fn.nodes.push({ ...parseNode(model, g), record: lenField(1, model.subarray(g.valueStart, g.valueEnd)) });
    }
    if (fn.domain === 'birdbox.stream') functions.set(fn.name, fn);
  }
  const spectrogram = functions.get('stream_spectrogram'), detect = functions.get('stream_detect');
  if (!spectrogram || !detect) throw new Error('BirdBox: the model has no streaming graphs');

  const graphField = [...fields(model)].find(f => f.field === 7);
  if (!graphField) throw new Error('No graph in ModelProto');
  const graphBytes = model.subarray(graphField.valueStart, graphField.valueEnd), top = parseGraph(graphBytes);
  const loopField = [...fields(graphBytes)].find(f => f.field === 1 && parseNode(graphBytes, f).op === 'Loop');
  if (!loopField) throw new Error('BirdBox: no segment Loop');
  let body = null;
  for (const a of fields(graphBytes, loopField.valueStart, loopField.valueEnd)) {
    if (a.field !== 5) continue;   // AttributeProto, field 6 = graph
    const g = [...fields(graphBytes, a.valueStart, a.valueEnd)].find(x => x.field === 6);
    if (g) body = parseGraph(graphBytes.subarray(g.valueStart, g.valueEnd));
  }
  if (!body) throw new Error('BirdBox: the segment Loop has no body');

  // ValueInfoProto { name, type { tensor_type { elem_type, shape } } }; without dims the shape stays open
  const ELEM = { float32: 1, int64: 7, bool: 9 };
  const valueInfo = (name, type, dims) => {
    if (!type) return lenField(1, Buffer.from(name));
    const shape = dims ? [lenField(2, Buffer.concat(dims.map(d => lenField(1, varintField(1, d)))))] : [];
    return Buffer.concat([lenField(1, Buffer.from(name)), lenField(2, lenField(1, Buffer.concat([varintField(1, ELEM[type]), ...shape])))]);
  };
  const build = (fn, types) => {
    const weights = fn.inputs.filter(name => body.inits.has(name));
    const outer = fn.inputs.filter(name => !body.inits.has(name) && (top.producers.has(name) || top.inits.has(name)));
    const prelude = ancestors(top, outer);
    const inputs = fn.inputs.filter(name => !weights.includes(name) && !outer.includes(name));
    const unknown = inputs.filter(name => !types[name]);
    if (unknown.length) throw new Error(`BirdBox: ${fn.name} reads ${unknown.join(', ')}, which the model does not have`);
    const graph = Buffer.concat([
      ...top.nodes.filter(n => prelude.nodes.has(n)).map(n => n.record),
      ...fn.nodes.map(n => n.record),
      lenField(2, Buffer.from(fn.name)),
      ...weights.map(name => body.inits.get(name)),
      ...[...prelude.leaves].filter(name => top.inits.has(name)).map(name => top.inits.get(name)),
      ...inputs.map(name => lenField(11, valueInfo(name, ...types[name]))),
      ...fn.outputs.map(name => lenField(12, valueInfo(name))),
    ]);
    return Buffer.concat([model.subarray(0, graphField.tagStart), lenField(7, graph), model.subarray(graphField.end)]);
  };
  const floored = floorCut(detect);
  if (!floored) throw new Error('BirdBox: stream_detect could not be cut at the noise floor (floorCut)');
  return {
    spectrogram: build(spectrogram, { audio: ['float32'] }),
    detect: build(floored, { db: ['float32'], first_frame: ['int64', []], times: ['float32'], conf: ['float32'], floor: ['float32'] }),
  };
}

/* ------------------------------------------------------------------ */
/* Main                                                                */
/* ------------------------------------------------------------------ */

function writeIfChanged(path, data) {
  mkdirSync(dirname(path), { recursive: true });
  const buf = Buffer.isBuffer(data) ? data : Buffer.from(data);
  if (existsSync(path) && statSync(path).size === buf.length && sha256(readFileSync(path)) === sha256(buf)) return;
  writeFileSync(path, buf);
}


function entry(relPath) {
  const buf = readFileSync(join(DIST, relPath));
  return { path: relPath, size: buf.length, sha256: sha256(buf) };
}

async function main() {
  console.log('Preparing BirdBox models (_models/dist)');
  mkdirSync(UPSTREAM, { recursive: true });
  const paths = {};
  for (const [key, file] of Object.entries(FILES)) paths[key] = await download(file);
  // Downloads of earlier model versions would ride along in the build cache
  const current = new Set(Object.values(FILES).flatMap(f => [f.name, `${f.name}.url`]));
  for (const name of readdirSync(UPSTREAM)) if (!current.has(name)) rmSync(join(UPSTREAM, name));

  // --- V3.0 audio model + labels: names the boxes --------------------------
  const labels = parseCsv(readFileSync(paths.v3Labels, 'utf8'));
  labels.forEach((l, i) => { if (+l.ix !== i) throw new Error(`V3 labels: ix ${l.ix} at row ${i}`); });
  const groupOf = l => (SOUND_GROUPS.has(l.taxon_group) ? 'Sound' : l.taxon_group);
  const present = new Set(labels.map(groupOf));
  const classes = [...CLASS_ORDER.filter(c => present.has(c)), ...[...present].filter(c => !CLASS_ORDER.includes(c)).sort()];
  // [scientific name, English common name, class index]
  const v3Species = labels.map(l => [l.scientific_name, l.common_name || l.scientific_name, classes.indexOf(groupOf(l))]);

  writeIfChanged(join(DIST, 'v3', V3_MODEL), readFileSync(paths.v3Model));
  writeIfChanged(join(DIST, 'v3', 'labels.json'), JSON.stringify({ classes, species: v3Species }));
  const names = {};
  for (const lang of NAME_LANGS) {
    if (lang === 'en') continue;   // English is the base list itself
    const column = `common_name_${lang}`;
    let found = 0;
    const list = labels.map(l => {
      const name = l[column] || '';
      if (name) found++;
      return name;
    });
    if (found / labels.length < MIN_NAME_COVERAGE) continue;
    writeIfChanged(join(DIST, 'v3', 'names', `${lang}.json`), JSON.stringify(list));
    names[lang] = { coverage: +(found / labels.length).toFixed(3) };
  }

  // --- BirdBox: its two streaming graphs -------------------------------------
  // The app runs BirdBox as a stream and merges the boxes of the windows itself
  // (birdbox-fusion.js).
  const birdbox = readFileSync(paths.birdboxModel), birdboxMeta = modelMetadata(birdbox), stream = streamGraphs(birdbox);
  const floorWindow = +birdboxMeta.floor_window_seconds, frameOrigin = +birdboxMeta.frame_origin;
  if (!(floorWindow > 0) || !Number.isInteger(frameOrigin)) throw new Error('BirdBox: the model has no rolling noise floor');
  const birdboxName = BIRDBOX_MODEL.replace(/\.onnx$/, '');
  writeIfChanged(join(DIST, 'birdbox', `${birdboxName}.spectrogram.onnx`), stream.spectrogram);
  writeIfChanged(join(DIST, 'birdbox', `${birdboxName}.detect.onnx`), stream.detect);

  // --- Manifest -----------------------------------------------------------
  const manifest = {
    sources: {
      zenodo: `https://zenodo.org/records/${ZENODO.record}`,
      birdbox: `https://github.com/${BIRDBOX.repo}/tree/${BIRDBOX.commit}/docs/models`,
    },
    v3: {
      name: 'BirdNET+ V3.0 Global 12K (FP16, pruned)',
      version: '3.0',
      sampleRate: 32000,
      species: labels.length,
      classes,
      model: entry(`v3/${V3_MODEL}`),
      labels: entry('v3/labels.json'),
      names: Object.fromEntries(Object.entries(names).map(([lang, info]) => [lang, { ...entry(`v3/names/${lang}.json`), ...info }])),
      input: 'input',
      output: 'predictions',
    },
    birdbox: {
      name: 'BirdBox Just-Bird (FP16)',
      sampleRate: 32000,
      model: entry(`birdbox/${birdboxName}.detect.onnx`),
      spectrogram: entry(`birdbox/${birdboxName}.spectrogram.onnx`),
      classNames: { 0: 'bird' },
      merge: 'fusion',
      defaultConf: 0.15,
      defaultPhraseGap: 0.05,
      clipSeconds: 3,
      clipHopSeconds: 1.5,
      floorWindowSeconds: floorWindow,
      frameOrigin,
    },
  };
  writeIfChanged(join(DIST, 'manifest.json'), JSON.stringify(manifest, null, 2));

  // Remove stale derived files from earlier model versions
  for (const dir of ['v3', 'birdbox']) {
    const keep = new Set(Object.values(manifest[dir]).flatMap(v => v && typeof v === 'object' ? (v.path ? [v.path] : Object.values(v).map(x => x?.path)) : []).filter(Boolean));
    const walk = rel => readdirSync(join(DIST, rel), { withFileTypes: true }).forEach(d => {
      const p = `${rel}/${d.name}`;
      if (d.isDirectory()) walk(p);
      else if (!keep.has(p)) rmSync(join(DIST, p));
    });
    walk(dir);
  }
  const total = manifest.v3.model.size + manifest.birdbox.model.size + manifest.birdbox.spectrogram.size;
  console.log(`BirdBox models ready: ${(total / 1048576).toFixed(1)} MB of ONNX models, ${Object.keys(names).length + 1} name languages`);
}

main().catch(err => {
  console.error(`\nfetch-models: ${err.message}`);
  process.exit(1);
});
