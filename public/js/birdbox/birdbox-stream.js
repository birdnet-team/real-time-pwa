/**
 * Ported from the BirdNET website (birdnet-team/birdnet-website, dev branch):
 * public/js/demo/birdbox-stream.js. Changed: live() lets go of audio it is done
 * with (see there).
 *
 * Runs BirdBox as a stream, so boxes appear a few seconds of audio at a time,
 * in a file and in a recording that is still in progress. A port of StreamGraph
 * in BirdBox's docs/javascripts/demo/chunked-graph.js.
 *
 * The model takes each 3 s window's noise floor from the 60 s before the
 * window's end, on one frame grid for the whole recording. So the spectrogram
 * is computed once, a few seconds of audio at a time, and kept as frames, and
 * each window goes through the network as soon as its frames are there. A
 * window's rows are final as soon as the audio reaches its end (readyWindows),
 * and they are the rows of one run of the whole graph, bit for bit.
 *
 * Two models do the work (fetch-models.mjs takes them out of the BirdBox model):
 *   spectrogram(audio) -> db (n_mels, frames), sound (frames)
 *   detect(db, sound, first_frame, times, conf) -> raw
 * Frame m covers samples m HOP_LENGTH + origin to m HOP_LENGTH + origin + N_FFT,
 * and window w is frames 128 w to 128 w + 251.
 *
 * Rows have the 9 columns of the graph's raw_all tensor: clip, t0, t1, f_lo,
 * f_hi, score, class, clip start, file end (see birdbox-fusion.js).
 *
 * run() reads a file's audio as the frames need it, from an array or from a source
 * that hands it out a block at a time (BlockSource: birdbox-worker.js gets it from
 * the page so), so the worker need not hold a copy of the whole file.
 */

const SAMPLE_RATE = 32000;
const CLIP_SECONDS = 3;
const CLIP_SAMPLES = CLIP_SECONDS * SAMPLE_RATE, HOP_SAMPLES = CLIP_SAMPLES / 2;
const HOP_SECONDS = HOP_SAMPLES / SAMPLE_RATE;
const OWNED_UNTIL = 0.75 * CLIP_SECONDS;   // a window owns its boxes up to here (birdbox-fusion.js)
const SEGMENT_CLIPS = 60 / HOP_SECONDS;    // column 0 counts the windows of each minute, as in the graph
// The model's spectrogram (onnx_export.py in BirdBox)
const N_MELS = 256, HOP_LENGTH = 375, N_FFT = 2048, CLIP_FRAMES = 252;
const WINDOW_FRAMES = HOP_SAMPLES / HOP_LENGTH;    // 128 frames from one window to the next
const LEFT_PAD = 16000;                            // audio the graph mirrors in front of the file: 0.5 s
const f32 = Math.fround;
const FLOOR_FRACTION = f32(20 / 100);             // FLOOR_PERCENTILE / 100 in BirdBox's onnx_export.py, in float32

/** File end of rows (column 8) while the recording is still growing. */
export const NO_END = 1e9;

/** Windows whose rows are final in a recording of n samples: the audio has reached their end. */
export const readyWindows = n => Math.max(0, Math.floor((n - CLIP_SAMPLES) / HOP_SAMPLES) + 1);

/** Samples the graph pads n samples to: the end of the last window. */
const paddedLength = n => CLIP_SAMPLES + Math.ceil(Math.max(0, n - CLIP_SAMPLES) / HOP_SAMPLES) * HOP_SAMPLES;

/** Windows in a recording of n samples. */
export const windowCount = n => (paddedLength(n) - CLIP_SAMPLES) / HOP_SAMPLES + 1;

/** File end of rows (column 8) for a recording of n samples: where its last window ends. */
export const fileEnd = n => f32(paddedLength(n) / SAMPLE_RATE);

/** Seconds up to which every window that owns a box has been seen, after `windows` windows. */
export const analyzedTo = windows => (windows - 1) * HOP_SECONDS + OWNED_UNTIL;

/** A source of samples for run(): an array, or { length, read(from, to) } (read may return a promise). */
const sourceOf = samples => (samples instanceof Float32Array
  ? { length: samples.length, read: (from, to) => samples.subarray(from, to) }
  : samples);

/** Samples from..to-1 of a source, and silence after its end, as the graph pads the file. */
async function readPadded(source, from, to) {
  const have = Math.min(to, source.length);
  if (have >= to) return source.read(from, to);
  const out = new Float32Array(to - from);
  if (have > from) out.set(await source.read(from, have));
  return out;
}

/**
 * A recording handed out a block at a time, as a source for run(): ask(from, to) asks for
 * samples from..to-1, and receive({ from, data }) takes them when they come. The next block
 * is asked for while half of the last one is still to be read, and the samples run() is
 * done with are dropped, so only a few blocks are held at any time.
 */
export class BlockSource {
  /**
   * @param {number} length  samples of the recording
   * @param {(from: number, to: number) => void} ask
   * @param {number} [block]  samples per block: 10 s
   */
  constructor(length, ask, block = 10 * SAMPLE_RATE) {
    Object.assign(this, { length, ask, block, base: 0, data: new Float32Array(0), keep: 0, asked: 0, waiting: null });
  }

  get end() { return this.base + this.data.length; }

  async read(from, to) {
    if (from < this.base) throw new Error(`samples from ${from} on were let go (${this.base})`);
    while (this.end < to) {
      this._ask();
      await new Promise(resolve => { this.waiting = resolve; });
    }
    if (this.asked - to < this.block / 2) this._ask();
    return this.data.subarray(from - this.base, to - this.base);
  }

  /** The samples before `before` are done with. */
  release(before) { this.keep = Math.max(this.keep, before); }

  /** Before the first read: the samples before `sample` are never needed, and are not asked for. */
  skipTo(sample) {
    if (this.asked || this.data.length) return;
    this.base = this.keep = this.asked = Math.max(0, Math.min(sample, this.length));
  }

  _ask() {
    if (this.asked >= this.length || this.asked > this.end) return;   // all asked for, or a block is on its way
    const from = this.asked, to = Math.min(this.length, from + this.block);
    this.asked = to;
    this.ask(from, to);
  }

  /** A block; the samples let go of are dropped. */
  receive({ from, data }) {
    if (from !== this.end) throw new Error(`block at ${from}, expected ${this.end}`);
    const keep = Math.min(Math.max(this.keep, this.base), this.end), joined = new Float32Array(this.end - keep + data.length);
    joined.set(this.data.subarray(keep - this.base));
    joined.set(data, this.end - keep);
    this.base = keep;
    this.data = joined;
    this.waiting?.();
    this.waiting = null;
  }
}

/** Spectrogram frames from frame `base` on, band by band (n_mels, frames) as the graph wants them. */
class Frames {
  constructor() { this.cap = 0; this.db = new Float32Array(0); this.sound = new Uint8Array(0); this.base = 0; this.count = 0; }

  get end() { return this.base + this.count; }

  /** Forget the frames before `frame`. */
  drop(frame) {
    const k = Math.min(Math.max(0, frame - this.base), this.count);
    if (!k) return;
    for (let b = 0; b < N_MELS; b++) this.db.copyWithin(b * this.cap, b * this.cap + k, b * this.cap + this.count);
    this.sound.copyWithin(0, k, this.count);
    this.base += k;
    this.count -= k;
  }

  /** Append n frames of a (n_mels, n) block. */
  append(db, sound, n) {
    if (this.count + n > this.cap) {
      const cap = Math.max(2 * this.cap, this.count + n, 4096), grown = new Float32Array(N_MELS * cap);
      for (let b = 0; b < N_MELS; b++) grown.set(this.db.subarray(b * this.cap, b * this.cap + this.count), b * cap);
      const s = new Uint8Array(cap);
      s.set(this.sound.subarray(0, this.count));
      Object.assign(this, { cap, db: grown, sound: s });
    }
    for (let b = 0; b < N_MELS; b++) this.db.set(db.subarray(b * n, (b + 1) * n), b * this.cap + this.count);
    this.sound.set(sound.subarray(0, n), this.count);
    this.count += n;
  }

  /** Frames from..to-1, as new arrays. */
  slice(from, to) {
    const n = to - from, o = from - this.base, db = new Float32Array(N_MELS * n);
    if (o < 0 || o + n > this.count) throw new Error(`frames ${from} to ${to - 1} are not computed (${this.base} to ${this.end - 1})`);
    for (let b = 0; b < N_MELS; b++) db.set(this.db.subarray(b * this.cap + o, b * this.cap + o + n), b * n);
    return { db, sound: this.sound.slice(o, o + n), n };
  }
}

/**
 * Each window's noise floor as StreamDetect takes it (log_floor in onnx_export.py): per mel
 * band, the FLOOR_PERCENTILE of the frames of its floor window that hold sound (all of them
 * if none does), interpolated in float32 between two sorted values. Here the frames of the
 * floor window are kept sorted per band, and from one window to the next only the frames
 * that leave and those that arrive (128 of about 5,100) are merged in, instead of the graph
 * selecting anew from all of them. The values are those of the graph, bit for bit
 * (tests/js/demo_chunk_parity.mjs).
 */
class RollingFloor {
  constructor() { Object.assign(this, { lo: 0, hi: 0, count: 0, cap: 0, sorted: null, spare: null }); }

  /**
   * The floor of frames lo..hi-1 into out[b * windows + i], for every band b. Windows come in
   * order; the frames from the last call's lo on must still be in `frames`.
   */
  at(frames, lo, hi, out, i, windows) {
    if (!this.sorted || lo < this.lo || lo > this.hi || hi < this.hi) this._fill(frames, lo, hi);
    else this._slide(frames, lo, hi);
    let { count, cap, sorted } = this;
    if (!count) {   // no frame with sound: all frames, as the graph does
      count = cap = hi - lo;
      sorted = new Float32Array(N_MELS * count);
      for (let b = 0; b < N_MELS; b++) {
        const row = sorted.subarray(b * count, (b + 1) * count);
        for (let f = lo; f < hi; f++) row[f - lo] = frames.db[b * frames.cap + f - frames.base];
        row.sort();
      }
    }
    const position = f32(f32(count - 1) * FLOOR_FRACTION), below = Math.floor(position);
    const fraction = f32(position - below), above = Math.min(below + 1, count - 1);
    for (let b = 0; b < N_MELS; b++) {
      const low = sorted[b * cap + below], high = sorted[b * cap + above];
      out[b * windows + i] = f32(low + f32(f32(high - low) * fraction));
    }
  }

  _room(count) {
    if (count <= this.cap) return;
    const cap = Math.max(count, Math.ceil(this.cap * 1.25)), grown = new Float32Array(N_MELS * cap);
    for (let b = 0; this.sorted && b < N_MELS; b++) grown.set(this.sorted.subarray(b * this.cap, b * this.cap + this.count), b * cap);
    Object.assign(this, { cap, sorted: grown, spare: new Float32Array(N_MELS * cap) });
  }

  /** Sort the frames with sound of lo..hi-1 anew. */
  _fill(frames, lo, hi) {
    const { db, sound, base } = frames;
    let count = 0;
    for (let f = lo; f < hi; f++) if (sound[f - base]) count++;
    this.count = 0;
    this._room(Math.max(1, count));
    for (let b = 0; b < N_MELS; b++) {
      const row = this.sorted.subarray(b * this.cap, b * this.cap + count), from = b * frames.cap - base;
      let j = 0;
      for (let f = lo; f < hi; f++) if (sound[f - base]) row[j++] = db[from + f];
      row.sort();
    }
    Object.assign(this, { lo, hi, count });
  }

  /** Move to lo..hi-1: take out the frames before lo, merge in those from the last hi on. */
  _slide(frames, lo, hi) {
    const { db, sound, base } = frames, leave = [], arrive = [];
    for (let f = this.lo; f < lo; f++) if (sound[f - base]) leave.push(f);
    for (let f = this.hi; f < hi; f++) if (sound[f - base]) arrive.push(f);
    const old = this.count, count = old - leave.length + arrive.length;
    this._room(Math.max(old, count));
    const { cap, sorted, spare } = this, gone = new Float32Array(leave.length), come = new Float32Array(arrive.length);
    for (let b = 0; b < N_MELS; b++) {
      const from = b * frames.cap - base, row = b * cap;
      for (let j = 0; j < leave.length; j++) gone[j] = db[from + leave[j]];
      for (let j = 0; j < arrive.length; j++) come[j] = db[from + arrive[j]];
      gone.sort();
      come.sort();
      // One pass: the old values without those leaving (all among them), and the new ones in order
      let s = 0, g = 0, c = 0, d = row;
      while (s < old || c < come.length) {
        if (s < old && g < gone.length && sorted[row + s] === gone[g]) { s++; g++; continue; }
        if (c >= come.length || (s < old && sorted[row + s] <= come[c])) spare[d++] = sorted[row + s++];
        else spare[d++] = come[c++];
      }
    }
    Object.assign(this, { lo, hi, count, sorted: spare, spare: sorted });
  }
}

export class StreamGraph {
  /**
   * @param ort  the ONNX Runtime Web module
   * @param {{spectrogram: ArrayBuffer, detect: ArrayBuffer}} models
   * @param {object} options  session options
   * @param {{floorWindowSeconds: number, frameOrigin: number}} config  from the models manifest
   */
  static async create(ort, models, options, config) {
    const spectrogram = await ort.InferenceSession.create(new Uint8Array(models.spectrogram), options);
    const detect = await ort.InferenceSession.create(new Uint8Array(models.detect), options);
    // fetch-models.mjs cuts detect at the floor (floorCut): the floor is computed here (RollingFloor)
    const floorInput = detect.inputNames.includes('floor');
    return Object.assign(new StreamGraph(), { ort, spectrogram, detect, floorInput, origin: config.frameOrigin, window: config.floorWindowSeconds * SAMPLE_RATE });
  }

  /** First frame of window w's floor window. */
  _first(w) {
    return Math.max(0, Math.ceil((w * HOP_SAMPLES + CLIP_SAMPLES - this.window - this.origin) / HOP_LENGTH));
  }

  /** Last frame window w needs. */
  _last(w) { return w * WINDOW_FRAMES + CLIP_FRAMES - 1; }

  /**
   * Compute frames up to `upTo` from the first n samples the graph sees (silence after the
   * source's end), `block` frames at most per run when audio allows. Afterwards the source
   * may let go of the samples before the next frame (release).
   */
  async _frames(frames, source, n, upTo, block) {
    const { ort, origin } = this, available = Math.floor((n - origin - N_FFT) / HOP_LENGTH);
    if (upTo > available) throw new Error(`frame ${upTo} needs more than ${n} samples`);
    while (frames.end <= upTo) {
      const m0 = frames.end, m1 = Math.min(available, Math.max(upTo, m0 + block - 1));
      const s0 = m0 * HOP_LENGTH + origin, s1 = m1 * HOP_LENGTH + origin + N_FFT;
      // Before the file start: the samples the graph's left pad puts there
      let piece = await readPadded(source, Math.max(0, s0), s1);
      if (s0 < 0) {
        const pre = new Float32Array(s1 - s0);
        pre.set(await readPadded(source, LEFT_PAD + s0, LEFT_PAD));
        pre.set(piece, -s0);
        piece = pre;
      }
      const out = await this.spectrogram.run({ audio: new ort.Tensor('float32', piece, [piece.length]) });
      const count = out.db.dims[1];
      if (count !== m1 - m0 + 1) throw new Error(`${count} frames from the spectrogram, ${m1 - m0 + 1} expected`);
      frames.append(out.db.data, out.sound.data, count);
    }
    source.release?.(Math.max(0, frames.end * HOP_LENGTH + origin));
  }

  /**
   * Rows of windows w0 .. w0+count-1 from the frames. With a floor input, the floors come
   * from the frames' RollingFloor, and the network gets only the windows' own frames; else
   * the graph gets the frames from the first floor window on.
   */
  async _detect(frames, w0, count, conf) {
    const { ort } = this, first = this._first(w0), end = this._last(w0 + count - 1) + 1;
    const times = new Float32Array(count);
    for (let i = 0; i < count; i++) times[i] = (w0 + i) * HOP_SECONDS;
    const feeds = {
      times: new ort.Tensor('float32', times, [count]),
      conf: new ort.Tensor('float32', Float32Array.of(conf), [1]),
    };
    let from = first;
    if (this.floorInput) {
      const floor = new Float32Array(N_MELS * count), rolling = frames.rolling ??= new RollingFloor();
      for (let i = 0; i < count; i++) rolling.at(frames, this._first(w0 + i), this._last(w0 + i) + 1, floor, i, count);
      feeds.floor = new ort.Tensor('float32', floor, [N_MELS, count, 1]);
      from = w0 * WINDOW_FRAMES;
    }
    frames.drop(first);   // the next windows start their floor windows later
    const { db, sound, n } = frames.slice(from, end);
    feeds.db = new ort.Tensor('float32', db, [N_MELS, n]);
    feeds.first_frame = new ort.Tensor('int64', BigInt64Array.of(BigInt(from)), []);
    if (!this.floorInput) feeds.sound = new ort.Tensor('bool', sound, [n]);
    const raw = (await this.detect.run(feeds)).raw;
    const rows = raw.dims[0] || 0, data = raw.data.slice();
    for (let r = 0; r < rows; r++) data[r * 9] = Math.round(data[r * 9 + 7] / HOP_SECONDS) % SEGMENT_CLIPS;
    return { data, rows };
  }

  /**
   * A whole recording: yields { data, rows, next, analyzedTo } after every `step`
   * windows. data holds the next rows; next counts the windows done, and a run
   * given it as `from` continues there.
   *
   * @param {Float32Array|{length: number, read: Function, release?: Function}} samples  mono audio
   *   at 32 kHz, or a source of it: read(from, to) gives (or resolves to) samples from..to-1,
   *   reads move forward, and release(before) says the samples before `before` are done with
   * @param {number} conf  confidence threshold
   * @param {number} step  windows per network run
   * @param {number} [from]  windows already done
   */
  async* run(samples, conf, step, from = 0) {
    const source = sourceOf(samples), n = source.length, target = paddedLength(n);
    const total = windowCount(n);
    // As the graph, in float32: after the file's last sample only silence was added
    const content = f32(f32(n) / SAMPLE_RATE), duration = f32(f32(target) / SAMPLE_RATE);
    const padded = duration > f32(content + f32(0.5 / SAMPLE_RATE));
    const frames = new Frames();
    // A run that continues needs the frames from its first floor window on, not the audio before
    frames.base = Math.min(this._first(from), from * WINDOW_FRAMES);
    source.skipTo?.(frames.base * HOP_LENGTH + this.origin);
    for (let w = from; w < total;) {
      const k = Math.min(step, total - w);
      // Only the frames of these windows: a run that also computed the next ones would take longer than the others
      await this._frames(frames, source, target, this._last(w + k - 1), k * WINDOW_FRAMES);
      const { data, rows } = await this._detect(frames, w, k, conf);
      let kept = 0;
      for (let r = 0; r < rows; r++) {
        const o = r * 9, t0 = data[o + 1], t1 = padded ? Math.min(data[o + 2], content) : data[o + 2];
        if (padded && !(t0 < content && t1 > t0)) continue;
        if (kept !== r) data.copyWithin(kept * 9, o, o + 9);
        data[kept * 9 + 2] = t1;
        data[kept * 9 + 8] = duration;
        kept++;
      }
      w += k;
      yield { data: data.slice(0, kept * 9), rows: kept, next: w, analyzedTo: analyzedTo(w) };
    }
  }

  /**
   * A recording in progress: push() its audio as it arrives, windows() gives the final
   * rows of the next windows once they are ready (readyWindows of the samples pushed),
   * with the file end (column 8) NO_END. Only new audio is turned into frames.
   *
   * Changed for BirdNET Live, which listens for as long as the person wants: the audio
   * is kept only until it has become spectrogram frames (the frames keep the floor
   * window, 60 s), so memory stays the same however long it runs. The rows are those
   * of the website's live() bit for bit: only samples no later frame reads are dropped.
   */
  live() {
    const graph = this, frames = new Frames();
    let audio = new Float32Array(30 * SAMPLE_RATE), base = 0, n = 0;   // audio[i] is sample base + i; n: samples pushed
    const source = {
      get length() { return n; },
      read(from, to) {
        if (from < base) throw new Error(`samples from ${from} on were let go (${base})`);
        return audio.subarray(from - base, to - base);
      },
    };
    return {
      get length() { return n; },
      push(pcm) {
        if (n - base + pcm.length > audio.length) {
          const grown = new Float32Array(Math.max(2 * audio.length, n - base + pcm.length));
          grown.set(audio.subarray(0, n - base));
          audio = grown;
        }
        audio.set(pcm, n - base);
        n += pcm.length;
      },
      async windows(conf, from, count) {
        if (from + count > readyWindows(n)) throw new Error(`windows ${from} to ${from + count - 1} are not ready in ${n} samples`);
        await graph._frames(frames, source, n, graph._last(from + count - 1), WINDOW_FRAMES);
        const { data, rows } = await graph._detect(frames, from, count, conf);
        for (let r = 0; r < rows; r++) data[r * 9 + 8] = NO_END;
        // The next frame reads from its first sample on; the first frames also read the
        // graph's left pad, from the first 0.5 s (LEFT_PAD), so nothing goes before that
        const keep = Math.max(0, frames.end * HOP_LENGTH + graph.origin);
        if (keep > LEFT_PAD && keep - base >= 10 * SAMPLE_RATE) {
          audio.copyWithin(0, keep - base, n - base);
          base = keep;
        }
        return { data, rows, next: from + count };
      },
    };
  }
}
