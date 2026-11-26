/**
 * BirdNET V3.0 for BirdBox's boxes: model download, one inference worker, and
 * the scoring of single boxes (scoreBox). The model takes audio of any length,
 * so a box is scored on its own time span and frequency band, without the
 * parts of it that belong to other boxes.
 * Ported from the BirdNET website (birdnet-team/birdnet-website, dev branch):
 * public/js/demo/v3-service.js, box scoring only, with one worker (the website
 * also uses one while the microphone records: each holds about 500 MB), and
 * reading the audio of a box from the app's recent audio instead of a file.
 */
import { fetchAsset, getManifest } from './model-store.js';
import { OrtPool } from './ort-pool.js';

export const SR = 32000;
const LOSS_WINDOW_MS = 120000;   // a worker lost to a failure is replaced, unless LOSS_MAX were lost within this time
const LOSS_MAX = 3;
const BOX_PARTS = 6;      // of a box longer than BOX_PARTS x BOX_MAX seconds, this many parts are scored, evenly spread
const BOX_MIN = 1;        // seconds: a shorter box is scored with the audio around it
const BOX_MAX = 10;       // seconds in one run: a longer box is scored in equal parts

/**
 * The audio a box is scored on, as [startSample, endSample] ranges: its time span,
 * or BOX_MIN seconds centred on a shorter box (moved, at the ends of the recording,
 * to where there is audio). A box longer than BOX_MAX is cut into equal parts.
 * `open`: the recording goes on, and null says that audio after the box is still to come.
 */
export function boxCrops(g, total, open = false) {
  const min = BOX_MIN * SR;
  let s = Math.max(0, Math.round(g.t0 * SR)), e = Math.round(g.t1 * SR);
  const short = e - s < min;
  if (short) {
    s = Math.max(0, Math.round((g.t0 + g.t1) / 2 * SR - min / 2));
    e = s + min;
  }
  if (e > total) {
    if (open) return null;
    if (short) s = Math.max(0, total - min);
    e = total;
  }
  if (e <= s) return [];
  const parts = Math.ceil((e - s) / (BOX_MAX * SR)), len = Math.ceil((e - s) / parts);
  return Array.from({ length: parts }, (_, i) => [s + i * len, Math.min(e, s + (i + 1) * len)]);
}

class V3Service extends EventTarget {
  constructor() {
    super();
    this.ready = null;
    this.pool = null;
  }

  async manifestEntry() {
    return (await getManifest()).v3;
  }

  /** The model's bytes, downloaded or restored from the cache, without starting a worker. */
  async download(onProgress) {
    await fetchAsset((await this.manifestEntry()).model, onProgress);
  }

  /** Download (or restore) the model and start the worker. */
  ensure(onProgress) {
    if (!this.ready) {
      const ready = this.ready = (async () => {
        const m = await this.manifestEntry();
        const model = await fetchAsset(m.model, onProgress);
        const pool = new OrtPool({ name: 'birdnet-v3', size: 1, retireOnError: true });
        pool.addEventListener('lost', () => this._replace());
        await pool.start(model, { input: m.input, output: m.output });
        if (this.ready !== ready) { pool.terminate(); throw new Error('stopped'); }
        this.pool = pool;
        return this;
      })();
      ready.catch(() => { if (this.ready === ready) { this.ready = null; this.pool = null; } });
    }
    return this.ready;
  }

  get workers() { return this.pool?.readyCount || 0; }

  /**
   * Score the sound of one box { t0, t1, flo, fhi }: cropped in time (boxCrops) from
   * the first `total` samples of the recording, limited to the box's band and without
   * the rectangles `cuts` (the sound of other boxes inside it), both in the worker
   * (region-cut.js), and run through the model at that length. `read(from, to)` gives
   * the samples from..to-1 of the recording, or null once they are gone. Resolves to
   * [[class, score], ...] for the classes scoring at least `min`: over the parts of a
   * long box, each class's highest score; null if the box's audio is gone. The parts
   * run one after the other, so a long box neither fills the queue nor holds a copy of
   * all its audio.
   */
  async scoreBox(read, total, g, cuts, min, opts) {
    const best = new Map();
    let crops = boxCrops(g, total) || [];
    if (crops.length > BOX_PARTS) crops = Array.from({ length: BOX_PARTS }, (_, i) => crops[Math.round(i * (crops.length - 1) / (BOX_PARTS - 1))]);
    for (const [s, e] of crops) {
      if (opts?.alive && !opts.alive()) break;   // the audio is gone: the parts left are not worth their runs
      const samples = read(s, e);
      if (!samples) return null;
      // Zero-padded only where the recording ends within BOX_MIN of the crop's start
      const audio = new Float32Array(Math.max(e - s, BOX_MIN * SR));
      audio.set(samples);
      // The cuts in this part, in seconds from its start
      const from = s / SR;
      const regions = Float32Array.from(cuts.filter(c => c.t1 * SR > s && c.t0 * SR < e).flatMap(c => [c.t0 - from, c.t1 - from, c.flo, c.fhi]));
      const r = await this.pool.run('v3box', { audio, sampleRate: SR, flo: g.flo, fhi: g.fhi, cuts: regions, min }, [audio.buffer], opts);
      r.index.forEach((c, k) => { if (!(best.get(c) >= r.score[k])) best.set(c, r.score[k]); });
    }
    return [...best];
  }

  /** A worker was lost to a failure: start another, unless they keep failing (out of memory, most likely). */
  _replace() {
    const now = performance.now();
    this.losses = (this.losses || []).filter(t => now - t < LOSS_WINDOW_MS);
    this.losses.push(now);
    if (this.losses.length <= LOSS_MAX) this.pool?.grow(async () => fetchAsset((await this.manifestEntry()).model)).catch(() => {});
  }

  /** Cancel the queued jobs tagged `tag`. */
  cancel(tag) {
    this.pool?.cancel(tag);
  }

  /** Stop the worker and give its memory back; the next ensure() starts a fresh one. */
  stop() {
    this.pool?.terminate();
    this.pool = null;
    this.ready = null;
  }
}

export const v3 = new V3Service();
