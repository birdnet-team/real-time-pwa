/**
 * BirdBox detector service: downloads the model's two streaming graphs and
 * runs them in a worker (birdbox-worker.js) on the microphone's audio.
 * Ported from the BirdNET website (birdnet-team/birdnet-website, dev branch):
 * public/js/demo/birdbox-service.js, live part only.
 *
 * A run reports the boxes of the 3 s windows at RAW_CONF, a few windows at a
 * time; birdbox-fusion.js turns them into the boxes at the app's confidence.
 */
import { CONFIG } from './config.js';
import { fetchAsset, getManifest } from './model-store.js';
import { CancelledError } from './ort-pool.js';

export const RAW_CONF = 0.05;   // lowest confidence the website's slider offers; the app fuses at its own

class BirdBoxService {
  constructor() { this.ready = null; this.worker = null; this.job = null; this.nextId = 1; this.models = null; }

  get loaded() { return !!this.worker; }

  /** The graphs' bytes, downloaded (or restored from the cache); onProgress gets { loaded, total, cached }. */
  async download(onProgress) {
    if (this.models) return this.models;
    const m = (await getManifest()).birdbox;
    const parts = [m.spectrogram, m.model], state = parts.map(() => ({ loaded: 0, cached: true }));
    const total = parts.reduce((sum, p) => sum + p.size, 0);
    const [spectrogram, detect] = await Promise.all(parts.map((entry, i) => fetchAsset(entry, p => {
      state[i] = p;
      onProgress?.({ loaded: state.reduce((sum, s) => sum + s.loaded, 0), total, cached: state.every(s => s.cached) });
    })));
    // Kept, so a stopped worker restarts without another download
    this.models = { spectrogram, detect };
    return this.models;
  }

  /** Download (or restore) the graphs and start the worker. */
  ensure(onProgress) {
    if (!this.ready) {
      const ready = this.ready = (async () => {
        const m = (await getManifest()).birdbox;
        await this.download(onProgress);
        if (this.ready !== ready) throw new CancelledError();
        const spectrogram = this.models.spectrogram.slice(0), detect = this.models.detect.slice(0);
        const worker = new Worker(new URL('./birdbox-worker.js', import.meta.url), { type: 'module', name: 'birdbox' });
        await new Promise((resolve, reject) => {
          worker.onmessage = ({ data }) => {
            if (data.type === 'ready') resolve();
            else if (data.type === 'error') reject(new Error(data.error));
          };
          worker.onerror = e => { e.preventDefault?.(); reject(new Error(e.message || 'birdbox: worker failed to start')); };
          worker.postMessage({
            type: 'init', ortModule: CONFIG.ortModule, wasmPaths: CONFIG.ortWasm, spectrogram, detect,
            config: { floorWindowSeconds: m.floorWindowSeconds, frameOrigin: m.frameOrigin },
          }, [spectrogram, detect]);
        }).catch(err => { worker.terminate(); throw err; });
        if (this.ready !== ready) { worker.terminate(); throw new CancelledError(); }
        worker.onmessage = ({ data }) => this._done(data);
        worker.onerror = e => {
          e.preventDefault?.();
          this._fail(new Error(e.message || 'birdbox: worker stopped (out of memory?)'));
        };
        this.worker = worker;
        return this;
      })();
      ready.catch(() => { if (this.ready === ready) this.ready = null; });
    }
    return this.ready;
  }

  /** A recording starts: its audio follows with liveAudio. */
  liveStart() {
    this.worker.postMessage({ type: 'live-start' });
  }

  /** Its next samples, 32 kHz (a copy is sent). */
  liveAudio(pcm) {
    const copy = pcm.slice();
    this.worker.postMessage({ type: 'live-audio', pcm: copy }, [copy.buffer]);
  }

  /**
   * Final rows of the next `count` windows of the recording, from window `from` on.
   * The audio sent so far must have reached their end (readyWindows).
   * @returns {Promise<{data: Float32Array, rows: number, next: number, ms: number}>}
   */
  liveWindows(from, count) {
    if (this.job) return Promise.reject(new Error('A detection is already running'));
    return new Promise((resolve, reject) => {
      const id = this.nextId++;
      this.job = { id, resolve, reject };
      this.worker.postMessage({ type: 'windows', id, from, count, conf: RAW_CONF });
    });
  }

  _done(msg) {
    const job = this.job;
    if (!job || msg.id !== job.id) return;
    this.job = null;
    if (msg.type === 'error') {
      // A session that failed (out of memory, most likely) is not used again: the next run starts a fresh worker
      this._reset();
      job.reject(new Error(msg.error));
    } else job.resolve(msg);
  }

  _fail(err) {
    const job = this.job;
    this._reset();
    job?.reject(err);
  }

  /** Stop the worker (a run in progress is cancelled); the next ensure() starts a fresh one. */
  stop() {
    const job = this.job;
    this._reset();
    job?.reject(new CancelledError());
  }

  _reset() {
    this.worker?.terminate();
    this.worker = null;
    this.job = null;
    this.ready = null;
  }
}

export const birdbox = new BirdBoxService();
