/**
 * Ported unchanged (apart from the config import) from the BirdNET website
 * (birdnet-team/birdnet-website, dev branch): public/js/demo/ort-pool.js.
 *
 * A pool of inference workers (ort-worker.js), each with its own
 * single-threaded ONNX Runtime session.
 *
 * Without cross-origin isolation (GitHub Pages cannot send COOP/COEP
 * headers) WASM threads are unavailable, but independent sessions in
 * parallel workers scale nearly as well and need no special headers.
 * Jobs start as soon as the first worker is ready; the others join the
 * queue when their session is loaded.
 */
import { CONFIG } from './config.js';

export class CancelledError extends Error {
  constructor() { super('cancelled'); this.cancelled = true; }
}

const SPAWN_TIMEOUT_MS = 120000;   // a worker that has not loaded its model by then is given up

/** The error of a job that never ran: the workers were gone before its turn. `lost` tells it from a job that failed. */
function lostError(cause) {
  const err = new Error(cause?.message || 'no worker could run the job');
  err.lost = true;
  return err;
}

export class OrtPool extends EventTarget {
  /**
   * @param {{name:string, size:number, retireOnError?:boolean}} options  retireOnError: a worker whose
   *   job failed is stopped (its session may be out of memory for good); a 'lost' event says so
   */
  constructor({ name, size, retireOnError = false }) {
    super();
    this.name = name;
    this.retireOnError = retireOnError;
    this.size = Math.max(1, size);
    this.slots = [];
    this.queue = [];
    this.nextId = 1;
    this.started = null;
    this.allReady = null;
    this.lastError = null;
    this.spawning = 0;
    this.limit = Infinity;   // workers allowed for now (limitTo)
  }

  get readyCount() { return this.slots.length; }

  /** The workers the pool runs: its size, or fewer while limited. */
  get target() { return Math.min(this.size, this.limit); }

  /**
   * Start the pool with a model. Resolves when the first worker can run jobs.
   * The model buffer is transferred, so callers must not reuse it.
   */
  start(model, config) {
    if (this.started) return this.started;
    this.config = config;
    const copies = [], n = this.target;
    for (let i = 0; i < n; i++) copies.push(i === n - 1 ? model : model.slice(0));
    this.started = (async () => {
      const first = await this._spawn(copies[0], config);
      if (!first) throw this.lastError || new Error(`${this.name}: worker failed to start`);
      this.allReady = Promise.all(copies.slice(1).map(buf => this._spawn(buf, config)));
    })();
    this.started.catch(() => { this.started = null; });
    return this.started;
  }

  _spawn(buffer, config) {
    this.spawning++;
    return new Promise(resolve => {
      const worker = new Worker(CONFIG.workerUrl, { type: 'module', name: this.name });
      const fail = err => {
        clearTimeout(timer);
        this.spawning--;
        worker.terminate();
        this.lastError = err;
        resolve(null);
      };
      const timer = setTimeout(() => fail(new Error(`${this.name}: worker did not start`)), SPAWN_TIMEOUT_MS);
      worker.onmessage = ({ data }) => {
        if (data.type === 'ready') {
          clearTimeout(timer);
          this.spawning--;
          if (this.slots.length >= this.target) {   // limited while it started
            worker.terminate();
            resolve(null);
            return;
          }
          const slot = { worker, job: null, ms: data.ms };
          worker.onmessage = e => this._onMessage(slot, e.data);
          worker.onerror = e => this._onCrash(slot, e);
          this.slots.push(slot);
          this.dispatchEvent(new Event('change'));
          this._pump();
          resolve(slot);
        } else if (data.type === 'error') {
          fail(new Error(data.error));
        }
      };
      worker.onerror = e => { e.preventDefault?.(); fail(new Error(e.message || `${this.name}: worker failed to start`)); };
      worker.postMessage({
        type: 'init',
        ortModule: CONFIG.ortModule,
        wasmPaths: CONFIG.ortWasm,
        model: buffer,
        config,
      }, [buffer]);
    });
  }

  /**
   * Queue a job. Higher priority runs first; jobs tagged with a generation
   * can be cancelled together.
   */
  run(type, payload, transfer = [], { gen = 0, priority = 0 } = {}) {
    return new Promise((resolve, reject) => {
      const job = { id: this.nextId++, type, payload, transfer, gen, priority, resolve, reject };
      let i = this.queue.length;
      while (i > 0 && this.queue[i - 1].priority < priority) i--;
      this.queue.splice(i, 0, job);
      this._pump();
    });
  }

  /** Reject queued jobs of a generation (or all queued jobs). Running jobs finish. */
  cancel(gen) {
    const keep = [];
    for (const job of this.queue) {
      if (gen === undefined || job.gen === gen) job.reject(new CancelledError());
      else keep.push(job);
    }
    this.queue = keep;
  }

  _pump() {
    for (const slot of this.slots) {
      if (slot.job || !this.queue.length) continue;
      const job = this.queue.shift();
      slot.job = job;
      slot.worker.postMessage({ type: job.type, id: job.id, ...job.payload }, job.transfer);
    }
  }

  _onMessage(slot, msg) {
    const job = slot.job;
    slot.job = null;
    const failed = !!job && msg.id === job.id && msg.type === 'error';
    if (job && msg.id === job.id) {
      if (failed) job.reject(this.lastError = new Error(msg.error));
      else job.resolve(msg);
    }
    if (failed && this.retireOnError) {
      this._retire(slot);
      this._lost();
    } else if (this.slots.length > this.target) this._retire(slot);
    this._pump();
  }

  /** A worker is gone for a failure. With none left and none starting, the queued jobs fail too. */
  _lost() {
    this.dispatchEvent(new Event('lost'));   // a listener may start a replacement, which takes the queue
    this._orphans();
  }

  /** With no worker left and none starting, the queued jobs can never run: they fail as lost. */
  _orphans() {
    if (this.slots.length || this.spawning || this.growing || !this.queue.length) return;
    const err = lostError(this.lastError);
    for (const job of this.queue) job.reject(err);
    this.queue = [];
  }

  /** Stop an idle worker. */
  _retire(slot) {
    slot.worker.terminate();
    this.slots = this.slots.filter(s => s !== slot);
    this.dispatchEvent(new Event('change'));
  }

  /**
   * Run at most `n` workers (Infinity: the pool's size again; grow() starts the
   * missing ones). Workers beyond it stop now if idle, or when their job is done.
   */
  limitTo(n) {
    this.limit = Math.max(1, n);
    for (const slot of [...this.slots]) {
      if (this.slots.length <= this.target) break;
      if (!slot.job) this._retire(slot);
    }
  }

  _onCrash(slot, event) {
    event.preventDefault?.();
    slot.worker.terminate();
    this.slots = this.slots.filter(s => s !== slot);
    this.lastError = new Error(event.message || `${this.name}: worker stopped (out of memory?)`);
    if (slot.job) slot.job.reject(this.lastError);
    this.dispatchEvent(new Event('change'));
    this._lost();
    this._pump();
  }

  /** Stop idle workers beyond `keep` to give their memory back; grow() starts them again. */
  trim(keep = 1) {
    if (this.queue.length || this.spawning) return; // still working through a job list
    for (const slot of [...this.slots]) {
      if (this.slots.length <= keep) break;
      if (slot.job) continue;
      slot.worker.terminate();
      this.slots = this.slots.filter(s => s !== slot);
    }
    this.dispatchEvent(new Event('change'));
  }

  /** Start workers up to the pool size (or its limit) again after trim(). */
  async grow(getModel) {
    if (this.target - this.slots.length - this.spawning <= 0 || !this.config) return;
    if (this.growing) { this.growAgain = true; return; }   // the limit may be higher by the time this one is done
    this.growing = true;
    try {
      const model = await getModel();
      const missing = this.target - this.slots.length - this.spawning;   // the limit may have changed meanwhile
      await Promise.all(Array.from({ length: Math.max(0, missing) }, (_, i) => this._spawn(i === missing - 1 ? model : model.slice(0), this.config)));
    } catch (err) {
      this.lastError = err;   // the model could not be fetched
      throw err;
    } finally {
      this.growing = false;
      this._orphans();   // no worker came of it: the jobs waiting for one fail instead of waiting forever
      if (this.growAgain) {
        this.growAgain = false;
        this.grow(getModel).catch(() => {});
      }
    }
  }

  terminate() {
    this.cancel();
    for (const slot of this.slots) {
      slot.worker.terminate();
      if (slot.job) slot.job.reject(new CancelledError());
    }
    this.slots = [];
    this.started = null;
  }
}
