/**
 * BirdBox live detection for BirdNET Live: boxes around bird sounds on the live
 * spectrogram, each named by BirdNET V3.0.
 *
 * This follows the BirdBox tab of the BirdNET website (birdnet-team/birdnet-website,
 * dev branch: public/js/demo/birdbox-tab.js) while it records:
 *
 * - The microphone's audio is converted to 32 kHz as it arrives (resample.js) and
 *   BirdBox (Just-Bird) runs on it as a stream (birdbox-stream.js): each 3 s window
 *   is detected once, as soon as the audio reaches its end, and its boxes are final.
 * - The windows' boxes are fused as BirdBox's graph does (birdbox-fusion.js), at the
 *   confidence and phrase gap of the app's settings (by default the website's: 0.15
 *   and 0.05 s).
 * - BirdNET V3.0 scores every box on its own audio: its time span (at least 1 s),
 *   with everything outside its frequency band, and the parts that smaller boxes
 *   inside it cover, replaced by background noise (region-cut.js). Only birds count.
 * - A box is named after its best-scoring bird among the species the recording surely
 *   holds: those that are the best bird of some box at the species confidence (a
 *   setting, 0.7 by default as on the website) or more. Such a species names a box
 *   from half that score.
 *   With the geolocation filter on, only species the app's location model expects
 *   there are candidates.
 *
 * Unlike the website, the app listens for as long as the person wants, so audio and
 * boxes are kept for KEEP_SECONDS only (the spectrogram shows at most 60 s), and the
 * "surely held" species are those of the whole time the app listens.
 *
 * The page (app.js, a classic script) uses it through window.BirdBoxLive.
 */
import { StreamConverter } from './resample.js';
import { birdbox, RAW_CONF } from './birdbox-service.js';
import { analyzedTo as analyzedAfter, readyWindows } from './birdbox-stream.js';
import { fuseBoxes } from './birdbox-fusion.js';
import { boxCrops, SR, v3 } from './v3-service.js';
import { catalog } from './species.js';
import { getManifest } from './model-store.js';

const CONF = 0.15;            // BirdBox's confidence until the app sets it (setBoxes): the website's default
const PHRASE_GAP = 0.05;      // seconds: BirdBox's phrase gap until the app sets it, the website's default
const SPECIES_CONF = 0.7;     // species confidence until the app sets it (setSure): the website's default
const SPECIES_LOW = 0.5;      // a species surely in the recording names a box from this share of the species confidence
const MIN_SCORE = 0.05;       // BirdNET V3.0 scores below this are not kept
const CANDIDATES = 5;         // species a box offers, best first
const HOP = 1.5;              // seconds from one BirdBox window to the next
const SETTLED = 1.5;          // a box this far (plus the phrase gap) behind the detection line no longer changes, unless linked to one that may
const CUT_MIN_DUR = 0.02;     // an overlap with another box counts from this many seconds
const CUT_MIN_BAND = 50;      // and this many Hz
const LIVE_MAX_WINDOWS = 8;   // windows per run when detection has fallen behind
const KEEP_SECONDS = 120;     // audio and boxes kept: the spectrogram shows at most 60 s
const KEYS_MAX = 2000;        // scores of this many sounds are kept; beyond, those of boxes that are gone are dropped
const SCORE_TRIES = 2;        // a box whose scoring failed this often gets no species
const SCORING_AHEAD = 2;      // boxes queued for BirdNET V3.0 at a time
const RETRY_MS = 10000;       // after BirdNET V3.0 failed, wait this long before trying again; twice as long each time in a row
const RETRY_MAX_MS = 160000;
const IDLE_RELEASE_MS = 120000;   // the workers (BirdNET V3.0 alone holds about 500 MB) stop this long after listening stops
const GEO_THRESHOLD = 0.05;   // the location model's score from which the app counts a species as expected (renderDetections)
const POP_MS = 420;           // a new box pops in over this time
const NONE = -1;              // no species for a box
const SPECIES_COLORS = ['#60a5fa', '#f87171', '#a3e635', '#c084fc', '#fb923c', '#f9a8d4', '#e7c9a0', '#818cf8'];
const DETECTED_COLOR = '#22d3ee';   // a box without a species
const LABEL_FONT = '600 11px system-ui, -apple-system, "Segoe UI", sans-serif';

/** A detected box, to the 0.1 ms and the hertz: its id while it stays the same. */
const boxKey = b => `${b.t0.toFixed(4)}|${b.t1.toFixed(4)}|${b.flo}|${b.fhi}`;
/** A rectangle of sound, to the millisecond and the hertz. */
const soundKey = g => `${g.t0.toFixed(3)}|${g.t1.toFixed(3)}|${Math.round(g.flo)}|${Math.round(g.fhi)}`;

/** The raw rows `data` (the first `rows` of it) with the rows `add` after them; the buffer grows by doubling. */
const appendRows = (data, rows, add) => {
  const used = rows * 9;
  if (used + add.length > data.length) {
    const grown = new Float32Array(Math.max(used + add.length, 2 * data.length, 1152));
    grown.set(data.subarray(0, used));
    data = grown;
  }
  data.set(add, used);
  return data;
};

class BirdBoxLive {
  constructor() {
    this.enabled = false;
    this.session = null;      // the time the app listens with BirdBox on (newSession)
    this.past = [];           // boxes of earlier sessions, as drawn when they ended: { t0, t1, flo, fhi, label, color }
    this.timeline = 0;        // seconds of the spectrogram the earlier sessions take: the current one starts here
    this.colors = new Map();  // species -> its color on the spectrogram
    this.geo = null;          // scientific name -> location score, or null when the filter is off
    this.labelLang = 'en_us';
    this.conf = CONF;         // BirdBox's confidence and phrase gap (setBoxes)
    this.gap = PHRASE_GAP;
    this.sureLevel = SPECIES_CONF;   // species confidence: a species this sure in some box names boxes (setSure)
    this.status = { state: 'off' };
    this.preloading = null;
    this.canvas = null;
    this.frame = 0;
  }

  /**
   * @param {{ parent: HTMLElement, view: () => object, onStatus: (status) => void }} options
   *   parent: the spectrogram's container; view(): the spectrogram's { width, height,
   *   secondsPerPx, minFreq, maxFreq }, whose right edge is now
   */
  init({ parent, view, onStatus }) {
    this.view = view;
    this.onStatus = onStatus;
    if (parent && !this.canvas) {
      const c = this.canvas = document.createElement('canvas');
      c.className = 'birdbox-overlay';
      c.setAttribute('aria-hidden', 'true');
      Object.assign(c.style, { position: 'absolute', top: '0', left: '0', pointerEvents: 'none', zIndex: '5' });
      parent.appendChild(c);
      window.addEventListener('resize', () => this.requestDraw());
    }
    this._setStatus(this.status);
  }

  /* ---------------- switching on and off ---------------- */

  /** BirdBox on: the models are downloaded (or restored) now, so listening starts with them. */
  enable() {
    if (this.enabled) return;
    this.enabled = true;
    this._preload();
  }

  /** BirdBox off: boxes go, and so do the workers and their memory. The models stay in the cache. */
  disable() {
    if (!this.enabled) return;
    this.enabled = false;
    this._endSession();
    clearTimeout(this.idleTimer);
    birdbox.stop();
    v3.stop();
    this.past = [];
    this.timeline = 0;
    this.colors.clear();
    this.preloading = null;
    this._setStatus({ state: 'off' });
    this.requestDraw();
  }

  _preload() {
    if (this.preloading) return this.preloading;
    const run = this.preloading = (async () => {
      const m = await getManifest();
      const sizes = { birdbox: m.birdbox.spectrogram.size + m.birdbox.model.size, v3: m.v3.model.size };
      const total = sizes.birdbox + sizes.v3, done = { birdbox: 0, v3: 0 };
      const progress = key => p => {
        done[key] = p.loaded;
        if (this.preloading === run && !p.cached) this._setStatus({ state: 'loading', pct: Math.floor(100 * (done.birdbox + done.v3) / total) });
      };
      await Promise.all([birdbox.download(progress('birdbox')), v3.download(progress('v3')), catalog.load().then(() => this._applyLabelLang())]);
    })();
    run.then(() => {
      if (this.preloading === run) this._setStatus({ state: this.session?.open ? 'active' : 'ready' });
    }, err => {
      if (this.preloading !== run) return;
      this.preloading = null;   // the next start tries again
      console.error('BirdBox:', err);
      this._setStatus({ state: 'error', message: err.message });
    });
    return run;
  }

  _setStatus(status) {
    this.status = status;
    this.onStatus?.(status);
  }

  /* ---------------- listening ---------------- */

  /** Listening starts, with audio at `sampleRate`, which push() hands in as it comes. */
  start(sampleRate) {
    if (!this.enabled) return;
    this._endSession();
    clearTimeout(this.idleTimer);
    const prev = this.loop;
    const s = this.session = this._newSession(sampleRate);
    if (this.status.state !== 'loading') this._setStatus({ state: 'active' });
    s.loop = this.loop = this._loop(s, prev);
    this._animate();
  }

  /** Listening stops: what has been detected stays on the spectrogram, and its last boxes get their species. */
  stop() {
    const s = this.session;
    if (!s?.open) return;
    s.open = false;
    s.wake?.();
    // No later window can change a box now
    s.pending = new Set();
    this._refresh(s);
    clearTimeout(this.idleTimer);
    this.idleTimer = setTimeout(() => {
      if (this.session?.open) return;
      birdbox.stop();
      v3.stop();
    }, IDLE_RELEASE_MS);
    if (this.enabled && this.status.state !== 'loading' && this.status.state !== 'error') this._setStatus({ state: 'ready' });
    this.requestDraw();
  }

  /** Audio as it arrives, at the sample rate given to start(). */
  push(pcm) {
    const s = this.session;
    if (!s?.open) return;
    s.inSamples += pcm.length;
    s.arrivedAt = performance.now();
    const out = s.converter.process(pcm);
    if (!out.length) return;
    if (s.n - s.base + out.length > s.audio.length) {
      // Only the last KEEP_SECONDS: the boxes before them are off the spectrogram
      const keep = Math.max(s.base, s.n + out.length - KEEP_SECONDS * SR);
      s.audio.copyWithin(0, keep - s.base, s.n - s.base);
      s.base = keep;
    }
    s.audio.set(out, s.n - s.base);
    s.n += out.length;
    this._sendAudio(s);
    s.wake?.();
    this._pumpScores(s);   // the audio after a box may be complete now
  }

  _newSession(sampleRate) {
    return {
      open: true, inRate: sampleRate, converter: new StreamConverter(sampleRate, SR), inSamples: 0, arrivedAt: 0,
      offset: this.timeline,                                  // its start on the spectrogram's timeline
      audio: new Float32Array((KEEP_SECONDS + 20) * SR), base: 0, n: 0,   // 32 kHz: audio[i] is sample base + i
      live: false, streamStart: 0, sent: 0,                   // the BirdBox stream: its first sample, and the samples it has
      data: new Float32Array(0), rows: 0, next: 0, fusion: {}, analyzedTo: null,   // its windows' boxes, in stream time
      songs: [], pending: new Set(), sounds: new Map(), born: new Map(),
      scores: new Map(), candidates: new Map(), wanted: new Map(), scoring: new Set(), failed: new Map(),
      sure: new Set(), retryAt: 0, retryMs: RETRY_MS, pumping: false, wake: null, loop: null,
    };
  }

  /** The session ends for good: its boxes become part of the past, as they are drawn now. */
  _endSession() {
    const s = this.session;
    if (!s) return;
    this.stop();
    for (const b of s.songs) {
      const sp = this._suggested(s, b);
      this.past.push({ t0: s.offset + b.t0, t1: s.offset + b.t1, flo: b.flo, fhi: b.fhi, label: sp === NONE ? '' : catalog.name(sp), color: this._color(sp) });
    }
    this.timeline = s.offset + s.inSamples / s.inRate;
    v3.cancel('birdbox');
    this.session = null;
  }

  /** New audio to the BirdBox worker, once its stream has started. */
  _sendAudio(s) {
    if (!s.live || s.n <= s.sent || !s.open) return;
    if (s.sent < s.base) {
      // Cannot happen while the worker takes the audio as it comes; were it ever, the stream would have a hole
      s.open = false;
      s.wake?.();
      this._setStatus({ state: 'error', message: 'BirdBox fell behind the microphone' });
      return;
    }
    birdbox.liveAudio(s.audio.subarray(s.sent - s.base, s.n - s.base));
    s.sent = s.n;
  }

  /**
   * Each window is run once, as soon as the audio reaches its end, and its rows are
   * final. A run takes every window that has become ready meanwhile (up to
   * LIVE_MAX_WINDOWS), so a slower device falls behind by a run, not more.
   */
  async _loop(s, prev) {
    try {
      await prev?.catch(() => {});   // the last session's run, if one was still on its way
      await this._preload();
      await birdbox.ensure();
      if (this.session !== s || !s.open) return;
      if (this.status.state !== 'error') this._setStatus({ state: 'active' });
      // The stream starts with the oldest audio kept: models loading for long lose some at the start
      birdbox.liveStart();
      s.live = true;
      s.streamStart = s.sent = s.base;
      this._sendAudio(s);
      while (s.open && this.session === s) {
        const ready = readyWindows(s.sent - s.streamStart);
        if (ready <= s.next) {
          await new Promise(resolve => { s.wake = resolve; });
          s.wake = null;
          continue;
        }
        const res = await birdbox.liveWindows(s.next, Math.min(ready - s.next, LIVE_MAX_WINDOWS));
        if (this.session !== s) return;
        s.data = appendRows(s.data, s.rows, res.data);
        s.rows += res.rows;
        s.next = res.next;
        this._trim(s);
        this._update(s);
      }
    } catch (err) {
      if (err.cancelled || this.session !== s || !this.enabled) return;
      console.error('BirdBox:', err);
      s.open = false;
      this._setStatus({ state: 'error', message: err.message });
    }
  }

  /** Rows of windows that ended more than KEEP_SECONDS ago go: their boxes are off the spectrogram. */
  _trim(s) {
    const cutoff = s.next * HOP - KEEP_SECONDS;
    if (!s.rows || !(s.data[7] < cutoff - 15)) return;
    let r = 0;
    while (r < s.rows && s.data[r * 9 + 7] < cutoff) r++;
    s.data = s.data.slice(r * 9, s.rows * 9);
    s.rows -= r;
    s.fusion = {};   // its rows changed: fused anew
  }

  /* ---------------- boxes ---------------- */

  /** quiet: the settings changed, not the audio; boxes that differ from the old ones do not pop in. */
  _update(s, quiet = false) {
    const off = s.streamStart / SR, before = s.songs, now = performance.now();
    // Rows still to come start with the next window: the boxes before them are fused once
    s.songs = fuseBoxes(s.data, s.rows, this.conf, this.gap, s.fusion, s.next * HOP)
      .map(b => ({ ...b, t0: b.t0 + off, t1: b.t1 + off }));
    for (const b of s.songs) {
      b.id = boxKey(b);
      if (!quiet && !before.some(o => o.t0 < b.t1 && b.t0 < o.t1 && o.flo < b.fhi && b.flo < o.fhi)) s.born.set(b.id, now);
    }
    if (s.born.size > 500) for (const [id, t] of s.born) if (now - t > POP_MS) s.born.delete(id);
    s.analyzedTo = off + analyzedAfter(s.next);
    this._markPending(s);
    this._refresh(s);
  }

  /**
   * The boxes that may still change: the next windows can fuse a box near the detection
   * line with theirs, join it into a phrase or hide it as nested, so a box ending less
   * than one window step (1.5 s) plus the phrase gap behind the line may change, and so
   * may a box linked to one that may (within the phrase gap, with overlapping bands).
   */
  _markPending(s) {
    s.pending = new Set();
    if (s.analyzedTo == null || !s.open) return;
    const after = s.analyzedTo - SETTLED - this.gap, todo = [];
    for (const b of s.songs) if (b.t1 > after) { s.pending.add(b); todo.push(b); }
    while (todo.length) {
      const p = todo.pop();
      for (const b of s.songs) {
        if (s.pending.has(b) || b.cls !== p.cls) continue;
        if (Math.max(b.t0, p.t0) - Math.min(b.t1, p.t1) > this.gap || b.fhi <= p.flo || p.fhi <= b.flo) continue;
        s.pending.add(b);
        todo.push(b);
      }
    }
  }

  _refresh(s) {
    this._buildSounds(s);
    s.wanted.clear();
    for (const b of s.songs) {
      const sound = s.sounds.get(b);
      if (!s.scores.has(sound.key)) s.wanted.set(sound.key, sound);
    }
    this._pumpScores(s);
    this.requestDraw();
  }

  /**
   * The audio every box is scored on: its rectangle, without the parts of it that
   * smaller boxes cover (`cuts`), so a box around another bird's call is not named
   * after that bird. `later`: the box, or one cut out of it, may still change.
   * The key names the rectangle and its cuts: the same key, the same audio.
   */
  _buildSounds(s) {
    const items = s.songs.map(g => ({ g, area: (g.t1 - g.t0) * (g.fhi - g.flo), cuts: [], later: s.pending.has(g) }));
    // The boxes come by start time: only the following ones that start before a box ends can overlap it
    for (let i = 0; i < items.length; i++) {
      const a = items[i];
      for (let j = i + 1; j < items.length && items[j].g.t0 < a.g.t1; j++) {
        const c = items[j], [small, large] = a.area <= c.area ? [a, c] : [c, a];
        const cut = { t0: c.g.t0, t1: Math.min(a.g.t1, c.g.t1), flo: Math.max(a.g.flo, c.g.flo), fhi: Math.min(a.g.fhi, c.g.fhi) };
        if (cut.t1 - cut.t0 < CUT_MIN_DUR || cut.fhi - cut.flo < CUT_MIN_BAND) continue;
        large.cuts.push(cut);
        if (small.later) large.later = true;
      }
    }
    s.sounds = new Map(items.map(it => [it.g, {
      key: [soundKey(it.g), ...it.cuts.map(soundKey).sort()].join('#'), g: it.g, cuts: it.cuts, later: it.later,
    }]));
    if (s.scores.size > KEYS_MAX) {
      const live = new Set([...s.sounds.values()].map(x => x.key));
      for (const map of [s.scores, s.candidates, s.failed]) for (const key of map.keys()) if (!live.has(key)) map.delete(key);
    }
  }

  /* ---------------- species: BirdNET V3.0 on each box ---------------- */

  /** Keep BirdNET V3.0 busy alongside the detection, whenever there may be a box to score. */
  async _pumpScores(s) {
    if (s.pumping || !s.wanted.size || performance.now() < s.retryAt || this.session !== s) return;
    s.pumping = true;
    try {
      await v3.ensure();
      await catalog.load();
    } catch (err) {
      if (this.session === s) this._scoreFailed(s, err);
      return;
    } finally {
      s.pumping = false;
    }
    for (let key; this.session === s && s.scoring.size < SCORING_AHEAD && (key = this._nextBox(s)) != null;) this._scoreBox(s, key, s.wanted.get(key));
  }

  /**
   * The box to score next: the earliest one that can no longer change. While the app
   * listens, a box that may still change waits, and a short box near the end waits for
   * the audio around it.
   */
  _nextBox(s) {
    for (const [key, want] of s.wanted) {
      if (s.scores.has(key)) { s.wanted.delete(key); continue; }
      if (s.scoring.has(key) || (want.later && s.open) || !boxCrops(want.g, s.n, s.open)) continue;
      return key;
    }
    return null;
  }

  /** BirdNET V3.0 on the audio of one box. Only birds are kept: BirdBox finds bird sounds. */
  async _scoreBox(s, key, sound) {
    s.scoring.add(key);
    const read = (from, to) => (from < s.base || to > s.n ? null : s.audio.slice(from - s.base, to - s.base));
    try {
      const out = await v3.scoreBox(read, s.n, sound.g, sound.cuts, MIN_SCORE, { gen: 'birdbox', priority: 5, alive: () => this.session === s });
      if (this.session !== s) return;
      // null: the box's audio is gone (scoring fell far behind): it stays without a species
      s.scores.set(key, (out || []).filter(([c]) => catalog.cls[c] === catalog.aves));
      s.retryMs = RETRY_MS;
      s.scoring.delete(key);
      s.wanted.delete(key);
      const top = this._candidates(s, key)?.[0];
      if (top && top.p >= this.sureLevel) s.sure.add(top.s);
      this.requestDraw();
    } catch (err) {
      if (this.session !== s || err.cancelled) return;
      s.scoring.delete(key);
      // A box that fails again (too long for the memory there is, most likely) goes without a species.
      // A box that only waited behind one that failed (err.lost) has not been tried.
      const tries = (s.failed.get(key) || 0) + (err.lost ? 0 : 1);
      s.failed.set(key, tries);
      if (tries >= SCORE_TRIES) {
        s.scores.set(key, []);
        s.wanted.delete(key);
      }
      this._scoreFailed(s, err);
      return;
    }
    this._pumpScores(s);
  }

  /** After a failure: wait, twice as long each time in a row, then go on with the boxes still waiting. */
  _scoreFailed(s, err) {
    console.error('BirdBox species:', err);
    if (performance.now() < s.retryAt) return;
    s.retryAt = performance.now() + s.retryMs;
    setTimeout(() => this._pumpScores(s), s.retryMs + 50);
    s.retryMs = Math.min(RETRY_MAX_MS, 2 * s.retryMs);
  }

  /** Up to CANDIDATES species for a box, best first, within the location filter; null while it is to be scored. */
  _candidates(s, key) {
    let cands = s.candidates.get(key);
    if (cands) return cands;
    const birds = s.scores.get(key);
    if (!birds) return null;
    const geo = this.geo;
    cands = birds.filter(([c]) => !geo || (geo.get(catalog.sci[c]) ?? -1) >= GEO_THRESHOLD)
      .map(([c, p]) => ({ s: c, p })).sort((a, b) => b.p - a.p).slice(0, CANDIDATES);
    s.candidates.set(key, cands);
    return cands;
  }

  /**
   * The species BirdNET V3.0 gives a box, or NONE: its best-scoring bird among the
   * species the recording surely holds, from SPECIES_LOW of the species confidence.
   * A single box is short, and its best bird at a middling score is often wrong; a
   * species that is also heard clearly elsewhere is right far more often (see the
   * website's README for the numbers).
   */
  _suggested(s, b) {
    const sound = s.sounds.get(b), cands = sound && this._candidates(s, sound.key);
    if (!cands?.length) return NONE;
    const pick = cands.find(c => s.sure.has(c.s));
    return pick && pick.p >= SPECIES_LOW * this.sureLevel ? pick.s : NONE;
  }

  _color(species) {
    if (species === NONE) return DETECTED_COLOR;
    if (!this.colors.has(species)) this.colors.set(species, SPECIES_COLORS[this.colors.size % SPECIES_COLORS.length]);
    return this.colors.get(species);
  }

  /* ---------------- settings from the page ---------------- */

  /**
   * The location filter: a Map of scientific name -> the app's location score
   * (BirdNET V2.4's area model), or null when it is off. A species the location
   * model does not know is not suggested while the filter is on.
   */
  setGeo(scores) {
    this.geo = scores && scores.size ? scores : null;
    const s = this.session;
    if (!s) return;
    s.candidates.clear();
    this._findSure(s);
  }

  /**
   * BirdBox's confidence and phrase gap, the website's sliders of those names (the
   * app's Box Confidence Threshold and Phrase Gap). The windows' boxes are kept from
   * RAW_CONF, so the boxes on the spectrogram are fused anew at once.
   */
  setBoxes(conf, gap) {
    if (!(conf >= RAW_CONF && conf < 1) || !(gap >= 0 && gap <= 5) || (conf === this.conf && gap === this.gap)) return;
    this.conf = conf;
    this.gap = gap;
    const s = this.session;
    if (!s || !s.rows) return;
    s.fusion = {};   // fused with the old values: anew
    this._update(s, true);
  }

  /**
   * The species confidence (0..1), the app's BirdBox Species Threshold: a box gets a
   * species once BirdNET V3.0 has been this sure of that species in some box, and its
   * own score is at least half of it. Boxes already scored are named again at once.
   */
  setSure(level) {
    if (!(level > 0 && level <= 1) || level === this.sureLevel) return;
    this.sureLevel = level;
    if (this.session) this._findSure(this.session);
  }

  /** The species the session surely holds, found again from all scores. */
  _findSure(s) {
    s.sure = new Set();
    for (const key of s.scores.keys()) {
      const top = this._candidates(s, key)?.[0];
      if (top && top.p >= this.sureLevel) s.sure.add(top.s);
    }
    this.requestDraw();
  }

  /** Species names in the app's label language. */
  setLabelLang(lang) {
    this.labelLang = lang;
    if (this.enabled) this._applyLabelLang();
  }

  /** One language change after the other, so the last one chosen wins. */
  _applyLabelLang() {
    this.langChange = (this.langChange || Promise.resolve())
      .then(() => catalog.setLabelLang(this.labelLang))
      .then(() => this.requestDraw(), err => console.error('BirdBox names:', err));
    return this.langChange;
  }

  /* ---------------- drawing ---------------- */

  _now() {
    const s = this.session;
    if (!s) return this.timeline;
    const t = s.inSamples / s.inRate;
    // Smooth between the pieces of audio, which arrive in bursts
    return s.offset + (s.open && s.inSamples ? t + Math.min(performance.now() - s.arrivedAt, 120) / 1000 : t);
  }

  _animate() {
    if (this.frame) return;
    const tick = () => {
      this.frame = 0;
      this._draw();
      if (this.session?.open || this._popping) this.frame = requestAnimationFrame(tick);
    };
    this.frame = requestAnimationFrame(tick);
  }

  /** Draw once more soon (the boxes changed while the spectrogram stands still). */
  requestDraw() {
    if (!this.frame) this.frame = requestAnimationFrame(() => { this.frame = 0; this._draw(); if (this._popping) this._animate(); });
  }

  _draw() {
    const c = this.canvas, v = this.view?.();
    if (!c || !v || !v.width || !v.height) return;
    const dpr = window.devicePixelRatio || 1, w = Math.round(v.width * dpr), h = Math.round(v.height * dpr);
    if (c.width !== w || c.height !== h) {
      c.width = w;
      c.height = h;
      c.style.width = `${v.width}px`;
      c.style.height = `${v.height}px`;
    }
    const ctx = c.getContext('2d');
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, v.width, v.height);
    this._popping = false;
    if (!this.enabled && !this.past.length) return;

    const now = this._now(), span = v.maxFreq - v.minFreq;
    const X = t => v.width - (now - t) / v.secondsPerPx;
    const Y = f => v.height * (1 - (f - v.minFreq) / span);
    const visible = (t0, t1) => X(t1) >= 0 && X(t0) <= v.width;
    this.past = this.past.filter(b => X(b.t1) >= -50);

    const items = this.past.filter(b => visible(b.t0, b.t1)).map(b => ({ ...b, dashed: false, pop: 1 }));
    const s = this.session, clock = performance.now(), calm = window.matchMedia?.('(prefers-reduced-motion: reduce)').matches;
    if (s) {
      // The detection line: where BirdBox has got to, behind the microphone
      if (s.open && s.analyzedTo != null) {
        const x = X(s.offset + s.analyzedTo);
        if (x >= 0 && x <= v.width) {
          ctx.fillStyle = 'rgba(34,211,238,0.45)';
          ctx.fillRect(Math.round(x), 0, 1, v.height);
        }
      }
      for (const b of s.songs) {
        const t0 = s.offset + b.t0, t1 = s.offset + b.t1;
        if (!visible(t0, t1)) continue;
        const sp = this._suggested(s, b), born = s.open ? s.born.get(b.id) : null;
        const pop = calm || born == null ? 1 : Math.min(1, (clock - born) / POP_MS);
        if (pop < 1) this._popping = true;
        items.push({ t0, t1, flo: b.flo, fhi: b.fhi, label: sp === NONE ? '' : catalog.name(sp), color: this._color(sp), dashed: s.pending.has(b), pop });
      }
    }

    ctx.font = LABEL_FONT;
    ctx.textBaseline = 'alphabetic';
    const tags = [];
    for (const it of items) {
      const x0 = X(it.t0), x1 = X(it.t1), y0 = Y(Math.min(it.fhi, v.maxFreq)), y1 = Y(Math.max(it.flo, v.minFreq));
      if (y1 <= 0 || y0 >= v.height) continue;   // outside the frequencies shown
      if (it.pop < 1) {
        // A ring that settles on the box as it appears
        const grow = (1 - it.pop) * 10, eased = it.pop * it.pop;
        ctx.strokeStyle = it.color;
        ctx.globalAlpha = (1 - eased) * 0.8;
        ctx.lineWidth = 1.5;
        ctx.strokeRect(x0 - grow, y0 - grow, Math.max(2, x1 - x0) + 2 * grow, Math.max(2, y1 - y0) + 2 * grow);
        ctx.globalAlpha = 1;
      }
      ctx.strokeStyle = it.color;
      ctx.lineWidth = 1.5;
      ctx.setLineDash(it.dashed ? [6, 3] : []);
      ctx.strokeRect(x0, y0, Math.max(2, x1 - x0), Math.max(2, y1 - y0));
      ctx.setLineDash([]);
      if (!it.label) continue;
      // The species on a tag above the box, unless another tag is there already
      const tw = ctx.measureText(it.label).width + 8, tx = Math.max(v.labelLeft || 0, Math.min(x0, v.width - tw)), ty = Math.max(14, y0 - 2);
      if (tags.some(o => o.x < tx + tw + 2 && o.x + o.w + 2 > tx && Math.abs(o.y - ty) < 16)) continue;
      tags.push({ x: tx, y: ty, w: tw });
      ctx.fillStyle = it.color;
      ctx.globalAlpha = 0.9;
      ctx.fillRect(tx, ty - 13, tw, 15);
      ctx.globalAlpha = 1;
      ctx.fillStyle = '#05040c';
      ctx.fillText(it.label, tx + 4, ty - 2);
    }
  }
}

window.BirdBoxLive = new BirdBoxLive();
window.dispatchEvent(new Event('birdboxlive-ready'));
