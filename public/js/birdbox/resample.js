// Ported unchanged from the BirdNET website (birdnet-team/birdnet-website, dev branch): public/js/demo/resample.js
/**
 * Sample rate conversion with a proper anti-aliasing filter.
 *
 * A Kaiser-windowed sinc low-pass with about 100 dB stopband attenuation,
 * evaluated as a polyphase filter at the exact output positions (n * in/out
 * input samples, as in librosa/soxr). It passes everything up to 91% of the
 * lower Nyquist frequency and removes everything from the Nyquist frequency
 * on: downsampling to 32 kHz keeps 0-14.6 kHz flat and nothing above 16 kHz
 * folds back into the band; upsampling a 22.05 kHz file leaves no mirror
 * image above 11 kHz.
 *
 * Large ratios (96 kHz and up) take two stages: an integer decimation to
 * 48-72 kHz with a short filter whose aliases all land above 16 kHz, then
 * the full filter, which removes them. That is several times cheaper than
 * one long filter at the input rate.
 *
 * Output ranges are computed independently and exactly as in a whole-file
 * run, so workers can convert a recording in parallel chunks, and a stream
 * can be converted as it arrives (StreamConverter).
 */

const ATTEN_DB = 100;
const PASS = 0.91;          // passband edge as a fraction of the lower Nyquist frequency
const MAX_PHASES = 4096;    // exact phase tables up to this many; beyond, interpolate between phases

const gcd = (a, b) => { while (b) [a, b] = [b, a % b]; return a; };

function besselI0(x) {
  let sum = 1, term = 1;
  for (let k = 1; k < 64; k++) {
    term *= (x / (2 * k)) ** 2;
    sum += term;
    if (term < 1e-15 * sum) break;
  }
  return sum;
}

/** One stage inRate -> outRate (integer Hz), passing up to fPass and attenuating from fStop. */
function makeStage(inRate, outRate, fPass, fStop) {
  const g = gcd(inRate, outRate), L = outRate / g, M = inRate / g;
  // Kaiser's estimate of the filter length for this attenuation and transition width
  const halfSeconds = (ATTEN_DB - 7.95) / (2.285 * 2 * Math.PI * (fStop - fPass)) / 2;
  const H = Math.max(1, Math.ceil(halfSeconds * inRate));
  return { L, M, H, T: 2 * H, fc: (fPass + fStop) / 2 / inRate, table: null };
}

/**
 * Filter taps per phase: row p holds h(i - H + 1 - p / P) for i = 0..T-1,
 * each row normalized to unit gain at DC.
 */
function buildTable(s) {
  const { L, H, T, fc } = s;
  const exact = L <= MAX_PHASES, P = exact ? L : MAX_PHASES, rows = exact ? P : P + 1;
  const beta = 0.1102 * (ATTEN_DB - 8.7), norm = besselI0(beta);
  const table = new Float64Array(rows * T);
  for (let r = 0; r < rows; r++) {
    const frac = r / P, o = r * T;
    let sum = 0;
    for (let i = 0; i < T; i++) {
      const tau = i - H + 1 - frac, u = tau / H;
      if (Math.abs(u) >= 1) continue;
      const x = 2 * fc * tau;
      const h = 2 * fc * (x === 0 ? 1 : Math.sin(Math.PI * x) / (Math.PI * x)) * besselI0(beta * Math.sqrt(1 - u * u)) / norm;
      table[o + i] = h;
      sum += h;
    }
    for (let i = 0; i < T; i++) table[o + i] /= sum;
  }
  Object.assign(s, { P, exact, table });
}

/** Input samples [i0, i1) that outputs [o0, o1) of a stage read. */
function stageSpan(s, o0, o1) {
  return [Math.floor(o0 * s.M / s.L) - s.H + 1, Math.floor((o1 - 1) * s.M / s.L) + s.H + 1];
}

/** Outputs [o0, o1) of a stage; x[0] is input sample x0 and x covers stageSpan(). */
function runStage(s, x, x0, o0, o1) {
  if (!s.table) buildTable(s);
  const { L, M, H, T, P, exact, table } = s, out = new Float32Array(o1 - o0);
  const q = Math.floor(M / L), r = M - q * L;
  let k = Math.floor(o0 * M / L), p = o0 * M - k * L;
  for (let n = 0; n < out.length; n++) {
    const base = k - H + 1 - x0;
    let acc = 0;
    if (exact) {
      const o = p * T;
      for (let i = 0; i < T; i++) acc += x[base + i] * table[o + i];
    } else {
      const pos = p * P / L, r0 = Math.floor(pos), w = pos - r0, a = r0 * T, b = a + T;
      for (let i = 0; i < T; i++) acc += x[base + i] * (table[a + i] + w * (table[b + i] - table[a + i]));
    }
    out[n] = acc;
    k += q;
    p += r;
    if (p >= L) { p -= L; k++; }
  }
  return out;
}

/** Number of output samples for nIn input samples (ceil, like librosa). */
const stageLength = (s, nIn) => Math.ceil(nIn * s.L / s.M);

/**
 * Converter for a recording of nIn samples at inRate. Returns null when
 * the rates match (no filtering at all, so 32 kHz audio stays bit-exact).
 */
export function makeConverter(inRate, outRate, nIn) {
  if (inRate === outRate) return null;
  const nyq = Math.min(inRate, outRate) / 2;
  let D = Math.floor(inRate / (1.5 * outRate));
  while (D >= 2 && inRate % D) D--;
  const stages = [];
  if (D >= 2) {
    const mid = inRate / D;
    // Aliases of this stage fold to above outRate / 2, where the next stage removes them
    stages.push(makeStage(inRate, mid, outRate / 2, mid - outRate / 2));
    stages.push(makeStage(mid, outRate, PASS * nyq, nyq));
  } else {
    stages.push(makeStage(inRate, outRate, PASS * nyq, nyq));
  }
  // Valid length of each stage's input; samples outside it are zero
  const lengths = [nIn];
  for (const s of stages) lengths.push(stageLength(s, lengths[lengths.length - 1]));

  return {
    length: lengths[lengths.length - 1],

    /** Input samples [i0, i1) needed for outputs [o0, o1). May reach past either end of the input. */
    span(o0, o1) {
      let a = o0, b = o1;
      for (let k = stages.length - 1; k >= 0; k--) {
        [a, b] = stageSpan(stages[k], a, b);
        if (k > 0) { a = Math.max(0, a); b = Math.min(lengths[k], b); }
      }
      return [a, b];
    },

    /** Outputs [o0, o1); x holds input samples from x0 on, covering span(o0, o1), zero outside the recording. */
    process(x, x0, o0, o1) {
      // Ranges each stage produces, from the last stage back
      const ranges = [[o0, o1]];
      for (let k = stages.length - 1; k > 0; k--) {
        const [a, b] = stageSpan(stages[k], ...ranges[0]);
        ranges.unshift([a, b]);
      }
      let cur = x, curStart = x0;
      for (let k = 0; k < stages.length; k++) {
        const [a, b] = ranges[k];
        if (k === stages.length - 1) return runStage(stages[k], cur, curStart, a, b);
        // Intermediate signal: zero outside its valid length, like a whole-file run
        const lo = Math.max(0, a), hi = Math.min(lengths[k + 1], b);
        const mid = new Float32Array(b - a);
        if (hi > lo) mid.set(runStage(stages[k], cur, curStart, lo, hi), lo - a);
        cur = mid;
        curStart = a;
      }
      return cur;
    },
  };
}

/**
 * Converts a stream as it arrives, such as a recording in progress: output
 * sample n is what a whole-file run of the stream gives.
 */
export class StreamConverter {
  constructor(inRate, outRate) {
    this.inRate = inRate;
    this.outRate = outRate;
    this.conv = makeConverter(inRate, outRate, Infinity);   // null when the rates match
    this.buffer = new Float32Array(1 << 15);
    this.first = 0;     // input sample at buffer[0]
    this.length = 0;    // samples in the buffer
    this.done = 0;      // output samples produced
  }

  /** Outputs done..to-1; the input they read is there, or silence after the end of the stream. */
  _run(to, conv = this.conv) {
    const [i0, i1] = conv.span(this.done, to), x = new Float32Array(i1 - i0);
    const from = Math.max(i0, this.first), end = Math.min(i1, this.first + this.length);
    if (end > from) x.set(this.buffer.subarray(from - this.first, end - this.first), from - i0);
    const out = conv.process(x, i0, this.done, to);
    this.done = to;
    // Keep what the next output samples still read
    const keep = Math.min(this.length, Math.max(0, this.conv.span(to, to + 1)[0] - this.first));
    this.buffer.copyWithin(0, keep, this.length);
    this.first += keep;
    this.length -= keep;
    return out;
  }

  /** @param {Float32Array} input  @returns {Float32Array} the output samples this input completes */
  process(input) {
    if (!this.conv) return input.slice();
    if (this.length + input.length > this.buffer.length) {
      const grown = new Float32Array(Math.max(2 * this.buffer.length, this.length + input.length));
      grown.set(this.buffer.subarray(0, this.length));
      this.buffer = grown;
    }
    this.buffer.set(input, this.length);
    this.length += input.length;
    const have = this.first + this.length;
    let to = Math.max(this.done, Math.floor(have * this.outRate / this.inRate));
    while (to > this.done && this.conv.span(this.done, to)[1] > have) to--;
    return to > this.done ? this._run(to) : new Float32Array(0);
  }

  /** The last samples, once the stream has ended. */
  flush() {
    if (!this.conv) return new Float32Array(0);
    // Now that the length is known, the converter of a file that long
    const conv = makeConverter(this.inRate, this.outRate, this.first + this.length);
    return conv.length > this.done ? this._run(conv.length, conv) : new Float32Array(0);
  }
}
