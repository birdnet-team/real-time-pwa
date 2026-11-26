// Ported unchanged from the BirdNET website (birdnet-team/birdnet-website, dev branch): public/js/demo/region-cut.js
/**
 * The audio of one box, for BirdNET: what lies outside the box's frequency band is
 * replaced by background noise, and so are time-frequency regions inside it that
 * belong to other boxes.
 *
 * Two things do not work here, measured on a busy dawn chorus (XC561949):
 * - A filter with a gentle slope, or turning the rest down. BirdNET still hears a
 *   bird 40 dB down, so a loud bird just above or below a box would name the box.
 * - Silence instead of the rest. The model then scores the box's own bird much lower.
 * So the rest is replaced by the recording's own background: on a short-time Fourier
 * transform (1024 samples, square-root Hann windows, half overlap), each replaced bin
 * gets its typical level over the audio with a random phase. That is steady noise
 * shaped like the recording's, without the sounds that come and go.
 *
 * How sharp it is: a steady tone 40 Hz outside the band is left at -29 dB, one 100 Hz
 * outside at -47 dB, one 300 Hz outside below -70 dB. Frames no region covers are not
 * touched at all; inside a covered frame the kept band loses only what it leaked into
 * the replaced bins (about -40 dB of it).
 */

const N = 1024, HOP = N / 2, BINS = N / 2 + 1;
// A bin's background is this quantile of its level over the audio: the median. Written back as it is, the
// noise comes out about 5 dB below the real background (the window and the median both take some). Raising
// it to the real level named boxes worse on the annotated recordings (89% right instead of 91%), so it stays.
const BACKGROUND = 0.5;
// Analysis and synthesis window: its square is a Hann window, which sums to 1 at half overlap
const WINDOW = Float64Array.from({ length: N }, (_, i) => Math.sin(Math.PI * (i + 0.5) / N));
const COS = Float64Array.from({ length: N / 2 }, (_, i) => Math.cos(2 * Math.PI * i / N));
const SIN = Float64Array.from({ length: N / 2 }, (_, i) => Math.sin(2 * Math.PI * i / N));
const REVERSED = Uint16Array.from({ length: N }, (_, i) => {
  let r = 0;
  for (let b = 1, v = i; b < N; b <<= 1, v >>= 1) r = (r << 1) | (v & 1);
  return r;
});

/** In-place radix-2 FFT of N complex values; `sign` -1 forward, +1 inverse (unscaled). */
function fft(re, im, sign) {
  for (let i = 0; i < N; i++) {
    const j = REVERSED[i];
    if (j > i) {
      let v = re[i]; re[i] = re[j]; re[j] = v;
      v = im[i]; im[i] = im[j]; im[j] = v;
    }
  }
  for (let size = 2; size <= N; size <<= 1) {
    const half = size >> 1, step = N / size;
    for (let start = 0; start < N; start += size) {
      for (let k = 0; k < half; k++) {
        const c = COS[k * step], s = sign * SIN[k * step];
        const a = start + k, b = a + half;
        const xr = re[b] * c - im[b] * s, xi = re[b] * s + im[b] * c;
        re[b] = re[a] - xr; im[b] = im[a] - xi;
        re[a] += xr; im[a] += xi;
      }
    }
  }
}

/**
 * Replace regions of x by the background noise of x, in place.
 * @param {Float32Array} x
 * @param {ArrayLike<number>} regions [t0, t1, flo, fhi] per region: seconds from the start of x, Hz
 * @returns {Float32Array} x
 */
export function cutRegions(x, sr, regions) {
  // Frame k holds the samples from k * HOP; a region covers the frames centred in its time span
  const kMin = Math.ceil((1 - N) / HOP), kMax = Math.floor((x.length - 1) / HOP), frames = kMax - kMin + 1;
  const spans = [], used = new Uint8Array(BINS);
  for (let r = 0; r + 3 < regions.length; r += 4) {
    const k0 = Math.max(kMin, Math.ceil((regions[r] * sr - N / 2) / HOP)), k1 = Math.min(kMax, Math.floor((regions[r + 1] * sr - N / 2) / HOP));
    const b0 = Math.max(0, Math.floor(regions[r + 2] * N / sr)), b1 = Math.min(N / 2, Math.ceil(regions[r + 3] * N / sr));
    if (k1 < k0 || b1 < b0) continue;
    spans.push([k0, k1, b0, b1]);
    used.fill(1, b0, b1 + 1);
  }
  if (!spans.length || frames < 1) return x;

  // The spectrum of every frame, for the background and for the frames to change
  const re = new Float64Array(N), im = new Float64Array(N);
  const specRe = new Float32Array(frames * BINS), specIm = new Float32Array(frames * BINS);
  for (let f = 0; f < frames; f++) {
    const start = (kMin + f) * HOP;
    for (let i = 0; i < N; i++) {
      const q = start + i;
      re[i] = q >= 0 && q < x.length ? x[q] * WINDOW[i] : 0;
      im[i] = 0;
    }
    fft(re, im, -1);
    specRe.set(re.subarray(0, BINS), f * BINS);
    specIm.set(im.subarray(0, BINS), f * BINS);
  }
  // Background of a bin: its median level over the frames, which a sound that comes and goes does not raise much
  const floor = new Float32Array(BINS), levels = new Float32Array(frames);
  for (let b = 0; b < BINS; b++) {
    if (!used[b]) continue;
    for (let f = 0; f < frames; f++) levels[f] = Math.hypot(specRe[f * BINS + b], specIm[f * BINS + b]);
    floor[b] = levels.sort()[Math.floor((frames - 1) * BACKGROUND)];
  }

  const cut = new Uint8Array(BINS), out = new Float64Array(x.length);   // out: what is taken from x
  let seed = 12345;   // the same audio gives the same result
  for (let f = 0; f < frames; f++) {
    const k = kMin + f;
    cut.fill(0);
    let any = false;
    for (const [k0, k1, b0, b1] of spans) {
      if (k < k0 || k > k1) continue;
      cut.fill(1, b0, b1 + 1);
      any = true;
    }
    if (!any) continue;
    // In the cut bins: the sound minus the background with a random phase; elsewhere nothing
    re.fill(0); im.fill(0);
    for (let b = 0; b < BINS; b++) {
      if (!cut[b]) continue;
      let dr = specRe[f * BINS + b], di = specIm[f * BINS + b];
      if (b > 0 && b < N / 2) {
        seed = (Math.imul(seed, 1103515245) + 12345) >>> 0;
        const phase = seed / 4294967296 * 2 * Math.PI;
        dr -= floor[b] * Math.cos(phase); di -= floor[b] * Math.sin(phase);
        re[N - b] = dr; im[N - b] = -di;
      }
      re[b] = dr; im[b] = di;
    }
    fft(re, im, 1);
    const start = k * HOP;
    for (let i = Math.max(0, -start), end = Math.min(N, x.length - start); i < end; i++) out[start + i] += re[i] * WINDOW[i];
  }
  for (let i = 0; i < x.length; i++) if (out[i] !== 0) x[i] -= out[i] / N;
  return x;
}

/**
 * Keep of x the band flo..fhi (Hz) without the regions `cuts`, in place; the rest
 * becomes background noise. The band is kept in whole bins (31 Hz at 32 kHz): from the
 * first bin wholly above flo to the one that holds fhi. A band edge within one bin of
 * the bottom or the top of the spectrum cuts nothing there.
 * @param {Float32Array} x
 * @param {ArrayLike<number>} cuts [t0, t1, flo, fhi] per region, as for cutRegions
 * @returns {Float32Array} x
 */
export function isolateBox(x, sr, flo, fhi, cuts = []) {
  const always = [-1, x.length / sr + 1], bin = sr / N, regions = [...cuts];
  if (flo > bin) regions.push(...always, 0, flo - bin);         // up to the bin below flo
  if (fhi < sr / 2 - bin) regions.push(...always, fhi + bin, sr / 2);
  return cutRegions(x, sr, regions);
}
