/**
 * BirdNET Live - Web Worker
 * Handles TensorFlow.js model loading, audio preprocessing (STFT/Mel),
 * inference, and post-processing (sensitivity, pooling, geolocation).
 * 
 * Code based on https://github.com/georg95/birdnet-web
 */

/* ==========================================================================
   1. IMPORTS & CONFIGURATION
   ========================================================================== */

const params = new URL(self.location.href).searchParams;
const TF_PATH = params.get('tf') || 'js/tfjs-4.14.0.min.js';
importScripts(TF_PATH);

// Paths
const ROOT = params.get('root') || 'models';
const BIRD_BASE = ROOT + '/birdnet';
const MODEL_PATH = BIRD_BASE + '/model.json';
const AREA_MODEL_PATH = BIRD_BASE + '/area-model/model.json';
const LABELS_DIR = BIRD_BASE + '/labels';

// Audio Constants
const SAMPLE_RATE = 48000;
const WINDOW_SAMPLES = 144000; // 3 seconds at 48kHz

/* ==========================================================================
   2. GLOBAL STATE
   ========================================================================== */

// Models
let birdModel = null;
let areaModel = null;

// Data
let birds = []; // Array of { scientificName, commonName, geoscore, ... }

// Inference Cache (for re-applying geo priors without re-running inference)
let lastPredictionList = null;
let lastMeans = null;
let lastHopSamples = null;
let lastNumFrames = 0;
let lastWindowSize = WINDOW_SAMPLES;

/* ==========================================================================
   3. CUSTOM LAYERS & KERNELS
   ========================================================================== */

/**
 * Custom Mel Spectrogram Layer.
 * Performs STFT and Mel-filterbank conversion within the TFJS graph.
 * Required because the model expects raw audio input.
 */
class MelSpecLayerSimple extends tf.layers.Layer {
  constructor(config) {
    super(config);
    this.sampleRate = config.sampleRate;
    this.specShape = config.specShape;
    this.frameStep = config.frameStep;
    this.frameLength = config.frameLength;
    this.melFilterbank = tf.tensor2d(config.melFilterbank);
  }

  build() {
    this.magScale = this.addWeight(
      'magnitude_scaling', [], 'float32',
      tf.initializers.constant({ value: 1.23 })
    );
    super.build();
  }

  computeOutputShape(inputShape) {
    return [inputShape[0], this.specShape[0], this.specShape[1], 1];
  }

  call(inputs) {
    return tf.tidy(() => {
      const x = inputs[0];
      // Process each item in batch
      return tf.stack(x.split(x.shape[0]).map(input => {
        let spec = input.squeeze();
        // Normalize
        spec = tf.sub(spec, tf.min(spec, -1, true));
        spec = tf.div(spec, tf.max(spec, -1, true).add(1e-6));
        spec = tf.sub(spec, 0.5).mul(2.0);
        
        // STFT (using custom kernel registered below)
        spec = tf.engine().runKernel('STFT', {
          signal: spec,
          frameLength: this.frameLength,
          frameStep: this.frameStep
        });
        
        // Mel Projection
        spec = tf.matMul(spec, this.melFilterbank).pow(2.0);
        spec = spec.pow(tf.div(1.0, tf.add(1.0, tf.exp(this.magScale.read()))));
        
        // Orientation fix
        spec = tf.reverse(spec, -1);
        spec = tf.transpose(spec).expandDims(-1);
        return spec;
      }));
    });
  }

  static get className() { return 'MelSpecLayerSimple'; }
}

// Register STFT Kernel (WebGL implementation for spectrogram generation)
tf.registerKernel({
  kernelName: 'STFT',
  backendName: 'webgl',
  kernelFunc: ({ backend, inputs: { signal, frameLength, frameStep } }) => {
    const innerDim = frameLength / 2;
    const batch = (signal.size - frameLength + frameStep) / frameStep | 0;

    // Stage 1: Windowing & Bit-Reversal
    let currentTensor = backend.runWebGLProgram({
      variableNames: ['x'],
      outputShape: [batch, frameLength],
      userCode: `void main(){
        ivec2 c=getOutputCoords();
        int p=c[1]%${innerDim};
        int k=0;
        for(int i=0;i<${Math.log2(innerDim)};++i){
          if((p & (1<<i))!=0){ k|=(1<<(${Math.log2(innerDim) - 1}-i)); }
        }
        int i=2*k;
        if(c[1]>=${innerDim}){ i=2*(k%${innerDim})+1; }
        int q=c[0]*${frameLength}+i;
        float val=getX((q/${frameLength})*${frameStep}+ q % ${frameLength});
        float cosArg=${2.0 * Math.PI / frameLength}*float(q);
        float mul=0.5-0.5*cos(cosArg);
        setOutput(val*mul);
      }`
    }, [signal], 'float32');

    // Stage 2: FFT Butterflies
    for (let len = 1; len < innerDim; len *= 2) {
      let prevTensor = currentTensor;
      currentTensor = backend.runWebGLProgram({
        variableNames: ['x'],
        outputShape: [batch, innerDim * 2],
        userCode: `void main(){
          ivec2 c=getOutputCoords();
          int b=c[0];
          int i=c[1];
          int k=i%${innerDim};
          int isHigh=(k%${len * 2})/${len};
          int highSign=(1 - isHigh*2);
          int baseIndex=k - isHigh*${len};
          float t=${Math.PI / len}*float(k%${len});
          float a=cos(t);
          float bsin=sin(-t);
          float oddK_re=getX(b, baseIndex+${len});
          float oddK_im=getX(b, baseIndex+${len + innerDim});
          if(i<${innerDim}){
            float evenK_re=getX(b, baseIndex);
            setOutput(evenK_re + (oddK_re*a - oddK_im*bsin)*float(highSign));
          } else {
            float evenK_im=getX(b, baseIndex+${innerDim});
            setOutput(evenK_im + (oddK_re*bsin + oddK_im*a)*float(highSign));
          }
        }`
      }, [prevTensor], 'float32');
      backend.disposeIntermediateTensorInfo(prevTensor);
    }

    // Stage 3: Real RFFT Output
    const real = backend.runWebGLProgram({
      variableNames: ['x'],
      outputShape: [batch, innerDim + 1],
      userCode: `void main(){
        ivec2 c=getOutputCoords();
        int b=c[0];
        int i=c[1];
        int zI=i%${innerDim};
        int conjI=(${innerDim}-i)%${innerDim};
        float Zk0=getX(b,zI);
        float Zk1=getX(b,zI+${innerDim});
        float Zk_conj0=getX(b,conjI);
        float Zk_conj1=-getX(b,conjI+${innerDim});
        float t=${-2.0 * Math.PI}*float(i)/float(${innerDim * 2});
        float diff0=Zk0 - Zk_conj0;
        float diff1=Zk1 - Zk_conj1;
        float result=(Zk0+Zk_conj0 + cos(t)*diff1 + sin(t)*diff0)*0.5;
        setOutput(result);
      }`
    }, [currentTensor], 'float32');
    backend.disposeIntermediateTensorInfo(currentTensor);
    return real;
  }
});

/* ==========================================================================
   4. INITIALIZATION
   ========================================================================== */

// Start initialization immediately
init();

async function init() {
  await tf.setBackend('webgl');
  tf.serialization.registerClass(MelSpecLayerSimple);

  // 1. Load Main Model
  birdModel = await tf.loadLayersModel(MODEL_PATH, {
    onProgress: p => postMessage({ message: 'load_model', progress: (p * 70) | 0 })
  });

  // 2. Warmup (avoids lag on first inference)
  postMessage({ message: 'warmup', progress: 70 });
  tf.tidy(() => {
    birdModel.predict(tf.zeros([1, WINDOW_SAMPLES]));
  });

  // 3. Load Geo Model (Optional)
  postMessage({ message: 'load_geomodel', progress: 90 });
  try {
    areaModel = await tf.loadGraphModel(AREA_MODEL_PATH);
  } catch (e) {
    console.warn("Geo model failed to load", e);
  }

  // 4. Load Labels
  postMessage({ message: 'load_labels', progress: 95 });
  await loadLabels();

  postMessage({ message: 'loaded' });
}

async function loadLabels(langOverride) {
  const navigatorLang = params.get('lang');
  const supportedLanguages = [
    'af', 'da', 'en_us', 'fr', 'ja', 'no', 'ro', 'sl', 'tr', 'ar', 'de', 'es', 'hu',
    'ko', 'pl', 'ru', 'sv', 'uk', 'cs', 'en_uk', 'fi', 'it', 'nl', 'pt', 'sk', 'th', 'zh'
  ];
  
  // Determine language
  const lang = (() => {
    if (langOverride) return langOverride;
    const req = params.get('lang');
    if (req) return req;
    if (!navigatorLang) return 'en_us';
    const base = navigatorLang.split('-')[0];
    return supportedLanguages.find(l => l.startsWith(base)) || 'en_us';
  })();

  // Fetch default (English) and localized lists
  const birdsList = (await fetch(LABELS_DIR + '/en_us.txt').then(r => r.text())).split('\n');
  let birdsListI18n;
  try {
    birdsListI18n = (await fetch(`${LABELS_DIR}/${lang}.txt`).then(r => r.text())).split('\n');
  } catch {
    birdsListI18n = birdsList;
  }

  // Merge into objects
  const newBirds = birdsList.map((base, i) => {
    const i18nLine = birdsListI18n[i] || base;
    const [sciBase, comBase] = base.split('_');
    const [sciLoc, comLoc] = i18nLine.split('_');
    return {
      geoscore: 1, // Default probability (will be updated by geo model)
      scientificName: sciBase || base,
      commonName: comBase || base,
      commonNameI18n: comLoc || comBase || base
    };
  });

  // Preserve geoscores if we already had birds loaded
  if (birds.length === newBirds.length) {
    for (let i = 0; i < birds.length; i++) {
      newBirds[i].geoscore = birds[i].geoscore;
    }
  }

  birds = newBirds;
}

/* ==========================================================================
   5. MESSAGE HANDLING
   ========================================================================== */

onmessage = async ({ data }) => {
  switch (data.message) {
    case 'predict':
      await handlePredict(data);
      break;
    case 'area-scores':
      await handleAreaScores(data);
      break;
    case 'load_labels':
      await loadLabels(data.lang);
      postMessage({ message: 'labels_loaded', lang: data.lang });
      break;
    case 'get_species_list':
      postMessage({ 
        message: 'species_list', 
        list: birds.map((b, i) => ({
          index: i,
          scientificName: b.scientificName,
          commonName: b.commonName,
          commonNameI18n: b.commonNameI18n,
          geoscore: b.geoscore
        }))
      });
      break;
  }
};

/* ==========================================================================
   6. CORE LOGIC: PREDICTION
   ========================================================================== */

async function handlePredict(data) {
  if (!birdModel) return;

  // 1. Prepare Audio Window
  const overlapSecRaw = parseFloat(data.overlapSec ?? 1.5);
  const overlapSec = Math.min(2.5, Math.max(0.0, Math.round(overlapSecRaw * 2) / 2));
  const overlapSamples = Math.round(overlapSec * SAMPLE_RATE);
  const hopSamples = Math.max(1, WINDOW_SAMPLES - overlapSamples);

  const pcm = data.pcmAudio || new Float32Array(0);
  const total = pcm.length;

  // Frame the audio (sliding window)
  const numFrames = Math.max(1, Math.ceil(Math.max(0, total - WINDOW_SAMPLES) / hopSamples) + 1);
  const framed = new Float32Array(numFrames * WINDOW_SAMPLES);
  for (let f = 0; f < numFrames; f++) {
    const start = f * hopSamples;
    const srcEnd = Math.min(start + WINDOW_SAMPLES, total);
    framed.set(pcm.subarray(start, srcEnd), f * WINDOW_SAMPLES);
  }

  // 2. Run Inference
  const audioTensor = tf.tensor2d(framed, [numFrames, WINDOW_SAMPLES]);
  const resTensor = birdModel.predict(audioTensor);
  let predictionList = await resTensor.array(); // [numFrames, numClasses]
  
  resTensor.dispose(); 
  audioTensor.dispose();

  // 3. Apply Sensitivity
  const sensitivity = parseFloat(data.sensitivity || 1.0);
  if (sensitivity !== 1.0) {
    predictionList = applySensitivity(predictionList, sensitivity);
  }

  // 4. Cache results (for geo updates)
  lastPredictionList = predictionList;
  lastHopSamples = hopSamples;
  lastNumFrames = numFrames;
  lastWindowSize = WINDOW_SAMPLES;

  // 5. Emit Segment Results
  emitSegments(predictionList, hopSamples, WINDOW_SAMPLES);

  // 6. Pool Results (Log-Mean-Exp) & Emit
  emitPooled(predictionList);
}

function emitSegments(predictionList, hopSamples, windowSize) {
  const segments = [];
  for (let f = 0; f < predictionList.length; f++) {
    const startSec = (f * hopSamples) / SAMPLE_RATE;
    const endSec = startSec + windowSize / SAMPLE_RATE;
    const preds = predictionList[f].map((conf, i) => ({
      index: i,
      confidence: conf,
      geoscore: birds[i].geoscore,
      scientificName: birds[i].scientificName,
      commonName: birds[i].commonName,
      commonNameI18n: birds[i].commonNameI18n
    }));
    segments.push({ start: startSec, end: endSec, preds });
  }
  postMessage({ message: 'segments', segments });
}

function emitPooled(predictionList) {
  const numClasses = predictionList[0]?.length || 0;
  const numFrames = predictionList.length;
  const ALPHA = 5.0; // Pooling factor
  
  const sumsExp = new Float64Array(numClasses);
  for (let f = 0; f < numFrames; f++) {
    const row = predictionList[f];
    for (let i = 0; i < numClasses; i++) {
      sumsExp[i] += Math.exp(ALPHA * row[i]);
    }
  }
  
  lastMeans = Array.from(sumsExp, s => Math.log(s / numFrames) / ALPHA);

  const pooled = lastMeans.map((m, i) => ({
    index: i,
    scientificName: birds[i].scientificName,
    commonName: birds[i].commonName,
    commonNameI18n: birds[i].commonNameI18n,
    confidence: m,
    geoscore: birds[i].geoscore
  }));
  
  postMessage({ message: 'pooled', pooled });
}

/**
 * Adjusts logits based on sensitivity slider.
 * Sensitivity > 1.0 boosts weak signals.
 */
function applySensitivity(list, sensitivity) {
  const bias = (sensitivity - 1.0) * 5.0; 
  return list.map(row => row.map(p => {
    const pp = Math.max(1e-7, Math.min(1 - 1e-7, p));
    const logit = Math.log(pp / (1 - pp));
    return 1 / (1 + Math.exp(-(logit + bias)));
  }));
}

/* ==========================================================================
   7. CORE LOGIC: GEOLOCATION
   ========================================================================== */

async function handleAreaScores(data) {
  if (!areaModel) return;

  // Calculate week of year
  tf.engine().startScope();
  const startOfYear = new Date(new Date().getFullYear(), 0, 1);
  startOfYear.setDate(startOfYear.getDate() + (1 - (startOfYear.getDay() % 7)));
  const week = Math.round((Date.now() - startOfYear.getTime()) / 604800000) + 1;

  // Predict occurrence probabilities
  const input = tf.tensor([[data.latitude, data.longitude, week]]);
  const areaScores = await areaModel.predict(input).data();
  tf.engine().endScope();

  // Update global state
  for (let i = 0; i < birds.length; i++) {
    birds[i].geoscore = areaScores[i];
  }
  
  postMessage({ message: 'area-scores' });

  // Re-emit cached results with new geo scores
  if (lastPredictionList && lastHopSamples != null) {
    emitSegments(lastPredictionList, lastHopSamples, lastWindowSize);
  }
  if (lastMeans) {
    const pooled = lastMeans.map((m, i) => ({
      index: i,
      scientificName: birds[i].scientificName,
      commonName: birds[i].commonName,
      commonNameI18n: birds[i].commonNameI18n,
      confidence: m,
      geoscore: birds[i].geoscore
    }));
    postMessage({ message: 'pooled', pooled });
  }
}