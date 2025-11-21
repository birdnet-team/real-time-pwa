const params = new URL(self.location.href).searchParams;
const TF_PATH = params.get('tf') || '/js/tfjs-4.14.0.min.js';
importScripts(TF_PATH);

const ROOT = params.get('root') || '/models';
const REQ_LANG = params.get('lang');

const BIRD_BASE = ROOT + '/birdnet';
const MODEL_PATH = BIRD_BASE + '/model.json';
const AREA_MODEL_PATH = BIRD_BASE + '/area-model/model.json';
const LABELS_DIR = BIRD_BASE + '/labels';

main();

async function main() {
  const navigatorLang = new URL(location.href).searchParams.get('lang');
  await tf.setBackend('webgl');

  // Custom layer (only used if referenced in model JSON)
  class MelSpecLayerSimple extends tf.layers.Layer {
    constructor(config){
      super(config);
      this.sampleRate   = config.sampleRate;
      this.specShape    = config.specShape;
      this.frameStep    = config.frameStep;
      this.frameLength  = config.frameLength;
      this.melFilterbank = tf.tensor2d(config.melFilterbank);
    }
    build(){
      this.magScale = this.addWeight(
        'magnitude_scaling', [], 'float32',
        tf.initializers.constant({ value: 1.23 })
      );
      super.build();
    }
    computeOutputShape(inputShape){
      return [inputShape[0], this.specShape[0], this.specShape[1], 1];
    }
    call(inputs){
      return tf.tidy(() => {
        const x = inputs[0];
        return tf.stack(x.split(x.shape[0]).map(input => {
          let spec = input.squeeze();
          spec = tf.sub(spec, tf.min(spec, -1, true));
            spec = tf.div(spec, tf.max(spec, -1, true).add(1e-6));
          spec = tf.sub(spec, 0.5).mul(2.0);
          spec = tf.engine().runKernel('STFT', {
            signal: spec,
            frameLength: this.frameLength,
            frameStep: this.frameStep
          });
          spec = tf.matMul(spec, this.melFilterbank).pow(2.0);
          spec = spec.pow(tf.div(1.0, tf.add(1.0, tf.exp(this.magScale.read()))));
          spec = tf.reverse(spec, -1);
          spec = tf.transpose(spec).expandDims(-1);
          return spec;
        }));
      });
    }
    static get className(){ return 'MelSpecLayerSimple'; }
  }
  tf.serialization.registerClass(MelSpecLayerSimple);

  // Load main model
  const birdModel = await tf.loadLayersModel(MODEL_PATH, {
    onProgress: p => postMessage({ message:'load_model', progress: (p * 70) | 0 })
  });

  // Warmup (shape [1,144000])
  postMessage({ message:'warmup', progress:70 });
  birdModel.predict(tf.zeros([1,144000])).dispose();

  // Optional geo model
  postMessage({ message:'load_geomodel', progress:90 });
  let areaModel = null;
  try { areaModel = await tf.loadGraphModel(AREA_MODEL_PATH); } catch {}

  // Labels
  postMessage({ message:'load_labels', progress:95 });
  const supportedLanguages = [
    'af','da','en_us','fr','ja','no','ro','sl','tr','ar','de','es','hu',
    'ko','pl','ru','sv','uk','cs','en_uk','fi','it','nl','pt','sk','th','zh'
  ];
  const lang = (() => {
    if (REQ_LANG) return REQ_LANG;
    if (!navigatorLang) return 'en_us';
    const base = navigatorLang.split('-')[0];
    return supportedLanguages.find(l => l.startsWith(base)) || 'en_us';
  })();

  const birdsList = (await fetch(LABELS_DIR + '/en_us.txt').then(r => r.text())).split('\n');
  let birdsListI18n;
  try { birdsListI18n = (await fetch(`${LABELS_DIR}/${lang}.txt`).then(r => r.text())).split('\n'); }
  catch { birdsListI18n = birdsList; }

  const birds = birdsList.map((base, i) => {
    const loc = birdsListI18n[i] || base;
    return {
      geoscore: 1,
      name: base.split('_')[1] || base,
      nameI18n: loc.split('_')[1] || base
    };
  });

  // Caches for geo re-emit
  let lastMeans = null;
  let lastPredictionList = null;
  let lastHopSamples = null;
  let lastNumFrames = 0;
  let lastWindowSize = 144000;

  postMessage({ message:'loaded' });

  onmessage = async ({ data }) => {
    if (data.message === 'predict') {
      const SAMPLE_RATE = 48000;
      const windowSize = 144000; // 3s
      const overlapSecRaw = parseFloat(data.overlapSec ?? 1.5);
      const overlapSec = Math.min(2.5, Math.max(0.0, Math.round(overlapSecRaw * 2) / 2));
      const overlapSamples = Math.round(overlapSec * SAMPLE_RATE);
      const hopSamples = Math.max(1, windowSize - overlapSamples);

      const pcm = data.pcmAudio || new Float32Array(0);
      const total = pcm.length;

      // Number of frames (ceil to pad tail)
      const numFrames = Math.max(1, Math.ceil(Math.max(0, total - windowSize) / hopSamples) + 1);
      const framed = new Float32Array(numFrames * windowSize);
      for (let f = 0; f < numFrames; f++) {
        const start = f * hopSamples;
        const srcEnd = Math.min(start + windowSize, total);
        framed.set(pcm.subarray(start, srcEnd), f * windowSize);
      }

      const audioTensor = tf.tensor2d(framed, [numFrames, windowSize]);
      const resTensor = birdModel.predict(audioTensor);
      const predictionList = await resTensor.array(); // shape [numFrames, numClasses]
      resTensor.dispose(); audioTensor.dispose();

      // Cache for geo updates
      lastPredictionList = predictionList;
      lastHopSamples = hopSamples;
      lastNumFrames = numFrames;
      lastWindowSize = windowSize;

      // Debug (first 3 frames)
      try {
        const dbg = predictionList.slice(0, Math.min(3, predictionList.length)).map((arr, b) => {
          const top = arr.map((v,i)=>({i,v}))
                         .sort((a,b)=>b.v-a.v)
                         .slice(0,10)
                         .map(({i,v}) => ({
                           index:i,
                           name: birds[i].nameI18n || birds[i].name,
                           confidence:v
                         }));
          return {
            frame: b,
            max: Math.max(...arr),
            mean: arr.reduce((a,c)=>a+c,0)/arr.length,
            top10: top
          };
        });
        postMessage({ message:'predict_debug', top10PerBatch: dbg });
      } catch {}

      // Segment-wise emission
      const segments = [];
      for (let f = 0; f < numFrames; f++) {
        const startSec = (f * hopSamples) / SAMPLE_RATE;
        const endSec = startSec + windowSize / SAMPLE_RATE;
        const preds = predictionList[f].map((conf,i)=> ({
          index: i,
          confidence: conf,
          geoscore: birds[i].geoscore,
          name: birds[i].name,
          nameI18n: birds[i].nameI18n
        }));
        segments.push({ start: startSec, end: endSec, preds });
      }
      postMessage({ message:'segments', segments });

      // Mean exponential pooling (log-mean-exp)
      const numClasses = predictionList[0]?.length || 0;
      const ALPHA = 5.0;
      const sumsExp = new Float64Array(numClasses);
      for (let f = 0; f < numFrames; f++) {
        const row = predictionList[f];
        for (let i = 0; i < numClasses; i++) sumsExp[i] += Math.exp(ALPHA * row[i]);
      }
      lastMeans = Array.from(sumsExp, s => Math.log(s / numFrames) / ALPHA);

      const pooled = lastMeans.map((m, i) => ({
        index: i,
        name: birds[i].name,
        nameI18n: birds[i].nameI18n,
        confidence: m,
        geoscore: birds[i].geoscore
      }));
      postMessage({ message:'pooled', pooled });
    }

    if (data.message === 'area-scores' && areaModel) {
      // Week-of-year input (approx; aligns first week to Monday)
      tf.engine().startScope();
      const startOfYear = new Date(new Date().getFullYear(),0,1);
      startOfYear.setDate(startOfYear.getDate() + (1 - (startOfYear.getDay() % 7)));
      const week = Math.round((Date.now() - startOfYear.getTime()) / 604800000) + 1;
      const input = tf.tensor([[data.latitude, data.longitude, week]]);
      const areaScores = await areaModel.predict(input).data();
      tf.engine().endScope();

      for (let i=0;i<birds.length;i++) birds[i].geoscore = areaScores[i];
      postMessage({ message:'area-scores' });

      // Re-emit segments with updated geo scores
      if (lastPredictionList && lastHopSamples != null) {
        const segments = [];
        for (let f=0; f<lastNumFrames; f++) {
          const startSec = (f * lastHopSamples) / 48000;
          const endSec = startSec + lastWindowSize / 48000;
          const preds = lastPredictionList[f].map((conf,i)=> ({
            index: i,
            confidence: conf,
            geoscore: birds[i].geoscore,
            name: birds[i].name,
            nameI18n: birds[i].nameI18n
          }));
          segments.push({ start: startSec, end: endSec, preds });
        }
        postMessage({ message:'segments', segments });
      }

      // Re-emit pooled with updated geo scores
      if (lastMeans) {
        const pooled = lastMeans.map((m, i) => ({
          index: i,
          name: birds[i].name,
          nameI18n: birds[i].nameI18n,
          confidence: m,
          geoscore: birds[i].geoscore
        }));
        postMessage({ message:'pooled', pooled });
      }
    }
  };
}

/* STFT kernel */
tf.registerKernel({
  kernelName:'STFT',
  backendName:'webgl',
  kernelFunc:({ backend, inputs:{ signal, frameLength, frameStep } })=>{
    const innerDim = frameLength / 2;
    const batch = (signal.size - frameLength + frameStep) / frameStep | 0;

    // Stage 1: window + bit-reverse
    let currentTensor = backend.runWebGLProgram({
      variableNames:['x'],
      outputShape:[batch, frameLength],
      userCode:`void main(){
        ivec2 c=getOutputCoords();
        int p=c[1]%${innerDim};
        int k=0;
        for(int i=0;i<${Math.log2(innerDim)};++i){
          if((p & (1<<i))!=0){ k|=(1<<(${Math.log2(innerDim)-1}-i)); }
        }
        int i=2*k;
        if(c[1]>=${innerDim}){ i=2*(k%${innerDim})+1; }
        int q=c[0]*${frameLength}+i;
        float val=getX((q/${frameLength})*${frameStep}+ q % ${frameLength});
        float cosArg=${2.0*Math.PI/frameLength}*float(q);
        float mul=0.5-0.5*cos(cosArg);
        setOutput(val*mul);
      }`
    },[signal],'float32');

    // Stage 2: FFT butterflies
    for(let len=1; len<innerDim; len*=2){
      let prevTensor = currentTensor;
      currentTensor = backend.runWebGLProgram({
        variableNames:['x'],
        outputShape:[batch, innerDim*2],
        userCode:`void main(){
          ivec2 c=getOutputCoords();
          int b=c[0];
          int i=c[1];
          int k=i%${innerDim};
          int isHigh=(k%${len*2})/${len};
          int highSign=(1 - isHigh*2);
          int baseIndex=k - isHigh*${len};
          float t=${Math.PI/len}*float(k%${len});
          float a=cos(t);
          float bsin=sin(-t);
          float oddK_re=getX(b, baseIndex+${len});
          float oddK_im=getX(b, baseIndex+${len+innerDim});
          if(i<${innerDim}){
            float evenK_re=getX(b, baseIndex);
            setOutput(evenK_re + (oddK_re*a - oddK_im*bsin)*float(highSign));
          } else {
            float evenK_im=getX(b, baseIndex+${innerDim});
            setOutput(evenK_im + (oddK_re*bsin + oddK_im*a)*float(highSign));
          }
        }`
      },[prevTensor],'float32');
      backend.disposeIntermediateTensorInfo(prevTensor);
    }

    // Stage 3: real RFFT output
    const real = backend.runWebGLProgram({
      variableNames:['x'],
      outputShape:[batch, innerDim+1],
      userCode:`void main(){
        ivec2 c=getOutputCoords();
        int b=c[0];
        int i=c[1];
        int zI=i%${innerDim};
        int conjI=(${innerDim}-i)%${innerDim};
        float Zk0=getX(b,zI);
        float Zk1=getX(b,zI+${innerDim});
        float Zk_conj0=getX(b,conjI);
        float Zk_conj1=-getX(b,conjI+${innerDim});
        float t=${-2.0*Math.PI}*float(i)/float(${innerDim*2});
        float diff0=Zk0 - Zk_conj0;
        float diff1=Zk1 - Zk_conj1;
        float result=(Zk0+Zk_conj0 + cos(t)*diff1 + sin(t)*diff0)*0.5;
        setOutput(result);
      }`
    },[currentTensor],'float32');
    backend.disposeIntermediateTensorInfo(currentTensor);
    return real;
  }
});