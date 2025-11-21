// BirdNET Live main client script
// NOTE: Refactor only; logic preserved. Added comments, grouped sections, minor style consistency.

/* -------------------------------------------------
 * Global state
 * ------------------------------------------------- */
let isListening = false;
let workerReady = false;
let birdnetWorker = null;

let audioContext;
let scriptNode;
let circularBuffer;
let circularWriteIndex = 0;
let currentStream;

const SAMPLE_RATE = 48000;
const WINDOW_SECONDS = 3;
const WINDOW_SAMPLES = SAMPLE_RATE * WINDOW_SECONDS;
const INFERENCE_INTERVAL_MS = 1000;

let geolocation = null;
let geoWatchId = null;
let geoEnabled = true;

let detectionThreshold = 0.25;
let latestDetections = [];
let inputGain = 1.0;

/* -------------------------------------------------
 * DOM accessors (lazy lookups)
 * ------------------------------------------------- */
const statusEl             = () => document.getElementById("statusText");
const recordButtonEl       = () => document.getElementById("recordButton");
const recordLabelTextEl    = () => document.querySelector(".record-label-text");
const detectionsList       = () => document.getElementById("detectionsList");
const geoStatusEl          = () => document.getElementById("geoStatusText");
const geoCoordsEl          = () => document.getElementById("geoCoordsText");
const settingsToggleEl     = () => document.getElementById("settingsToggle");
const settingsDrawerEl     = () => document.getElementById("settingsDrawer");
const settingsOverlayEl    = () => document.getElementById("settingsOverlay");
const statusDebugEl        = () => document.getElementById("statusDebugInfo"); // optional (may not exist)

/* -------------------------------------------------
 * Spectrogram config
 * ------------------------------------------------- */
let spectroCanvas, spectroCtx;
let spectroAnimationId = null;
let analyser;
let dataArray;
let bufferLength;

const SPECTRO_FFT_SIZE = 512;
const SPECTRO_DEFAULT_DURATION_SEC = 20;
const SPECTRO_DEFAULT_ZOOM = 0.75;
const SPECTRO_DEFAULT_GAIN = 1.0;
const SPECTRO_OUTPUT_GAMMA = 0.8;

let spectroDurationSec = SPECTRO_DEFAULT_DURATION_SEC;
let spectroZoom = SPECTRO_DEFAULT_ZOOM;
let spectroGain = SPECTRO_DEFAULT_GAIN;
let spectroColumnSeconds = 0;
let lastSpectroColumnTime = 0;

/* -------------------------------------------------
 * Boot
 * ------------------------------------------------- */
document.addEventListener("DOMContentLoaded", () => {
  initWorker();
  setupRecordButton();
  initUIControls();
  setupSettingsToggle();
  if (geoEnabled) {
    getGeolocation();
  } else {
    updateGeoDisplay("Geolocation disabled.", null);
  }
});

/* -------------------------------------------------
 * Worker setup
 * ------------------------------------------------- */
function initWorker() {
  const tfPath = "/js/tfjs-4.14.0.min.js";
  const root   = "/models";
  const lang   = navigator.language || "en-US";

  const params = new URLSearchParams({ tf: tfPath, root, lang });
  birdnetWorker = new Worker(`/js/birdnet-worker.js?${params.toString()}`);

  birdnetWorker.onmessage = (event) => {
    const data = event.data || {};
    switch (data.message) {
      case "load_model":
      case "warmup":
      case "load_geomodel":
      case "load_labels":
        if (typeof data.progress === "number") {
          statusEl().textContent = `Loading BirdNET… ${data.progress}%`;
        }
        break;
      case "loaded":
        workerReady = true;
        statusEl().textContent = "Model ready. Tap Listen to start.";
        if (geolocation) sendAreaScores();
        break;
      case "predict_debug":
        // Debug info optional
        break;
      case "segments":
        // Per-segment data (not visualized yet)
        break;
      case "pooled":
        renderDetections(data.pooled);
        break;
      case "area-scores":
        // Geo priors updated; worker handles refresh
        break;
      default:
        break;
    }
  };

  birdnetWorker.onerror = (err) => {
    console.error("Worker error", err);
    statusEl().textContent = "BirdNET worker error.";
  };
}

/* -------------------------------------------------
 * UI controls & ranges
 * ------------------------------------------------- */
function initUIControls() {
  const geoToggle = document.getElementById("geoToggle");
  if (geoToggle) {
    geoToggle.checked = geoEnabled;
    geoToggle.addEventListener("change", () => {
      geoEnabled = geoToggle.checked;
      if (geoEnabled) {
        updateGeoDisplay("Requesting geolocation…", null);
        getGeolocation();
      } else {
        geolocation = null;
        updateGeoDisplay("Geolocation disabled.", null);
        renderDetections();
      }
    });
  }

  bindRange("durationRange", spectroDurationSec, (v) => {
    spectroDurationSec = v;
    if (spectroCanvas && spectroCanvas.width > 0) {
      spectroColumnSeconds = spectroDurationSec / spectroCanvas.width;
    }
  }, (v) => `${v}s`);

  bindRange("zoomRange", spectroZoom, (v) => {
    spectroZoom = v;
  }, (v) => `${Math.round(v * 100)}%`);

  bindRange("gainRange", spectroGain, (v) => {
    spectroGain = v;
  }, (v) => `${v.toFixed(1)}×`);

  bindRange("thresholdRange", detectionThreshold * 100, (v) => {
    detectionThreshold = v / 100;
    renderDetections();
  }, (v) => `${Math.round(v)}%`);

  bindRange("inputGainRange", inputGain, (v) => {
    inputGain = v;
  }, (v) => `${v.toFixed(1)}×`);
}

function bindRange(id, initialValue, onChange, format) {
  const input = document.getElementById(id);
  const label = document.querySelector(`[id='${id.replace("Range", "Value")}']`);
  if (!input) return;
  if (typeof initialValue === "number") {
    input.value = initialValue;
  }
  const setLabel = (val) => {
    if (label) label.textContent = format ? format(val) : val;
  };
  setLabel(parseFloat(input.value));
  input.addEventListener("input", () => {
    const val = parseFloat(input.value);
    onChange(val, input);
    setLabel(val);
  });
}

/* -------------------------------------------------
 * Spectrogram init
 * ------------------------------------------------- */
function initSpectrogramCanvas() {
  if (spectroCanvas) return;
  spectroCanvas = document.getElementById("liveSpectrogram");
  if (!spectroCanvas) {
    console.warn("[Spectrogram] canvas not found");
    return;
  }
  resizeSpectrogramCanvas();
  window.addEventListener("resize", resizeSpectrogramCanvas);
}

function resizeSpectrogramCanvas() {
  if (!spectroCanvas) return;
  const cssW = spectroCanvas.clientWidth || 600;
  const cssH = spectroCanvas.clientHeight || 220;

  let snapshot = null;
  if (spectroCtx) {
    try {
      snapshot = spectroCtx.getImageData(0, 0, spectroCanvas.width, spectroCanvas.height);
    } catch (_) {}
  }

  spectroCanvas.width = cssW;
  spectroCanvas.height = cssH;

  spectroCtx = spectroCanvas.getContext("2d");
  spectroCtx.fillStyle = "#000";
  spectroCtx.fillRect(0, 0, cssW, cssH);

  if (snapshot) spectroCtx.putImageData(snapshot, 0, 0);

  spectroColumnSeconds = cssW > 0 ? spectroDurationSec / cssW : 0.05;
  lastSpectroColumnTime = audioContext ? audioContext.currentTime : 0;
}

function startSpectrogram(source) {
  initSpectrogramCanvas();
  if (!spectroCanvas) return;

  analyser = audioContext.createAnalyser();
  analyser.fftSize = SPECTRO_FFT_SIZE;
  analyser.smoothingTimeConstant = 0.6;
  source.connect(analyser);

  bufferLength = analyser.frequencyBinCount;
  dataArray = new Uint8Array(bufferLength);

  lastSpectroColumnTime = audioContext.currentTime;
  if (!spectroColumnSeconds) {
    const w = spectroCanvas.width || 600;
    spectroColumnSeconds = spectroDurationSec / w;
  }
  if (!spectroAnimationId) {
    spectroAnimationId = requestAnimationFrame(drawSpectrogram);
  }
}

function stopSpectrogram() {
  if (spectroAnimationId) {
    cancelAnimationFrame(spectroAnimationId);
    spectroAnimationId = null;
  }
  if (analyser) {
    try { analyser.disconnect(); } catch (_) {}
    analyser = null;
  }
}

function drawSpectrogram() {
  spectroAnimationId = requestAnimationFrame(drawSpectrogram);
  if (!analyser || !audioContext) return;

  analyser.getByteFrequencyData(dataArray);
  const w = spectroCanvas.width;
  const h = spectroCanvas.height;
  if (!w || !h) return;

  if (!spectroColumnSeconds) {
    spectroColumnSeconds = spectroDurationSec / Math.max(1, w);
  }

  const audioNow = audioContext.currentTime;
  let columnsNeeded = Math.floor((audioNow - lastSpectroColumnTime) / spectroColumnSeconds);
  if (columnsNeeded <= 0) return;
  columnsNeeded = Math.min(columnsNeeded, w - 1);
  lastSpectroColumnTime += columnsNeeded * spectroColumnSeconds;

  // Shift left
  spectroCtx.drawImage(
    spectroCanvas,
    columnsNeeded, 0, w - columnsNeeded, h,
    0, 0, w - columnsNeeded, h
  );

  const displayLen = Math.floor(bufferLength * spectroZoom);
  const startBin = 0;
  const endBin = Math.max(startBin + 1, displayLen);
  const barHeight = h / (endBin - startBin);

  for (let c = 0; c < columnsNeeded; c++) {
    const x = w - columnsNeeded + c;
    for (let i = startBin; i < endBin; i++) {
      const magnitude = Math.max(1e-6, dataArray[i] / 255);
      let norm = Math.log1p(magnitude * spectroGain) / Math.log1p(1 + spectroGain);
      norm = Math.pow(Math.min(1, Math.max(0, norm)), SPECTRO_OUTPUT_GAMMA);
      const y = h - ((i - startBin) / (endBin - startBin)) * h;
      spectroCtx.fillStyle = d3.interpolateMagma(norm);
      spectroCtx.fillRect(x, y - barHeight, 1, barHeight);
    }
  }
}

/* -------------------------------------------------
 * Record button wiring
 * ------------------------------------------------- */
function setupRecordButton() {
  const btn = recordButtonEl();
  if (!btn) return;
  btn.addEventListener("click", async () => {
    if (!isListening) {
      await startListening();
    } else {
      stopListening();
    }
  });
}

/* -------------------------------------------------
 * Audio graph (mic stream + buffer + inference)
 * ------------------------------------------------- */
async function startListening() {
  if (!workerReady) {
    statusEl().textContent = "BirdNET model is still loading…";
    return;
  }
  try {
    isListening = true;
    const button = recordButtonEl();
    if (button) button.classList.add("recording");
    const label = recordLabelTextEl();
    if (label) label.textContent = "Stop";
    statusEl().textContent = "Requesting microphone access…";

    currentStream = await navigator.mediaDevices.getUserMedia({
      audio: {
        channelCount: 1,
        sampleRate: SAMPLE_RATE,
        echoCancellation: false,
        noiseSuppression: false,
        autoGainControl: false
      }
    });

    setupAudioGraphFromStream(currentStream);
    statusEl().textContent = "Listening…";
  } catch (e) {
    console.error(e);
    statusEl().textContent = "Microphone access failed.";
    isListening = false;
    const button = recordButtonEl();
    if (button) button.classList.remove("recording");
    const label = recordLabelTextEl();
    if (label) label.textContent = "Start";
  }
}

function stopListening() {
  isListening = false;
  const button = recordButtonEl();
  if (button) button.classList.remove("recording");
  const label = recordLabelTextEl();
  if (label) label.textContent = "Start";
  statusEl().textContent = "Stopped. Tap Listen to start.";

  if (currentStream) {
    currentStream.getTracks().forEach(t => t.stop());
    currentStream = null;
  }
  if (scriptNode) {
    scriptNode.disconnect();
    scriptNode.onaudioprocess = null;
    scriptNode = null;
  }
  if (audioContext) {
    stopSpectrogram();
    audioContext.close();
    audioContext = null;
  }
}

function setupAudioGraphFromStream(stream) {
  audioContext = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: SAMPLE_RATE });
  const source = audioContext.createMediaStreamSource(stream);
  startSpectrogram(source);

  circularBuffer = new Float32Array(WINDOW_SAMPLES);
  circularWriteIndex = 0;

  const scriptBufferSize = 2048;
  scriptNode = audioContext.createScriptProcessor(scriptBufferSize, 1, 1);
  source.connect(scriptNode);
  scriptNode.connect(audioContext.destination);

  scriptNode.onaudioprocess = (ev) => {
    const input = ev.inputBuffer.getChannelData(0);
    for (let i = 0; i < input.length; i++) {
      circularBuffer[circularWriteIndex] = input[i];
      circularWriteIndex = (circularWriteIndex + 1) % circularBuffer.length;
    }
  };

  startInferenceLoop();
}

/* -------------------------------------------------
 * Inference loop
 * ------------------------------------------------- */
function startInferenceLoop() {
  const tick = () => {
    if (!isListening || !workerReady || !circularBuffer || !birdnetWorker) return;
    const windowed = getCurrentWindow();
    if (windowed) {
      const geoCtx = geolocation ? {
        latitude: geolocation.lat,
        longitude: geolocation.lon
      } : {};
      birdnetWorker.postMessage(
        { message: "predict", pcmAudio: windowed, overlapSec: 1.5, ...geoCtx },
        [windowed.buffer]
      );
    }
    if (isListening) setTimeout(tick, INFERENCE_INTERVAL_MS);
  };
  tick();
}

function getCurrentWindow() {
  if (!circularBuffer) return null;
  const result = new Float32Array(WINDOW_SAMPLES);
  let idx = circularWriteIndex;
  for (let i = 0; i < WINDOW_SAMPLES; i++) {
    result[i] = Math.max(-1, Math.min(1, circularBuffer[idx] * inputGain));
    idx = (idx + 1) % circularBuffer.length;
  }
  return result;
}

/* -------------------------------------------------
 * Detections rendering
 * ------------------------------------------------- */
function renderDetections(pooled) {
  if (Array.isArray(pooled)) latestDetections = pooled;

  const container = detectionsList();
  if (!container) return;

  const useGeoFilter = geoEnabled && !!geolocation;
  const all = latestDetections || [];

  // Binary geo filter: if geo is active and we have a fix,
  // drop any species with geoscore < 0.05 (5%) or missing.
  const afterGeo = useGeoFilter
    ? all.filter(p => typeof p.geoscore === "number" && p.geoscore >= 0.05)
    : all;

  // Apply audio confidence threshold after geo filtering.
  const afterAudio = afterGeo.filter(p => p.confidence >= detectionThreshold);

  // Sort and cap.
  const top = afterAudio
    .sort((a, b) => b.confidence - a.confidence)
    .slice(0, 20);

  container.innerHTML = "";

  if (!top.length) {
    container.innerHTML = `
      <div class="text-muted small">
        No detections above ${Math.round(detectionThreshold * 100)}% confidence yet.
        ${useGeoFilter ? "Geo filter may be hiding low geo-score species." : "Keep listening or lower the threshold."}
      </div>
    `;
    return;
  }

  top.forEach(p => {
    const confPct = (p.confidence * 100).toFixed(1);
    const geoInfo = useGeoFilter && typeof p.geoscore === "number"
      ? ` · Geo score: ${(p.geoscore * 100).toFixed(1)}%`
      : "";
    const name = p.nameI18n || p.name || `Class ${p.index}`;

    const card = document.createElement("div");
    card.className = "card";
    card.innerHTML = `
      <div class="card-body py-2 px-3">
        <div class="d-flex justify-content-between align-items-center">
          <div>
            <div class="fw-semibold small">${name}</div>
            <div class="small text-muted">
              Confidence: ${confPct}%${geoInfo}
            </div>
            ${
              geolocation
                ? `<div class="small text-muted">
                     ${geolocation.lat.toFixed(3)}, ${geolocation.lon.toFixed(3)}
                   </div>`
                : ""
            }
          </div>
        </div>
      </div>
    `;
    container.appendChild(card);
  });
}

/* -------------------------------------------------
 * Settings (drawer) toggle
 * ------------------------------------------------- */
function setupSettingsToggle() {
  const toggle = settingsToggleEl();
  const drawer = settingsDrawerEl();
  const overlay = settingsOverlayEl();
  const closeBtn = document.getElementById("settingsClose");
  if (!toggle || !drawer) return;

  const setState = (open) => {
    toggle.setAttribute("aria-expanded", open ? "true" : "false");
    drawer.setAttribute("aria-hidden", open ? "false" : "true");
    drawer.classList.toggle("open", open);
    if (overlay) {
      overlay.classList.toggle("active", open);
      overlay.setAttribute("aria-hidden", open ? "false" : "true");
    }
  };

  setState(false);

  toggle.addEventListener("click", () => {
    const open = !drawer.classList.contains("open");
    setState(open);
  });

  if (overlay) overlay.addEventListener("click", () => setState(false));
  if (closeBtn) closeBtn.addEventListener("click", () => setState(false));
  document.addEventListener("keydown", (evt) => {
    if (evt.key === "Escape") setState(false);
  });
}

/* -------------------------------------------------
 * Geolocation
 * ------------------------------------------------- */
function updateGeoDisplay(message, coords) {
  const status = geoStatusEl();
  const coordsEl = geoCoordsEl();
  if (status) status.textContent = message;
  if (coordsEl) {
    coordsEl.textContent = coords
      ? `${coords.lat.toFixed(4)}, ${coords.lon.toFixed(4)} (±${Math.round(coords.accuracy)}m)`
      : "—";
  }
}

function getGeolocation() {
  if (!navigator.geolocation) {
    updateGeoDisplay("Geolocation not supported.", null);
    return;
  }
  if (geoWatchId !== null) {
    navigator.geolocation.clearWatch(geoWatchId);
    geoWatchId = null;
  }
  updateGeoDisplay("Requesting geolocation…", null);

  geoWatchId = navigator.geolocation.watchPosition(
    (pos) => {
      geolocation = {
        lat: pos.coords.latitude,
        lon: pos.coords.longitude,
        accuracy: pos.coords.accuracy,
        timestamp: pos.timestamp
      };
      updateGeoDisplay("Location acquired.", geolocation);
      sendAreaScores();
      renderDetections(); // re-filter with geo prior active
    },
    (err) => {
      console.warn("Geolocation error", err);
      geolocation = null;
      updateGeoDisplay("Geolocation failed.", null);
      renderDetections();
    },
    {
      enableHighAccuracy: true,
      maximumAge: 15000,
      timeout: 20000
    }
  );
}

function sendAreaScores() {
  if (!birdnetWorker || !geolocation) return;
  const now = new Date();
  const startYear = new Date(now.getFullYear(), 0, 1);
  const week = Math.min(
    52,
    Math.max(
      1,
      Math.floor((now - startYear) / (7 * 24 * 60 * 60 * 1000)) + 1
    )
  );
  const hour = now.getHours(); // local hour
  birdnetWorker.postMessage({
    message: "area-scores",
    latitude: geolocation.lat,    // renamed
    longitude: geolocation.lon,   // renamed
    week,
    hour
  });
}
