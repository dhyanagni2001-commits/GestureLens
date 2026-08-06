import { startCamera, stopCamera } from "./camera.js";
import { loadModel, predictFrame, isModelLoaded } from "./inference.js";
import { updatePredictionUI, drawOverlay } from "./overlay.js";

const videoEl = document.getElementById("webcam");
const canvasEl = document.getElementById("overlay");
const startBtn = document.getElementById("start-btn");
const stopBtn = document.getElementById("stop-btn");
const statusEl = document.getElementById("status");
const labelEl = document.getElementById("pred-label");
const confidenceEl = document.getElementById("pred-confidence");

const INFERENCE_INTERVAL_MS = 200; // ~5 fps inference; [TODO] tune for perf/UX balance

let stream = null;
let inferenceTimer = null;

function setStatus(text) {
  statusEl.textContent = text;
}

async function handleStart() {
  startBtn.disabled = true;
  try {
    if (!isModelLoaded()) {
      await loadModel(setStatus);
    }

    setStatus("Requesting camera…");
    stream = await startCamera(videoEl);
    setStatus("Running");

    stopBtn.disabled = false;

    inferenceTimer = setInterval(() => {
      if (videoEl.readyState < 2) return; // not enough data yet
      const result = predictFrame(videoEl);
      updatePredictionUI(labelEl, confidenceEl, result);
      drawOverlay(canvasEl, videoEl, result);
    }, INFERENCE_INTERVAL_MS);
  } catch (err) {
    console.error(err);
    setStatus(`Error: ${err.message}`);
    startBtn.disabled = false;
  }
}

function handleStop() {
  if (inferenceTimer) {
    clearInterval(inferenceTimer);
    inferenceTimer = null;
  }
  stopCamera(stream);
  stream = null;

  startBtn.disabled = false;
  stopBtn.disabled = true;
  setStatus("Stopped");
  labelEl.textContent = "—";
  confidenceEl.textContent = "";
}

startBtn.addEventListener("click", handleStart);
stopBtn.addEventListener("click", handleStop);

// Pre-load the model as soon as the page opens so Start feels instant.
loadModel(setStatus).catch((err) => {
  console.error(err);
  setStatus(`Model failed to load: ${err.message}`);
});
