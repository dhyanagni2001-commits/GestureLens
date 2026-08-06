/**
 * Loads the GestureLens TF.js model and runs per-frame inference against
 * the live webcam feed.
 *
 * Assumes model.json + shard files live at ../../models/tfjs/ relative to
 * this file's served path — adjust MODEL_URL if you host the model
 * elsewhere (e.g. a CDN or separate static bucket).
 */
import * as tf from "https://cdnjs.cloudflare.com/ajax/libs/tensorflow/4.17.0/tf.min.js";

const MODEL_URL = "../models/tfjs/model.json"; // [TODO] confirm final hosted path
const IMG_SIZE = 224; // [TODO] must match training input size

// [TODO] Replace with the actual class list saved during training
// (see training/models/keras/class_names.txt)
export const CLASS_NAMES = [
  "call",
  "fist",
  "like",
  "ok",
  "palm",
  "peace",
  "stop",
];

let model = null;

export async function loadModel(onStatus = () => {}) {
  onStatus("Loading model…");
  model = await tf.loadLayersModel(MODEL_URL);
  onStatus("Model loaded");
  return model;
}

/**
 * Runs a single inference pass on the current video frame.
 * Returns { label, confidence } for the top prediction.
 */
export function predictFrame(videoEl) {
  if (!model) {
    throw new Error("Model not loaded yet — call loadModel() first.");
  }

  return tf.tidy(() => {
    const frame = tf.browser
      .fromPixels(videoEl)
      .resizeBilinear([IMG_SIZE, IMG_SIZE])
      .toFloat()
      .div(255.0)
      .expandDims(0);

    const logits = model.predict(frame);
    const probs = logits.dataSync();

    let maxIdx = 0;
    for (let i = 1; i < probs.length; i++) {
      if (probs[i] > probs[maxIdx]) maxIdx = i;
    }

    return {
      label: CLASS_NAMES[maxIdx] ?? `class_${maxIdx}`,
      confidence: probs[maxIdx],
    };
  });
}

export function isModelLoaded() {
  return model !== null;
}
