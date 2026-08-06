/**
 * Renders the current gesture prediction into the UI — both the text
 * readout and a lightweight canvas overlay on top of the video feed.
 */

export function updatePredictionUI(labelEl, confidenceEl, { label, confidence }) {
  labelEl.textContent = label;
  confidenceEl.textContent = `${(confidence * 100).toFixed(1)}%`;
}

export function drawOverlay(canvasEl, videoEl, { label, confidence }) {
  const ctx = canvasEl.getContext("2d");

  if (canvasEl.width !== videoEl.videoWidth || canvasEl.height !== videoEl.videoHeight) {
    canvasEl.width = videoEl.videoWidth;
    canvasEl.height = videoEl.videoHeight;
  }

  ctx.clearRect(0, 0, canvasEl.width, canvasEl.height);

  const text = `${label} (${(confidence * 100).toFixed(0)}%)`;
  ctx.font = "20px sans-serif";
  const padding = 8;
  const textWidth = ctx.measureText(text).width;

  ctx.fillStyle = "rgba(0, 0, 0, 0.6)";
  ctx.fillRect(12, 12, textWidth + padding * 2, 32);

  ctx.fillStyle = "#ffffff";
  ctx.fillText(text, 12 + padding, 12 + 22);
}
