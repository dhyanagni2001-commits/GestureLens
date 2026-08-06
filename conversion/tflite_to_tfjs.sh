#!/usr/bin/env bash
# Convert a Keras SavedModel/.h5 to TensorFlow.js format for browser deployment.
#
# Note: tensorflowjs_converter converts from Keras/SavedModel, not from
# .tflite directly. The TFLite artifact (models/tflite/) is kept as the
# mobile/edge-optimized reference model; the TF.js artifact (models/tfjs/)
# is what actually ships to the browser.
#
# Usage:
#   ./tflite_to_tfjs.sh <input_keras_model> <output_tfjs_dir>
# Example:
#   ./tflite_to_tfjs.sh ../models/keras/gesturelens.h5 ../models/tfjs

set -euo pipefail

INPUT_MODEL="${1:-../models/keras/gesturelens.h5}"
OUTPUT_DIR="${2:-../models/tfjs}"

if ! command -v tensorflowjs_converter &> /dev/null; then
    echo "[GestureLens] tensorflowjs_converter not found."
    echo "[GestureLens] Install it with: pip install tensorflowjs"
    exit 1
fi

mkdir -p "$OUTPUT_DIR"

echo "[GestureLens] Converting $INPUT_MODEL -> $OUTPUT_DIR"

tensorflowjs_converter \
    --input_format=keras \
    --quantize_float16 \
    "$INPUT_MODEL" \
    "$OUTPUT_DIR"

echo "[GestureLens] Done. model.json + shard files written to $OUTPUT_DIR"
