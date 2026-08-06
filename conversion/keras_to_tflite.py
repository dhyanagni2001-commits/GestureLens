"""
Convert a trained Keras GestureLens model to a quantized TFLite model.

Usage:
    python keras_to_tflite.py --model ../models/keras/gesturelens.h5 \
        --out ../models/tflite/gesturelens_quant.tflite --quantize
"""
import argparse
import os

import tensorflow as tf


def representative_dataset_gen(img_size=(224, 224), num_samples=100):
    """
    Yields random calibration samples for full-integer post-training
    quantization. Replace with real validation images for better accuracy
    on the quantized model.
    [TODO] Swap this out for a generator over real held-out images.
    """
    for _ in range(num_samples):
        data = tf.random.uniform((1,) + img_size + (3,), minval=0, maxval=1)
        yield [data]


def convert(model_path: str, out_path: str, quantize: bool, img_size=(224, 224)):
    model = tf.keras.models.load_model(model_path)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)

    if quantize:
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = lambda: representative_dataset_gen(img_size)
        # Dynamic-range quantization by default (safe, no need for full-int
        # calibration). Uncomment below for full-integer quantization if
        # targeting integer-only accelerators:
        # converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        # converter.inference_input_type = tf.uint8
        # converter.inference_output_type = tf.uint8

    tflite_model = converter.convert()

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "wb") as f:
        f.write(tflite_model)

    orig_size = os.path.getsize(model_path) / (1024 * 1024)
    new_size = os.path.getsize(out_path) / (1024 * 1024)
    print(f"[GestureLens] Original model: {orig_size:.2f} MB")
    print(f"[GestureLens] TFLite model:   {new_size:.2f} MB")
    print(f"[GestureLens] Saved to {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Convert Keras model to TFLite")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--out", type=str, default="../models/tflite/gesturelens_quant.tflite")
    parser.add_argument("--quantize", action="store_true")
    parser.add_argument("--img-size", type=int, nargs=2, default=[224, 224])
    args = parser.parse_args()

    convert(args.model, args.out, args.quantize, tuple(args.img_size))


if __name__ == "__main__":
    main()
