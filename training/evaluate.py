"""
Evaluate a trained GestureLens model: overall accuracy, per-class report,
and confusion matrix.

Usage:
    python evaluate.py --model ../models/keras/gesturelens.h5 --data-dir ./data/hagrid
"""
import argparse
import os
import sys

import numpy as np
import tensorflow as tf

sys.path.append(os.path.join(os.path.dirname(__file__), "data_prep"))
from preprocess import build_datasets  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description="Evaluate GestureLens model")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--img-size", type=int, nargs=2, default=[224, 224])
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    model = tf.keras.models.load_model(args.model)
    _, val_ds, class_names = build_datasets(
        args.data_dir, img_size=tuple(args.img_size), batch_size=args.batch_size
    )

    loss, acc = model.evaluate(val_ds)
    print(f"[GestureLens] Val loss: {loss:.4f} | Val accuracy: {acc:.4f}")

    y_true, y_pred = [], []
    for images, labels in val_ds:
        preds = model.predict(images, verbose=0)
        y_true.extend(labels.numpy().tolist())
        y_pred.extend(np.argmax(preds, axis=1).tolist())

    try:
        from sklearn.metrics import classification_report, confusion_matrix

        print("\n[GestureLens] Per-class report:")
        print(classification_report(y_true, y_pred, target_names=class_names))

        print("[GestureLens] Confusion matrix:")
        print(confusion_matrix(y_true, y_pred))
    except ImportError:
        print("[GestureLens] Install scikit-learn for a detailed per-class report: "
              "pip install scikit-learn")


if __name__ == "__main__":
    main()
