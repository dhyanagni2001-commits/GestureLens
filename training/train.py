"""
Fine-tune MobileNetV2 on the HaGRID gesture dataset.

Mirrors the Kaggle training notebook so training can also be run as a
standalone script (e.g. in CI or on a local/cloud GPU box).

Usage:
    python train.py --data-dir ./data/hagrid --epochs 15 --batch-size 32 \
        --out ../models/keras/gesturelens.h5
"""
import argparse
import os
import sys

import tensorflow as tf

sys.path.append(os.path.join(os.path.dirname(__file__), "data_prep"))
from preprocess import build_datasets  # noqa: E402


def build_model(num_classes: int, img_size=(224, 224), fine_tune_at: int = 100):
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=img_size + (3,),
        include_top=False,
        weights="imagenet",
    )
    base_model.trainable = True
    # Freeze all layers before `fine_tune_at`, fine-tune the rest.
    for layer in base_model.layers[:fine_tune_at]:
        layer.trainable = False

    inputs = tf.keras.Input(shape=img_size + (3,))
    x = base_model(inputs, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)
    model = tf.keras.Model(inputs, outputs)
    return model


def main():
    parser = argparse.ArgumentParser(description="Train GestureLens MobileNetV2 classifier")
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--img-size", type=int, nargs=2, default=[224, 224])
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--fine-tune-at", type=int, default=100)
    parser.add_argument("--out", type=str, default="../models/keras/gesturelens.h5")
    args = parser.parse_args()

    img_size = tuple(args.img_size)
    train_ds, val_ds, class_names = build_datasets(
        args.data_dir, img_size=img_size, batch_size=args.batch_size
    )
    print(f"[GestureLens] Training on classes: {class_names}")

    model = build_model(num_classes=len(class_names), img_size=img_size, fine_tune_at=args.fine_tune_at)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=args.lr),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.summary()

    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=3, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2),
    ]

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        callbacks=callbacks,
    )

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    model.save(args.out)
    print(f"[GestureLens] Model saved to {args.out}")

    with open(os.path.join(os.path.dirname(args.out), "class_names.txt"), "w") as f:
        f.write("\n".join(class_names))

    best_val_acc = max(history.history.get("val_accuracy", [0]))
    print(f"[GestureLens] Best val accuracy: {best_val_acc:.4f}")


if __name__ == "__main__":
    main()
