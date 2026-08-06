"""
Preprocessing pipeline for HaGRID images feeding into MobileNetV2 fine-tuning.

Handles: resize, normalization, and basic augmentation via tf.keras
image_dataset_from_directory + a tf.data augmentation pipeline.
"""
import argparse

import tensorflow as tf

IMG_SIZE = (224, 224)  # [TODO: confirm final input resolution]
BATCH_SIZE = 32


def build_datasets(data_dir: str, img_size=IMG_SIZE, batch_size=BATCH_SIZE, val_split=0.2, seed=42):
    train_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=val_split,
        subset="training",
        seed=seed,
        image_size=img_size,
        batch_size=batch_size,
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=val_split,
        subset="validation",
        seed=seed,
        image_size=img_size,
        batch_size=batch_size,
    )
    class_names = train_ds.class_names

    normalization_layer = tf.keras.layers.Rescaling(1.0 / 255)
    augmentation = tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.05),
            tf.keras.layers.RandomZoom(0.1),
            tf.keras.layers.RandomBrightness(0.1),
        ]
    )

    def prep_train(x, y):
        x = augmentation(x)
        x = normalization_layer(x)
        return x, y

    def prep_val(x, y):
        x = normalization_layer(x)
        return x, y

    train_ds = train_ds.map(prep_train).prefetch(tf.data.AUTOTUNE)
    val_ds = val_ds.map(prep_val).prefetch(tf.data.AUTOTUNE)

    return train_ds, val_ds, class_names


def main():
    parser = argparse.ArgumentParser(description="Build train/val datasets from HaGRID directory")
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--img-size", type=int, nargs=2, default=list(IMG_SIZE))
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    args = parser.parse_args()

    train_ds, val_ds, class_names = build_datasets(
        args.data_dir, img_size=tuple(args.img_size), batch_size=args.batch_size
    )
    print(f"[GestureLens] Classes found: {class_names}")
    print(f"[GestureLens] Train batches: {tf.data.experimental.cardinality(train_ds).numpy()}")
    print(f"[GestureLens] Val batches: {tf.data.experimental.cardinality(val_ds).numpy()}")


if __name__ == "__main__":
    main()
