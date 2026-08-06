"""
Basic sanity tests for the preprocessing pipeline.
Run with: pytest tests/
"""
import os
import sys

import numpy as np
import pytest
import tensorflow as tf

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "training", "data_prep"))
from preprocess import IMG_SIZE  # noqa: E402


def test_img_size_is_valid():
    assert len(IMG_SIZE) == 2
    assert all(dim > 0 for dim in IMG_SIZE)


def test_normalization_range():
    """Pixel values should be rescaled into [0, 1]."""
    dummy = tf.constant(np.random.randint(0, 256, size=(1, 224, 224, 3)), dtype=tf.float32)
    normalization_layer = tf.keras.layers.Rescaling(1.0 / 255)
    normalized = normalization_layer(dummy)
    assert float(tf.reduce_min(normalized)) >= 0.0
    assert float(tf.reduce_max(normalized)) <= 1.0


def test_augmentation_preserves_shape():
    dummy = tf.random.uniform((1, 224, 224, 3))
    augmentation = tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.05),
        ]
    )
    augmented = augmentation(dummy)
    assert augmented.shape == dummy.shape


if __name__ == "__main__":
    pytest.main([__file__])
