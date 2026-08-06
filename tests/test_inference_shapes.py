"""
Sanity tests for the trained model's expected input/output shapes.
These are skipped automatically if no trained model artifact is present,
so the suite still passes in a fresh checkout before training has run.

Run with: pytest tests/
"""
import os

import pytest

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "keras", "gesturelens.h5")
EXPECTED_INPUT_SIZE = (224, 224, 3)  # [TODO] confirm against training config


@pytest.mark.skipif(not os.path.exists(MODEL_PATH), reason="No trained model artifact present yet")
def test_model_input_output_shapes():
    import tensorflow as tf

    model = tf.keras.models.load_model(MODEL_PATH)

    input_shape = model.input_shape[1:]
    assert input_shape == EXPECTED_INPUT_SIZE, (
        f"Expected input shape {EXPECTED_INPUT_SIZE}, got {input_shape}"
    )

    output_shape = model.output_shape[1:]
    assert len(output_shape) == 1, "Expected a flat softmax output over classes"


@pytest.mark.skipif(not os.path.exists(MODEL_PATH), reason="No trained model artifact present yet")
def test_model_predicts_valid_probability_distribution():
    import numpy as np
    import tensorflow as tf

    model = tf.keras.models.load_model(MODEL_PATH)
    dummy_input = np.random.rand(1, *EXPECTED_INPUT_SIZE).astype("float32")
    preds = model.predict(dummy_input, verbose=0)

    assert preds.shape[0] == 1
    assert np.isclose(preds.sum(), 1.0, atol=1e-3), "Softmax output should sum to ~1.0"
