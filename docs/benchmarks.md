# Benchmarks

Fill in after running `training/evaluate.py` and browser latency profiling.

## Accuracy

| Split | Top-1 Accuracy | Notes |
|---|---|---|
| Train | `[TODO]` | |
| Validation | `[TODO]` | |
| Test | `[TODO]` | |

## Per-class performance

`[TODO: paste classification_report output from evaluate.py]`

## Model size

| Format | Size (MB) |
|---|---|
| Keras (.h5) | `[TODO]` |
| TFLite (quantized) | `[TODO]` |
| TF.js (float16) | `[TODO]` |

## Browser inference latency

Measured on: `[TODO: device/browser, e.g. 2019 Intel MacBook Pro, 8GB RAM, Chrome 12x]`

| Metric | Value |
|---|---|
| Avg inference time / frame | `[TODO] ms` |
| Effective FPS | `[TODO]` |
| Model load time (cold) | `[TODO] s` |

## Methodology notes

`[TODO: describe how latency was measured — e.g. wrapped predictFrame() in
performance.now() timers averaged over N frames]`
