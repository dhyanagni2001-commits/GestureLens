# Model Card: GestureLens v1

## Overview
- **Task**: Static hand-gesture image classification
- **Architecture**: MobileNetV2 (ImageNet-pretrained backbone, fine-tuned top layers)
- **Training data**: HaGRID (Hand Gesture Recognition Image Dataset)
- **Training environment**: Kaggle Notebooks (GPU)

## Input / Output
- **Input shape**: `[TODO: e.g. 224 x 224 x 3, RGB, normalized to 0-1]`
- **Output**: Softmax distribution over gesture classes
- **Classes**: `[TODO: list class names, e.g. call, fist, like, ok, palm, peace, stop]`

## Artifacts
| Stage | Format | Location |
|---|---|---|
| Trained model | Keras `.h5` / SavedModel | `models/keras/` |
| Edge-optimized | TFLite (quantized) | `models/tflite/` |
| Browser-deployed | TensorFlow.js | `models/tfjs/` |

## Performance
| Metric | Value |
|---|---|
| Top-1 accuracy (val) | `[TODO]` |
| Top-1 accuracy (test) | `[TODO]` |
| Keras model size | `[TODO] MB` |
| TFLite (quantized) size | `[TODO] MB` |
| Browser inference latency | `[TODO] ms/frame` |
| Validated device | 2019 Intel MacBook Pro, 8GB RAM, no dGPU |

## Known limitations
- `[TODO: note lighting/background sensitivity, if observed]`
- `[TODO: note any class confusion pairs from the confusion matrix]`
- Single-hand, single-gesture-per-frame assumption (no multi-hand tracking)

## Intended use
Portfolio / demo project showing an end-to-end edge-ML deployment lifecycle
(train → quantize → convert → browser inference). Not validated for
safety-critical or accessibility-critical gesture-control use cases.
