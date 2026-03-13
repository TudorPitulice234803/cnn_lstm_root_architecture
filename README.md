# Beyond Static Segmentation: LSTM-Enhanced Architectures for Temporal Consistency in Root Analysis

This repository contains the code for the paper *"Beyond Static Segmentation: LSTM-Enhanced Architectures for Temporal Consistency in Root Analysis"*, submitted to **Plant Methods**.

## Overview

Current deep learning models for root phenotyping segment each frame independently, causing lateral roots to appear and vanish between consecutive timepoints. This study investigates whether integrating ConvLSTM2D layers into established segmentation architectures can resolve this temporal inconsistency.

We compare seven architectures across three pairs:

| Static baseline | LSTM-enhanced variant |
|---|---|
| U-Net (~1.9M params) | LSTM-U-Net (~6M params) |
| ResNet (~4.6M params) | LSTM-ResNet (~4.6M params) |
| ViT with MLP (~1M params) | — |
| ViT without MLP (~730K params) | LSTM-ViT (~1.6M params) |

All LSTM variants use a **replacement strategy**: a single Conv2D layer at the bottleneck is replaced with ConvLSTM2D, preserving network depth and isolating temporal recurrence as the sole experimental variable.

## Requirements

- Python 3.11
- TensorFlow 2.x with GPU support (CUDA enabled)
- NVIDIA GPU with sufficient VRAM (experiments used an RTX 6000 Ada Generation, 48 GB)

### Dependencies

```
tensorflow
numpy
opencv-python
scikit-learn
scikit-image
patchify
skan
networkx
matplotlib
scipy
pandas
```

Install with:

```bash
pip install tensorflow numpy opencv-python scikit-learn scikit-image patchify skan networkx matplotlib scipy pandas
```

### GPU Setup

Training was conducted on an NVIDIA RTX 6000 Ada Generation GPU (48 GB VRAM) with CUDA 13.0. A dedicated GPU with CUDA support is strongly recommended for reproducing results. For Apple Silicon machines, `tensorflow-metal` can be used for local development but training times will be significantly longer.

Set legacy Keras if using TensorFlow 2.16+:

```python
import os
os.environ['TF_USE_LEGACY_KERAS'] = '1'
```

## Data

The dataset consists of time-series root images of *Arabidopsis thaliana* grown on agar plates, acquired at the National Plant Eco-phenotyping Center (NPEC). Images are 8-bit greyscale at 2643×1891 pixels, collected from 20 plates imaged over 15 consecutive days.

**Preprocessing pipeline:**
1. Automatic cropping via edge detection to remove black borders
2. Padding to ensure dimensions are divisible by 256
3. Tiling into non-overlapping 256×256 patches
4. Discarding patches with no foreground across all 15 days

**Dataset splits:**

| Split | Images | Patches |
|---|---|---|
| Training (5 plates) | 75 | 3,285 |
| Validation (1 plate) | 15 | 108 |
| Test (1 plate) | 15 | 161 |

**Input formats:**
- U-Net and ResNet: RGB — `(256, 256, 3)`
- ViT variants: Grayscale — `(256, 256, 1)`
- LSTM variants: 15-frame sequences — `(15, 256, 256, C)`

The dataset is available from the corresponding author upon reasonable request.

## Training

Each model is trained across **5 independent runs** with matched random seeds `[42, 123, 456, 789, 999]` using the Adam optimiser (`lr=1e-3`), binary cross-entropy loss, and batch size of 8.

| Architecture pair | Epochs | Steps/epoch |
|---|---|---|
| U-Net / LSTM-U-Net | 3 | 1,000 |
| ResNet / LSTM-ResNet | 4 | 1,000 |
| ViT / ViT-no-MLP / LSTM-ViT | 7 | 1,500 |

No early stopping or data augmentation was applied.

## Evaluation

### Frame-wise Metrics
- **F1-score**
- **mIoU** (mean Intersection over Union)
- **Precision**
- **Recall**

### Temporal Consistency Score (TCS)

A custom cascade-based penalty metric that quantifies frame-to-frame prediction coherence against ground truth primary root growth. TCS penalises biologically implausible behaviours such as root disappearance, shrinkage during growth, stagnation, and drift. Lower TCS indicates better temporal consistency. See Section 3.5 of the paper for the full formulation.

### RISE Saliency Analysis

RISE (Randomized Input Sampling for Explanation) saliency maps are generated per-patch and reassembled into full-image composites to visualise how each architecture allocates spatial attention.

## Results Summary

| Model | F1 | mIoU | TCS |
|---|---|---|---|
| U-Net | 0.60 ± 0.16 | 0.44 ± 0.15 | 2.18 ± 1.15 |
| LSTM-U-Net | 0.76 ± 0.01 | 0.61 ± 0.02 | 1.15 ± 0.23 |
| ResNet | 0.76 ± 0.02 | 0.61 ± 0.02 | 1.11 ± 0.19 |
| LSTM-ResNet | 0.76 ± 0.01 | 0.62 ± 0.00 | 1.50 ± 0.29 |
| ViT (MLP) | 0.69 ± 0.03 | 0.53 ± 0.04 | 2.94 ± 0.15 |
| ViT (no MLP) | 0.71 ± 0.02 | 0.55 ± 0.02 | 2.89 ± 0.21 |
| LSTM-ViT | 0.77 ± 0.00 | 0.62 ± 0.01 | 1.11 ± 0.29 |

LSTM integration improved both segmentation quality and temporal consistency in U-Net and ViT, but degraded TCS in ResNet due to residual connections bypassing the ConvLSTM2D during training.

## Citation

If you use this code, please cite:

```
Pitulice, T., Noyan, A., & Indermun, S. (2025). Beyond Static Segmentation:
LSTM-Enhanced Architectures for Temporal Consistency in Root Analysis.
Plant Methods. [submitted]
```

## License

This project is associated with Breda University of Applied Sciences. Please contact the corresponding author for licensing enquiries.
