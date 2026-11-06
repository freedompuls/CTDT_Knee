# Knee Osteoarthritis KL Grading via Error-Complementarity Stacking Ensemble

[![Python](https://img.shields.io/badge/Python-3.10-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0.1-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

Official implementation of our paper on automated Kellgren–Lawrence (KL) grading of knee osteoarthritis (KOA) from plain radiographs.

## Overview

The core idea of this work is to exploit the **measured error complementarity** between a modern CNN (ConvNeXt-Tiny) and a deformable-attention Transformer (DAT-Tiny), rather than simply enlarging model capacity.

We first quantify the hard-example overlap between the two base models. Their intersection covers only **3.5%** of the union, indicating that the same image rarely challenges both architectures. Based on this observation, we design a feature-level stacking ensemble together with a two-stage training schedule and hard-example-targeted moving least squares (MLS) augmentation.

## Method

1. **Heterogeneous feature extraction** — ConvNeXt-Tiny and DAT-Tiny each output a 768-d feature vector after global average pooling.
2. **Feature-level stacking** — the two vectors are concatenated into a 1536-d joint feature and mapped to 5 classes by a learnable linear fusion layer.
3. **Two-stage progressive training**
   - Stage 1: freeze both backbones, train only the fusion layer.
   - Stage 2: unfreeze the last two stages of each backbone and fine-tune jointly at a lower learning rate.
4. **Hard-example mining** — samples misclassified over consecutive epochs by either base model are collected; the union is used as the hard-example set.
5. **Targeted MLS augmentation** — distortion and stretching deformations are applied only to the hard-example union; augmented copies are added to the training set only.

## Dataset

Experiments use the public **Knee Osteoarthritis Severity Grading Dataset** derived from the Osteoarthritis Initiative (OAI).

| Split      | Images |
|------------|--------|
| Training   | 5,778  |
| Validation | 826    |
| Test       | 1,656  |

## Results

| Method | Overall Acc. (%) | Macro F1 (%) |
|--------|-----------------|--------------|
| ConvNeXt-Tiny (single) | 70.34 | 65.20 |
| DAT-Tiny (single) | 70.22 | 64.60 |
| Simple stacking | 72.64 | 68.00 |
| **Proposed (full)** | **74.88** | **70.98** |

- 95% CI of the overall accuracy: **72.73–76.91%**
- **86.5%** of all errors are confined to adjacent KL grades
- Ablation: two-stage training **+0.79** pp, MLS targeted augmentation **+1.45** pp

## Installation

```bash
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>
pip install -r requirements.txt
```
![Confusion matrix](<images/Confusion matrix of the proposed method on the test set2.png>)
