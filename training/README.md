# Training Module

## Scope

This module defines the training contract for Robotera VLA using a pi0.5-adjusted model baseline.

## Status

Implementation details are intentionally pending and should be filled with owner-provided training specifics.

## Installation

### 1. Create a Python environment

This project expects Python `3.10`.

```bash
conda create -n robotera_vla python=3.10
conda activate robotera_vla
```

### 2. Install this repository

```bash
pip install -e .
```

## Model Checkpoints

### Base Models

The base model (`π₀.₅`) is sourced from the [Physical Intelligence openpi](https://github.com/Physical-Intelligence/openpi) project.

| Model       | Checkpoint Path                              |
| ----------- | -------------------------------------------- |
| π₀.₅ base   | `gs://openpi-assets/checkpoints/pi05_base`   |

See the [openpi Model Checkpoints](https://github.com/Physical-Intelligence/openpi?tab=readme-ov-file#model-checkpoints) section for details.

### Fine-Tuned Models

We provide a fine-tuned model for the pick-and-place task.
| Model       | Checkpoint Path                                              |
| ----------- | -------------------------------------------------------------|
| π₀.₅ M7_PAP | `https://huggingface.co/roboterax/M7_pickplace_example_ckpt` |


## Fine-Tuning Base Models on Your Own Data

We will fine-tune the $\pi_{0.5}$ model on the [robotera example dataset](https://huggingface.co/datasets/roboterax/M7_pickplace_example) as a running example for how to fine-tune a base model on your own data. 

Compute the normalization statistics for the training data.:

```bash
python scripts/compute_norm_stats.py --config-name pi05_M7_pp_opensource
```

training:

```bash
XLA_PYTHON_CLIENT_MEM_FRACTION=0.8 python scripts/train.py pi05_M7_pp_opensource --exp-name=my_experiment --overwrite
```

## Test Results

### Training Environment

| Item | Value |
|---|---|
| OS | Ubuntu 22.04 |
| GPU | NVIDIA A800 (8 cards) |
| Training Duration | 24 hours |
| Dataset | robotera example dataset |

### Training Status

Training has been successfully validated on the above hardware configuration. The model was trained on 8x A800 GPUs for approximately 24 hours using the test dataset, demonstrating stable training convergence and proper hardware utilization.