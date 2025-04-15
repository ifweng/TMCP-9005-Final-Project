# RNA Structure Prediction Model

This repository contains tools and scripts for training and predicting RNA structure using a transformer-based deep learning model.

## Hardware Requirements

This model has been optimized to run on Google Colab Premium with the following specifications:

| Component | Specification |
|-----------|---------------|
| GPU | NVIDIA T4 (16GB GDDR6) |
| vCPUs | ~4-8 cores (varies) |
| RAM | ~25-30GB (high-RAM runtime) |
| Storage | ~100GB (Premium tier) |
| OS | Ubuntu 20.04 LTS |

## Environment Setup for Google Colab

To set up the environment in Google Colab Premium, use one of the following methods:

### Option 1: Using conda environment file

Use the `colab_environment.yml` file in your Google Drive:

Then run these commands in a Colab cell:

```python
!pip install -q condacolab
import condacolab
condacolab.install()
!mamba env create -f /content/drive/MyDrive/path/to/colab_environment.yml
!mamba activate rna_reconstruct
```

### Option 2: Direct pip installation

Since Colab already has many packages pre-installed, you can simply install the remaining packages:

```python
!pip install fastai==2.7.13 einops==0.7.0 pytorch-lightning==2.0.0 tensorboardx==2.2 \
    torchmetrics==0.11.4 torchscan==0.1.2 x-transformers==1.24.5 kaggle==1.5.16 \
    tensorly==0.8.1 tensorly-torch==0.4.0
```

## Installation

Follow these steps to set up the data:

1. **Download Data**: 
   ```bash
   ./setup_data.sh
   ```
   This script downloads the required training and test datasets.

2. **Install Dependencies**: 
   ```bash
   ./setup_tools.sh
   ```
   This installs all required software packages and libraries.

3. **Feature Extraction**: 
   ```bash
   ./calculate_features.sh
   ```
   This step calculates the features required for model training and inference.

## Training

To train the RNA structure prediction model, use the `utils/train_uni_adjnet.py` script.

### Recommended Training Configuration

```bash
python3 utils/train_uni_adjnet_se.py \
  --bpp_path /content/drive/MyDrive/rna/eterna/ \
  --train_path /content/drive/MyDrive/rna/train_data/train_data.parquet \
  --out_path outmodel_dir \
  --device 0 \
  --num_workers 4 \
  --wd 0.05 \
  --epoch 270 \
  --lr_max 5e-3 \
  --pct_start 0.05 \
  --batch_cnt 1791 \
  --sgd_lr 5e-5 \
  --sgd_epochs 25 \
  --sgd_batch_cnt 500 \
  --sgd_wd 0.05 \
  --fold 0 \
  --nfolds 1000 \
  --pos_embedding dyn \
  --adj_ks 3 \
  --seed 42 \
  --use_se
```

> **Note**: Adjust batch sizes and worker counts based on available memory in Colab.

## Inference

To make predictions on a new test set, use the `utils/predict.py` script:

```bash
python3 utils/predict.py \
  --bpp_path /content/drive/MyDrive/rna/eterna/ \
  --test_path /content/drive/MyDrive/rna/test_data.csv \
  --model_path /content/drive/MyDrive/rna/trained_models/model_0.pth \
  --out_path /content/drive/MyDrive/rna/predictions/ \
  --device 0 \
  --pos_embedding dyn \
  --adj_ks 3 \
  --num_workers 4 \
  --use_se
```

## Key Parameters

| Parameter | Description |
|-----------|-------------|
| `--bpp_path` | Path to base pair probability files |
| `--pos_embedding` | Positional embedding type (dyn, xpos, alibi) |
| `--adj_ks` | Kernel size for adjacency convolutions |
| `--use_se` | Use Squeeze-and-Excitation blocks |
| `--device` | GPU device index (typically 0) |
| `--num_workers` | Number of data loading workers |

## Performance Considerations for Colab

- For Colab environments, limit workers (4-8) to avoid memory issues
- Store intermediate results to Google Drive to persist across sessions
- Consider reducing batch size if encountering OOM errors
- Use T4-optimized settings for best performance
- Set restart_runtime=True in Colab to apply environment changes
- Mount your Google Drive with `drive.mount('/content/drive')` to access saved data