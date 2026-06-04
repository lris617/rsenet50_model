# PyTorch_nn: ResNet50 and CNN-LSTM Regression for Delta Vth Prediction

This repository contains PyTorch experiments for predicting the continuous target `delta_vth` from image-based data. It includes two main modeling pipelines:

- **Single-image regression** with a custom ResNet50 model.
- **Sequence-image regression** with a CNN/ResNet feature extractor followed by an LSTM.

Although the dataset CSV files may contain fields such as `label`, `delta`, and `vth`, the main training scripts use `delta_vth` as the regression target.

## Project Structure

```text
PyTorch_nn/
├── dataset/                       # Single-image dataset split
│   ├── train/
│   ├── val/
│   └── test/
├── dataset14/                     # Another single-image dataset split
│   ├── train/
│   ├── val/
│   └── test/
├── dataset4lstm14/                # Sequence-image dataset for CNN-LSTM
│   ├── train/
│   ├── val/
│   ├── test/
│   └── models*/                   # Historical checkpoints and prediction results
├── model/
│   ├── __init__.py
│   ├── resnet.py                  # Custom ResNet implementation
│   ├── cnnlstm.py                 # CNN-LSTM regression model
│   ├── cnn.py                     # Simple CNN baseline
│   └── vit.py                     # Patch embedding prototype
├── report/                        # Checkpoints, loss logs, figures
├── train.py                       # Train single-image ResNet50 regression
├── test_regression.py             # Test single-image ResNet50 regression
├── train_cnnlstm.py               # Train CNN-LSTM regression
├── train_resnrt_lstm.py           # CNN-LSTM training variant with ResNet weights
├── train_111.py                   # CNN-LSTM training variant
├── test_regressionm_cnnlstm.py    # Test CNN-LSTM regression
├── test_cnn_lstm.py               # CNN-LSTM testing variant
├── plot_colormap.py               # Colormap and loss visualization
├── analysis_and_visualization.py  # Metrics and visualization utilities
└── lstm_identity_mapping.py       # Standalone LSTM identity-mapping experiment
```

## Environment

The project depends on:

```text
torch
torchvision
pandas
numpy
Pillow
matplotlib
scikit-learn
```

Install PyTorch according to your CUDA version, then install the remaining packages with `pip` or `conda`.

Example:

```bash
pip install pandas numpy pillow matplotlib scikit-learn
```

## Recommended Working Directory

Several scripts use paths such as `PyTorch_nn/dataset/...` and `PyTorch_nn/report/...`. For that reason, run commands from the parent directory:

```bash
cd /home/swh/rsenet50_model
```

If you run scripts from inside `PyTorch_nn`, some relative paths may not resolve correctly.

## Dataset Format

### Single-Image Dataset

The single-image datasets are stored in `dataset/` and `dataset14/`.

Expected layout:

```text
dataset14/
├── train/
│   ├── labels.csv
│   └── sample_*.png
├── val/
│   ├── labels.csv
│   └── sample_*.png
└── test/
    ├── labels.csv
    └── sample_*.png
```

Each `labels.csv` should contain at least:

```text
filename,delta_vth
```

The training code loads each image using `filename` and uses `delta_vth` as the regression label.

### Sequence-Image Dataset

The CNN-LSTM pipeline uses `dataset4lstm14/`. Each sample is a directory containing a sequence of images.

Expected layout:

```text
dataset4lstm14/train/
├── labels.csv
├── 14_001/
│   ├── combined_1.png
│   ├── combined_2.png
│   ├── combined_3.png
│   └── combined_4.png
└── 14_002/
    ├── combined_1.png
    ├── combined_2.png
    ├── combined_3.png
    └── combined_4.png
```

The current CNN-LSTM scripts use four frames per sample:

```text
combined_1.png
combined_2.png
combined_3.png
combined_4.png
```

## Model Pipelines

### 1. ResNet50 Single-Image Regression

The custom ResNet50 is defined in:

```text
model/resnet.py
```

For ResNet50, the final convolutional feature after global average pooling is 2048-dimensional. The project then applies a final fully connected layer:

```text
2048 -> 1
```

So the model output is one regression value, not a 2048-dimensional vector.

Train:

```bash
python PyTorch_nn/train.py
```

Default training inputs:

```text
PyTorch_nn/dataset/train/labels.csv
PyTorch_nn/dataset/val/labels.csv
```

Default training settings:

```text
batch_size = 32
num_epochs = 1500
lr = 1e-3
optimizer = Adam
loss = MSELoss
```

Training outputs:

```text
PyTorch_nn/report/day7/resnet50_peak2peak_best_epoch*_val*.pth
PyTorch_nn/report/train_val_loss7.csv
```

Test:

```bash
python PyTorch_nn/test_regression.py
```

Default test inputs:

```text
PyTorch_nn/dataset14/test/labels.csv
PyTorch_nn/dataset14/test/*.png
```

Default checkpoint:

```text
PyTorch_nn/report/resnet50_peak2peak_best_epoch1472_val0.0023.pth
```

Test outputs:

```text
PyTorch_nn/dataset14/test/pred_results14.csv
PyTorch_nn/dataset14/test/bias_plot14.png
```

### 2. CNN-LSTM Sequence Regression

The CNN-LSTM model is defined in:

```text
model/cnnlstm.py
```

Input tensor shape:

```text
(batch, seq, C, H, W)
```

Default sequence length is 4.

Pipeline:

1. Read four `combined_*.png` images for each sample.
2. Resize each frame to `224 x 224`.
3. Use ResNet18 or ResNet50 to extract frame-level features.
4. Project features to `feature_dim`.
5. Feed the feature sequence into an LSTM.
6. Use the last LSTM output for regression.
7. Predict a single `delta_vth` value.

Train:

```bash
python PyTorch_nn/train_cnnlstm.py
```

Default training inputs:

```text
PyTorch_nn/dataset4lstm14/train/labels.csv
PyTorch_nn/dataset4lstm14/val/labels.csv
```

Default training settings:

```text
batch_size = 8
seq_len = 4
img_size = 224
num_epochs = 1500
lr = 1e-4
cnn_type = 'resnet50'
feature_dim = 256
lstm_hidden_dim = 64
lstm_layers = 5
loss = MSELoss
```

Test:

```bash
python PyTorch_nn/test_regressionm_cnnlstm.py
```

The test script looks for:

```text
cnn_lstm_best_epoch*_val*.pth
```

and selects the checkpoint with the lowest validation loss encoded in the filename.

## Visualization

Generate colormap and loss-curve visualizations:

```bash
python PyTorch_nn/plot_colormap.py
```

This script:

- reads prediction results,
- reshapes the first 49 predictions into a `7 x 7` grid,
- draws a predicted `delta_vth` colormap,
- computes MSE, MAE, and R2,
- plots training and validation loss curves.

Additional analysis:

```bash
python PyTorch_nn/analysis_and_visualization.py
```

Note: check the file paths inside analysis scripts before running them, especially if you move result folders.

## Path Notes

The repository has historically used a mix of:

- `PyTorch_nn/report/...`
- `report/...`
- `dataset...`
- hard-coded checkpoint paths

Recent cleanup removed the absolute `/home/swh/...` path from:

- `model/__init__.py`
- `plot_colormap.py`

However, several scripts still use `PyTorch_nn/...` relative paths. The safest way to run the project is still:

```bash
cd /home/swh/rsenet50_model
python PyTorch_nn/<script_name>.py
```

## Common Issues

### FileNotFoundError for datasets

Check that you are running from the parent directory:

```bash
pwd
```

Expected:

```text
/home/swh/rsenet50_model
```

### FileNotFoundError for checkpoints

Some test scripts use fixed checkpoint paths. If you train a new model, update the `.pth` path in the corresponding test script.

### CNN-LSTM sample loading error

Make sure each sample directory contains:

```text
combined_1.png
combined_2.png
combined_3.png
combined_4.png
```

Also make sure the `filename` column in `labels.csv` matches the sample directory names.

## Suggested Future Improvements

- Add a `requirements.txt` or `environment.yml`.
- Move all paths and hyperparameters to a config file or command-line arguments.
- Merge duplicate training and testing scripts.
- Standardize output folders under one `report/` directory.
- Add a small demo dataset or script for quick smoke testing.
- Add unit tests for dataset loading and model forward passes.

