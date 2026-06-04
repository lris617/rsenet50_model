# Sample Data for Quick Testing

This folder contains a small subset of the original project data for smoke testing data loading, model forward passes, and inference scripts.

The original datasets are not modified.

## Structure

```text
sample_data/
├── single_image/
│   ├── train/   # 8 samples
│   ├── val/     # 3 samples
│   └── test/    # 4 samples
└── sequence_image/
    ├── train/   # 6 samples
    ├── val/     # 2 samples
    └── test/    # 3 samples
```

## Single-Image Data

Source:

```text
PyTorch_nn/dataset14/
```

Each split contains:

```text
labels.csv
sample_*.png
```

The CSV keeps the original columns. The key columns used by the project are:

```text
filename
delta_vth
```

## Sequence-Image Data

Source:

```text
PyTorch_nn/dataset4lstm14/
```

Each split contains:

```text
labels.csv
sample_id/
```

Each sequence sample directory keeps the four frames used by the CNN-LSTM scripts:

```text
combined_1.png
combined_2.png
combined_3.png
combined_4.png
```

## Usage Notes

Most existing scripts still use hard-coded paths such as:

```text
PyTorch_nn/dataset14/...
PyTorch_nn/dataset4lstm14/...
```

To run scripts directly on this sample data, update the dataset paths in the target script to:

```text
PyTorch_nn/sample_data/single_image/...
PyTorch_nn/sample_data/sequence_image/...
```

Recommended working directory:

```bash
cd /home/swh/rsenet50_model
```

