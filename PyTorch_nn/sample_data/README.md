# Sample Data for Quick Testing

This folder now contains two small sample datasets with different layouts.

They are meant for quick validation and path testing after the project cleanup.

## Structure

```text
sample_data/
├── cnn_lstm_single_frame/
│   ├── train/   # 8 samples
│   ├── val/     # 3 samples
│   └── test/    # 4 samples
└── multi_frame_sequence/
    ├── train/   # 6 samples
    ├── val/     # 2 samples
    └── test/    # 3 samples
```

## 1. `cnn_lstm_single_frame`

This dataset uses a flat image layout:

```text
split/
├── labels.csv
├── sample_001.png
├── sample_002.png
└── ...
```

It matches the current behavior of `train_cnn_lstm.py` and `test_cnn_lstm.py`, because those scripts currently do:

```python
seq_len = 1
img_path = os.path.join(root_dir, f"{sample_id}.png")
```

So even though the script names still contain `cnn_lstm`, the current code path is closer to a single-image regression pipeline wrapped in an LSTM-shaped input.

## 2. `multi_frame_sequence`

This dataset uses a per-sample sequence directory layout:

```text
split/
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

This layout is suitable for a real multi-frame CNN-LSTM loader, but it does not match the current implementation of `train_cnn_lstm.py` / `test_cnn_lstm.py` as they exist right now.

## CSV Columns

Both datasets keep the original CSV columns. The key columns used by the project are:

```text
filename
delta_vth
```

Examples:

- In `cnn_lstm_single_frame`, `filename` looks like `sample_055.png`
- In `multi_frame_sequence`, `filename` looks like `14_055`

## Which dataset matches the current scripts

### Current `train_cnn_lstm.py` / `test_cnn_lstm.py`

Use:

```text
PyTorch_nn/sample_data/cnn_lstm_single_frame/...
```

Recommended path changes:

```python
train_img_dir = "PyTorch_nn/sample_data/cnn_lstm_single_frame/train"
val_img_dir = "PyTorch_nn/sample_data/cnn_lstm_single_frame/val"
test_img_dir = "PyTorch_nn/sample_data/cnn_lstm_single_frame/test"
```

### Future or custom real sequence loader

Use:

```text
PyTorch_nn/sample_data/multi_frame_sequence/...
```

and read frames from:

```text
combined_1.png
combined_2.png
combined_3.png
combined_4.png
```

## Recommended Working Directory

```bash
cd D:\rsenet50_model
```
