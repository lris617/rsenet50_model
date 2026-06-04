# PyTorch_nn

这个仓库当前主要维护 `PyTorch_nn/` 目录，用于基于图像的阈值预测实验。

当前项目包含：

- `model/`：训练过程中保存的模型文件
- `report/`：实验输出与权重文件
- `sample_data/`：示例数据集，按 `train/`、`val/`、`test/` 划分
- `train_cnn_lstm.py`：训练脚本
- `test_cnn_lstm.py`：测试脚本
- `utils.py`：数据与训练辅助函数

目录结构如下：

```text
PyTorch_nn/
|-- model/
|-- report/
|-- sample_data/
|   |-- train/
|   |-- val/
|   `-- test/
|-- train_cnn_lstm.py
|-- test_cnn_lstm.py
`-- utils.py
```

`sample_data` 中每个数据划分目录通常包含：

```text
labels.csv
sample_*.png
```

其中 `labels.csv` 至少包含以下字段：

```text
filename
delta_vth
```

当前运行方式：

```bash
python PyTorch_nn/train_cnn_lstm.py
python PyTorch_nn/test_cnn_lstm.py
```

说明：如果脚本里仍保留旧数据路径，请将其改为 `PyTorch_nn/sample_data/train`、`PyTorch_nn/sample_data/val` 和 `PyTorch_nn/sample_data/test`。
