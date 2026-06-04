# PyTorch_nn

当前仓库的主要内容是 `PyTorch_nn/`，这是一个精简后的 PyTorch 图像回归实验目录。

当前保留下来的核心内容包括：

- `ResNet` / `CNN-LSTM` 相关模型定义
- 一套训练脚本和一套测试脚本
- 两套用于快速验证的数据样例

当前预测目标字段为：

```text
delta_vth
```

## 当前目录

```text
PyTorch_nn/
├── model/
├── sample_data/
├── train_cnn_lstm.py
├── test_cnn_lstm.py
├── utils.py
└── 项目运行说明.md
```

## sample_data 说明

`PyTorch_nn/sample_data/` 下当前有两套样例数据：

```text
sample_data/
├── cnn_lstm_single_frame/
└── multi_frame_sequence/
```

### `cnn_lstm_single_frame`

使用扁平图片结构：

```text
split/
├── labels.csv
├── sample_*.png
```

样本数量：

- `train`：8
- `val`：3
- `test`：4

这套数据和当前保留下来的 `train_cnn_lstm.py` / `test_cnn_lstm.py` 是匹配的。

### `multi_frame_sequence`

使用多帧序列目录结构：

```text
split/
├── labels.csv
├── 14_001/
│   ├── combined_1.png
│   ├── combined_2.png
│   ├── combined_3.png
│   └── combined_4.png
```

样本数量：

- `train`：6
- `val`：2
- `test`：3

这套数据适合真实的多帧 CNN-LSTM 数据加载逻辑，但和当前脚本实现并不直接匹配。

## 当前代码行为

虽然脚本名仍然是 `cnn_lstm`，但按当前代码实现：

```python
seq_len = 1
img_path = os.path.join(self.root_dir, f"{sample_id}.png")
```

所以它实际更接近：

```text
单张图片输入 + LSTM 外壳的回归实验
```

这也是为什么当前脚本应该配合 `cnn_lstm_single_frame/` 使用，而不是 `multi_frame_sequence/`。

## 如何运行

建议从仓库根目录运行：

```bash
cd D:\rsenet50_model
python PyTorch_nn/train_cnn_lstm.py
python PyTorch_nn/test_cnn_lstm.py
```

## 运行前需要注意

当前脚本里仍然保留了旧路径，默认不会直接指向现在的 `sample_data`。

训练脚本里原本写的是：

```text
PyTorch_nn/dataset14/train
PyTorch_nn/dataset14/val
```

测试脚本里原本写的是：

```text
PyTorch_nn/dataset14/test
```

如果要使用当前样例数据，请改成：

```text
PyTorch_nn/sample_data/cnn_lstm_single_frame/train
PyTorch_nn/sample_data/cnn_lstm_single_frame/val
PyTorch_nn/sample_data/cnn_lstm_single_frame/test
```

另外，训练脚本里的预训练权重路径如果不存在，也需要一起修改或临时设为 `None`。

## 文档

更详细的说明见：

- [PyTorch_nn/项目运行说明.md](D:/rsenet50_model/PyTorch_nn/项目运行说明.md)
- [PyTorch_nn/sample_data/README.md](D:/rsenet50_model/PyTorch_nn/sample_data/README.md)
