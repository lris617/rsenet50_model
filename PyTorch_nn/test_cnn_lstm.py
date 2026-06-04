import torch
import torch.nn as nn
import os
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import glob
import time
import numpy as np
from model import CNNLSTMRegressor
from model import CNNLSTMRegressorResnetSeq
# --------- 数据集定义 ---------
class LSTMImageDataset(Dataset):
    def __init__(self, root_dir, label_csv, seq_len=4, img_size=224, transform=None):
        self.root_dir = root_dir
        self.df = pd.read_csv(label_csv)
        self.seq_len = seq_len
        self.img_size = img_size
        self.transform = transform or transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        sample_id = row['filename'].replace('.png', '')
        imgs = []
        for i in range(1, self.seq_len+1):
            img_path = os.path.join(self.root_dir, f"{sample_id}.png")
            img = Image.open(img_path).convert("RGB")
            img = self.transform(img)
            imgs.append(img)
        imgs = torch.stack(imgs, dim=0)  # (seq, C, H, W)
        target = torch.tensor([row['delta_vth']], dtype=torch.float32)
        return imgs, target, sample_id

# --------- 路径配置 ---------
model_save_dir = "report/cnnlstm14.12/cnnlstm/resnet50_bs16_lr0.0001_feature1_lstm_layers1_hiddendim256"
result_dir = model_save_dir
result_csv = os.path.join(result_dir, "test_pred_results.csv")
scatter_plot_path = os.path.join(result_dir, "test_pred_vs_true.png")
test_img_dir = "PyTorch_nn/dataset14/test"
test_label_csv = os.path.join(test_img_dir, "labels.csv")
model_files_pattern = os.path.join(model_save_dir, "cnn_lstm_best_epoch*_val*.pth")
model_files = glob.glob(model_files_pattern)
if not model_files:
    raise FileNotFoundError("未找到已保存的模型权重文件")
model_files.sort(key=lambda x: float(x.split("_val")[-1].replace(".pth", "")))
best_model_path = model_files[0]

if __name__ == "__main__":
    os.makedirs(result_dir, exist_ok=True)
    batch_size = 1
    seq_len = 1
    img_size = 224
    cnn_type = 'resnet50'
    feature_dim = 1
    lstm_hidden_dim = 256
    lstm_layers = 1

    # 加载test集
    test_dataset = LSTMImageDataset(test_img_dir, test_label_csv, seq_len=seq_len, img_size=img_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    print(f"加载模型: {best_model_path}")

    model = CNNLSTMRegressorResnetSeq(
        cnn_type=cnn_type,
        feature_dim=feature_dim,
        lstm_hidden_dim=lstm_hidden_dim,
        lstm_layers=lstm_layers,
        out_dim=1,
        pretrained=False
    )
    model.load_state_dict(torch.load(best_model_path, map_location="cpu"))
    model.eval()

    # 读取真实标签
    label_df = pd.read_csv(test_label_csv)
    label_map = {row['filename'].replace('.png', ''): row for _, row in label_df.iterrows()}

    # 推理
    pred_deltas = []
    true_deltas = []
    sample_ids = []
    delta_biases = []
    start_time = time.time()
    with torch.no_grad():
        for batch in test_loader:
            input_tensor, target, sample_ids_batch = batch
            pred = model(input_tensor)
            pred = pred.cpu().numpy().flatten()
            target = target.cpu().numpy().flatten()
            for i, sample_id in enumerate(sample_ids_batch):
                label_row = label_map.get(sample_id, None)
                true_delta = float(label_row['delta_vth']) if label_row is not None else None
                pred_delta = float(pred[i])
                print(f"{sample_id}: pred_delta_vth={pred_delta:.6f}, true_delta_vth={true_delta:.6f}, diff={pred_delta-true_delta if (true_delta is not None) else float('nan'):.6f}")
                sample_ids.append(sample_id)
                true_deltas.append(true_delta)
                pred_deltas.append(pred_delta)
                delta_biases.append(pred_delta - true_delta if true_delta is not None else None)
    end_time = time.time()
    total_samples = len(test_dataset)
    avg_time_per_sample = (end_time - start_time) / total_samples
    print(f"单个样本平均推理时间: {avg_time_per_sample:.6f} 秒")

    # 计算误差指标
    pred_deltas_np = np.array(pred_deltas)
    true_deltas_np = np.array(true_deltas)
    abs_bias = np.abs(pred_deltas_np - true_deltas_np)
    sum_abs_bias = np.sum(abs_bias)
    mae_loss = np.mean(abs_bias)
    mse_loss = np.mean((pred_deltas_np - true_deltas_np) ** 2)
    mean_bias = np.mean(pred_deltas_np - true_deltas_np)
    print(f"所有样本 bias 绝对值总和: {sum_abs_bias:.6f}")
    print(f"bias绝对值的平均值: {mae_loss:.6f}")
    print(f"平均 bias: {mean_bias:.6f}")
    print(f"MAE loss: {mae_loss:.6f}")
    print(f"MSE loss: {mse_loss:.6f}")

    # 保存结果
    result_df = pd.DataFrame({
        "sample_id": sample_ids,
        "true_delta": true_deltas,
        "pred_delta": pred_deltas,
        "delta_bias": delta_biases
    })
    result_df.to_csv(result_csv, index=False)
    print(f"测试集预测结果已保存到: {result_csv}")

    # 保存整体指标
    metrics_csv = os.path.join(result_dir, "test_metrics.csv")
    with open(metrics_csv, "w") as f:
        f.write("mae_loss,avg_time_per_sample\n")
        f.write(f"{mae_loss},{avg_time_per_sample}\n")
    print(f"测试集MAE和平均推理时间已保存到: {metrics_csv}")

    # 可视化
    import matplotlib
    plt.rcParams['font.family'] = 'Arial-Regular'
    plt.rcParams['font.size'] = 18

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(true_deltas, pred_deltas,  s=80, alpha=0.7, edgecolors='w', linewidths=1.2)
    ax.plot([min(true_deltas), max(true_deltas)], [min(true_deltas), max(true_deltas)], 'r--', linewidth=2)

    ax.set_xlabel(r"True $\Delta V_{TH}$", fontsize=18, fontname="Arial-Regular", labelpad=8)
    ax.set_ylabel(r"Predicted $\Delta V_{TH}$", fontsize=18, fontname="Arial-Regular", labelpad=8)
    ax.set_title(r"Test Set: True vs Predicted  $\Delta V_{TH}$", fontsize=18, fontname="Arial-Regular", loc='center', pad=12)

    ax.tick_params(axis='both', which='major', labelsize=18, direction='in', length=8, width=2, top=True, right=True)
    for spine in ax.spines.values():
        spine.set_linewidth(2)
        spine.set_position(('outward', 0))

    plt.subplots_adjust(right=0.7)
    plt.tight_layout()
    plt.savefig(scatter_plot_path, dpi=150)
    plt.close()
    print(f"预测对比图已保存为{scatter_plot_path}")
