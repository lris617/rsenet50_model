import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# 获取脚本所在目录
script_dir = os.path.dirname(os.path.abspath(__file__))

# 预测结果csv绝对路径
csv_path = os.path.join(script_dir, "/dataset/test/labels.csv")

# loss文件绝对路径
loss_path = os.path.abspath(os.path.join(script_dir, "report/cnnlstm14.5/cnnlstm_lossresnet50_bs16_lr0.0001_feature1_lstm_layers1_hiddendim16.csv"))

# 目标图片保存目录
csv_dir = os.path.basename(script_dir)
pic_dir = os.path.abspath(os.path.join(script_dir, "report/cnnlstm14.5", csv_dir))
os.makedirs(pic_dir, exist_ok=True)

# 1. 读取预测结果，画7x7采样点colormap
df = pd.read_csv(csv_path)
# 解析 pred_delta_vth
pred_delta = df['pred_delta_vth'].astype(float).values[:49]
if len(pred_delta) < 49:
    raise ValueError("数据不足49个采样点")
pred_delta_2d = pred_delta.reshape(7, 7)

plt.figure(figsize=(5, 4))
im = plt.imshow(pred_delta_2d, cmap='viridis', aspect='auto')
plt.colorbar(im, label='Predicted ΔVth')
plt.title('Predicted ΔVth Colormap (7x7)')
plt.xlabel('X')
plt.ylabel('Y')
plt.tight_layout()
plt.savefig(os.path.join(pic_dir, "colormap_pred_delta_7x7.png"), dpi=300)
plt.close()

# 2. 模型性能评估指标
import re
def parse_tensor_str(s):
    # 解析字符串如 "tensor(0.1324, dtype=torch.float64)" 为 float
    match = re.match(r'tensor\(([-+]?\d*\.\d+|\d+),', str(s))
    if match:
        return float(match.group(1))
    else:
        return float(s)

y_true = df['true_delta_vth'].apply(parse_tensor_str)
y_pred = df['pred_delta_vth'].astype(float)
mse = mean_squared_error(y_true, y_pred)
mae = mean_absolute_error(y_true, y_pred)
r2 = r2_score(y_true, y_pred)

with open(os.path.join(pic_dir, "model_metrics.txt"), "w") as f:
    f.write(f"MSE: {mse:.6f}\n")
    f.write(f"MAE: {mae:.6f}\n")
    f.write(f"R2: {r2:.6f}\n")

print(f"模型评估指标:\nMSE: {mse:.6f}\nMAE: {mae:.6f}\nR2: {r2:.6f}")

# 3. Loss vs Epoch 曲线
loss_df = pd.read_csv(loss_path)
plt.figure(figsize=(6, 4))
plt.plot(loss_df['epoch'], loss_df['train_loss'], label='Train Loss')
plt.plot(loss_df['epoch'], loss_df['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss vs Epoch')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(pic_dir, "loss_vs_epoch.png"), dpi=300)
plt.close()
