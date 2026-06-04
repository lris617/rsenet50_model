import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from matplotlib import font_manager
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# ===== 统一风格设置（与 test_cnn_lstm.py 保持一致） =====
font_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "arial-regular.ttf")
if os.path.exists(font_path):
    font_manager.fontManager.addfont(font_path)
    plt.rcParams['font.family'] = 'Arial-Regular'
    plt.rcParams['font.sans-serif'] = ['Arial-Regular']
else:
    plt.rcParams['font.family'] = 'DejaVu Sans'
    print("警告：未找到arial-regular.ttf，继续使用默认字体。")
plt.rcParams['font.size'] = 18
plt.rcParams['axes.linewidth'] = 2
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
plt.rcParams['xtick.major.width'] = 2
plt.rcParams['ytick.major.width'] = 2
plt.rcParams['xtick.labelsize'] = 18
plt.rcParams['ytick.labelsize'] = 18
plt.rcParams['axes.labelsize'] = 18
plt.rcParams['axes.titlesize'] = 20
plt.rcParams['legend.fontsize'] = 18

# 获取脚本所在目录
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, ".."))

# 预测结果csv路径
csv_path = os.path.join(project_root, "report/cnnlstm14.12/cnnlstm/resnet50_bs16_lr0.0001_feature1_lstm_layers1_hiddendim256/test_pred_results.csv")

# loss文件路径
loss_path = os.path.join(project_root, "report/cnnlstm14.12/cnnlstm_lossresnet50_bs16_lr0.0001_feature1_lstm_layers1_hiddendim256.csv")

# 目标图片保存目录
csv_dir = os.path.basename(script_dir)
pic_dir = os.path.join(project_root, "report/cnnlstm14.12/pic", csv_dir)
os.makedirs(pic_dir, exist_ok=True)

# 1. 读取预测结果，画7x7采样点colormap
df = pd.read_csv(csv_path)
pred_delta = df['pred_delta'].values[:49]
if len(pred_delta) < 49:
    raise ValueError("数据不足49个采样点")
pred_delta_2d = pred_delta.reshape(7, 7)

fig, ax = plt.subplots(figsize=(8, 6))
im = ax.imshow(pred_delta_2d, cmap='viridis', aspect='auto')
cbar = plt.colorbar(im, ax=ax)
cbar.set_label(r'Predicted $\Delta V_{TH}$', fontsize=18, labelpad=10)
cbar.ax.tick_params(labelsize=18)
ax.set_title(r'Predicted $\Delta V_{TH}$ Colormap (7$\times$7)', fontsize=18, fontname="Arial-Regular", loc='center', pad=12)
ax.set_xlabel('X', fontsize=18, labelpad=8, fontname="Arial-Regular")
ax.set_ylabel('Y', fontsize=18, labelpad=8, fontname="Arial-Regular")
ax.tick_params(axis='both', which='major', labelsize=18, direction='in', length=8, width=2, top=True, right=True)
for spine in ax.spines.values():
    spine.set_linewidth(2)
    spine.set_position(('outward', 0))
plt.subplots_adjust(right=0.7)
plt.tight_layout()
plt.savefig(os.path.join(pic_dir, "colormap_pred_delta_7x7.png"), dpi=150)
plt.close()

# 2. 模型性能评估指标
y_true = df['true_delta']
y_pred = df['pred_delta']
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
fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(loss_df['epoch'], loss_df['train_loss'], label='Train Loss')
ax.plot(loss_df['epoch'], loss_df['val_loss'], label='Validation Loss')
ax.set_xlabel('Epoch', fontsize=18, labelpad=8, fontname="Arial-Regular")
ax.set_ylabel('Loss', fontsize=18, labelpad=8, fontname="Arial-Regular")
ax.set_title('Loss vs Epoch', fontsize=18, fontname="Arial-Regular", loc='center', pad=12)
ax.legend(fontsize=18, loc='best')
ax.tick_params(axis='both', which='major', labelsize=18, direction='in', length=8, width=2, top=True, right=True)
for spine in ax.spines.values():
    spine.set_linewidth(2)
    spine.set_position(('outward', 0))
plt.subplots_adjust(right=0.7)
plt.tight_layout()
plt.savefig(os.path.join(pic_dir, "loss_vs_epoch.png"), dpi=150)
plt.close()
