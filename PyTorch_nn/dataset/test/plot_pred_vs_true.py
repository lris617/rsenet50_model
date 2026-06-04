import pandas as pd
import matplotlib.pyplot as plt
import re

csv_path = "PyTorch_nn/dataset/test/pred_results7.csv"
df = pd.read_csv(csv_path)

def extract_tensor_value(s):
    # 提取 "tensor(xxx, dtype=torch.float64)" 里的 xxx
    match = re.search(r"tensor\(([-\d\.eE]+)", str(s))
    if match:
        return float(match.group(1))
    try:
        return float(s)
    except Exception:
        return None

true = df["true_delta_vth"].apply(extract_tensor_value)
pred = df["pred_delta_vth"].astype(float)

plt.figure(figsize=(6, 5))
plt.scatter(true, pred, alpha=0.7)
plt.xlabel("True Delta")
plt.ylabel("Predicted Delta")
plt.title("Test Set: True vs Predicted")
min_val = min(true.min(), pred.min())
max_val = max(true.max(), pred.max())
plt.plot([min_val, max_val], [min_val, max_val], 'r--')
plt.tight_layout()
plt.savefig("PyTorch_nn/dataset/test/pred_vs_true7.png")
plt.close()
print("已保存图片: PyTorch_nn/dataset/test/pred_vs_true7.png")
