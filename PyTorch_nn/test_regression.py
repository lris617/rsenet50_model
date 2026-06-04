import os
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from model import resnet50
import torch.nn as nn
import matplotlib.pyplot as plt
import time

class Peak2PeakDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        self.data = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        img_path = os.path.join(self.img_dir, row['filename'])
        image = Image.open(img_path).convert('RGB')
        label = float(row['delta_vth'])
        if self.transform:
            image = self.transform(image)
        return image, row['filename'], label

def main():
    # 只需指定主卡，DataParallel会自动用所有可用GPU（如0和1），无需写"cuda:0,1"
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    test_csv = "PyTorch_nn/dataset14/test/labels.csv"
    test_img_dir = "PyTorch_nn/dataset14/test"
    batch_size = 1
    THRESHOLD = 3.5

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    test_dataset = Peak2PeakDataset(test_csv, test_img_dir, transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    model = resnet50()
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs for inference.")
        model = nn.DataParallel(model)
    model = model.to(device)
    # 加载权重时兼容DataParallel和单卡
    state_dict = torch.load("PyTorch_nn/report/resnet50_peak2peak_best_epoch1472_val0.0023.pth", map_location=device)
    if isinstance(model, nn.DataParallel):
        model.module.load_state_dict(state_dict)
    else:
        model.load_state_dict(state_dict)
    model.eval()

    results = []
    start_time = time.time()
    with torch.no_grad():
        for images, filenames, true_delta_vths in test_loader:
            images = images.to(device)
            preds = model(images).cpu().squeeze(1).numpy()
            for fname, pred, true_delta_vth in zip(filenames, preds, true_delta_vths):
                delta = pred - THRESHOLD
                bias = pred - float(true_delta_vth)
                results.append({
                    "filename": fname,
                    "pred_delta_vth": pred,
                    "true_delta_vth": true_delta_vth,
                    "delta": delta,
                    "bias": bias
                })
                # 明确比较模型输出与label['delta_vth']
                print(f"{fname}: pred_delta_vth={pred:.6f}, true_delta_vth={true_delta_vth:.6f}, diff={pred-true_delta_vth:.6f}, delta={delta:.6f}, bias={bias:.6f}")
    end_time = time.time()
    total_samples = len(test_dataset)
    avg_time_per_sample = (end_time - start_time) / total_samples
    print(f"单个样本平均推理时间: {avg_time_per_sample:.6f} 秒")

    # 保存结果
    df = pd.DataFrame(results)
    df.to_csv("PyTorch_nn/dataset14/test/pred_results14.csv", index=False)
    # 统计 bias、mae 和 mse
    mean_bias = df["bias"].abs().mean()
    mae_loss = (df["pred_delta_vth"] - df["true_delta_vth"]).abs().mean()
    mse_loss = ((df["pred_delta_vth"] - df["true_delta_vth"]) ** 2).mean()
    print(f"平均 bias (绝对值): {mean_bias:.6f}")
    print(f"MAE loss: {mae_loss:.6f}")
    print(f"MSE loss: {mse_loss:.6f}")
    print("测试完成，结果已保存到 test/pred_results.csv。")

    # 绘制每个样本的 bias 曲线图
    plt.figure(figsize=(10, 5))
    plt.plot(df["bias"].values, marker="o", linestyle="-")
    plt.title("Bias (pred - true) for Each Sample")
    plt.xlabel("Sample Index")
    plt.ylabel("Bias")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("PyTorch_nn/dataset14/test/bias_plot14.png")
    print("bias 曲线图已保存到 test/bias_plot.png。")


if __name__ == "__main__":
    main()