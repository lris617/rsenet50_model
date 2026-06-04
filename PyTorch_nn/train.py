import os
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from model import resnet50
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np

# 设置随机种子，保证实验可复现
seed = 33
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

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
        return image, torch.tensor(label, dtype=torch.float32)

def main():
    # 只需指定主卡，DataParallel会自动用所有可用GPU（如0和1），无需写"cuda:0,1"
    print("torch.cuda.is_available():", torch.cuda.is_available())
    print("torch.cuda.device_count():", torch.cuda.device_count())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    base_dir = os.path.dirname(os.path.abspath(__file__))
    train_csv = os.path.join(base_dir, "dataset/train/labels.csv")
    train_img_dir = os.path.join(base_dir, "dataset/train")
    val_csv = os.path.join(base_dir, "dataset/val/labels.csv")
    val_img_dir = os.path.join(base_dir, "dataset/val")
    batch_size = 32
    num_epochs = 1500
    lr = 1e-3

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    train_dataset = Peak2PeakDataset(train_csv, train_img_dir, transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataset = Peak2PeakDataset(val_csv, val_img_dir, transform)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model = resnet50(pretrained=True)
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs for training.")
        model = nn.DataParallel(model)
    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # 维护top3最优模型列表 [(val_loss, path)]
    best_models = []

    # 新增：准备loss记录文件
    loss_record_path = os.path.join(base_dir, "report/train_val_loss7.csv")
    if not os.path.exists(loss_record_path):
        with open(loss_record_path, "w") as f:
            f.write("epoch,train_loss,val_loss\n")

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for images, targets in train_loader:
            images, targets = images.to(device), targets.to(device).unsqueeze(1)
            outputs = model(images)
            loss = criterion(outputs, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)
        epoch_loss = running_loss / len(train_loader.dataset)

        # 验证
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, targets in val_loader:
                images, targets = images.to(device), targets.to(device).unsqueeze(1)
                outputs = model(images)
                loss = criterion(outputs, targets)
                val_loss += loss.item() * images.size(0)
        val_loss = val_loss / len(val_loader.dataset)
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {epoch_loss:.4f}, Val Loss: {val_loss:.4f}")

        # 记录loss到csv
        with open(loss_record_path, "a") as f:
            f.write(f"{epoch+1},{epoch_loss},{val_loss}\n")

        # 只在进入top3时保存模型
        save_path = os.path.join(base_dir, f"report/day7/resnet50_peak2peak_best_epoch{epoch+1}_val{val_loss:.4f}.pth")
        best_models.append((val_loss, save_path))
        best_models = sorted(best_models, key=lambda x: x[0])[:3]
        if (val_loss, save_path) in best_models:
            if isinstance(model, nn.DataParallel):
                torch.save(model.module.state_dict(), save_path)
            else:
                torch.save(model.state_dict(), save_path)
            print(f"Model saved: {save_path}")
        # 删除非top3的旧模型
        all_paths = [path for _, path in best_models]
        model_dir = os.path.dirname(save_path)
        for f in os.listdir(model_dir):
            if f.startswith("resnet50_peak2peak_best_epoch") and f.endswith(".pth"):
                full_path = os.path.join(model_dir, f)
                if full_path not in all_paths:
                    os.remove(full_path)
                    print(f"Removed old model: {full_path}")

    print("训练完成。")

if __name__ == "__main__":
    main()
