import os
import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from model import CNNLSTMRegressor
from model import CNNLSTMRegressorResnetSeq

# 设置随机种子，保证实验可复现
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

class LSTMImageDataset(Dataset):
    """
    每个样本为一组时序图片（如4帧）和一个回归标签
    """
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
        for i in range(1, self.seq_len + 1):
            img_path = os.path.join(self.root_dir, f"{sample_id}.png")
            img = Image.open(img_path).convert("RGB")
            img = self.transform(img)
            imgs.append(img)
        imgs = torch.stack(imgs, dim=0)  # (seq, C, H, W)
        label = torch.tensor([row['delta_vth']], dtype=torch.float32)
        return imgs, label

def main():
    # 配置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    batch_size = 16
    seq_len = 1
    img_size = 224
    num_epochs = 1500
    lr = 1e-4
    cnn_type = 'resnet50'
    feature_dim = 1
    lstm_hidden_dim = 256
    lstm_layers = 1

    # 数据路径
    train_img_dir = "PyTorch_nn/sample_data/train"
    train_label_csv = os.path.join(train_img_dir, "labels.csv")
    val_img_dir = "PyTorch_nn/sample_data/val"
    val_label_csv = os.path.join(val_img_dir, "labels.csv")

    # 数据集
    train_dataset = LSTMImageDataset(train_img_dir, train_label_csv, seq_len=seq_len, img_size=img_size)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataset = LSTMImageDataset(val_img_dir, val_label_csv, seq_len=seq_len, img_size=img_size)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # 模型
    model = CNNLSTMRegressorResnetSeq(
        cnn_type=cnn_type,
        feature_dim=feature_dim,
        lstm_hidden_dim=lstm_hidden_dim,
        lstm_layers=lstm_layers,
        out_dim=1,
        pretrained=True
    ).to(device)
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs for training.")
        model = nn.DataParallel(model)

    # ===== 加载ResNet预训练权重（自定义）并冻结参数 =====
    # 如果你有单独训练好的ResNet权重（只包含CNN部分），可在此处加载
    # 例如: resnet_weights_path = "your_resnet_weights.pth"
    resnet_weights_path = "PyTorch_nn/report/resnet50_peak2peak_best_epoch1472_val0.0023.pth"  # 修改为你的权重路径，如 "fudan/day7/resnet50_custom.pth"
    if resnet_weights_path is not None:
        state_dict = torch.load(resnet_weights_path, map_location=device)
        # 兼容DataParallel
        target_cnn = model.module.cnn if isinstance(model, nn.DataParallel) else model.cnn
        target_cnn.load_state_dict(state_dict, strict=False)
        # 冻结ResNet参数
        # for param in target_cnn.parameters():
        #     param.requires_grad = False
        # print(f"Loaded ResNet weights from {resnet_weights_path} and froze its parameters.")

        for param in target_cnn.parameters():
            param.requires_grad = False
        # 只解冻layer3和layer4参数
        for name, module in target_cnn.named_children():
            if name in ['layer3', 'layer4']:
                for param in module.parameters():
                    param.requires_grad = True
    # =========================================

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # 记录loss
    loss_record_path = f"report/cnnlstm14.12/cnnlstm_loss{cnn_type}_bs{batch_size}_lr{lr}_feature{feature_dim}_lstm_layers{lstm_layers}_hiddendim{lstm_hidden_dim}.csv"
    if not os.path.exists(loss_record_path):
        with open(loss_record_path, "w") as f:
            f.write("epoch,train_loss,val_loss,batch_size,lr\n")

    # 保存top3模型
    model_save_dir = f"report/cnnlstm14.12/cnnlstm/{cnn_type}_bs{batch_size}_lr{lr}_feature{feature_dim}_lstm_layers{lstm_layers}_hiddendim{lstm_hidden_dim}"
    os.makedirs(model_save_dir, exist_ok=True)
    best_models = []

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for imgs, targets in train_loader:
            imgs, targets = imgs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * imgs.size(0)
        epoch_loss = running_loss / len(train_loader.dataset)

        # 验证
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for imgs, targets in val_loader:
                imgs, targets = imgs.to(device), targets.to(device)
                outputs = model(imgs)
                loss = criterion(outputs, targets)
                val_loss += loss.item() * imgs.size(0)
        val_loss = val_loss / len(val_loader.dataset)
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {epoch_loss:.4f}, Val Loss: {val_loss:.4f}")

        # 记录loss到csv
        with open(loss_record_path, "a") as f:
            f.write(f"{epoch+1},{epoch_loss},{val_loss},{batch_size},{lr}\n")

        # 保存top3模型
        save_path = os.path.join(model_save_dir, f"cnn_lstm_best_epoch{epoch+1}_val{val_loss:.4f}.pth")
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
        for f in os.listdir(model_save_dir):
            if f.startswith("cnn_lstm_best_epoch") and f.endswith(".pth"):
                full_path = os.path.join(model_save_dir, f)
                if full_path not in all_paths:
                    os.remove(full_path)
                    print(f"Removed old model: {full_path}")

    print("训练完成。")

if __name__ == "__main__":
    main()