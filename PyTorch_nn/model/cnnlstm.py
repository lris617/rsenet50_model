import torch
import torch.nn as nn
from torchvision import models

class CNNLSTMRegressor(nn.Module):
    """
    CNN-LSTM回归模型：每帧图片用CNN提特征，序列特征送入LSTM，最后全连接输出回归结果。
    """
    def __init__(self, cnn_type='resnet18', feature_dim=128, lstm_hidden_dim=64, lstm_layers=2, out_dim=1, pretrained=True):
        super().__init__()
        if cnn_type == 'resnet18':
            base_cnn = models.resnet18(pretrained=pretrained)
            cnn_out_dim = base_cnn.fc.in_features
        elif cnn_type == 'resnet50':
            base_cnn = models.resnet50(pretrained=pretrained)
            cnn_out_dim = base_cnn.fc.in_features
        else:
            raise ValueError("cnn_type仅支持'resnet18'或'resnet50'")

        self.use_fc_feature = False  # 默认用FC前特征图
        # 如果想用FC输出作为LSTM输入，设置self.use_fc_feature = True

        if self.use_fc_feature:
            # 兼容原有实现：FC输出1维
            base_cnn.fc = nn.Linear(base_cnn.fc.in_features, 1)
            self.cnn = base_cnn
            self.lstm = nn.LSTM(
                input_size=1,
                hidden_size=lstm_hidden_dim,
                num_layers=lstm_layers,
                batch_first=True
            )
        else:
            # 只保留到avgpool，输出(B*S, cnn_out_dim, 1, 1)
            self.cnn = nn.Sequential(*list(base_cnn.children())[:-1])
            self.feature_proj = nn.Linear(cnn_out_dim, feature_dim)
            self.lstm = nn.LSTM(
                input_size=feature_dim,
                hidden_size=lstm_hidden_dim,
                num_layers=lstm_layers,
                batch_first=True
            )
        self.reg_head = nn.Linear(lstm_hidden_dim, out_dim)

    def forward(self, x):
        # x: (batch, seq, C, H, W)
        B, S, C, H, W = x.shape
        x = x.view(B * S, C, H, W)
        if self.use_fc_feature:
            feat = self.cnn(x)  # (B*S, 1)
            if isinstance(feat, tuple):
                feat = feat[0]
            feat = feat.view(B * S, -1)  # (B*S, 1)
            feat = feat.view(B, S, 1)    # (B, S, 1)
        else:
            feat = self.cnn(x)  # (B*S, cnn_out_dim, 1, 1)
            feat = feat.view(B * S, -1)  # (B*S, cnn_out_dim)
            feat = self.feature_proj(feat)  # (B*S, feature_dim)
            feat = feat.view(B, S, -1)  # (B, S, feature_dim)
        lstm_out, _ = self.lstm(feat)  # (B, S, lstm_hidden_dim)
        last_feat = lstm_out[:, -1, :]  # 取最后时刻
        out = self.reg_head(last_feat)  # (B, out_dim)
        return out

class CNNLSTMRegressorResnetSeq(CNNLSTMRegressor):
    """
    兼容旧代码：强制使用ResNet FC前特征图序列化输入LSTM
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.use_fc_feature = False
