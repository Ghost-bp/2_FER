"""
轻量级视觉编码器 — 替代 CLIP ViT-B-32。

每个编码器输出 512 维特征向量，与 ResNet-18 门控融合兼容。

支持的编码器：
  - MobileNetV3-Small (约 2.5M 参数)
  - ResNet-18 (复用项目已有结构，作为无 CLIP 时的纯 ResNet 基线)
"""

import torch
import torch.nn as nn
from torchvision import models


class MobileNetV3Encoder(nn.Module):
    """
    MobileNetV3-Small 编码器。
    输入 224×224 RGB → 输出 512 维特征向量。
    参数量约 2.5M（不含最后的投影层）。
    """
    def __init__(self, output_dim=512, pretrained=True):
        super().__init__()
        # 加载 ImageNet 预训练的 MobileNetV3-Small
        weights = models.MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = models.mobilenet_v3_small(weights=weights)

        # 去掉最后的分类头，保留特征提取部分
        self.features = backbone.features  # 输出 576 通道
        self.avgpool = backbone.avgpool    # (1, 1) 自适应平均池化

        # 投影层：576 → output_dim
        self.project = nn.Sequential(
            nn.Linear(576, output_dim),
            nn.BatchNorm1d(output_dim),
            nn.ReLU(inplace=True),
            nn.Linear(output_dim, output_dim),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)  # (B, 576)
        x = self.project(x)         # (B, 512)
        return x


class ResNet18Encoder(nn.Module):
    """
    纯 ResNet-18 编码器（无 CLIP）。
    相比 models.py 中的 Model 类，这个版本去掉了 CLIP 分支，
    直接将 ResNet-18 特征送入 FC 分类。

    输入 224×224 RGB → 输出 output_dim 维特征向量。
    参数量约 11.2M。
    """
    def __init__(self, output_dim=512, pretrained_msceleb_path=None):
        super().__init__()
        from models import ResNet, BasicBlock

        self.resnet = ResNet(
            block=BasicBlock,
            n_blocks=[2, 2, 2, 2],
            channels=[64, 128, 256, 512],
            output_dim=1000,
        )

        # 加载 MSCeleb 预训练权重（如果提供）
        if pretrained_msceleb_path is not None:
            checkpoint = torch.load(pretrained_msceleb_path, map_location='cpu')
            state_dict = checkpoint.get('state_dict', checkpoint)
            self.resnet.load_state_dict(state_dict, strict=False)

        # 去掉 fc 层，用 avgpool 后的 512 维特征
        self.features = nn.Sequential(*list(self.resnet.children())[:-2])  # conv1~layer4
        self.avgpool = nn.Sequential(*list(self.resnet.children())[-2:-1])  # avgpool

        # 投影层
        self.project = nn.Linear(512, output_dim)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)  # (B, 512)
        x = self.project(x)         # (B, output_dim)
        return x
