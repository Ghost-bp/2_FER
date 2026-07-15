"""
Backbone 工厂函数 — 支持多种 CNN 架构。
用于 Model 类中替换默认的 ResNet-18。

支持的架构:
  - resnet18 (默认): BasicBlock, 512维输出
  - resnet34: BasicBlock, 512维输出 (更深)
  - resnet50: Bottleneck, 2048维输出 (需投影到512)
  - efficientnet-b0: MBConv, 1280维输出 (需投影到512)
"""

import torch
import torch.nn as nn
from torchvision import models


def create_torchvision_backbone(arch='resnet18', pretrained=True,
                                pretrained_path=None, device='cpu'):
    """
    创建 torchvision 预训练 backbone。

    Args:
        arch: 'resnet18' | 'resnet34' | 'resnet50' | 'efficientnet-b0'
        pretrained: 是否使用 ImageNet 预训练权重
        pretrained_path: 自定义权重路径（优先级高于 pretrained）
        device: 加载权重的设备

    Returns:
        tuple: (features_module, avgpool_module, fc_in_dim)
    """
    # --- ResNet 系列 ---
    if arch in ('resnet18', 'resnet34', 'resnet50'):
        return _create_resnet(arch, pretrained, pretrained_path, device)

    # --- EfficientNet 系列 ---
    elif arch == 'efficientnet-b0':
        return _create_efficientnet(pretrained, pretrained_path, device)

    else:
        raise ValueError(
            f"不支持的架构: {arch}。可选: resnet18, resnet34, resnet50, efficientnet-b0"
        )


def _create_resnet(arch, pretrained, pretrained_path, device):
    """创建 ResNet backbone (resnet18/34/50)。"""
    arch_config = {
        'resnet18': (models.resnet18, models.ResNet18_Weights.IMAGENET1K_V1),
        'resnet34': (models.resnet34, models.ResNet34_Weights.IMAGENET1K_V1),
        'resnet50': (models.resnet50, models.ResNet50_Weights.IMAGENET1K_V1),
    }

    model_fn, weights_cls = arch_config[arch]
    weights = weights_cls if pretrained else None
    model = model_fn(weights=weights)

    # 加载自定义预训练权重（如果提供）
    if pretrained_path is not None:
        checkpoint = torch.load(pretrained_path, map_location=device)
        state_dict = checkpoint.get('state_dict', checkpoint)
        model.load_state_dict(state_dict, strict=False)

    # 切片：features = conv1~layer4, avgpool = AdaptiveAvgPool2d
    features = nn.Sequential(*list(model.children())[:-2])
    avgpool = nn.Sequential(*list(model.children())[-2:-1])
    fc_in_dim = model.fc.in_features  # res18/34→512, res50→2048

    return features, avgpool, fc_in_dim


def _create_efficientnet(pretrained, pretrained_path, device):
    """创建 EfficientNet-B0 backbone。"""
    weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.efficientnet_b0(weights=weights)

    if pretrained_path is not None:
        checkpoint = torch.load(pretrained_path, map_location=device)
        state_dict = checkpoint.get('state_dict', checkpoint)
        model.load_state_dict(state_dict, strict=False)

    # EfficientNet 结构: features → avgpool → classifier
    features = model.features
    avgpool = model.avgpool
    fc_in_dim = model.classifier[1].in_features  # 1280

    return features, avgpool, fc_in_dim


def count_backbone_params(features, avgpool, fc_in_dim=None):
    """
    统计 backbone 参数量。

    Args:
        features: 特征提取层
        avgpool: 池化层
        fc_in_dim: 输出维度（可选）

    Returns:
        dict: {total, trainable, frozen}
    """
    total = 0
    trainable = 0
    for module in [features, avgpool]:
        for p in module.parameters():
            n = p.numel()
            total += n
            if p.requires_grad:
                trainable += n

    return {
        'total': total,
        'trainable': trainable,
        'frozen': total - trainable,
    }
