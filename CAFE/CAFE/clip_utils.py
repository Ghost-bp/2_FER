"""
CLIP 模型层数调整工具。
不修改 CLIP 源码，通过切片 resblocks 实现层数调整。
"""

import torch.nn as nn


def reduce_clip_layers(clip_model, num_layers):
    """
    减少 CLIP 视觉编码器的 Transformer 层数。

    CLIP ViT-B-32 原始有 12 层 Transformer。
    此函数将 resblocks 截断为前 num_layers 层。

    Args:
        clip_model: 完整的 CLIP 模型（已加载权重）
        num_layers: 保留的层数（1-12）

    Returns:
        修改后的 clip_model（原地修改）
    """
    visual = clip_model.visual
    original_layers = visual.transformer.layers

    if num_layers >= original_layers:
        print(f"  保持原始层数: {original_layers}")
        return clip_model

    # 截断 resblocks
    old_resblocks = list(visual.transformer.resblocks.children())
    new_resblocks = old_resblocks[:num_layers]
    visual.transformer.resblocks = nn.Sequential(*new_resblocks)
    visual.transformer.layers = num_layers

    print(f"  CLIP 层数: {original_layers} → {num_layers}")
    return clip_model


def get_clip_num_layers(clip_model):
    """获取 CLIP 视觉编码器当前的 Transformer 层数。"""
    return clip_model.visual.transformer.layers
