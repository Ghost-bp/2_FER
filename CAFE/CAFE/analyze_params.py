"""
模型参数量分析脚本。
统计 CAFE 模型各组件（CLIP、ResNet、FC）的参数量，
区分可训练 vs 冻结参数，输出表格和 JSON。
"""

import os
import sys
sys.path.append(os.path.dirname(__file__))

import json
import torch
import clip as clip_module
from models import Model
from config import (
    CLIP_MODEL_PATH, RESNET_PRETRAINED_PATH,
    NUM_CLASSES, EMOTION_LABELS,
)


def count_parameters(model, prefix_filter=None):
    """统计模型参数：总数、可训练数、冻结数。"""
    total = 0
    trainable = 0
    frozen = 0

    for name, param in model.named_parameters():
        if prefix_filter and not name.startswith(prefix_filter):
            continue
        n = param.numel()
        total += n
        if param.requires_grad:
            trainable += n
        else:
            frozen += n

    return total, trainable, frozen


def format_params(n):
    """格式化参数数量为人类可读格式。"""
    if n >= 1e9:
        return f"{n/1e9:.2f}B"
    elif n >= 1e6:
        return f"{n/1e6:.2f}M"
    elif n >= 1e3:
        return f"{n/1e3:.2f}K"
    else:
        return str(n)


def analyze():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"设备: {device}")
    print(f"CLIP 模型路径: {CLIP_MODEL_PATH}")
    print(f"ResNet 预训练路径: {RESNET_PRETRAINED_PATH}")

    # 加载模型
    print("\n加载 CLIP 模型...")
    clip_model, _ = clip_module.load(CLIP_MODEL_PATH, device=device)

    print("构建 CAFE 模型...")
    model = Model(
        clip_model=clip_model,
        num_classes=NUM_CLASSES,
        resnet_pretrained_path=RESNET_PRETRAINED_PATH,
        device=device,
    ).to(device)

    # === 统计各组件参数量 ===
    results = {}

    # 1. CLIP 图像编码器（visual 部分）
    clip_total, clip_trainable, clip_frozen = count_parameters(
        clip_model.visual
    )
    results["CLIP_ViT-B-32 (visual)"] = {
        "total": clip_total,
        "trainable": clip_trainable,
        "frozen": clip_frozen,
        "formatted": format_params(clip_total),
    }

    # 2. ResNet 特征提取器 (features)
    resnet_total = 0
    resnet_trainable = 0
    resnet_frozen = 0
    for name, param in model.features.named_parameters():
        n = param.numel()
        resnet_total += n
        if param.requires_grad:
            resnet_trainable += n
        else:
            resnet_frozen += n
    results["ResNet-18 features (conv1~layer4)"] = {
        "total": resnet_total,
        "trainable": resnet_trainable,
        "frozen": resnet_frozen,
        "formatted": format_params(resnet_total),
    }

    # 3. ResNet avgpool (features2)
    avgpool_total = 0
    avgpool_trainable = 0
    avgpool_frozen = 0
    for name, param in model.features2.named_parameters():
        n = param.numel()
        avgpool_total += n
        if param.requires_grad:
            avgpool_trainable += n
    # avgpool 无参数，但保留统计位
    results["ResNet-18 avgpool (features2)"] = {
        "total": avgpool_total,
        "trainable": avgpool_trainable,
        "frozen": 0,
        "formatted": format_params(avgpool_total),
    }

    # 4. FC 分类头
    fc_total, fc_trainable, fc_frozen = count_parameters(model.fc)
    results["FC Classifier (512→" + str(NUM_CLASSES) + ")"] = {
        "total": fc_total,
        "trainable": fc_trainable,
        "frozen": fc_frozen,
        "formatted": format_params(fc_total),
    }

    # 5. 总计
    grand_total = clip_total + resnet_total + avgpool_total + fc_total
    grand_trainable = clip_trainable + resnet_trainable + avgpool_trainable + fc_trainable
    grand_frozen = clip_frozen + resnet_frozen + avgpool_frozen + fc_frozen
    results["**总计**"] = {
        "total": grand_total,
        "trainable": grand_trainable,
        "frozen": grand_frozen,
        "formatted": format_params(grand_total),
    }

    # === 输出 ===
    print("\n" + "=" * 80)
    print("  CAFE 模型参数量分析")
    print("=" * 80)
    print(f"  {'组件':<38} {'总数':>10} {'可训练':>10} {'冻结':>10} {'占比':>8}")
    print("  " + "-" * 76)

    for name, stats in results.items():
        pct = stats["total"] / grand_total * 100 if grand_total > 0 else 0
        print(f"  {name:<38} {stats['formatted']:>10} "
              f"{format_params(stats['trainable']):>10} "
              f"{format_params(stats['frozen']):>10} "
              f"{pct:>7.1f}%")

    print("  " + "-" * 76)
    print(f"\n  总参数: {format_params(grand_total)}")
    print(f"  可训练: {format_params(grand_trainable)} ({grand_trainable/grand_total*100:.1f}%)")
    print(f"  冻结:   {format_params(grand_frozen)} ({grand_frozen/grand_total*100:.1f}%)")
    print("=" * 80)

    # 保存 JSON
    output_dir = os.path.join(os.path.dirname(__file__), "..", "..", "experiments", "results")
    os.makedirs(output_dir, exist_ok=True)
    json_path = os.path.join(output_dir, "param_analysis.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n结果已保存: {json_path}")

    return results


if __name__ == "__main__":
    analyze()
