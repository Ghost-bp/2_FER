# 实验 E1: ColorJitter 增强（手动终止）

- **分支**: `exp2/aug-color`
- **日期**: 2026-07-16（**手动终止于 Fold 2**）
- **描述**: ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5) 颜色增强
- **终止原因**: Fold 1/2 仅 15.87%/16.67%——低于随机水平（6类=16.67%）。默认参数过于激进，同一张人脸在不同 epoch 颜色剧烈变化，模型无法学习一致的面部表情特征。
- **建议**: 后续如需测试颜色增强，使用 milder 参数（brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1）并设为 CLI 可选项。
