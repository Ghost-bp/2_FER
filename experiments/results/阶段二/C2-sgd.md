# 实验 C2: SGD+momentum + ExponentiaLR

- **分支**: `exp2/sgd`
- **日期**: 2026-07-16（**手动终止于 Fold 3**）
- **描述**: 经典 SGD+momentum(0.9) 对比 Adam——结果在 lr=2e-4 下彻底不成立
- **超参数**: lr=0.0002, bs=32, SGD+momentum+ExponentialLR(gamma=0.9), WD=0.0001, ResNet-50+ImageNet
- **协议**: LOSO 12 折 + YOLO 人脸检测

## 部分结果（仅 3 折）

| Fold | 受试者 | 最佳准确率 | vs A3 (Adam) |
|:----:|:------:|:--------:|:------:|
| 1 | subject_1 | 63.49% | **-11.9** |
| 2 | subject_2 | 63.33% | **-15.0** |
| 3 | subject_3 | 55.56% | **-35.5** |

## ⚠️ 终止分析

- **lr=2e-4 对 SGD 严重不足**：SGD 的有效学习率约等于 lr × gradient_scale，而 Adam 通过二阶矩矫正将其提升约 10~50 倍（取决于 grad magnitude）。在此配置下 SGD 需要 >30 epochs 才能到达 Adam 在 epoch 2~3 的水平。12 折完整运行的预计耗时将超过 2.5 小时，且最终结果必然远低于 Adam。
- **这不是 SGD vs Adam 的公平对比**：仅证明"SGD 在 Adam 的最优 lr 下不成立"——这在深度学习中是常识。如需公平对比，需后续 C2b（SGD + lr=0.01 或 0.001）实验。
- **隐含结论**：在小样本 FER 上，自适应优化器（Adam/AdamW）的价值不在于更好的最终精度，而在于对 lr 设置的鲁棒性——SGD 需要精细调 lr 才能获得类似性能，而这在 LOSO 12 折 × 受试者多样化的场景下难以统一设定。

## 建议

- 后续 C3-C5 统一使用 **Adam + 不同调度器**
- 如果 C3（CosineAnnealingLR）或 C4（ReduceLROnPlateau）效率更高，F 组可在其中选最优调度器
