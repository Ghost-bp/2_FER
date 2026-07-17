# 🏆 实验 F1: AdamW + Cosine + Geometric（最终最优组合）

- **分支**: `exp2/f1-best-combo`
- **日期**: 2026-07-17（完成于 ~08:21，总耗时 94.6 分钟）
- **描述**: Phase 2 所有消融实验的最优因素组合：AdamW (C5最优优化器) + Cosine (C组最优调度器) + Geometric (E组最优增强)
- **超参数**: lr=2e-4, bs=32, AdamW+CosineAnnealingLR, WD=1e-4, ResNet-50+ImageNet, **aug_geometric=on**
- **协议**: LOSO 12 折 + **YOLO-only** 人脸检测

## 12 折结果

| Fold | 受试者 | 最佳准确率 | vs C5 (无增强) | 判定 |
|:----:|:------:|:--------:|:------:|:----:|
| 1 | subject_1 | 89.68% | **+21.4** | ✅✅✅ |
| 2 | subject_2 | 83.33% | +1.7 | ✅ |
| 3 | subject_3 | 97.78% | +8.9 | ✅✅ |
| 4 | subject_4 | **100.00%** 🔥 | **+16.7** | ✅✅✅ |
| 5 | subject_5 | **99.17%** 🔥 | +9.2 | ✅✅ |
| 6 | subject_6 | 89.17% | **+16.7** | ✅✅✅ |
| 7 | subject_7 | 93.00% | **+10.0** | ✅✅ |
| 8 | subject_8 | **100.00%** 🔥 | **+15.5** | ✅✅✅ |
| 9 | subject_9 | 92.86% | +7.1 | ✅✅ |
| 10 | subject_10 | 87.50% | +5.0 | ✅ |
| 11 | subject_11 | 75.00% | 0.0 | ≈ |
| 12 | subject_12 | 64.29% | -1.4 | ≈ |
| **平均** | — | **89.31% ± 10.38%** 🏆 | **+9.22** | |

## 分析

- **89.31% 是 Phase 2 最终最优结果！** 比 C5 (80.09%) 高 **9.22pp**，比 Phase 1 基线 (71.94%) 高 **17.37pp**
- **S4 和 S8 双双满分（100%）**：两位受试者都拥有全部 6 类表情，AdamW+Cos+Geometric 在特征完整的受试者上做到完美识别
- **5 折超过 90%**（S3 97.78%、S5 99.17%、S7 93.00%、S4/S8 100%）
- **关键洞察：Geometric × YOLO 的协同效应**：
  - E2 (Haar-first): Geometric +1.18pp vs C3
  - F1 (YOLO-only): Geometric **+9.22pp** vs C5
  - 解释：YOLO 的 wider crop 保留了更多空间上下文（头部姿态、肩膀），Geometric 增强在 wider crop 上训练出真正的姿态不变性——不是"忽略姿态"而是"学会在任意姿态下找到表情特征"。Haar 的 tight crop 已经裁掉了大部分姿态信息，Geometric 在 tight crop 上的收益天然受限
- **S11/S12 仍是瓶颈**（75%/64%），但它们缺少关键表情类别——这不是模型能力问题而是数据约束

## Phase 2 完整排行榜

| 排名 | 实验 | 配置 | 准确率 | 标准差 | 协议 |
|:---:|------|------|:------:|:------:|:--:|
| 🥇 | **F1** | RN50+AdamW+Cos+Geo | **89.31%** | 10.38% | YOLO-only |
| 🥈 | C5 | RN50+AdamW+Cos | 80.09% | **7.55%** | YOLO-only |
| 🥉 | D2 | RN50+AdamW+Cos+Drop05 | 79.18% | 11.78% | YOLO-only |
| 4 | C2b | RN50+SGD+Cos | 77.78% | 11.71% | YOLO-only |
| 5 | D4 | RN50+AdamW+Cos+WD1e-3 | 77.10% | 14.72% | YOLO-only |
| — | E3 | +Geo+Color | 24.78% | — | 💀 崩了 |

## 结论

**KMU-FED 6 类表情识别最优配置**：
- Backbone: ResNet-50 (ImageNet 预训练)
- 优化器: AdamW (lr=2e-4, WD=1e-4)
- 调度器: CosineAnnealingLR (T_max=60)
- 增强: RandomHorizontalFlip + RandomErasing + **RandomRotation(±15°) + RandomAffine(±10%)**
- 人脸检测: **YOLO-only** (不加载 Haar Cascade)
- 协议: LOSO 12 折
- **无 Dropout、无 LabelSmoothing、无 ColorJitter、无 CLIP**
