# 实验 C5: AdamW + CosineAnnealingLR

- **分支**: `exp2/adamw-cosine`
- **日期**: 2026-07-17（完成于 01:28，总耗时 88.0 分钟）
- **描述**: AdamW 的 decoupled weight decay 配合 Cosine 长高 lr 窗口——C1 证明 AdamW+Exp 惨败(-6.11pp)，但 Cosine 可能给 WD 足够时间生效
- **超参数**: lr=2e-4, bs=32, AdamW+CosineAnnealingLR, WD=1e-4, ResNet-50+ImageNet
- **协议**: LOSO 12 折 + **YOLO-only** 人脸检测 ⚠️ (新协议，不加载 Haar)

## 12 折结果

| Fold | 受试者 | 最佳准确率 |
|:----:|:------:|:--------:|
| 1 | subject_1 | 68.25% |
| 2 | subject_2 | 81.67% |
| 3 | subject_3 | 88.89% |
| 4 | subject_4 | 83.33% |
| 5 | subject_5 | 90.00% |
| 6 | subject_6 | 72.50% |
| 7 | subject_7 | 83.00% |
| 8 | subject_8 | 84.55% |
| 9 | subject_9 | 85.71% |
| 10 | subject_10 | 82.50% |
| 11 | subject_11 | 75.00% |
| 12 | subject_12 | 65.71% |
| **平均** | — | **80.09% ± 7.55%** 🏆 |

## 分析

- **80.09% ± 7.55% 为 Phase 2 新纪录！** 首次突破 80% 大关，同时标准差 7.55% 是所有实验中最低——均值最高 + 最稳定，双重突破
- **AdamW + Cosine = 绝配**：C1 (AdamW+Exp) = 70.81%，C5 (AdamW+Cos) = 80.09%。Cosine 的平滑高 lr 窗口让 AdamW 的 decoupled weight decay 有足够时间逐步正则化——而 Exponential 的快速 lr 衰减让 WD 还没来得及生效 lr 就没了
- **所有折都在 65%+**，没有任何一折崩溃——这是之前所有实验都没做到的（E2 的 S11=61%, S10=65%）
- **S10 从 65% 跃升到 82.50%（+17.5pp vs E2）**：该受试者缺少 FE 表情，AdamW 的更强正则化帮助了跨类别泛化
- **⚠️ 协议差异**：C5 是首个 YOLO-only 实验。与之前 Haar-first 实验不可直接比较。部分提升可能来自 YOLO wider crop

## 全部实验排行

| 排名 | 实验 | 配置 | 准确率 | 标准差 | 协议 |
|:---:|------|------|:------:|:------:|:--:|
| 🥇 | **C5** | RN50+AdamW+Cos | **80.09%** | **7.55%** ⭐ | YOLO-only |
| 🥈 | E2 | RN50+Cos+Geometric | 79.62% | 8.62% | Haar-first |
| 🥉 | C3 | RN50+Cos | 78.44% | 9.55% | Haar-first |
| 4 | A3 | RN50+Exp | 76.92% | 11.86% | Haar-first |
| 5 | D1 | RN50+Cos+Drop03 | 73.68% | 9.48% | Haar-first |
