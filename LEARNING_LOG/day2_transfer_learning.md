# Day 2: 迁移学习与预训练模型

**日期**: 2026-07-14
**学习时长**: ~4 小时
**状态**: ✅ 完成（理解概念，等看项目代码时深化）

## 学习资源

| 类型 | 内容 |
|------|------|
| 📹 视频 | 李沐《动手学深度学习》— 迁移学习 / Fine-tuning 章节 |
| 📹 视频 | [同济子豪兄：迁移学习 Fine-tuning 实战](https://www.bilibili.com/video/BV1Ng411C7WY/) |
| 💡 参考 | CAFE 项目 `CAFE/CAFE/models.py` — `Model.__init__` 预训练权重加载 |

## 核心知识点

### 1. 什么是迁移学习

```
大模型 (ImageNet预训练)  →  你的任务 (表情识别)
     ↓                           ↓
  1000类分类能力             6类表情分类能力
     ↓                           ↓
  冻结底层 → 只替换最后一层 → 微调
```

- **预训练（Pretraining）**: 在大数据集上先训练好模型
- **微调（Fine-tuning）**: 把预训练模型适配到你的具体任务
- **特征提取（Feature Extraction）**: 冻结预训练模型，只训练新的分类头

### 2. 两个关键操作

```python
# 操作1: 加载预训练权重，跳过不匹配的层
res18.load_state_dict(msceleb_model['state_dict'], strict=False)
# strict=False 的含义:
#   - 匹配的层 → 加载预训练权重 ✅
#   - 不匹配的层（如最后的 FC 1000→6）→ 跳过 ✅
#   - 新层随机初始化

# 操作2: 冻结 CLIP，不让它更新
with torch.no_grad():                    # 方式A: 不记录梯度（更快）
    image_features = clip_model.encode_image(x)

# 或
clip_model.requires_grad_(False)          # 方式B: 显式冻结
```

### 3. CAFE 项目的特殊之处：双重迁移学习

CAFE 同时用了两个预训练模型，这是它的核心创新：

| 预训练模型 | 原始任务 | 预训练数据 | 在 CAFE 中的角色 |
|-----------|----------|-----------|-----------------|
| **ResNet-18** | 人脸识别 | MSCeleb（名人脸） | 提取面部表情特征 → **参与训练** |
| **CLIP ViT-B-32** | 图文匹配 | 4亿图文对 | 提供通用视觉语义 → **冻结不动** |

```python
# models.py Model.forward() 核心逻辑:
image_features = clip_model.encode_image(x)    # CLIP: 冻结，不训练
x = self.features(x)                            # ResNet: 参与训练
x = image_features * torch.sigmoid(x)           # 门控融合
out = self.fc(x)                                # 分类
```

### 4. Fine-tune 的三种策略

| 策略 | 做什么 | 适用场景 |
|------|--------|----------|
| **全量微调** | 所有层都训练 | 大数据量 + 目标域差异大 |
| **部分微调** | 冻结底层，只训顶层 | 小数据量 + 底层特征通用 |
| **线性探针** | 冻结全部，只加分类头 | 极小数据量 + 快速实验 |

> CAFE 用的是混合策略：CLIP 冻结（线性探针）+ ResNet 全量微调。

## 与 CAFE 项目的关联

```
你学的迁移学习概念         →    CAFE 项目代码位置
─────────────────────────────────────────────────
预训练权重加载              →  models.py:158-161  load_state_dict(strict=False)
冻结策略 (no_grad)          →  models.py:169      with torch.no_grad()
冻结 vs 微调的选择          →  CLIP冻结 / ResNet训练
替换分类头                  →  models.py:173      fc = nn.Linear(512, 6)
```

## 收获

- ✅ 明白了迁移学习的本质：借用一个"见过世面"的模型，适配到自己的任务
- ✅ 理解了 `strict=False` 的含义——允许部分加载
- ✅ 知道了 CAFE 为什么冻结 CLIP 但训练 ResNet
- ⚠️ 还不熟：具体什么时候用哪种微调策略、学习率怎么设

## 疑问（等看项目源码时解决）

1. MSCeleb 预训练的 ResNet 和 ImageNet 预训练的有什么区别？
2. 为什么冻结 CLIP 而不是冻结 ResNet？反过来行不行？
3. `with torch.no_grad()` 和 `requires_grad=False` 效果一样吗？

## 下一步

**Day 3: CLIP 与多模态特征融合**
- 理解 CLIP 的双塔架构（图像编码器 + 文本编码器）
- 理解对比学习原理
- 重点是 `sigmoid(ResNet特征) × CLIP特征` 这个门控机制
- 视频: [CLIP 论文模型讲解](https://www.bilibili.com/video/BV1xM4m1m7vA/)
