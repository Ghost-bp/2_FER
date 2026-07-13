# CAFE 表情识别项目 — 学习路线指南

> 针对本项目所需知识点的系统性学习路径。你已经了解 RNN 和 Transformer 基础，本指南
> 帮你从当前水平过渡到能完全理解并修改这个项目。
>
> 📝 **学习进度追踪**: 每天学完后说 "记录学习"，自动生成 [LEARNING_LOG/](LEARNING_LOG/) 学习日志。

---

## 项目技术栈总览

```
人脸检测 (YOLOv8) → 特征提取 (ResNet-18 + CLIP) → 门控融合 (Sigmoid Gate)
→ 监督正则 (Mask+Supervisor) → 分类 (FC) → 评估 (LOSO交叉验证)
```

---

## 第一阶段：CNN 与 ResNet（2-3 天） ✅ **已完成 — Day 1**

> 📝 [学习记录](LEARNING_LOG/day1_resnet.md) — 2026-07-13

### 为什么必须学

项目中 `models.py` 的核心是 `BasicBlock` 和 `ResNet` 类——这是提取面部特征的骨干网络。
ResNet-18 的 4 个 layer 提取从低级纹理到高级语义的特征，最终 512 维向量代表一张脸。

### 本项目对应代码

- [models.py:42-100](CAFE/CAFE/models.py#L42-L100) — `BasicBlock`（残差块实现，`x += i` 就是残差连接）
- [models.py:103-140](CAFE/CAFE/models.py#L103-L140) — `ResNet`（4 层结构：[2,2,2,2] 配置）

### 知识点

1. 卷积层（Conv2d）、BatchNorm、ReLU、MaxPool 的作用
2. 残差连接：`x += identity` 为什么能解决深层网络退化
3. ResNet-18 vs ResNet-50 的结构差异（BasicBlock vs Bottleneck）
4. `AdaptiveAvgPool2d` 如何让任意尺寸输入得到固定尺寸输出

### 🎬 推荐视频

| 视频 | 链接 |
|------|------|
| 一口气搞懂 ResNet：残差思想 + 代码 | [BV1xdadzNECa](https://www.bilibili.com/video/BV1xdadzNECa/) |
| ResNet 残差神经网络硬核讲解（手撸代码） | [av1356262764](https://www.bilibili.com/video/av1356262764/) |
| 李沐《动手学深度学习》ResNet | B 站搜 "李沐 ResNet" |

---

## 第二阶段：迁移学习与预训练模型（1-2 天） ✅ **已完成 — Day 2**

> 📝 [学习记录](LEARNING_LOG/day2_transfer_learning.md) — 2026-07-14

### 为什么必须学

本项目同时使用了两个预训练模型：

- **ResNet-18** 在 MSCeleb（人脸识别数据集）上预训练，已经学会了"看脸"
- **CLIP ViT-B-32** 在 4 亿图文对上预训练，学会了通用的视觉语义

关键在于：**冻结 CLIP，只训练 ResNet 分支**。理解"什么时候冻结、什么时候微调"是这个项目的核心技能。

### 本项目对应代码

- [models.py:158-161](CAFE/CAFE/models.py#L158-L161) — `load_state_dict(msceleb_model['state_dict'], strict=False)`
- [CAFE_KMU-FED_train.py:16-17](CAFE/CAFE/CAFE_KMU-FED_train.py#L16-L17) — CLIP 在 `torch.no_grad()` 下调用

### 知识点

1. ImageNet 预训练 vs 领域专用预训练（MSCeleb 人脸 vs ImageNet 通用物体）
2. `load_state_dict(strict=False)` 的含义：只加载匹配的层，不匹配的跳过
3. 冻结策略：`with torch.no_grad()` vs `requires_grad=False`
4. Fine-tune vs Feature Extraction 的区别

### 🎬 推荐视频

| 视频 | 链接 |
|------|------|
| 同济子豪兄：迁移学习 Fine-tuning 实战（强烈推荐） | [BV1Ng411C7WY](https://www.bilibili.com/video/BV1Ng411C7WY/) |
| 迁移学习精讲 + PyTorch 实战（CV方向） | [BV1Kjtgz8EXT](https://www.bilibili.com/video/BV1Kjtgz8EXT/) |

---

## 第三阶段：CLIP 与多模态特征融合（2-3 天）

### 为什么必须学

这是本项目**最核心的创新点**。CAFE 的做法是：

1. CLIP 编码图像得到 512 维特征
2. ResNet 编码人脸得到 512 维特征
3. `sigmoid(ResNet特征)` 作为门控信号，逐元素乘以 CLIP 特征
4. 结果：ResNet 决定了 CLIP 的哪些维度"被激活"

理解这个门控机制是掌握整个项目的关键。

### 本项目对应代码

- [models.py:167-180](CAFE/CAFE/models.py#L167-L180) — `Model.forward()` 中的 `image_features * torch.sigmoid(x)`
- [models.py:192-200](CAFE/CAFE/models.py#L192-L200) — `supervisor()` 损失函数

### 知识点

1. CLIP 的双塔架构：图像编码器（ViT）+ 文本编码器（Transformer）
2. 对比学习原理：最大化图文对相似度，最小化非配对相似度
3. Sigmoid 门控的意义：输出 0~1 之间，实现"软选择"而非"硬开关"
4. 为什么用 CLIP 而不是直接分类：CLIP 的 512 维特征蕴含丰富的视觉语义

### 🎬 推荐视频

| 视频 | 链接 |
|------|------|
| CLIP 论文模型讲解（有字幕） | [BV1xM4m1m7vA](https://www.bilibili.com/video/BV1xM4m1m7vA/) |
| 多模态大模型：VIT/CLIP/BLIP 原理+实战 | [BV1ofSNYrEVM](https://www.bilibili.com/video/BV1ofSNYrEVM/) |
| 对比学习+多模态 2 小时精讲 | [BV1JuqABeEuX](https://www.bilibili.com/video/BV1JuqABeEuX/) |

---

## 第四阶段：损失函数设计与正则化（1-2 天）

### 为什么必须学

本项目的损失函数不是简单的 CrossEntropyLoss，而是三部分组成：

```
Total Loss = 1.0 × CE_Loss + 5.0 × Diversity_Loss + 1.5 × Masked_CE_Loss
```

理解每个部分的作用是调参和改进模型的基础。

### 本项目对应代码

- [CAFE_KMU-FED_train.py:142-146](CAFE/CAFE/CAFE_KMU-FED_train.py#L142-L146) — 损失组合
- [models.py:110-130](CAFE/CAFE/models.py#L110-L130) — `Mask()` 生成随机二值掩码
- [models.py:133-155](CAFE/CAFE/models.py#L133-L155) — `supervisor()` 两部分的计算

### 知识点

1. **CE Loss**（交叉熵）：标准的分类损失
2. **Masked CE Loss**（mc_loss[0]）：随机屏蔽 10/73 ≈ 14% 的特征维度后做分类——强迫模型不依赖少数"捷径"特征
3. **Diversity Loss**（mc_loss[1]）：惩罚特征集中在少数 chunk——鼓励 7 个 chunk 均匀贡献
4. **设计意图**：防止模型只靠"嘴巴形状"判断开心，强迫它同时利用眼睛、眉毛、额头等信息

### 🎬 推荐视频

| 视频 | 链接 |
|------|------|
| 损失函数详解（交叉熵、正则化） | B 站搜 "李沐 损失函数" 或 "交叉熵损失" |

---

## 第五阶段：交叉验证与评估协议（1 天）

### 为什么必须学

本项目的评估不是简单的 train/test split，而是 **Leave-One-Subject-Out (LOSO)**：

- 12 个受试者，每人轮流做一次验证集
- 这确保评估的是"模型对没见过的人的泛化能力"，而非"记住了某个人的脸"

### 本项目对应代码

- [CAFE_KMU-FED_train.py:264-280](CAFE/CAFE/CAFE_KMU-FED_train.py#L264-L280) — LOSO 划分逻辑
- [CAFE_KMU-FED_train.py:82-92](CAFE/CAFE/CAFE_KMU-FED_train.py#L82-L92) — 数据集按 subject_id 组织

### 知识点

1. K-Fold Cross-Validation：将数据分为 K 份，轮流做验证
2. LOSO（Leave-One-Subject-Out）：每人轮流做验证——适合受试者数量少的场景
3. 数据泄漏（Data Leakage）：为什么同一人的图片不能同时出现在训练和验证集
4. 为什么 KFold 在 12 人上不可靠：每折只有 1-2 人，一个人的表现决定整折准确率

### 🎬 推荐视频

| 视频 | 链接 |
|------|------|
| 交叉验证 Cross-Validation 详解 | [BV1GQ4y1P7Tv](https://www.bilibili.com/video/BV1GQ4y1P7Tv/) |

---

## 第六阶段：PyTorch 工程实践（持续学习）

### 知识点

1. `Dataset` / `DataLoader`：如何自定义数据加载（本项目用 `KMU_FED` 类）
2. `transform`：训练/验证使用不同的数据增强（不对称增强）
3. `SummaryWriter`（TensorBoard）：记录训练曲线
4. `argparse`：命令行参数管理
5. `torch.save` / `torch.load`：模型保存和加载

### 🎬 推荐视频

| 视频 | 链接 |
|------|------|
| PyTorch 零基础入门（含 ResNet 章节） | [BV1fTN3evEDX](https://www.bilibili.com/video/BV1fTN3evEDX/) |
| PyTorch 入门到精通 2025 版 | [BV1fqcvetEQD](https://www.bilibili.com/video/BV1fqcvetEQD/) |

---

## 学习顺序建议

```
第1周：CNN基础 → ResNet原理 → 手写BasicBlock代码
第2周：迁移学习 → 预训练模型加载 → 冻结/微调实验
第3周：CLIP原理 → 多模态融合 → 理解门控机制
第4周：损失函数 → 监督分支 → 调参实验
第5周：交叉验证 → LOSO → 评估指标
```

---

## 🎯 检验标准

完成学习后，你应该能回答以下问题：

1. `models.py` 第 175 行 `image_features * torch.sigmoid(x)` 中，sigmoid 输出范围是多少？如果 sigmoid 输出全为 1，模型退化成什么？
2. 为什么 `Mask()` 函数中每个 chunk 是 73 维而非 512/7≈73.14？最后一个 chunk 为什么是 74 维？
3. 如果把 LOSO 换成随机 80/20 划分，准确率会如何变化？为什么？
4. 如果 CLIP 不冻结，让它的参数也参与训练，会有什么后果？

---

## 📁 附加：表情识别专项

| 资源 | 说明 |
|------|------|
| 基于深度学习的表情识别（毕设项目） | B 站搜 "表情识别 PyTorch" |
| OpenCV+Python 人脸表情识别入门 | [BV1Sf421B7ZZ](https://www.bilibili.com/video/BV1Sf421B7ZZ/) |
| FER2013 表情识别实战 | [BV1Ba411p7V9](https://www.bilibili.com/video/BV1Ba411p7V9/) |
