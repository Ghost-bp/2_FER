# models.py 代码导读

> 边看代码边对照本文，逐个理解每个类/方法的作用。
> 代码文件：[CAFE/CAFE/models.py](../CAFE/CAFE/models.py)（共 231 行）

---

## 整体结构速览

```
models.py
│
├── my_MaxPool2d      (L15-34)   自定义池化层 ──→ 被 supervisor() 调用
├── BasicBlock         (L38-78)   ResNet-18 残差块 ──→ 被 ResNet 调用
├── ResNet             (L81-127)  ResNet-18 特征提取器 ──→ 被 Model 拆成 features / features2
├── Mask()             (L131-151) 随机掩码生成 ──→ 被 supervisor() 调用
├── supervisor()       (L154-173) 监督分支损失 ──→ 被 Model.forward() 在训练时调用
└── Model              (L177-230) CAFE 完整模型 ──→ 被训练/推理脚本调用
```

---

## 一、my_MaxPool2d（L15-34）— 自定义池化层

### 作用

在**维度 1（通道维）**上做 max pooling，而不是空间维度。
PyTorch 自带的 `MaxPool2d` 只支持空间维度（H×W），这里用 `transpose` 绕过去。

### 为什么需要

`supervisor()` 需要把 512 维特征按 chunk 分组做池化，这要求在通道方向滑动。

### 数据流

```
输入: (B, 512, 1, 1)
  ↓ transpose(3,1)  → (B, 1, 1, 512)   // 把 512 维换到空间维
  ↓ max_pool2d       → (B, 1, 1, 7)     // 73维一组取最大值
  ↓ transpose(3,1)  → (B, 7, 1, 1)     // 换回来
输出: (B, 7, 1, 1)
```

### 构造参数

| 参数 | 含义 | 本项目用法 |
|------|------|-----------|
| `kernel_size` | 池化窗口大小 | `(1, 73)` — 在 73 维上做池化 |
| `stride` | 滑动步长 | `(1, 73)` — 不重叠，7 组各取 1 个最大值 |
| `padding/dilation` | 填充/膨胀 | 默认 0/1 |
| `return_indices` | 是否返回最大值位置 | `False` |
| `ceil_mode` | 输出尺寸向上取整 | `False` |

---

## 二、BasicBlock（L38-78）— 残差块

### 作用

ResNet-18 的基本构建块。实现公式：**输出 = ReLU( F(x) + identity )**

### 结构

```
输入 x (in_channels)
    │
    ├──→ Conv3×3 → BN → ReLU → Conv3×3 → BN ──→ F(x)
    │                                              │
    └──→ [1×1 下采样分支，仅当尺寸不匹配时] ──→ identity
                                                   │
                                            x = F(x) + identity
                                                   │
                                               ReLU → 输出
```

### 构造参数

| 参数 | 含义 | 第一个块示例 | 后续块示例 |
|------|------|-------------|-----------|
| `in_channels` | 输入通道数 | 64 | 64 |
| `out_channels` | 输出通道数 | 64 (layer1) / 128 (layer2) | 同左 |
| `stride` | 步长 | 2（layer2-4 第一个块下采样）| 1 |
| `downsample` | 是否需要 1×1 分支 | stride≠1 或通道数不匹配 | False |

### forward() 逐行解读

```python
i = x                          # L66: 保存 identity
x = self.conv1(x)              # L67: 3×3 卷积
x = self.bn1(x)                # L68: 批归一化
x = self.relu(x)               # L69: 激活
x = self.conv2(x)              # L70: 3×3 卷积
x = self.bn2(x)                # L71: 批归一化
if self.downsample is not None:
    i = self.downsample(i)     # L74: 尺寸不匹配时，1×1 卷积对齐
x += i                         # L76: ⭐ 残差连接（你 Day1 学的核心）
x = self.relu(x)               # L77: 最终激活
return x
```

### ⚠️ 注意：本项目用 `downsample` 参数而非自动判断

```
标准 ResNet 实现：stride≠1 或 in_channels≠out_channels 时自动添加 downsample
本项目实现：在 get_resnet_layer() 中手动计算后传入 downsample 参数
```

---

## 三、ResNet（L81-127）— 特征提取器

### 作用

ResNet-18 骨干网络，从人脸图片提取 512 维特征向量。

### 整体架构

```
输入 (B, 3, 224, 224)
  ↓ conv1:  7×7, stride=2, 64 通道  → (B, 64, 112, 112)
  ↓ bn1 + ReLU
  ↓ maxpool: 3×3, stride=2          → (B, 64, 56, 56)
  ↓ layer1: 2×BasicBlock, 64通道   → (B, 64, 56, 56)    ← 不改变尺寸
  ↓ layer2: 2×BasicBlock, 128通道  → (B, 128, 28, 28)  ← stride=2 下采样
  ↓ layer3: 2×BasicBlock, 256通道  → (B, 256, 14, 14)
  ↓ layer4: 2×BasicBlock, 512通道  → (B, 512, 7, 7)
  ↓ avgpool: AdaptiveAvgPool2d(1,1) → (B, 512, 1, 1)
  ↓ view 展平                      → (B, 512)           ← 特征向量 h
  ↓ fc: Linear(512→1000)            → (B, 1000)         ← 分类输出 x
输出: (x, h)
```

### forward() 返回值

| 返回值 | 形状 | 含义 |
|--------|------|------|
| `x` | `(B, 1000)` | 全连接层输出（1000 类，MSCeleb 原始任务） |
| `h` | `(B, 512)` | avgpool 后的特征向量 |

### get_resnet_layer() 方法（L102-113）

创建 1 个 layer（包含 N 个 BasicBlock）：

```python
# 示例: layer1 = get_resnet_layer(BasicBlock, 2, 64, stride=1)
#   → [BasicBlock(64→64, stride=1), BasicBlock(64→64, stride=1)]

# 示例: layer2 = get_resnet_layer(BasicBlock, 2, 128, stride=2)
#   → [BasicBlock(64→128, stride=2, downsample=True), BasicBlock(128→128, stride=1)]
#                                                    ↑ 第一个块做下采样
```

### ⚠️ 构造参数中的 `output_dim`

```python
ResNet(block=BasicBlock, n_blocks=[2,2,2,2], channels=[64,128,256,512], output_dim=1000)
```

- `output_dim=1000` 对应 MSCeleb 的 1000 个身份类别
- 在 CAFE 中，这个 1000 维输出**不会被用到**——Model 会把这层拆掉，只取 512 维特征

---

## 四、Mask()（L131-151）— 随机掩码生成

### 作用

生成形状为 `(nb_batch, 512, 1, 1)` 的二值掩码。每次调用随机打乱，确保每个 batch 屏蔽不同的位置。

### 为什么要 7 个 chunk

512 维特征分为 7 组，每组 73 维（最后一组 74 维，补偿 512÷7 的余数）：

- 每组都有 10 个位置被随机置 0
- 屏蔽率 = 10/73 ≈ 13.7%

### 数据构造过程

```
每组 (73维): [1]×63 + [0]×10  → shuffle → 共7组
最后一组:    [1]×64 + [0]×10  → shuffle
拼接: 512 维向量 (7×73=511维 + 补偿1维 = 512)
扩展: nb_batch 份 → (nb_batch, 512, 1, 1)
```

### 设计意图

```
如果不用 mask → 模型只靠"嘴巴形状"就能判断"开心"
用 mask 后   → 每次随机屏蔽 14% 的特征维度
              → 模型必须同时学会看眼睛、眉毛、额头等
              → 提升泛化能力
```

---

## 五、supervisor()（L154-173）— 监督分支损失

### 作用

计算两部分的监督损失，**作用在门控融合后的特征**（`image_features × sigmoid(ResNet特征)`）上。

### 函数签名

```python
def supervisor(x, targets, cnum=73, device='cuda'):
    # x:       (B, 512) — 门控融合后的特征
    # targets: (B,)     — 真实标签 (0~5，共6类)
    # cnum:    73       — 每个 chunk 的特征维度
    # 返回:    [loss_1, loss_2]
```

### loss_2：多样性正则（L165）

```python
branch = x.reshape(B, 512, 1, 1)              # → (B, 512, 1, 1)
branch = MaxPool2d(1,73) 在 512 维上滑窗       # → (B, 7, 1, 1)
branch = branch.reshape(B, 7)                  # → (B, 7)  7个chunk各取1个最大值
loss_2 = 1.0 - mean(sum(branch, dim=1)) / 73  # 鼓励每个 chunk 都有较高激活值
```

**直觉理解**：如果 7 个 chunk 中只有一个 chunk 有高激活值，`sum(branch)` 就小，`loss_2` 就大。
模型被迫让所有 7 个 chunk 都做出贡献。

### loss_1：掩码后分类损失（L167-171）

```python
mask = Mask(B)                                  # 随机掩码
branch_1 = x * mask                             # 屏蔽部分维度
branch_1 = MaxPool2d(1,73) → reshape(B, 7)     # 7个chunk各取1个最大值
loss_1 = CrossEntropyLoss(branch_1, targets)    # 用7维特征做分类
```

**直觉理解**：即使用了随机 mask，模型依然能正确分类——说明它不依赖少数"捷径"特征。

### 实际调用（L219-222）

```python
MC_loss = supervisor(
    image_features * torch.sigmoid(x),   # ← 注意：门控融合后的特征，不是原始 ResNet 特征
    targets,
    cnum=73,
    device=self.device
)
```

---

## 六、Model（L177-230）— CAFE 完整模型 ⭐

### 作用

整合所有组件的完整模型，是训练和推理的统一入口。

### **init**() 逐行解读（L189-207）

```python
# L192: 保存传入的 CLIP 模型（外部加载好传进来，不在内部加载）
self.clip_model = clip_model

# L195-196: 创建 ResNet-18（1000 类输出，对应 MSCeleb 预训练）
res18 = ResNet(BasicBlock, [2,2,2,2], [64,128,256,512], output_dim=1000)

# L198-201: 加载 MSCeleb 人脸预训练权重
# ① torch.load 读取 .pth 文件 → ② strict=False 只加载匹配的层(跳过1000类FC)
res18.load_state_dict(msceleb_model['state_dict'], strict=False)

# L203: 拆出 features = ResNet 去掉最后两层 (fc + avgpool)
# res18.children() = [conv1, bn1, relu, maxpool, layer1-4, avgpool, fc]
# [:-2] 取前 11 个 → 包含 conv1~layer4
self.features = nn.Sequential(*list(res18.children())[:-2])

# L204: 拆出 features2 = avgpool（第 12 个）
self.features2 = nn.Sequential(*list(res18.children())[-2:-1])

# L206-207: 新的分类头：512 → 6（KMU-FED 的 6 类表情）
fc_in_dim = list(res18.children())[-1].in_features  # = 512
self.fc = nn.Linear(512, num_classes)
```

### ResNet 拆分示意图

```
原始 res18 (1000类，MSCeleb预训练):
┌──────────────────────────────────────────────────────┐
│ conv1 → bn1 → relu → maxpool → layer1 → layer2       │
│ → layer3 → layer4 → avgpool → fc(512→1000)           │
└──────┬───────────────┬──────────┬────────────────────┘
       │               │          │
       └─ self.features ─┘        │
                      └─ self.features2 (=avgpool)
                                  └─ self.fc (新的 512→6)
```

### forward() 核心数据流（L209-230）

```
输入 x: (B, 3, 224, 224)
│
├─→ CLIP 编码（冻结，不计算梯度）         L210-211
│   image_features = clip_model.encode_image(x)
│   输出: (B, 512)
│
├─→ ResNet 特征提取（参与训练）           L213-215
│   x = self.features(x)        → (B, 512, 7, 7)
│   x = self.features2(x)       → (B, 512, 1, 1)   ← avgpool
│   x = x.view(B, 512)          → (B, 512)          ← 展平
│
├─→ 监督分支（仅训练模式）               L218-222
│   MC_loss = supervisor(image_features * sigmoid(x), targets)
│   输出: [loss_1, loss_2]
│
├─→ 门控融合 ⭐                          L224
│   x = image_features * torch.sigmoid(x)
│   输出: (B, 512)
│   ↑ CLIP语义  ↑ 0~1之间的门控信号
│
└─→ 分类头                               L225
    out = self.fc(x)            → (B, 6) 或 (B, 7)
```

### ⭐ 门控融合为什么用 sigmoid

```
sigmoid(x) 输出范围 (0, 1)，而非 ReLU 的 [0, ∞) 或 tanh 的 (-1, 1)

好处:
  - 输出有上限(1)，防止某一维绝对值过大主导整个向量
  - 软选择而非硬开关：每个维度被"部分激活"
  - 如果输出全 1 → 模型退化为"直接用 CLIP 特征分类"
  - 如果输出全 0 → CLIP 特征完全不贡献

实际上 sigmoid 的值在 0.3~0.7 之间均匀分布，
表示 ResNet 在"有选择地"激活 CLIP 的不同语义维度。
```

---

## 完整数据流总图

```
                              ┌──────────────────┐
                              │   输入图片        │
                              │ (B, 3, 224, 224)  │
                              └──────┬───────────┘
                                     │
                    ┌────────────────┼────────────────┐
                    │                                   │
                    ▼                                   ▼
        ┌──────────────────┐              ┌──────────────────┐
        │  CLIP ViT-B-32   │              │  ResNet-18       │
        │  (冻结，不训练)   │              │  (MSCeleb预训练)  │
        └────────┬─────────┘              └────────┬─────────┘
                 │                                 │
                 ▼                                 ▼
        image_features                         x (512维)
        (B, 512)                               │
                 │                                 │
                 │                          sigmoid(x)
                 │                          (B, 512, 0~1)
                 │                                 │
                 └──────────┬──────────────────────┘
                            │
                            ▼
                 ┌──────────────────┐
                 │  门控融合         │
                 │  output =         │
                 │  image_features   │
                 │  * sigmoid(x)     │
                 └────────┬─────────┘
                          │
                ┌─────────┴─────────┐
                │                   │
         (训练模式)           (验证/推理模式)
                │                   │
                ▼                   ▼
        ┌──────────────┐    ┌──────────────┐
        │ supervisor() │    │  fc(512→6)   │
        │ 返回:         │    │ → 直接分类    │
        │ [loss_1,      │    └──────────────┘
        │  loss_2]      │
        └──────┬────────┘
               │
               ▼
        ┌──────────────┐
        │  fc(512→6)   │
        │ + CE_Loss    │
        └──────────────┘

训练总损失 = 1.0×CE_Loss + 5.0×loss_2 + 1.5×loss_1
```

---

## 阅读顺序建议

```
Step 1: BasicBlock.forward()  (L65-78)   — 6行，你手写过，快速过
Step 2: ResNet.forward()      (L115-127) — 11行，理解每层的维度变化
Step 3: Model.__init__()      (L189-207) — 理解 ResNet 怎么被拆成三段
Step 4: Model.forward()       (L209-230) — ⭐ 最核心，逐行对照上面数据流
Step 5: supervisor()          (L154-173) — 理解两个 loss 的含义
Step 6: Mask()                (L131-151) — 被 supervisor 调用时才需要看
Step 7: my_MaxPool2d          (L15-34)   — 工具层，最后看
```

对照代码阅读时，遇到不懂的直接问我。
