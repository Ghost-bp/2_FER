# Day 1: ResNet 残差网络

**日期**: 2026-07-13
**学习时长**: ~6 小时
**状态**: ✅ 完成

## 学习资源

| 类型 | 内容 |
|------|------|
| 📹 视频 | 李沐《动手学深度学习》ResNet 章节 |
| 📹 视频 | [一口气搞懂 ResNet（B站）](https://www.bilibili.com/video/BV1xdadzNECa/) |
| 📹 视频 | [ResNet 手撸代码（B站）](https://www.bilibili.com/video/BV1Si421h7YW/) |
| 📄 论文 | *Deep Residual Learning for Image Recognition* (He et al., 2015) |
| 💻 代码 | `D:\手搓残差神经网络\PyTorch_nn\model\resnet.py` |

## 核心知识点

### 1. 残差连接（Residual Connection）— 核心创新

```python
# models.py 第 60 行: 这就是残差连接的全部
out += identity  # x += i
```

- **问题**: 网络越深，训练误差反而上升（不是过拟合，是优化困难）
- **方案**: 学习残差映射 F(x) 而非直接学习目标映射 H(x)，H(x) = F(x) + x
- **为什么有效**: 反向传播时加法不衰减梯度，深层也能收到有效梯度信号

### 2. BasicBlock — ResNet-18/34 的残差块

```python
# expansion = 1: 输入输出通道数不变
# 结构: Conv3×3 → BN → ReLU → Conv3×3 → BN → +skip → ReLU
```

- 两个 3×3 卷积，通道数不变
- 当 stride≠1 或尺寸不匹配时，用 1×1 卷积分支做下采样

### 3. Bottleneck — ResNet-50/101/152 的残差块

```python
# expansion = 4: 输出通道 = 输入通道 × 4
# 结构: Conv1×1(压缩) → Conv3×3 → Conv1×1(恢复×4) → +skip → ReLU
```

- 1×1 卷积先压缩通道数（减少计算量），3×3 提取特征，1×1 恢复并扩展通道
- 参数量远小于两个 3×3 卷积，适合深层网络

### 4. ResNet 主框架 & _make_layer

```python
# 创建 4 个 stage，每个 stage 包含 blocks_num[i] 个残差块
self.layer1 = self._make_layer(block, 64,  blocks_num[0])   # 56×56
self.layer2 = self._make_layer(block, 128, blocks_num[1])   # 28×28 (stride=2)
self.layer3 = self._make_layer(block, 256, blocks_num[2])   # 14×14
self.layer4 = self._make_layer(block, 512, blocks_num[3])   # 7×7
```

### 5. ResNet 五个版本

| 模型 | Block | blocks_num | Params | 适用场景 |
|------|-------|------------|--------|----------|
| **ResNet-18** | BasicBlock | [2,2,2,2] | 11.7M | **本项目使用** |
| ResNet-34 | BasicBlock | [3,4,6,3] | 21.8M | 中等任务 |
| ResNet-50 | Bottleneck | [3,4,6,3] | 25.6M | 迁移学习首选 |
| ResNet-101 | Bottleneck | [3,4,23,3] | 44.5M | 高精度 |
| ResNet-152 | Bottleneck | [3,8,36,3] | 60.2M | 极致精度 |

## 可视化

![ResNet Architecture](images/resnet_architecture.png)

## 与 CAFE 项目的关联

你写的 `resnet.py` 和项目中 `CAFE/CAFE/models.py` 的 ResNet 部分**几乎完全一致**：

| 你的代码 | 项目代码 | 对应关系 |
|----------|----------|----------|
| `BasicBlock` (第4行) | `BasicBlock` (models.py:42) | ✅ 完全对应 |
| `Bottleneck` (第32行) | 项目用 ResNet-18，不需要 Bottleneck | 🔗 了解即可 |
| `ResNet.__init__` (第66行) | `ResNet.__init__` (models.py:103) | ✅ 结构一致 |
| `_make_layer` (第92行) | `get_resnet_layer` (models.py:154) | ✅ 逻辑相同 |
| `resnet18()` (第143行) | `Model.__init__` (models.py:147) | ✅ 直接调用 |

> 项目中用的是 MSCeleb 人脸预训练的 ResNet-18，不是 ImageNet 预训练。
> `load_state_dict(state_dict, strict=False)` 只加载匹配的层，最后一层 1000→7(6) 类会被跳过。

## 收获

- ✅ 理解了残差连接为什么能解决"深层网络退化"问题
- ✅ 能区分 BasicBlock（轻量）和 Bottleneck（深层优化）
- ✅ 知道了 `[2,2,2,2]` 等 blocks_num 的含义
- ✅ 能看懂 `_make_layer` 的逻辑：第一个块处理下采样，后面全是实线残差
- ⚠️ 还有点懵：Kaiming 初始化的原理、不同版本 FLOPs 的计算

## 下一步

**Day 2: 迁移学习与预训练模型**
- 学习 `load_state_dict(strict=False)` 的含义
- 理解 "冻结 CLIP，只训练 ResNet" 的策略
- 视频：同济子豪兄 迁移学习 Fine-tuning 实战 [BV1Ng411C7WY](https://www.bilibili.com/video/BV1Ng411C7WY/)
