# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目概述

面部表情识别（FER）研究项目，在 **KMU-FED** 自定义数据集上进行实验。7 类表情：愤怒(AN)、厌恶(DI)、恐惧(FE)、开心(HA)、悲伤(SA)、惊讶(SU)、中性(NE)。

### CAFE — CLIP 辅助面部表情识别

核心方法位于 `CAFE/CAFE/`，架构如下：

1. **ResNet-18** 骨干网络（MSCeleb 人脸识别预训练）提取面部特征
2. **CLIP ViT-B-32** 并行编码图像（冻结权重，不参与训练）
3. **Sigmoid 门控机制**：ResNet 特征经过 sigmoid 后与 CLIP 特征逐元素相乘，选择性激活与表情相关的 CLIP 维度
4. **监督分支（Supervisor）**：通过随机掩码机制，强制模型利用多样化的特征维度，避免过度依赖少数几个容易区分的特征
5. **YOLOv8n** 在数据加载阶段检测人脸（检测失败时回退使用整张图片）

训练采用 **按受试者划分的 10 折交叉验证**（受试者独立评估协议）。

### KMU-FED 数据集

图片命名规则：`{受试者ID}_{表情代码}_{会话}_{序号}.jpg`，例如 `03_FE_s02_054.jpg`。数据位于 `KMU-FED/KMU-FED/`。

## 关键文件

| 文件 | 用途 |
| ---- | ---- |
| `CAFE/CAFE/CAFE_KMU-FED_train.py` | 训练脚本：数据集类、10 折交叉验证主循环、TensorBoard 日志 |
| `CAFE/CAFE/video.py` | 视频推理：人脸检测 + 逐帧表情识别 + 耗时统计 |
| `CAFE/CAFE/generate_video.py` | 工具脚本：将指定受试者的图片合成为 mp4 视频 |
| `CAFE/CAFE/models.py` | 共享模型定义（BasicBlock、ResNet、Model、Mask、supervisor） |
| `CAFE/CAFE/config.py` | 统一配置管理（路径、超参数，支持环境变量覆盖） |
| `CAFE/CAFE/clip/clip.py` | OpenAI CLIP 加载器（模型下载、预处理、分词） |
| `CAFE/CAFE/clip/model.py` | CLIP 模型架构：ViT、ModifiedResNet、Transformer、注意力池化 |
| `CAFE/CAFE/clip/simple_tokenizer.py` | CLIP 文本分词器（BPE 编码） |
| `requirements.txt` | 项目 pip 依赖 |

## Agent 系统

| Agent | 文件 | 用途 |
| ----- | ---- | ---- |
| training-logger | `.claude/agents/training-logger.md` | 读取训练指标 JSON，整理为结构化训练记录 |
| training-analyzer | `.claude/agents/training-analyzer.md` | 分析训练数据，生成可视化图表和分析报告 |

用法：训练完成后，在 Claude Code 中输入"**分析训练结果**"即可自动调用两个 Agent 生成完整分析报告。

## 必需的模型文件（不在仓库中）

以下文件需手动下载/放置：

- `CAFE/CAFE/clip/ViT-B-32.pt` — CLIP 视觉模型权重
- `CAFE/CAFE/clip/resnet18_msceleb.pth` — MSCeleb 人脸数据集预训练的 ResNet-18
- `face_yolov8n.pt` — YOLOv8n 人脸检测模型（代码中引用路径：`/home/chenruimin/chenruimin/mini_Xception-main/face_yolov8n.pt`）

## 环境配置

所有路径优先使用环境变量，找不到时使用 `config.py` 中的默认值：

| 环境变量 | 说明 | 默认值 |
| -------- | ---- | ------ |
| `KMU_FED_ROOT` | 数据集目录 | `../../KMU-FED/KMU-FED` |
| `YOLO_MODEL_PATH` | YOLO 人脸检测模型 | `../../face_yolov8n.pt` |
| `CLIP_MODEL_PATH` | CLIP 权重 | `clip/ViT-B-32.pt` |
| `OUTPUT_DIR` | 训练输出目录 | `../../KMU-FED/output_kmu_fed_clip` |
| `TENSORBOARD_DIR` | TensorBoard 日志 | `<OUTPUT_DIR>/tensorboard` |

## 运行训练

```bash
pip install -r requirements.txt
cd CAFE/CAFE

# 默认配置
python CAFE_KMU-FED_train.py

# 自定义配置
python CAFE_KMU-FED_train.py \
    --data_dir ../../KMU-FED/KMU-FED \
    --yolo ../../face_yolov8n.pt \
    --epochs 60 --batch_size 32 --lr 0.0002

# CPU 训练（低配机器）
python CAFE_KMU-FED_train.py --device cpu --batch_size 8

# 监控训练
tensorboard --logdir ../../KMU-FED/output_kmu_fed_clip/tensorboard
```

关键超参数：

- `num_classes = 7`、`batch_size = 32`、`lr = 0.0002`、`num_epochs = 60`
- 损失函数：`1.0 × CrossEntropyLoss + 5.0 × mc_loss[1] + 1.5 × mc_loss[0]`
- `mc_loss[0]`：掩码后特征的交叉熵损失（鼓励单个 chunk 内的判别能力）
- `mc_loss[1]`：特征多样性正则项（鼓励各 chunk 均匀贡献）

## 运行视频推理

```bash
cd CAFE/CAFE

# 默认配置
python video.py

# 自定义输入输出
python video.py --input my_video.mp4 --output result.mp4 --weights ./fold_8_best.pth
```

## 重要架构细节

- **监督损失（Supervisor Loss）**：两部分的复合损失作用于 `image_features * sigmoid(x)`，而非原始特征。`Mask` 函数生成随机二值掩码（每个 73 维 chunk 中 63 个 1 + 10 个 0，共 7 个 chunk 对应 512 维），每个 batch 随机打乱。
- **受试者独立划分**：`KFold` 基于排序后的唯一受试者 ID 进行划分，而非随机打乱图片索引。这是为了防止同一人的不同图片同时出现在训练集和验证集中（数据泄漏）。
- **训练/验证数据增强不对称**：训练使用 `RandomHorizontalFlip` + `RandomErasing`；验证仅使用 Resize + Normalize。
- **设备选择**：自动检测 CUDA 可用性，不可用时静默回退到 CPU。
- **CLIP 特征冻结**：`clip_model.encode_image(x)` 在 `torch.no_grad()` 下调用，CLIP 在整个训练过程中不更新参数。
- **代码组织**：所有共享模型定义在 `models.py`，所有路径/超参数在 `config.py`（支持环境变量和 CLI 参数双层覆盖）。训练脚本内置早停机制（`--patience 10`）。
