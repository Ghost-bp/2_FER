# 项目修改记录

**修改日期**: 2026-07-10 (第一/二轮), 2026-07-11 (第三轮)
**修改范围**: 全项目重构 + 数据问题修复 + 评估协议改进 + 人脸检测修复

---

## 第三轮修改: 人脸检测修复 & 数据质量分析（2026-07-11）

### 🐛 关键发现: YOLO person 检测 ≠ 人脸检测

对比测试发现 YOLOv8n 检测 "person"（全身）:
- YOLO person 裁剪: 1541×1194 px（占画面 **95.8%** — 几乎等于整张图）
- Haar Cascade 人脸裁剪: 115×115 px（占画面 **0.7%** — 真正的面部区域）

这意味着模型一直在学习**衣服、身体姿态、背景**，而非面部表情。

### 修复: 多级人脸检测回退

在 `KMU_FED.__getitem__` 中实现三级回退策略:
```
级别1: Haar Cascade 精确人脸检测（优先，90% 命中率）
  ↓ 失败
级别2: YOLO person 检测（备选）
  ↓ 失败
级别3: 整张图片（最终回退）
```

Haar Cascade XML 文件下载到项目根目录 `haarcascade_frontalface_default.xml`（909KB）。

### 数据集质量报告

生成 `KMU-FED/DATASET_REPORT.md`，包含:
- 每受试者每表情矩阵（12×6）
- 缺失表情标注
- Subject 7 100% 异常标记
- 改进建议

### Subject 7 排查结果

- 无重复图片（所有文件大小唯一）
- 图片尺寸一致（1600×1200）
- 缺 DI（disgust）类，仅 5 类
- 100%（100/100 正确）仍然异常高，需人工复查

### 文件变更

| 操作 | 文件 | 说明 |
|------|------|------|
| 修改 | `CAFE/CAFE/CAFE_KMU-FED_train.py` | 人脸检测: Haar Cascade + YOLO 三级回退 |
| 新增 | `haarcascade_frontalface_default.xml` | OpenCV Haar 人脸检测模型（909KB） |
| 新增 | `KMU-FED/DATASET_REPORT.md` | 数据集质量报告 |
| 新增 | `LEARNING_GUIDE.md` | 6 阶段学习路线（含B站视频链接） |
| 安装 | `opencv-contrib-python` | 替换基础 opencv-python 以支持完整功能 |

---

## 第二轮修改：数据问题修复 & 评估协议改进（2026-07-10 下午）

### 🐛 Bug 发现：首次训练分析

对首次训练的深入分析发现 3 个关键问题：

**Bug 1 — NE（中性）类别无数据**：
`EMOTION_MAP` 和 `EMOTION_LABELS` 包含 `"NE": 6`（neutral/中性），但 KMU-FED 数据集中该类别有 **0 张图片**。模型 `num_classes=7` 的第 7 个输出神经元永远接收不到正向训练信号，浪费模型容量，且可能干扰其他类别的判别。

**Bug 2 — 10-Fold CV 在小数据集上的不可靠性**：
12 个受试者用 10 折 CV 意味着每折只有 1-2 人做验证。一个人的表现决定整折准确率。首次训练结果中出现：
- Fold 6: 31.4%（该受试者表情难以识别）
- Fold 10: 99.0%（验证集太小或受试者表情极端）
- 标准差 18% 说明评估协议不稳定

**Bug 3 — 受试者表情分布严重不均**：
- Subject 2: 缺少厌恶(0张)和悲伤(0张)
- Subject 4: 缺少愤怒(0张)和厌恶(0张)
- Subject 11: 缺少开心(0张)
- Subject 12: 缺少恐惧(0张)、惊讶(0张)

当这些受试者作为验证集时，某些表情类别完全无法评估。

### 修复 1: `config.py` — 6 类 + 移除 NE

```python
# 修改前
NUM_CLASSES = 7
EMOTION_MAP = {..., "NE": 6}
EMOTION_LABELS = {..., 6: "neutral"}

# 修改后
NUM_CLASSES = 6  # KMU-FED 无中性样本
EMOTION_MAP = {"AN":0, "DI":1, "FE":2, "HA":3, "SA":4, "SU":5}
EMOTION_LABELS = {0:"angry", 1:"disgust", 2:"fear", 3:"happy", 4:"sad", 5:"surprise"}
```

### 修复 2: `CAFE_KMU-FED_train.py` — LOSO 评估协议

**问题**: KFold 在 12 人上做 10 折导致每折验证集极小（1-2 人）。

**方案**: 新增 `--cv_method loso`（默认），采用 Leave-One-Subject-Out：
- 12 折，每折 1 人做验证，其余 11 人训练
- 记录 **每位受试者** 的准确率（`subject_results`）
- 训练前打印完整数据分布报告（每受试者每表情的数量）
- 每折打印验证集的受试者 ID 和表情分布

### 修复 3: `video.py` — 类别数与 config 同步

将硬编码的 `num_classes=7` 改为 `num_classes=NUM_CLASSES`，从 config.py 导入。

### 新增功能: 数据分布自动报告

训练启动时自动打印完整数据集分析矩阵。

### 文件变更

| 操作 | 文件 | 说明 |
|------|------|------|
| 修改 | `CAFE/CAFE/config.py` | NUM_CLASSES 7→6，移除 NE |
| 重构 | `CAFE/CAFE/CAFE_KMU-FED_train.py` | LOSO 评估协议 + 数据分布报告 + per-subject 日志 |
| 修复 | `CAFE/CAFE/video.py` | 类别数改为引用 config |

---

## 第二轮修改：数据问题修复 & 评估协议改进（2026-07-10 下午）

### 🐛 Bug 发现：首次训练分析

对首次训练的深入分析发现 3 个关键问题：

**Bug 1 — NE（中性）类别无数据**：
`EMOTION_MAP` 和 `EMOTION_LABELS` 包含 `"NE": 6`（neutral/中性），但 KMU-FED 数据集中该类别有 **0 张图片**。模型 `num_classes=7` 的第 7 个输出神经元永远接收不到正向训练信号，浪费模型容量，且可能干扰其他类别的判别。

**Bug 2 — 10-Fold CV 在小数据集上的不可靠性**：
12 个受试者用 10 折 CV 意味着每折只有 1-2 人做验证。一个人的表现决定整折准确率。首次训练结果中出现：
- Fold 6: 31.4%（该受试者表情难以识别）
- Fold 10: 99.0%（验证集太小或受试者表情极端）
- 标准差 18% 说明评估协议不稳定

**Bug 3 — 受试者表情分布严重不均**：
- Subject 2: 缺少厌恶(0张)和悲伤(0张)
- Subject 4: 缺少愤怒(0张)和厌恶(0张)
- Subject 11: 缺少开心(0张)
- Subject 12: 缺少恐惧(0张)、惊讶(0张)

当这些受试者作为验证集时，某些表情类别完全无法评估。

### 修复 1: `config.py` — 6 类 + 移除 NE

```python
# 修改前
NUM_CLASSES = 7
EMOTION_MAP = {..., "NE": 6}
EMOTION_LABELS = {..., 6: "neutral"}

# 修改后
NUM_CLASSES = 6  # KMU-FED 无中性样本
EMOTION_MAP = {"AN":0, "DI":1, "FE":2, "HA":3, "SA":4, "SU":5}
EMOTION_LABELS = {0:"angry", 1:"disgust", 2:"fear", 3:"happy", 4:"sad", 5:"surprise"}
```

### 修复 2: `CAFE_KMU-FED_train.py` — LOSO 评估协议

**问题**: KFold 在 12 人上做 10 折导致每折验证集极小（1-2 人）。

**方案**: 新增 `--cv_method loso`（默认），采用 Leave-One-Subject-Out：
- 12 折，每折 1 人做验证，其余 11 人训练
- 记录 **每位受试者** 的准确率（`subject_results`）
- 训练前打印完整数据分布报告（每受试者每表情的数量）
- 每折打印验证集的受试者 ID 和表情分布

### 修复 3: `video.py` — 类别数与 config 同步

将硬编码的 `num_classes=7` 改为 `num_classes=NUM_CLASSES`，从 config.py 导入。

### 新增功能: 数据分布自动报告

训练启动时自动打印：
```
============================================================
  数据集分析
============================================================
  受试者总数: 12
  图片总数: 1106
  类别数: 6
    angry: 196 ###################
    disgust: 120 ############
    fear: 200 ####################
    happy: 210 #####################
    sad: 180 ##################
    surprise: 200 ####################
  每受试者图片数:
    受试者 1: 126 张  [A:26, D:20, F:20, H:20, SA:20, SU:20]
    ...
============================================================
```

### 文件变更

| 操作 | 文件 | 说明 |
|------|------|------|
| 修改 | `CAFE/CAFE/config.py` | NUM_CLASSES 7→6，移除 NE |
| 重构 | `CAFE/CAFE/CAFE_KMU-FED_train.py` | LOSO 评估协议 + 数据分布报告 + per-subject 日志 |
| 修复 | `CAFE/CAFE/video.py` | 类别数改为引用 config |

---

## 1. 新增 `CAFE/CAFE/config.py` — 统一配置管理

**问题**: 原代码中数据集路径、模型路径、超参数散落在 3 个文件中，且使用硬编码绝对路径（`/data/chenruimin/...`），更换环境必须逐文件修改。  
**方案**: 新建 `config.py`，所有路径和超参数集中管理，支持**环境变量覆盖**。

**批注**: 
- 每条配置都有环境变量备选（如 `KMU_FED_ROOT`、`YOLO_MODEL_PATH`），可以在不改代码的情况下通过 `.env` 或 `export` 切换环境
- 路径使用 `os.path.join(os.path.dirname(__file__), ...)` 计算相对路径，项目移动后自动适配
- 超参数（Batch Size、LR 等）也集中管理，命令行 `argparse` 可以覆盖

```python
# 例如：通过环境变量切换数据集
# shell: export KMU_FED_ROOT=/new/data/path
# 或放在 .env 文件中
DATASET_ROOT = os.environ.get("KMU_FED_ROOT", <默认值>)
```

---

## 2. 新增 `CAFE/CAFE/models.py` — 共享模型定义

**问题**: `BasicBlock`、`ResNet`、`Model`、`my_MaxPool2d`、`Mask`、`supervisor` 在 `CAFE_KMU-FED_train.py` 和 `video.py` 中各有一份拷贝，约 200 行重复代码。修改模型结构时需两处同步，容易遗漏。

**方案**: 提取到 `models.py`，训练和推理脚本统一 `from models import Model`。

**批注**:
- `clip_model` 改为构造参数传入而非模块顶层加载，推理和训练可使用不同 CLIP 实例
- `Mask` 函数的 `np.random.shuffle` → `random.shuffle` 改为 `np.random.shuffle`（保持原行为，加注释说明）
- `supervisor` 函数新增 `device` 参数，自动将 mask 张量放到正确设备
- `ResNet` 的 `get_resnet_layer` 保持原参数默认值，确保完全兼容

---

## 3. 重构 `CAFE/CAFE/CAFE_KMU-FED_train.py` — 训练脚本

### 3.1 添加 argparse CLI 接口

**问题**: 原脚本所有配置硬编码，改参数需编辑源码。  
**方案**: 新增 `--data_dir`、`--epochs`、`--batch_size`、`--lr`、`--yolo`、`--device` 等 12 个命令行参数。

```bash
# 覆盖数据集和训练轮数
python CAFE_KMU-FED_train.py --data_dir ./KMU-FED --epochs 100

# 使用 CPU 训练（低配机器）
python CAFE_KMU-FED_train.py --device cpu --batch_size 8

# 禁用 TensorBoard
python CAFE_KMU-FED_train.py --no_tensorboard
```

### 3.2 添加 TensorBoard 日志

**问题**: 原代码只有 `print`，无法可视化训练曲线、对比不同实验。  
**方案**: 集成 `torch.utils.tensorboard.SummaryWriter`，记录以下标量：
- `Fold_N/Train_Loss`、`Fold_N/Train_Acc`、`Fold_N/Val_Loss`、`Fold_N/Val_Acc`
- `Fold_N/CE_Loss`、`Fold_N/MC_Loss0`、`Fold_N/MC_Loss1`（损失分解）
- `Fold_N/LearningRate`（学习率衰减）
- `Summary/Mean_Accuracy`、`Summary/Std_Accuracy`

```bash
# 启动 TensorBoard 查看
tensorboard --logdir KMU-FED/output_kmu_fed_clip/tensorboard
```

### 3.3 训练指标 JSON 导出

新增自动保存 `training_metrics.json`，包含每折每 epoch 的完整指标，供 Agent 分析使用。

### 3.4 早停机制

新增 `patience` 参数，验证准确率连续 N 轮不提升自动停止该折训练。

### 3.5 运行时统计

新增每 epoch 耗时输出和训练总耗时统计。

---

## 4. 重构 `CAFE/CAFE/video.py` — 推理脚本

**问题**: 原脚本硬编码视频路径、模型权重路径，不支持命令行传参，且包含约 200 行重复的模型定义代码。

**方案**:
- 导入 `models.py` 和 `config.py`，消除重复代码
- 新增 argparse：`--input`、`--output`、`--weights`、`--clip_model`、`--yolo`、`--device`、`--conf`
- 抽取 `preprocess_face()` 函数，提高可读性
- 所有 print 添加 emoji 前缀，更易区分日志类型

```bash
# 自定义输入输出
python video.py --input my_video.mp4 --output annotated.mp4

# 使用特定权重和设备
python video.py --weights ./fold_8_best.pth --device cuda:0
```

---

## 5. 修复 `CAFE/CAFE/generate_video.py` — Bug 修复 + 功能增强

### 🐛 Bug 修复

**Bug 1 — 注释/打印与实际代码不一致**:
- 原注释: `# 只提取 第二个人（02_xxx）的所有图片`
- 原打印: `✅ 第二个人（subject=02）视频合成完成！`
- 实际代码: `subject_id = "08"`（提取的是第 8 个人）
- 输出文件: `subject_8_video.mp4`（文件名为 8，与代码一致）

**修复**: 移除硬编码，改为 CLI 参数 `--subject`，注释和打印自动匹配实际值。

**Bug 2 — 硬编码路径**:
- `IMG_DIR = "/data/chenruimin/KMU-FED"` → `--data_dir` 参数

### 功能增强
- 新增 `--subject`、`--fps`、`--frame_size` 命令行参数
- 新增空结果检测（未找到图片时提示而非生成空视频）
- 输出路径自动根据 subject_id 生成

---

## 6. 新增 `requirements.txt`

**问题**: 项目没有依赖声明，环境复现靠手工试错。  
**内容**: 列出所有必须的 pip 包及最低版本要求。`clip/` 目录下的 vendored CLIP 无需 pip 安装，但其依赖 `ftfy`、`regex` 仍列在 requirements 中。

---

## 7. 新增 Agent: `training-logger` (`.claude/agents/training-logger.md`)

**作用**: 读取训练输出的 `training_metrics.json` 和 TensorBoard 日志，提取并整理为结构化 Markdown 记录。只做数据转录，不做分析。

**触发**: "记录训练"、"训练日志"、"查看训练结果"

---

## 8. 新增 Agent: `training-analyzer` (`.claude/agents/training-analyzer.md`)

**作用**: 读取 training-logger 的输出，进行多维度分析（过拟合检测、收敛速度、交叉验证一致性、损失分解），并用 matplotlib 生成 4 张可视化图表。输出分析报告。

**触发**: "分析训练结果"、"可视化训练"、"我的模型训练得怎么样"

**生成图表**:
1. `fold_accuracies.png` — 各折准确率柱状图
2. `loss_curves.png` — 10 折训练/验证 Loss 曲线
3. `acc_curves.png` — 10 折训练/验证准确率曲线
4. `loss_decomposition.png` — CE Loss + MC Loss 分解

---

## 文件变更汇总

| 操作 | 文件 | 说明 |
|------|------|------|
| 新增 | `CAFE/CAFE/config.py` | 统一配置管理 |
| 新增 | `CAFE/CAFE/models.py` | 共享模型定义（消除约200行重复） |
| 新增 | `requirements.txt` | 项目依赖声明 |
| 新增 | `.claude/agents/training-logger.md` | 训练记录 Agent |
| 新增 | `.claude/agents/training-analyzer.md` | 训练分析 Agent |
| 重构 | `CAFE/CAFE/CAFE_KMU-FED_train.py` | 添加 argparse + TensorBoard + JSON 导出 + 早停 |
| 重构 | `CAFE/CAFE/video.py` | 添加 argparse + 消除重复代码 |
| 修复 | `CAFE/CAFE/generate_video.py` | 修复 subject_id bug + 添加 argparse |
| 更新 | `CLAUDE.md` | 反映新架构 |

---

## 终端训练指令

```bash
# 0. 安装依赖（首次运行前）
pip install -r requirements.txt

# 1. 确保权重文件就位
#    - CAFE/CAFE/clip/ViT-B-32.pt
#    - CAFE/CAFE/clip/resnet18_msceleb.pth
#    - face_yolov8n.pt（放在项目根目录）

# 2. 默认配置训练
cd CAFE/CAFE
python CAFE_KMU-FED_train.py

# 3. 自定义配置训练
python CAFE_KMU-FED_train.py \
    --data_dir ../../KMU-FED/KMU-FED \
    --yolo ../../face_yolov8n.pt \
    --epochs 60 \
    --batch_size 32 \
    --lr 0.0002 \
    --device cuda:0

# 4. 启动 TensorBoard 监控训练
tensorboard --logdir ../../KMU-FED/output_kmu_fed_clip/tensorboard

# 5. 训练完成后，分析结果
# 在 Claude Code 中说："分析训练结果"（会自动调用 training-analyzer agent）

# 6. 视频推理
python video.py --input ../../KMU-FED/subject_8_video.mp4 --output result.mp4

# 7. 生成测试视频
python generate_video.py --subject 08 --fps 10
```
