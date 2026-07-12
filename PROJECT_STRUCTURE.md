# CAFE 表情识别项目文件结构

```
d:/MM_Emotion/2_FER/
│
├── CLAUDE.md                              # 项目指南（给 Claude AI 看的项目说明）
├── LEARNING_GUIDE.md                      # 6阶段学习路线 + B站视频链接
├── MODIFICATIONS.md                       # 完整修改记录（三轮共20+项修改）
├── PROJECT_STRUCTURE.md                   # ★ 本文件 — 项目文件结构地图
├── requirements.txt                       # pip 依赖清单
├── haarcascade_frontalface_default.xml    # OpenCV Haar 人脸检测模型 (909KB)
│
├── KMU-FED/                               # 数据集 & 训练输出
│   ├── DATASET_REPORT.md                  #   数据集质量报告（12人×6类矩阵）
│   ├── KMU-FED/                           #   原始图片 (1106张, 1600×1200)
│   │   ├── 01_AN_mr_002.jpg              #     命名: {受试者}_{表情}_{会话}_{序号}.jpg
│   │   ├── 01_FE_mr_010.jpg
│   │   └── ... (1106 files)
│   └── output_kmu_fed_clip/              #   训练输出
│       ├── training_metrics.json          #     LOSO 训练指标 JSON
│       ├── tensorboard/                   #     TensorBoard 事件文件
│       │   └── 20260710_*/               #       按时间戳组织的日志
│       └── analysis/                      #     可视化图表
│           ├── fold_accuracies.png        #       KFold 各折准确率柱状图
│           ├── all_folds_val_acc.png      #       全部折验证准确率曲线
│           ├── loss_decomposition.png     #       损失分解图
│           ├── kfold_vs_loso_comparison.png #    KFold vs LOSO 对比
│           └── accuracy_vs_size.png       #       准确率 vs 数据量散点图
│
└── CAFE/CAFE/                             # ★ 核心代码
    ├── config.py                          #   统一配置（路径、超参数、表情映射）
    │                                      #   支持环境变量 + argparse 双层覆盖
    ├── models.py                          #   共享模型定义
    │                                      #   - BasicBlock (ResNet残差块)
    │                                      #   - ResNet (4层特征提取器)
    │                                      #   - Mask (随机掩码生成)
    │                                      #   - supervisor (监督损失函数)
    │                                      #   - Model (CAFE完整模型)
    ├── CAFE_KMU-FED_train.py              #   训练主脚本
    │                                      #   - KMU_FED 数据集类（Haar+YOLO人脸检测）
    │                                      #   - LOSO/KFold 交叉验证
    │                                      #   - TensorBoard 日志
    │                                      #   - 早停机制
    ├── video.py                           #   视频推理脚本
    │                                      #   - 人脸检测 + 逐帧表情识别
    │                                      #   - 耗时统计
    ├── generate_video.py                  #   工具: 受试者图片 → mp4 视频
    └── clip/                              #   OpenAI CLIP 源码 (vendored)
        ├── __init__.py
        ├── clip.py                        #     CLIP 加载器（模型下载/预处理/分词）
        ├── model.py                       #     CLIP 架构（ViT, ResNet, Transformer）
        └── simple_tokenizer.py            #     BPE 文本分词器
```

## 核心数据流

```
训练:
KMU-FED图片 → Haar人脸检测(115×115) → ResNet-18编码(512维)
                                      ↘
                          sigmoid门控 × CLIP特征(512维) → FC(6类) → 损失
                                      ↗
           KMU-FED图片 → CLIP ViT-B-32编码(512维, 冻结)

推理:
视频帧 → Haar人脸检测 → ResNet-18 → sigmoid × CLIP → argmax → 表情标签
```

## 两轮训练对比

| 指标 | 第一轮 (KFold + 7类) | 第二轮 (LOSO + 6类) |
|------|----------------------|---------------------|
| 平均准确率 | 71.66% | 71.85% |
| 标准差 | 18.01% ❌ | 11.72% ✅ |
| 最低 | 31.43% | 54.29% |
| 最高 | 99.0% | 100% (Subj 7) ⚠️ |
| 人脸检测 | YOLO person (全身) | YOLO person (全身) |
| 总耗时 | 113 分钟 | 149 分钟 |

## 文件用途速查

| 你想做什么 | 打开这个文件 |
|-----------|-------------|
| 学这个项目 | [LEARNING_GUIDE.md](LEARNING_GUIDE.md) |
| 查看修改历史 | [MODIFICATIONS.md](MODIFICATIONS.md) |
| 了解数据集 | [KMU-FED/DATASET_REPORT.md](KMU-FED/DATASET_REPORT.md) |
| 改配置/路径 | [CAFE/CAFE/config.py](CAFE/CAFE/config.py) |
| 看模型结构 | [CAFE/CAFE/models.py](CAFE/CAFE/models.py) |
| 开始训练 | [CAFE/CAFE/CAFE_KMU-FED_train.py](CAFE/CAFE/CAFE_KMU-FED_train.py) |
| 运行推理 | [CAFE/CAFE/video.py](CAFE/CAFE/video.py) |
