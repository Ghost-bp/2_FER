"""
统一配置文件 —— 所有路径和超参数集中管理。
可通过命令行参数覆盖，也可直接修改此文件的默认值。
"""

import os

# ==================== 路径配置 ====================
# 数据集根目录
DATASET_ROOT = os.environ.get(
    "KMU_FED_ROOT",
    os.path.join(os.path.dirname(__file__), "..", "..", "KMU-FED", "KMU-FED")
)

# CLIP 模型权重
CLIP_MODEL_PATH = os.environ.get(
    "CLIP_MODEL_PATH",
    os.path.join(os.path.dirname(__file__), "clip", "ViT-B-32.pt")
)

# ResNet-18 MSCeleb 预训练权重
RESNET_PRETRAINED_PATH = os.environ.get(
    "RESNET_PRETRAINED_PATH",
    os.path.join(os.path.dirname(__file__), "clip", "resnet18_msceleb.pth")
)

# YOLOv8 人脸检测模型
YOLO_MODEL_PATH = os.environ.get(
    "YOLO_MODEL_PATH",
    os.path.join(os.path.dirname(__file__), "..", "..", "face_yolov8n.pt")
)

# 训练输出目录
OUTPUT_DIR = os.environ.get(
    "OUTPUT_DIR",
    os.path.join(os.path.dirname(__file__), "..", "..", "KMU-FED", "output_kmu_fed_clip")
)

# 推理权重（默认 fold_8 best）
INFERENCE_WEIGHTS = os.environ.get(
    "INFERENCE_WEIGHTS",
    os.path.join(os.path.dirname(__file__), "..", "domain_finetune",
                 "output_joint_fer_raf_finetune_kmu_fed",
                 "fold_best_weights_9_1", "fold_8_best.pth")
)

# TensorBoard 日志目录
TENSORBOARD_DIR = os.environ.get(
    "TENSORBOARD_DIR",
    os.path.join(OUTPUT_DIR, "tensorboard")
)

# 训练指标 JSON 日志
METRICS_LOG_PATH = os.environ.get(
    "METRICS_LOG_PATH",
    os.path.join(OUTPUT_DIR, "training_metrics.json")
)

# ==================== 训练超参数 ====================
NUM_CLASSES = 6  # 原为7，但KMU-FED数据集中无NE(中性)样本
INPUT_SIZE = (224, 224)
BATCH_SIZE = 32
LEARNING_RATE = 0.0002
NUM_EPOCHS = 60
PATIENCE = 10  # 10轮没有提升就停止
NUM_WORKERS = 0
NUM_FOLDS = 10

# 损失权重
LOSS_WEIGHT_CE = 1.0       # 交叉熵损失
LOSS_WEIGHT_DIVERSITY = 5.0   # mc_loss[1] 特征多样性
# 比下面参数高 说明作认为 防止特征集中比分类准确更加重要
LOSS_WEIGHT_MASKED = 1.5      # mc_loss[0] 掩码特征交叉熵

# ==================== 表情映射 ====================
EMOTION_MAP = {
    "AN": 0, "DI": 1, "FE": 2, "HA": 3,
    "SA": 4, "SU": 5
    # 注意: KMU-FED 数据集中无 NE(中性) 样本，已移除
}

EMOTION_LABELS = {
    0: "angry", 1: "disgust", 2: "fear", 3: "happy",
    4: "sad", 5: "surprise"
}
