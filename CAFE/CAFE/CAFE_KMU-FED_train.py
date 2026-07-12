"""
CAFE 模型训练脚本（KMU-FED 数据集）。
用法：
    python CAFE_KMU-FED_train.py                           # 使用 config.py 中的默认路径
    python CAFE_KMU-FED_train.py --data_dir ./data         # 指定数据集路径
    python CAFE_KMU-FED_train.py --epochs 100 --lr 0.0001  # 覆盖超参数
    python CAFE_KMU-FED_train.py --yolo ./face_yolov8n.pt  # 指定 YOLO 模型路径
"""

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse
import json
import time
from datetime import datetime

import torch
import numpy as np
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms
import torch.nn as nn
import torch.nn.functional as F
import cv2
from ultralytics import YOLO
from torch.utils.tensorboard import SummaryWriter

import clip
from config import (
    DATASET_ROOT, CLIP_MODEL_PATH, RESNET_PRETRAINED_PATH,
    YOLO_MODEL_PATH, OUTPUT_DIR, TENSORBOARD_DIR, METRICS_LOG_PATH,
    NUM_CLASSES, INPUT_SIZE, BATCH_SIZE, LEARNING_RATE,
    NUM_EPOCHS, PATIENCE, NUM_WORKERS, NUM_FOLDS,
    LOSS_WEIGHT_CE, LOSS_WEIGHT_DIVERSITY, LOSS_WEIGHT_MASKED,
    EMOTION_MAP, EMOTION_LABELS,
)
from models import Model


# ===================== 命令行参数 =====================
def parse_args():
    parser = argparse.ArgumentParser(description="CAFE 面部表情识别训练")
    # 路径
    parser.add_argument("--data_dir", type=str, default=DATASET_ROOT,
                        help="KMU-FED 数据集目录")
    parser.add_argument("--clip_model", type=str, default=CLIP_MODEL_PATH,
                        help="CLIP 模型权重路径")
    parser.add_argument("--resnet_pretrained", type=str, default=RESNET_PRETRAINED_PATH,
                        help="MSCeleb 预训练 ResNet-18 路径")
    parser.add_argument("--yolo", type=str, default=YOLO_MODEL_PATH,
                        help="YOLOv8n 人脸检测模型路径")
    parser.add_argument("--output_dir", type=str, default=OUTPUT_DIR,
                        help="训练输出目录")
    # 超参数
    parser.add_argument("--epochs", type=int, default=NUM_EPOCHS, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE, help="批次大小")
    parser.add_argument("--lr", type=float, default=LEARNING_RATE, help="学习率")
    parser.add_argument("--num_workers", type=int, default=NUM_WORKERS, help="DataLoader workers")
    parser.add_argument("--folds", type=int, default=NUM_FOLDS, help="K-Fold 折数")
    parser.add_argument("--patience", type=int, default=PATIENCE, help="早停耐心值")
    # 其他
    parser.add_argument("--device", type=str, default=None,
                        help="设备 (cuda:0 / cpu)，默认自动检测")
    parser.add_argument("--no_tensorboard", action="store_true",
                        help="禁用 TensorBoard")
    parser.add_argument("--cv_method", type=str, default="loso",
                        choices=["loso", "kfold"],
                        help="交叉验证方法: loso(留一受试者) / kfold(K折)")
    return parser.parse_args()


# ===================== KMU-FED 数据集 =====================
class KMU_FED(Dataset):
    def __init__(self, root_dir, input_size=(224, 224), transform=None, face_detector=None):
        self.root_dir = root_dir
        self.img_paths = []
        self.labels = []
        self.subject_ids = []

        for fname in os.listdir(root_dir):
            if fname.lower().endswith(('jpg', 'jpeg', 'png')):
                parts = fname.split('_')
                subj_id = str(int(parts[0]))
                emo_code = parts[1]
                if emo_code in EMOTION_MAP:
                    self.img_paths.append(os.path.join(root_dir, fname))
                    self.labels.append(EMOTION_MAP[emo_code])
                    self.subject_ids.append(subj_id)

        self.transform = transform
        self.input_size = input_size
        self.face_detector = face_detector

        # 预计算所有人脸边界框（init 时一次性完成，避免每 epoch 重复检测）
        self._face_bboxes = {}
        self._precompute_faces()

    def _detect_face(self, src):
        """多级回退: Haar Cascade → YOLO → None(整图)"""
        h_img, w_img = src.shape[:2]
        if self.face_detector is not None and hasattr(self.face_detector, 'detectMultiScale'):
            scale = 640.0 / max(h_img, w_img)
            small = cv2.resize(src, (int(w_img*scale), int(h_img*scale)))
            gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
            faces = self.face_detector.detectMultiScale(gray, 1.1, 5)
            if len(faces) > 0:
                x, y, w, h = faces[0]
                x, y, w, h = int(x/scale), int(y/scale), int(w/scale), int(h/scale)
                pad_w, pad_h = int(w*0.2), int(h*0.2)
                return (max(0, x-pad_w), max(0, y-pad_h),
                        min(w_img, x+w+pad_w), min(h_img, y+h+pad_h))
        if hasattr(self, '_yolo_detector') and self._yolo_detector is not None:
            results = self._yolo_detector(src, conf=0.4, verbose=False)
            if len(results) > 0 and len(results[0].boxes) > 0:
                return tuple(map(int, results[0].boxes[0].xyxy[0]))
        return None

    def _precompute_faces(self):
        """预计算所有图片的人脸 bbox（仅 init 调用一次）"""
        for idx, path in enumerate(self.img_paths):
            src = cv2.imread(path)
            if src is not None:
                self._face_bboxes[idx] = self._detect_face(src)

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        src = cv2.imread(self.img_paths[idx])
        img_rgb = cv2.cvtColor(src, cv2.COLOR_BGR2RGB)

        bbox = self._face_bboxes.get(idx)
        if bbox is not None:
            x1, y1, x2, y2 = bbox
            face = img_rgb[y1:y2, x1:x2]
        else:
            face = img_rgb

        if self.transform is not None:
            face = self.transform(face)

        label = self.labels[idx]
        return face, label


# ===================== 训练一折 =====================
def train_one_fold(model, train_loader, val_loader, fold_idx, args, writer):
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
    best_val_acc = 0.0
    patience_counter = 0
    fold_dir = os.path.join(args.output_dir, f"fold_{fold_idx}")
    os.makedirs(fold_dir, exist_ok=True)

    fold_history = {"fold": fold_idx, "epochs": []}

    for epoch in range(args.epochs):
        # ---- 训练阶段 ----
        model.train()
        total_loss = 0.0
        total_ce = 0.0
        total_mc0 = 0.0
        total_mc1 = 0.0
        correct = 0
        total = 0

        t0 = time.time()
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(args.device), labels.to(args.device)
            outputs, mc_loss = model(imgs, labels, phase='train')

            loss_ce = F.cross_entropy(outputs, labels)
            loss = (LOSS_WEIGHT_CE * loss_ce +
                    LOSS_WEIGHT_DIVERSITY * mc_loss[1] +
                    LOSS_WEIGHT_MASKED * mc_loss[0])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_ce += loss_ce.item()
            total_mc0 += mc_loss[0].item()
            total_mc1 += mc_loss[1].item()
            correct += (outputs.argmax(1) == labels).sum().item()
            total += labels.size(0)

        train_acc = correct / total
        train_loss = total_loss / len(train_loader)
        train_time = time.time() - t0

        # ---- 验证阶段 ----
        model.eval()
        val_correct = 0
        val_total = 0
        val_loss_sum = 0.0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(args.device), labels.to(args.device)
                outputs = model(imgs, phase='test')
                val_loss = F.cross_entropy(outputs, labels)
                val_loss_sum += val_loss.item()
                val_correct += (outputs.argmax(1) == labels).sum().item()
                val_total += labels.size(0)

        val_acc = val_correct / val_total
        val_loss = val_loss_sum / len(val_loader)
        scheduler.step()

        # ---- 早停检查 ----
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            torch.save(model.state_dict(), os.path.join(fold_dir, "best.pth"))
        else:
            patience_counter += 1

        # ---- 日志记录 ----
        epoch_info = {
            "epoch": epoch + 1,
            "train_loss": round(train_loss, 6),
            "train_acc": round(train_acc, 4),
            "val_loss": round(val_loss, 6),
            "val_acc": round(val_acc, 4),
            "ce_loss": round(total_ce / len(train_loader), 6),
            "mc_loss0": round(total_mc0 / len(train_loader), 6),
            "mc_loss1": round(total_mc1 / len(train_loader), 6),
            "lr": round(scheduler.get_last_lr()[0], 8),
            "train_time_s": round(train_time, 1),
        }
        fold_history["epochs"].append(epoch_info)

        # TensorBoard
        global_step = (fold_idx - 1) * args.epochs + epoch
        if writer is not None:
            writer.add_scalar(f"Fold_{fold_idx}/Train_Loss", train_loss, epoch)
            writer.add_scalar(f"Fold_{fold_idx}/Train_Acc", train_acc, epoch)
            writer.add_scalar(f"Fold_{fold_idx}/Val_Loss", val_loss, epoch)
            writer.add_scalar(f"Fold_{fold_idx}/Val_Acc", val_acc, epoch)
            writer.add_scalar(f"Fold_{fold_idx}/CE_Loss", total_ce / len(train_loader), epoch)
            writer.add_scalar(f"Fold_{fold_idx}/MC_Loss0", total_mc0 / len(train_loader), epoch)
            writer.add_scalar(f"Fold_{fold_idx}/MC_Loss1", total_mc1 / len(train_loader), epoch)
            writer.add_scalar(f"Fold_{fold_idx}/LearningRate", scheduler.get_last_lr()[0], epoch)

        print(f"Fold{fold_idx:2d} Epoch{epoch+1:3d} | "
              f"Train Loss:{train_loss:.4f} Acc:{train_acc:.4f} | "
              f"Val Loss:{val_loss:.4f} Acc:{val_acc:.4f} | "
              f"Time:{train_time:.0f}s")

        if patience_counter >= args.patience:
            print(f"  ⏹ 早停于 Epoch {epoch+1}（{args.patience} 轮未提升）")
            break

    fold_history["best_val_acc"] = round(best_val_acc, 4)
    print(f"  ✅ Fold{fold_idx} Best Val Acc: {best_val_acc:.4f}")
    return best_val_acc, fold_history


# ===================== 交叉验证主循环 =====================
def run_cross_validation(args, writer):
    train_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(INPUT_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.RandomHorizontalFlip(),
        transforms.RandomErasing(scale=(0.02, 0.25)),
    ])
    val_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(INPUT_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # 加载人脸检测器
    yolo_detector = None
    face_cascade = None
    try:
        yolo_detector = YOLO(args.yolo)
        print(f"人脸检测: YOLO 已加载 ({args.yolo})")
    except Exception as e:
        print(f"YOLO 加载失败 ({e})")

    # 尝试加载 Haar Cascade（比 YOLO 更精确的人脸检测）
    import os as _os
    cascade_paths = [
        _os.path.join(_os.path.dirname(__file__), "..", "..", "haarcascade_frontalface_default.xml"),
        "haarcascade_frontalface_default.xml",
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml" if hasattr(cv2, 'data') else None,
    ]
    for cp in cascade_paths:
        if cp and _os.path.exists(cp):
            try:
                face_cascade = cv2.CascadeClassifier(cp)
                if not face_cascade.empty():
                    print(f"人脸检测: Haar Cascade 已加载")
                    break
            except Exception:
                continue

    if face_cascade is None and yolo_detector is None:
        print("人脸检测: 不可用，将使用整张图片")
    elif face_cascade is None:
        print("人脸检测: Haar 不可用，使用 YOLO (person-level)")

    # 构建数据集
    dataset = KMU_FED(args.data_dir, transform=val_transform, face_detector=face_cascade)
    dataset._yolo_detector = yolo_detector  # YOLO 作为回退
    unique_subjects = np.array(sorted(set(dataset.subject_ids), key=int))

    # ---- 数据分布报告 ----
    print(f"\n{'='*60}")
    print(f"  数据集分析")
    print(f"{'='*60}")
    print(f"  数据集: {args.data_dir}")
    print(f"  受试者总数: {len(unique_subjects)}")
    print(f"  图片总数: {len(dataset)}")
    print(f"  类别数: {NUM_CLASSES}")
    from collections import Counter
    emo_counts = Counter(dataset.labels)
    for label_id, name in sorted(EMOTION_LABELS.items()):
        count = emo_counts.get(label_id, 0)
        bar = '#' * (count // 10)
        print(f"    {name:>10} (id={label_id}): {count:4d} {bar}")
    print(f"\n  每受试者图片数:")
    for s in unique_subjects:
        count = sum(1 for sid in dataset.subject_ids if sid == s)
        emo_dist = Counter(dataset.labels[i] for i, sid in enumerate(dataset.subject_ids) if sid == s)
        emo_str = ', '.join(f'{EMOTION_LABELS[e][0].upper()}:{c}' for e, c in sorted(emo_dist.items()))
        print(f"    受试者 {s}: {count:3d} 张  [{emo_str}]")
    print(f"{'='*60}\n")

    # ---- 交叉验证划分 ----
    if args.cv_method == "loso":
        # Leave-One-Subject-Out: 每人轮流做验证集
        fold_list = [(s, [t for t in unique_subjects if t != s], [s])
                     for s in unique_subjects]
        method_name = f"LOSO ({len(unique_subjects)} folds)"
    else:
        # K-Fold
        skf = KFold(n_splits=args.folds, shuffle=True, random_state=42)
        fold_list = []
        for tr_idx, va_idx in skf.split(unique_subjects):
            fold_list.append((f"fold_{len(fold_list)+1}",
                              unique_subjects[tr_idx].tolist(),
                              unique_subjects[va_idx].tolist()))
        method_name = f"{args.folds}-Fold CV"

    print(f"评估方法: {method_name}")
    print(f"总轮次: {len(fold_list)}")

    subject_accs = {}  # {subject_id: best_acc}
    fold_accs = []
    fold_histories = []

    for fold_idx, (fold_name, tr_persons, va_persons) in enumerate(fold_list):
        tr_set = set(tr_persons)
        va_set = set(va_persons)

        tr_indices = [i for i, p in enumerate(dataset.subject_ids) if p in tr_set]
        va_indices = [i for i, p in enumerate(dataset.subject_ids) if p in va_set]

        print(f"\n{'='*60}")
        print(f"  Fold {fold_idx+1}/{len(fold_list)}: {fold_name}")
        if args.cv_method == "loso":
            print(f"  验证受试者: {va_persons[0]} ({len(va_indices)} 张)")
            val_emos = Counter(dataset.labels[i] for i in va_indices)
            print(f"  验证表情分布: {dict(val_emos)}")
        print(f"  训练: {len(tr_indices)} 张 | 验证: {len(va_indices)} 张")
        print(f"{'='*60}")

        train_dataset = KMU_FED(args.data_dir, transform=train_transform, face_detector=face_cascade)
        val_dataset = dataset

        train_loader = DataLoader(Subset(train_dataset, tr_indices),
                                  args.batch_size, shuffle=True, num_workers=args.num_workers)
        val_loader = DataLoader(Subset(val_dataset, va_indices),
                                args.batch_size, shuffle=False, num_workers=args.num_workers)

        model = Model(
            clip_model=clip_model,
            num_classes=NUM_CLASSES,
            resnet_pretrained_path=args.resnet_pretrained,
            device=args.device,
        ).to(args.device)

        best_acc, fold_history = train_one_fold(
            model, train_loader, val_loader, fold_idx + 1, args, writer
        )
        fold_accs.append(best_acc)
        fold_histories.append(fold_history)

        # 记录每个受试者的准确率
        for s in va_persons:
            subject_accs[f"subject_{s}"] = round(best_acc, 4)
        fold_histories.append(fold_history)

    # ---- 汇总 ----
    print("\n" + "=" * 60)
    print("  交叉验证结果汇总")
    print("=" * 60)
    for i, acc in enumerate(fold_accs):
        fold_name = fold_list[i][0]
        print(f"  {fold_name}: {acc:.4f}")
    mean_acc = np.mean(fold_accs)
    std_acc = np.std(fold_accs)
    print(f"\n  平均精度: {mean_acc:.4f} +- {std_acc:.4f}")

    # Per-subject accuracies
    if args.cv_method == "loso":
        print(f"\n  每位受试者准确率:")
        for s in sorted(subject_accs.keys(), key=lambda x: int(x.split('_')[1])):
            acc = subject_accs[s]
            bar_len = int(acc * 30)
            bar = '#' * bar_len + '-' * (30 - bar_len)
            print(f"    {s}: {acc:.4f} [{bar}]")
        min_subj = min(subject_accs, key=subject_accs.get)
        max_subj = max(subject_accs, key=subject_accs.get)
        print(f"  最低: {min_subj} = {subject_accs[min_subj]:.4f}")
        print(f"  最高: {max_subj} = {subject_accs[max_subj]:.4f}")

    # TensorBoard 汇总
    if writer is not None:
        writer.add_scalar("Summary/Mean_Accuracy", mean_acc, 0)
        writer.add_scalar("Summary/Std_Accuracy", std_acc, 0)
        for i, acc in enumerate(fold_accs):
            writer.add_scalar("Summary/Fold_Accuracy", acc, i + 1)

    # 保存 JSON 指标
    metrics = {
        "timestamp": datetime.now().isoformat(),
        "dataset": args.data_dir,
        "cv_method": args.cv_method,
        "num_classes": NUM_CLASSES,
        "config": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "folds": len(fold_list),
        },
        "fold_results": {f"fold_{i+1}": acc for i, acc in enumerate(fold_accs)},
        "subject_results": subject_accs,
        "mean_accuracy": round(mean_acc, 4),
        "std_accuracy": round(std_acc, 4),
        "fold_histories": fold_histories,
    }
    metrics_path = os.path.join(args.output_dir, "training_metrics.json")
    os.makedirs(args.output_dir, exist_ok=True)
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print(f"\n📝 训练指标已保存: {metrics_path}")

    return mean_acc, std_acc


# ===================== 入口 =====================
if __name__ == "__main__":
    args = parse_args()

    # 设备
    if args.device is None:
        args.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print(f"🖥️  设备: {args.device}")

    # 输出目录
    os.makedirs(args.output_dir, exist_ok=True)

    # CLIP
    clip_model, _ = clip.load(args.clip_model, device=args.device)
    print(f"✅ CLIP 模型加载成功: {args.clip_model}")

    # TensorBoard
    if args.no_tensorboard:
        writer = None
        print("⏭️  TensorBoard 已禁用")
    else:
        tb_dir = os.path.join(args.output_dir, "tensorboard",
                              datetime.now().strftime("%Y%m%d_%H%M%S"))
        writer = SummaryWriter(log_dir=tb_dir)
        print(f"📈 TensorBoard 日志: {tb_dir}")

    # 启动训练
    t_start = time.time()
    run_cross_validation(args, writer)
    elapsed = time.time() - t_start
    print(f"\n⏱️  总耗时: {elapsed/60:.1f} 分钟")

    if writer is not None:
        writer.close()
