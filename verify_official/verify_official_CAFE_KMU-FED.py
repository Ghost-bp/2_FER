"""
验证实验：使用官方 CAFE 源码的精确架构，在 KMU-FED 上运行 KFold+YOLO。
目的：验证本地代码的 60.91% CLIP-12 基线是否准确。

架构来源：https://github.com/zyh-uaiaaaa/Generalizable-FER
→ code/ours_CAFE.py 中的 Model / ResNet / BasicBlock / supervisor / Mask 类原封不动复制。
"""

import os
import sys
# 添加 CAFE clip 模块路径
sys.path.insert(0, os.path.abspath(os.path.join(
    os.path.dirname(__file__), '..', 'CAFE', 'CAFE')))
import argparse
import json
import time
from datetime import datetime

import cv2
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset, Subset
from sklearn.model_selection import KFold
from ultralytics import YOLO

# ==================== 官方源码精确复制 - START ====================
# 以下代码从 https://github.com/zyh-uaiaaaa/Generalizable-FER/blob/main/code/ours_CAFE.py
# 逐字复制，不做任何修改。


class my_MaxPool2d(nn.Module):
    def __init__(self, kernel_size, stride=None, padding=0, dilation=1,
                 return_indices=False, ceil_mode=False):
        super(my_MaxPool2d, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride or kernel_size
        self.padding = padding
        self.dilation = dilation
        self.return_indices = return_indices
        self.ceil_mode = ceil_mode

    def forward(self, input):
        input = input.transpose(3, 1)
        input = F.max_pool2d(input, self.kernel_size, self.stride,
                             self.padding, self.dilation, self.ceil_mode,
                             self.return_indices)
        input = input.transpose(3, 1).contiguous()
        return input


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=False):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        if downsample:
            conv = nn.Conv2d(in_channels, out_channels, kernel_size=1,
                             stride=stride, bias=False)
            bn = nn.BatchNorm2d(out_channels)
            downsample = nn.Sequential(conv, bn)
        else:
            downsample = None
        self.downsample = downsample

    def forward(self, x):
        i = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        if self.downsample is not None:
            i = self.downsample(i)
        x += i
        x = self.relu(x)
        return x


class ResNet(nn.Module):
    def __init__(self, block, n_blocks, channels, output_dim):
        super().__init__()
        self.in_channels = channels[0]
        assert len(n_blocks) == len(channels) == 4
        self.conv1 = nn.Conv2d(3, self.in_channels, kernel_size=7,
                               stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self.get_resnet_layer(block, n_blocks[0], channels[0])
        self.layer2 = self.get_resnet_layer(block, n_blocks[1], channels[1], stride=2)
        self.layer3 = self.get_resnet_layer(block, n_blocks[2], channels[2], stride=2)
        self.layer4 = self.get_resnet_layer(block, n_blocks[3], channels[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(self.in_channels, output_dim)

    def get_resnet_layer(self, block=BasicBlock, n_blocks=[2, 2, 2, 2],
                         channels=[64, 128, 256, 512], stride=1):
        layers = []
        if self.in_channels != block.expansion * channels:
            downsample = True
        else:
            downsample = False
        layers.append(block(self.in_channels, channels, stride, downsample))
        for i in range(1, n_blocks):
            layers.append(block(block.expansion * channels, channels))
        self.in_channels = block.expansion * channels
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        h = x.view(x.shape[0], -1)
        x = self.fc(h)
        return x, h


def Mask(nb_batch):
    """官方原版 Mask 函数（硬编码 cuda）"""
    bar = []
    for i in range(7):
        foo = [1] * 63 + [0] * 10
        if i == 6:
            foo = [1] * 64 + [0] * 10
        random.shuffle(foo)
        bar += foo
    bar = [bar for i in range(nb_batch)]
    bar = np.array(bar).astype("float32")
    bar = bar.reshape(nb_batch, 512, 1, 1)
    bar = torch.from_numpy(bar)
    bar = bar.cuda()
    bar = Variable(bar)
    return bar


def supervisor(x, targets, cnum):
    """官方原版 supervisor 函数（硬编码 cuda）"""
    branch = x
    branch = branch.reshape(branch.size(0), branch.size(1), 1, 1)
    branch = my_MaxPool2d(kernel_size=(1, cnum), stride=(1, cnum))(branch)
    branch = branch.reshape(branch.size(0), branch.size(1),
                            branch.size(2) * branch.size(3))
    loss_2 = 1.0 - 1.0 * torch.mean(torch.sum(branch, 2)) / cnum

    mask = Mask(x.size(0))
    branch_1 = x.reshape(x.size(0), x.size(1), 1, 1) * mask
    branch_1 = my_MaxPool2d(kernel_size=(1, cnum), stride=(1, cnum))(branch_1)
    branch_1 = branch_1.view(branch_1.size(0), -1)
    loss_1 = nn.CrossEntropyLoss()(branch_1, targets)
    return [loss_1, loss_2]


class OfficialModel(nn.Module):
    """官方原版 Model 类 —— 逐字复制，仅改类名避免冲突"""
    def __init__(self, pretrained=True, num_classes=7, drop_rate=0,
                 resnet_pretrained_path=None):
        super(OfficialModel, self).__init__()

        res18 = ResNet(block=BasicBlock, n_blocks=[2, 2, 2, 2],
                       channels=[64, 128, 256, 512], output_dim=1000)
        msceleb_model = torch.load(resnet_pretrained_path, map_location='cpu')
        state_dict = msceleb_model['state_dict']
        res18.load_state_dict(state_dict, strict=False)

        self.drop_rate = drop_rate
        self.features = nn.Sequential(*list(res18.children())[:-2])
        self.features2 = nn.Sequential(*list(res18.children())[-2:-1])

        fc_in_dim = list(res18.children())[-1].in_features  # 512
        self.fc = nn.Linear(fc_in_dim, num_classes)

        self.parm = {}
        for name, parameters in self.fc.named_parameters():
            print(name, ':', parameters.size())
            self.parm[name] = parameters

    def forward(self, x, clip_model, targets, phase='train'):
        with torch.no_grad():
            image_features = clip_model.encode_image(x)

        x = self.features(x)
        x = self.features2(x)
        x = x.view(x.size(0), -1)

        if phase == 'train':
            MC_loss = supervisor(image_features * torch.sigmoid(x),
                                 targets, cnum=73)

        x = image_features * torch.sigmoid(x)
        out = self.fc(x)

        if phase == 'train':
            return out, MC_loss
        else:
            return out, out


# ==================== 官方源码精确复制 - END ====================


# ==================== KMU-FED 数据集 ====================
EMOTION_MAP = {
    'AN': 0,  # Angry
    'DI': 1,  # Disgust
    'FE': 2,  # Fear
    'HA': 3,  # Happy
    'SA': 4,  # Sad
    'SU': 5,  # Surprise
}
EMOTION_LABELS = {
    0: 'Angry', 1: 'Disgust', 2: 'Fear',
    3: 'Happy', 4: 'Sad', 5: 'Surprise',
}
NUM_CLASSES = 6


class KMU_FED(Dataset):
    """KMU-FED 数据集加载器（YOLO 人脸检测）"""
    def __init__(self, root_dir, transform=None, yolo_detector=None):
        self.root_dir = root_dir
        self.img_paths = []
        self.labels = []
        self.subject_ids = []
        self.transform = transform
        self.yolo_detector = yolo_detector

        for fname in sorted(os.listdir(root_dir)):
            if fname.lower().endswith(('jpg', 'jpeg', 'png')):
                parts = fname.split('_')
                subj_id = str(int(parts[0]))
                emo_code = parts[1]
                if emo_code in EMOTION_MAP:
                    self.img_paths.append(os.path.join(root_dir, fname))
                    self.labels.append(EMOTION_MAP[emo_code])
                    self.subject_ids.append(subj_id)

        # 预计算所有人脸 bbox
        self._face_bboxes = {}
        self._precompute_faces()

    def _precompute_faces(self):
        for idx, path in enumerate(self.img_paths):
            src = cv2.imread(path)
            if src is not None and self.yolo_detector is not None:
                try:
                    results = self.yolo_detector(src, conf=0.4, verbose=False)
                    if len(results) > 0 and len(results[0].boxes) > 0:
                        self._face_bboxes[idx] = tuple(
                            map(int, results[0].boxes[0].xyxy[0]))
                except Exception:
                    pass

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


# ==================== 训练 ====================
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='../../KMU-FED/KMU-FED')
    parser.add_argument('--yolo', type=str, default='../../yolov8n.pt')
    parser.add_argument('--resnet_pretrained', type=str,
                        default='../../CAFE/CAFE/clip/resnet18_msceleb.pth')
    parser.add_argument('--clip_model', type=str,
                        default='../../CAFE/CAFE/clip/ViT-B-32.pt')
    parser.add_argument('--output_dir', type=str,
                        default='../../KMU-FED/output_verify_official')
    parser.add_argument('--epochs', type=int, default=60)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.0002)
    parser.add_argument('--folds', type=int, default=10)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--num_workers', type=int, default=0)
    args = parser.parse_args()

    setup_seed(3407)

    print(f"{'='*60}")
    print(f"  验证实验: 官方原版架构 + KMU-FED + KFold+YOLO")
    print(f"  CLIP-12 + ResNet-18 (MSCeleb) + Sigmoid Gate + Supervisor")
    print(f"{'='*60}")
    print(f"  数据: {args.data_dir}")
    print(f"  YOLO: {args.yolo}")
    print(f"  协议: KFold {args.folds}折")
    print(f"  超参: lr={args.lr}, bs={args.batch_size}, epochs={args.epochs}")
    print(f"{'='*60}\n")

    # 加载 CLIP
    import clip
    device = torch.device(args.device)
    clip_model, preprocess = clip.load(args.clip_model, device=device)
    print(f"✅ CLIP 加载成功: {args.clip_model}")

    # 加载 YOLO
    yolo_detector = None
    try:
        yolo_detector = YOLO(args.yolo)
        print(f"✅ YOLO 加载成功: {args.yolo}")
    except Exception as e:
        print(f"⚠️  YOLO 加载失败: {e}")

    # 数据增强（匹配官方原版）
    train_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
        transforms.RandomHorizontalFlip(),
        transforms.RandomErasing(scale=(0.02, 0.25)),
    ])
    val_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    # 构建数据集
    dataset = KMU_FED(args.data_dir, transform=val_transform,
                      yolo_detector=yolo_detector)
    unique_subjects = np.array(sorted(set(dataset.subject_ids), key=int))

    print(f"\n📊 数据集分析")
    print(f"   受试者: {len(unique_subjects)}, 图片: {len(dataset)}")
    print(f"   类别: {NUM_CLASSES}\n")

    # KFold 划分
    skf = KFold(n_splits=args.folds, shuffle=True, random_state=42)
    fold_list = []
    for tr_idx, va_idx in skf.split(unique_subjects):
        fold_list.append((unique_subjects[tr_idx].tolist(),
                          unique_subjects[va_idx].tolist()))

    fold_accs = []
    fold_histories = []

    for fold_idx, (tr_persons, va_persons) in enumerate(fold_list):
        tr_set = set(tr_persons)
        va_set = set(va_persons)

        tr_indices = [i for i, p in enumerate(dataset.subject_ids) if p in tr_set]
        va_indices = [i for i, p in enumerate(dataset.subject_ids) if p in va_set]

        print(f"\n{'─'*50}")
        print(f"  Fold {fold_idx+1}/{args.folds}")
        print(f"  训练: {len(tr_indices)} 张 | 验证: {len(va_indices)} 张 "
              f"(受试者 {va_persons})")
        print(f"{'─'*50}")

        train_dataset = KMU_FED(args.data_dir, transform=train_transform,
                                yolo_detector=yolo_detector)

        train_loader = DataLoader(Subset(train_dataset, tr_indices),
                                  args.batch_size, shuffle=True,
                                  num_workers=args.num_workers)
        val_loader = DataLoader(Subset(dataset, va_indices),
                                args.batch_size, shuffle=False,
                                num_workers=args.num_workers)

        # === 官方原版 Model ===
        model = OfficialModel(
            num_classes=NUM_CLASSES,
            resnet_pretrained_path=args.resnet_pretrained,
        ).to(device)
        model.train()

        # === 官方原版优化器 + 调度器 ===
        optimizer = torch.optim.Adam(
            model.parameters(), lr=args.lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer, gamma=0.9)

        best_val_acc = 0.0
        patience_counter = 0
        fold_history = {"fold": fold_idx + 1, "epochs": []}

        for epoch in range(args.epochs):
            # ---- 训练 ----
            model.train()
            total_loss = 0.0
            correct = 0
            total = 0
            t0 = time.time()

            for imgs, labels in train_loader:
                imgs, labels = imgs.to(device), labels.to(device)

                output, MC_loss = model(imgs, clip_model, labels, phase='train')

                loss1 = nn.CrossEntropyLoss()(output, labels)
                loss = loss1 + 5 * MC_loss[1] + 1.5 * MC_loss[0]

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                _, predicts = torch.max(output, 1)
                correct += torch.eq(predicts, labels).sum().item()
                total += labels.size(0)

            scheduler.step()
            train_acc = correct / total
            train_loss = total_loss / len(train_loader)
            train_time = time.time() - t0

            # ---- 验证 ----
            model.eval()
            val_correct = 0
            val_total = 0
            val_loss_sum = 0.0
            with torch.no_grad():
                for imgs, labels in val_loader:
                    imgs, labels = imgs.to(device), labels.to(device)
                    outputs, _ = model(imgs, clip_model, labels, phase='test')
                    val_loss = nn.CrossEntropyLoss()(outputs, labels)
                    val_loss_sum += val_loss.item()
                    _, predicts = torch.max(outputs, 1)
                    val_correct += torch.eq(predicts, labels).sum().item()
                    val_total += labels.size(0)

            val_acc = val_correct / val_total
            val_loss = val_loss_sum / len(val_loader)

            epoch_info = {
                "epoch": epoch + 1,
                "train_loss": round(train_loss, 6),
                "train_acc": round(train_acc, 4),
                "val_loss": round(val_loss, 6),
                "val_acc": round(val_acc, 4),
                "lr": round(scheduler.get_last_lr()[0], 8),
                "train_time_s": round(train_time, 1),
            }
            fold_history["epochs"].append(epoch_info)

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
            else:
                patience_counter += 1

            print(f"  Epoch {epoch+1:2d} | "
                  f"Train Loss:{train_loss:.4f} Acc:{train_acc:.4f} | "
                  f"Val Loss:{val_loss:.4f} Acc:{val_acc:.4f} | "
                  f"Time:{train_time:.0f}s")

            if patience_counter >= args.patience:
                print(f"  ⏹ 早停于 Epoch {epoch+1}")
                break

        fold_history["best_val_acc"] = round(best_val_acc, 4)
        fold_accs.append(best_val_acc)
        fold_histories.append(fold_history)
        print(f"  ✅ Fold {fold_idx+1} Best: {best_val_acc:.4f}")

    # ---- 汇总 ----
    mean_acc = np.mean(fold_accs)
    std_acc = np.std(fold_accs)

    print(f"\n{'='*60}")
    print(f"  验证实验汇总")
    print(f"{'='*60}")
    for i, acc in enumerate(fold_accs):
        print(f"  Fold {i+1}: {acc:.4f}")
    print(f"\n  平均: {mean_acc:.4f} ± {std_acc:.4f}")
    print(f"  官方 CLIP-12 基线 (本地代码): 60.91% ± 15.67%")
    print(f"  差值: {(mean_acc - 0.6091) * 100:.2f} pp")
    print(f"{'='*60}")

    # 保存结果
    metrics = {
        "timestamp": datetime.now().isoformat(),
        "experiment": "官方原版架构验证",
        "dataset": args.data_dir,
        "cv_method": "kfold",
        "num_classes": NUM_CLASSES,
        "config": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "folds": args.folds,
        },
        "fold_results": {f"fold_{i+1}": acc for i, acc in enumerate(fold_accs)},
        "mean_accuracy": round(mean_acc, 4),
        "std_accuracy": round(std_acc, 4),
        "fold_histories": fold_histories,
    }
    os.makedirs(args.output_dir, exist_ok=True)
    metrics_path = os.path.join(args.output_dir, "training_metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print(f"\n📝 结果已保存: {metrics_path}")


if __name__ == '__main__':
    main()
