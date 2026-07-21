"""
================================================================================
CAFE 参数 & 结构优化训练脚本
================================================================================
基于: CAFE_KMU-FED_train_run.py (师兄原版)
修复:
  ① 增强顺序: ToTensor→Normalize 移至 HFlip+Erasing 之后
  ② 早停生效: 连续 patience 轮 val_acc 不提升自动 break
  ③ YOLO 预计算: __init__ 一次性检测人脸缓存 bbox
  ④ 6类分类: emotion_map 移除 NE
  ⑤ val_loss: 验证阶段同步计算并记录
新增:
  - argparse 控制全部超参数
  - ResNet-18/34/50 backbone 切换
  - MSCeleb/ImageNet 预训练切换
  - Adam/AdamW 优化器 + Exp/Cos/Plateau 调度器
  - 数据增强选项 (Geometric/Color)
  - 损失权重可调 + Dropout + LabelSmoothing
  - training_metrics.json 保存完整指标
================================================================================
"""
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse
import json
import time
import random
from datetime import datetime

import torch
import numpy as np
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms, models as tv_models
import torch.nn as nn
import torch.nn.functional as F
import cv2
from ultralytics import YOLO
import clip
from torch.autograd import Variable

# ===================== 参数解析 =====================
def parse_args():
    p = argparse.ArgumentParser(description="CAFE 参数优化训练")
    # 实验标识
    p.add_argument("--exp_name", type=str, required=True, help="实验名称")
    p.add_argument("--seed", type=int, default=42, help="随机种子")
    # 路径
    p.add_argument("--data_dir", type=str, default="../../../KMU-FED/KMU-FED")
    p.add_argument("--yolo", type=str, default="../../../yolov8n.pt")
    p.add_argument("--clip_model", type=str, default="clip/ViT-B-32.pt")
    p.add_argument("--msceleb_path", type=str, default="clip/resnet18_msceleb.pth")
    # 模型架构
    p.add_argument("--backbone", type=str, default="resnet18",
                   choices=["resnet18", "resnet34", "resnet50"])
    p.add_argument("--pretrained", type=str, default="msceleb",
                   choices=["msceleb", "imagenet"])
    p.add_argument("--dropout", type=float, default=0.0, help="Dropout 比率")
    # 训练超参数
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--num_workers", type=int, default=0)
    # 优化器/调度器
    p.add_argument("--optimizer", type=str, default="adam",
                   choices=["adam", "adamw"])
    p.add_argument("--scheduler", type=str, default="exponential",
                   choices=["exponential", "cosine", "plateau"])
    p.add_argument("--weight_decay", type=float, default=1e-4)
    # 正则化
    p.add_argument("--label_smoothing", type=float, default=0.0)
    # 数据增强
    p.add_argument("--aug_geometric", action="store_true", default=False)
    p.add_argument("--aug_color", action="store_true", default=False)
    # 损失权重
    p.add_argument("--loss_div", type=float, default=5.0,
                   help="MC 多样性损失权重")
    p.add_argument("--loss_mask", type=float, default=1.5,
                   help="MC 掩码损失权重")
    # 设备
    p.add_argument("--device", type=str, default=None)
    return p.parse_args()


# ===================== 全局设置 =====================
args = parse_args()
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(args.seed)

device_str = args.device or ('cuda:0' if torch.cuda.is_available() else 'cpu')
device = torch.device(device_str)

# 输出目录
base_output = f"KMU-FED/output_optimize/{args.exp_name}/"
os.makedirs(base_output, exist_ok=True)
log_dir = "../../experiments/results/阶段三/"
os.makedirs(log_dir, exist_ok=True)

# 模型参数
num_classes = 6
input_size = (224, 224)

emotion_map = {  # ④ 移除 NE，6类分类
    "AN": 0, "DI": 1, "FE": 2, "HA": 3,
    "SA": 4, "SU": 5,
}

# 打印配置
print("=" * 60)
print(f"  实验: {args.exp_name}")
print(f"  Backbone: {args.backbone} | 预训练: {args.pretrained} | Dropout: {args.dropout}")
print(f"  优化器: {args.optimizer} | 调度器: {args.scheduler} | LR: {args.lr} | WD: {args.weight_decay}")
print(f"  损失权重: CE=1.0, Div={args.loss_div}, Mask={args.loss_mask}")
print(f"  LabelSmoothing: {args.label_smoothing}")
print(f"  Batch: {args.batch_size} | Epochs: {args.epochs} | Patience: {args.patience}")
print(f"  增强: Geometric={args.aug_geometric}, Color={args.aug_color}")
print(f"  设备: {device_str} | Seed: {args.seed}")
print(f"  输出: {base_output}")
print("=" * 60)

# ===================== CLIP 模型 =====================
clip_model, preprocess = clip.load(args.clip_model, device=device)
print(f"CLIP 加载成功: {args.clip_model}")

# ===================== ResNet Backbone 定义 =====================
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
        x = self.conv1(x); x = self.bn1(x); x = self.relu(x)
        x = self.conv2(x); x = self.bn2(x)
        if self.downsample is not None:
            i = self.downsample(i)
        x += i; x = self.relu(x)
        return x


class Bottleneck(nn.Module):
    """ResNet-50 使用的瓶颈块，expansion=4"""
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1, downsample=False):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion,
                               kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)
        self.relu = nn.ReLU(inplace=True)

        if downsample:
            conv = nn.Conv2d(in_channels, out_channels * self.expansion,
                             kernel_size=1, stride=stride, bias=False)
            bn = nn.BatchNorm2d(out_channels * self.expansion)
            downsample = nn.Sequential(conv, bn)
        else:
            downsample = None
        self.downsample = downsample

    def forward(self, x):
        i = x
        x = self.conv1(x); x = self.bn1(x); x = self.relu(x)
        x = self.conv2(x); x = self.bn2(x); x = self.relu(x)
        x = self.conv3(x); x = self.bn3(x)
        if self.downsample is not None:
            i = self.downsample(i)
        x += i; x = self.relu(x)
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

    def get_resnet_layer(self, block, n_blocks, channels, stride=1):
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
        x = self.conv1(x); x = self.bn1(x); x = self.relu(x); x = self.maxpool(x)
        x = self.layer1(x); x = self.layer2(x); x = self.layer3(x); x = self.layer4(x)
        x = self.avgpool(x)
        h = x.view(x.shape[0], -1)
        x = self.fc(h)
        return x, h


def get_backbone_config(name):
    """返回 (block_class, n_blocks, channels)"""
    cfgs = {
        "resnet18":  (BasicBlock, [2, 2, 2, 2], [64, 128, 256, 512]),
        "resnet34":  (BasicBlock, [3, 4, 6, 3], [64, 128, 256, 512]),
        "resnet50":  (Bottleneck, [3, 4, 6, 3], [64, 128, 256, 512]),
    }
    return cfgs[name]

# ===================== Supervisor 模块 (原版) =====================
class my_MaxPool2d(nn.Module):
    def __init__(self, kernel_size, stride=None, padding=0, dilation=1,
                 return_indices=False, ceil_mode=False):
        super().__init__()
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


def Mask(nb_batch):
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
    branch = x
    branch = branch.reshape(branch.size(0), branch.size(1), 1, 1)
    branch = my_MaxPool2d(kernel_size=(1, cnum), stride=(1, cnum))(branch)
    branch = branch.reshape(branch.size(0), branch.size(1), branch.size(2) * branch.size(3))
    loss_2 = 1.0 - 1.0 * torch.mean(torch.sum(branch, 2)) / cnum

    mask = Mask(x.size(0))
    branch_1 = x.reshape(x.size(0), x.size(1), 1, 1) * mask
    branch_1 = my_MaxPool2d(kernel_size=(1, cnum), stride=(1, cnum))(branch_1)
    branch_1 = branch_1.view(branch_1.size(0), -1)
    loss_1 = nn.CrossEntropyLoss()(branch_1, targets)

    return [loss_1, loss_2]


# ===================== CAFE Model =====================
class Model(nn.Module):
    def __init__(self, backbone_name="resnet18", pretrained="msceleb",
                 num_classes=6, dropout=0.0, msceleb_path="clip/resnet18_msceleb.pth"):
        super().__init__()

        block_cls, n_blocks, channels = get_backbone_config(backbone_name)
        resnet = ResNet(block=block_cls, n_blocks=n_blocks, channels=channels, output_dim=1000)

        # 加载预训练权重
        if pretrained == "msceleb" and backbone_name == "resnet18":
            msceleb_model = torch.load(msceleb_path, map_location='cpu', weights_only=True)
            state_dict = msceleb_model['state_dict']
            resnet.load_state_dict(state_dict, strict=False)
            print(f"预训练: MSCeleb → {backbone_name}")
        elif pretrained == "imagenet":
            # 从 torchvision 加载 ImageNet 预训练权重
            tv_map = {
                "resnet18": tv_models.resnet18,
                "resnet34": tv_models.resnet34,
                "resnet50": tv_models.resnet50,
            }
            tv_model = tv_map[backbone_name](weights='IMAGENET1K_V1')
            tv_state = tv_model.state_dict()
            resnet.load_state_dict(tv_state, strict=False)
            print(f"预训练: ImageNet → {backbone_name}")
        elif pretrained == "msceleb" and backbone_name != "resnet18":
            # MSCeleb 只有 ResNet-18 权重，非18 backbone 回退 ImageNet
            print(f"⚠ MSCeleb 仅支持 ResNet-18，{backbone_name} 回退 ImageNet")
            tv_map = {
                "resnet18": tv_models.resnet18,
                "resnet34": tv_models.resnet34,
                "resnet50": tv_models.resnet50,
            }
            tv_model = tv_map[backbone_name](weights='IMAGENET1K_V1')
            tv_state = tv_model.state_dict()
            resnet.load_state_dict(tv_state, strict=False)
            print(f"预训练: ImageNet (fallback) → {backbone_name}")
        else:
            print(f"预训练: 无 → {backbone_name}")

        self.features = nn.Sequential(*list(resnet.children())[:-2])
        self.features2 = nn.Sequential(*list(resnet.children())[-2:-1])

        fc_in_dim = list(resnet.children())[-1].in_features

        # 投影层: 将 backbone 特征映射到 CLIP 的 512 维空间 (ResNet-50 为 2048 维)
        self.proj = nn.Linear(fc_in_dim, 512) if fc_in_dim != 512 else nn.Identity()

        # Dropout
        self.dropout_layer = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        self.fc = nn.Linear(512, num_classes)  # gate 输出固定 512 维

    def forward(self, x, targets=None, phase='train'):
        with torch.no_grad():
            image_features = clip_model.encode_image(x)

        x = self.features(x)
        x = self.features2(x)
        x = x.view(x.size(0), -1)
        x = self.proj(x)  # 投影到 512 维以匹配 CLIP

        if phase == 'train':
            MC_loss = supervisor(image_features * torch.sigmoid(x), targets, cnum=73)

        x = self.dropout_layer(x)
        x = image_features * torch.sigmoid(x)
        out = self.fc(x)

        if phase == 'train':
            return out, MC_loss
        else:
            return out


# ===================== KMU-FED 数据集 (③ YOLO预计算) =====================
class KMU_FED(Dataset):
    def __init__(self, root_dir, input_size=(224, 224), transform=None,
                 face_detector=None, precompute_bboxes=False):
        self.root_dir = root_dir
        self.img_paths = []
        self.labels = []
        self.subject_ids = []

        for fname in sorted(os.listdir(root_dir)):
            if fname.lower().endswith(('jpg', 'jpeg', 'png')):
                parts = fname.split('_')
                subj_id = str(int(parts[0]))
                emo_code = parts[1]
                if emo_code in emotion_map:
                    self.img_paths.append(os.path.join(root_dir, fname))
                    self.labels.append(emotion_map[emo_code])
                    self.subject_ids.append(subj_id)

        self.transform = transform
        self.input_size = input_size
        self.face_detector = face_detector

        # ③ YOLO 预计算: __init__ 一次性检测所有人脸 bbox
        self.bbox_cache = {}
        if precompute_bboxes and face_detector is not None:
            print(f"  预计算 YOLO bbox ({len(self.img_paths)} 张)...", end=" ", flush=True)
            for idx in range(len(self.img_paths)):
                img = cv2.imread(self.img_paths[idx])
                if img is None:
                    self.bbox_cache[idx] = None
                    continue
                results = face_detector(img, conf=0.4)
                if len(results) > 0 and len(results[0].boxes) > 0:
                    x1, y1, x2, y2 = map(int, results[0].boxes[0].xyxy[0])
                    self.bbox_cache[idx] = (x1, y1, x2, y2)
                else:
                    self.bbox_cache[idx] = None  # fallback 整张图
            print("完成")

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        src = cv2.imread(self.img_paths[idx])
        img_rgb = cv2.cvtColor(src, cv2.COLOR_BGR2RGB)

        # ③ 使用缓存的 bbox（若无缓存则实时检测）
        if idx in self.bbox_cache:
            bbox = self.bbox_cache[idx]
        elif self.face_detector is not None:
            results = self.face_detector(src, conf=0.4)
            if len(results) > 0 and len(results[0].boxes) > 0:
                x1, y1, x2, y2 = map(int, results[0].boxes[0].xyxy[0])
                bbox = (x1, y1, x2, y2)
            else:
                bbox = None
        else:
            bbox = None

        if bbox is not None:
            x1, y1, x2, y2 = bbox
            face = img_rgb[y1:y2, x1:x2]
        else:
            face = img_rgb

        if self.transform is not None:
            face = self.transform(face)

        label = self.labels[idx]
        return face, label


# ===================== 训练一折 (②早停 + ⑤val_loss) =====================
def train_one_fold(model, train_loader, val_loader, fold_idx):
    # 优化器
    if args.optimizer == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                       weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
                                      weight_decay=args.weight_decay)

    # 调度器
    if args.scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs)
    elif args.scheduler == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', patience=5, factor=0.5)
    else:
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)

    # 损失函数
    if args.label_smoothing > 0:
        ce_loss_fn = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    else:
        ce_loss_fn = nn.CrossEntropyLoss()

    best_val_acc = 0.0
    patience_counter = 0  # ② 早停计数器
    fold_dir = os.path.join(base_output, f"fold_{fold_idx}")
    os.makedirs(fold_dir, exist_ok=True)

    fold_history = {"epochs": [], "best_val_acc": 0.0}

    scheduler_name = args.scheduler

    for epoch in range(args.epochs):
        t0 = time.time()

        # ---- 训练 ----
        model.train()
        total_loss = 0.0
        total_ce = 0.0
        total_mc0 = 0.0
        total_mc1 = 0.0
        correct = 0
        total = 0

        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs, mc_loss = model(imgs, labels, phase='train')

            loss_ce = ce_loss_fn(outputs, labels)
            loss = loss_ce + args.loss_div * mc_loss[1] + args.loss_mask * mc_loss[0]

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

        # ---- 验证 (⑤ 记录 val_loss) ----
        model.eval()
        val_correct = 0
        val_total = 0
        val_loss_sum = 0.0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                outputs = model(imgs, phase='test')
                v_loss = ce_loss_fn(outputs, labels)
                val_loss_sum += v_loss.item()
                val_correct += (outputs.argmax(1) == labels).sum().item()
                val_total += labels.size(0)

        val_acc = val_correct / val_total
        val_loss = val_loss_sum / len(val_loader)
        train_time = time.time() - t0

        # 调度器 step
        if scheduler_name == 'plateau':
            scheduler.step(val_acc)
        else:
            scheduler.step()

        # ---- ② 早停检查 ----
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            torch.save(model.state_dict(), os.path.join(fold_dir, "best.pth"))
        else:
            patience_counter += 1

        # 日志
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

        print(f"Fold{fold_idx:2d} Epoch{epoch+1:3d} | "
              f"Train Loss:{train_loss:.4f} Acc:{train_acc:.4f} | "
              f"Val Loss:{val_loss:.4f} Acc:{val_acc:.4f} | "
              f"Time:{train_time:.0f}s")

        if patience_counter >= args.patience:
            print(f"  ⏹ 早停 (Epoch {epoch+1}, {args.patience}轮未提升)")
            break

    fold_history["best_val_acc"] = round(best_val_acc, 4)
    print(f"  ✅ Fold{fold_idx} Best: {best_val_acc:.4f}")
    return best_val_acc, fold_history


# ===================== 10折交叉验证 =====================
def run_10fold():
    # ---- ① 增强顺序修正: HFlip + Erasing 在 ToTensor/Normalize 之后 ----
    aug_list = [
        transforms.ToPILImage(),
        transforms.Resize(input_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
        transforms.RandomHorizontalFlip(),
    ]
    if args.aug_geometric:
        aug_list.append(transforms.RandomRotation(degrees=15))
        aug_list.append(transforms.RandomAffine(
            degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)))
    if args.aug_color:
        aug_list.append(transforms.ColorJitter(
            brightness=0.15, contrast=0.15, saturation=0.15, hue=0.1))
    aug_list.append(transforms.RandomErasing(scale=(0.02, 0.25)))
    train_transform = transforms.Compose(aug_list)

    val_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(input_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    aug_info = "HFlip+Erasing"
    if args.aug_geometric:
        aug_info += "+Geo"
    if args.aug_color:
        aug_info += "+Color"
    print(f"增强: {aug_info}")

    # ---- 加载 YOLO ----
    yolo_detector = None
    try:
        yolo_detector = YOLO(args.yolo)
        print(f"YOLO 加载: {args.yolo}")
    except Exception as e:
        print(f"YOLO 加载失败 ({e})")

    # ③ YOLO 实时检测模式（precompute 后续优化）
    dataset = KMU_FED(args.data_dir, transform=val_transform,
                      face_detector=yolo_detector, precompute_bboxes=False)

    # 受试者划分
    unique_subjects = np.array(sorted(set(dataset.subject_ids), key=int))
    print(f"受试者: {len(unique_subjects)}人 | 图片: {len(dataset)}张 | 类别: {num_classes}")

    skf = KFold(n_splits=10, shuffle=True, random_state=args.seed)
    fold_accs = []
    fold_histories = []

    for fold, (tr_idx, va_idx) in enumerate(skf.split(unique_subjects)):
        print(f"\n{'='*50}")
        print(f"  Fold {fold+1}/10")
        tr_persons = unique_subjects[tr_idx]
        va_persons = unique_subjects[va_idx]
        tr_set = set(tr_persons)

        tr_indices = [i for i, p in enumerate(dataset.subject_ids) if p in tr_set]
        va_indices = [i for i, p in enumerate(dataset.subject_ids) if p not in tr_set]

        print(f"  训练: {len(tr_indices)}张 | 验证: {len(va_indices)}张")
        print(f"{'='*50}")

        # 每 fold 创建新的 train dataset (带增强) 复用 val dataset
        train_dataset = KMU_FED(args.data_dir, transform=train_transform,
                                 face_detector=yolo_detector, precompute_bboxes=False)
        val_dataset_fold = dataset

        train_loader = DataLoader(Subset(train_dataset, tr_indices),
                                  args.batch_size, shuffle=True, num_workers=args.num_workers)
        val_loader = DataLoader(Subset(val_dataset_fold, va_indices),
                                args.batch_size, shuffle=False, num_workers=args.num_workers)

        model = Model(
            backbone_name=args.backbone,
            pretrained=args.pretrained,
            num_classes=num_classes,
            dropout=args.dropout,
            msceleb_path=args.msceleb_path,
        ).to(device)

        best_acc, fold_history = train_one_fold(model, train_loader, val_loader, fold + 1)
        fold_accs.append(best_acc)
        fold_histories.append(fold_history)

    # ---- 汇总 ----
    print("\n" + "=" * 60)
    print("  结果汇总")
    print("=" * 60)
    for i, acc in enumerate(fold_accs):
        print(f"  Fold {i+1}: {acc:.4f}")
    mean_acc = np.mean(fold_accs)
    std_acc = np.std(fold_accs)
    print(f"\n  平均: {mean_acc:.4f} ± {std_acc:.4f}")

    # ---- 保存 JSON ----
    metrics = {
        "timestamp": datetime.now().isoformat(),
        "exp_name": args.exp_name,
        "dataset": args.data_dir,
        "num_classes": num_classes,
        "config": {
            "backbone": args.backbone,
            "pretrained": args.pretrained,
            "dropout": args.dropout,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "optimizer": args.optimizer,
            "scheduler": args.scheduler,
            "weight_decay": args.weight_decay,
            "label_smoothing": args.label_smoothing,
            "aug_geometric": args.aug_geometric,
            "aug_color": args.aug_color,
            "loss_div": args.loss_div,
            "loss_mask": args.loss_mask,
            "patience": args.patience,
            "seed": args.seed,
        },
        "fold_results": {f"fold_{i+1}": acc for i, acc in enumerate(fold_accs)},
        "mean_accuracy": round(mean_acc, 4),
        "std_accuracy": round(std_acc, 4),
        "fold_histories": fold_histories,
    }
    metrics_path = os.path.join(base_output, "training_metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print(f"\n指标已保存: {metrics_path}")

    # 保存到 experiments/results/阶段三/
    log_path = os.path.join(log_dir, f"{args.exp_name}.log")
    with open(log_path, "w", encoding="utf-8") as f:
        f.write(f"实验: {args.exp_name}\n")
        f.write(f"时间: {datetime.now().isoformat()}\n")
        f.write(f"配置: {json.dumps(metrics['config'], ensure_ascii=False)}\n")
        f.write(f"结果: {mean_acc:.4f} ± {std_acc:.4f}\n")
        for i, acc in enumerate(fold_accs):
            f.write(f"  Fold {i+1}: {acc:.4f}\n")
    print(f"日志已保存: {log_path}")

    return mean_acc, std_acc


if __name__ == "__main__":
    t_start = time.time()
    run_10fold()
    elapsed = time.time() - t_start
    print(f"\n总耗时: {elapsed/60:.1f} min")
