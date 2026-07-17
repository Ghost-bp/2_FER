import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import cv2
import numpy as np
import torch
import time
import torch.nn as nn
import torch.nn.functional as F
import clip
from ultralytics import YOLO
from torch.autograd import Variable

# ===================== 1. 设备配置 =====================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

# ===================== 2. 加载 CLIP =====================
clip_model, preprocess = clip.load("clip/ViT-B-32.pt", device=device)

# ===================== 3. 【完全和训练一致】模型结构 =====================
class my_MaxPool2d(nn.Module):
    def __init__(self, kernel_size, stride=None, padding=0, dilation=1, return_indices=False, ceil_mode=False):
        super(my_MaxPool2d, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride or kernel_size
        self.padding = padding
        self.dilation = dilation
        self.return_indices = return_indices
        self.ceil_mode = ceil_mode

    def forward(self, input):
        input = input.transpose(3,1)
        input = F.max_pool2d(input, self.kernel_size, self.stride, self.padding, self.dilation, self.ceil_mode, self.return_indices)
        input = input.transpose(3,1).contiguous()
        return input

class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_channels, out_channels, stride=1, downsample=False):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        if downsample:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.downsample = None

    def forward(self, x):
        i = x
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        if self.downsample is not None:
            i = self.downsample(i)
        x += i
        return self.relu(x)

class ResNet(nn.Module):
    def __init__(self, block, n_blocks, channels, output_dim):
        super().__init__()
        self.in_channels = channels[0]
        self.conv1 = nn.Conv2d(3, self.in_channels, 7, 2, 3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(3, 2, 1)
        self.layer1 = self.get_resnet_layer(block, n_blocks[0], channels[0])
        self.layer2 = self.get_resnet_layer(block, n_blocks[1], channels[1], stride=2)
        self.layer3 = self.get_resnet_layer(block, n_blocks[2], channels[2], stride=2)
        self.layer4 = self.get_resnet_layer(block, n_blocks[3], channels[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(self.in_channels, output_dim)

    def get_resnet_layer(self, block, n_blocks, channels, stride=1):
        layers = []
        downsample = self.in_channels != block.expansion * channels
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

class Model(nn.Module):
    def __init__(self, num_classes=7):
        super().__init__()
        res18 = ResNet(BasicBlock, [2,2,2,2], [64,128,256,512], output_dim=1000)
        msceleb_model = torch.load('clip/resnet18_msceleb.pth', map_location=device)
        state_dict = msceleb_model['state_dict']
        res18.load_state_dict(state_dict, strict=False)

        self.features = nn.Sequential(*list(res18.children())[:-2])
        self.features2 = nn.Sequential(*list(res18.children())[-2:-1])
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        with torch.no_grad():
            image_features = clip_model.encode_image(x)
        x = self.features(x)
        x = self.features2(x)
        x = x.view(x.size(0), -1)
      
        x = image_features * torch.sigmoid(x)
        out = self.fc(x)
        return out

# ===================== 4. 加载训练好的权重 =====================
emotion_model_path = 'domain_finetune/output_joint_fer_raf_finetune_kmu_fed/fold_best_weights_9_1/fold_8_best.pth'
emotion_labels = {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy', 4: 'sad', 5: 'surprise', 6: 'neutral'}

# 加载模型
model = Model(num_classes=7).to(device)
model.load_state_dict(torch.load(emotion_model_path, map_location=device))
model.eval()

# 人脸检测器
face_detector = YOLO("/home/chenruimin/chenruimin/mini_Xception-main/face_yolov8n.pt")

# ===================== 5. 视频路径 =====================
video_input_path = "KMU-FED/subject_8_video.mp4"
video_output_path = "KMU-FED/clip_resnet_result.mp4"
SHOW_REALTIME_WINDOW = False

# ===================== 6. 视频初始化 =====================
cap = cv2.VideoCapture(video_input_path)
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
frame_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
video_duration = frame_total / fps if fps > 0 else 0
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video_writer = cv2.VideoWriter(video_output_path, fourcc, fps, (width, height))

print(f"视频时长: {video_duration:.2f}s | 总帧数: {frame_total}")

# ===================== 7. 时间统计 =====================
frame_count = 0
total_start = time.time()

total_det = 0.0    # YOLO 检测
total_prep = 0.0   # 预处理
total_infer = 0.0  # 表情推理
total_draw = 0.0   # 绘制

# ===================== 8. 推理主循环 =====================
with torch.no_grad():
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # 1. YOLO 人脸检测
        t0 = time.time()
        det_results = face_detector(frame, conf=0.4, verbose=False)
        total_det += time.time() - t0

        curr_prep = 0.0
        curr_infer = 0.0
        curr_draw = 0.0

        for res in det_results:
            for box in res.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                face_crop = frame[y1:y2, x1:x2]
                if face_crop.size == 0:
                    continue

                # 2. 图像预处理
                t1 = time.time()
                rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
                rgb = cv2.resize(rgb, (224, 224))
                img = torch.from_numpy(rgb).permute(2,0,1).float() / 255.0
                mean = torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)
                std = torch.tensor([0.229, 0.224, 0.225]).view(3,1,1)
                img = (img - mean) / std
                img = img.unsqueeze(0).to(device)
                curr_prep += time.time() - t1

                # 3. 表情推理
                t2 = time.time()
                pred = model(img)
                emotion_idx = pred.argmax(1).item()
                emotion_name = emotion_labels[emotion_idx]
                curr_infer += time.time() - t2

                # 4. 绘制
                t3 = time.time()
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(frame, emotion_name, (x1, max(y1-10, 20)),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
                curr_draw += time.time() - t3

        total_prep += curr_prep
        total_infer += curr_infer
        total_draw += curr_draw
        frame_count += 1
        video_writer.write(frame)

# ===================== 9. 输出时间统计 =====================
total_elapsed = time.time() - total_start

avg_total = (total_det + total_prep + total_infer + total_draw) / frame_count

print("\n" + "="*80)
print("📊 推理速度统计")
print(f"🎬 视频时长: {video_duration:.2f}s")
print(f"⚡ 总推理时间: {total_elapsed:.2f}s")
print(f"✅ 人脸检测(YOLO): {total_det/frame_count*1000:.2f} ms/帧")
print(f"✅ 图像预处理: {total_prep/frame_count*1000:.2f} ms/帧")
print(f"✅ 表情推理: {total_infer/frame_count*1000:.2f} ms/帧")
print(f"✅ 绘制: {total_draw/frame_count*1000:.2f} ms/帧")
print(f"✅ 平均单帧: {avg_total*1000:.2f} ms")
print(f"✅ FPS: {1/avg_total:.1f}")
print("="*80)

cap.release()
video_writer.release()
print(f"\n✅ 推理完成！输出视频：{video_output_path}")