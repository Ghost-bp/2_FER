"""
CAFE 视频推理脚本（面部表情识别）。
用法：
    python video.py                                           # 使用 config.py 中的默认路径
    python video.py --input video.mp4 --output result.mp4     # 指定输入/输出
    python video.py --weights ./fold_8_best.pth --device cpu  # 指定权重和设备
"""

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse
import cv2
import numpy as np
import torch
import time

import clip
from ultralytics import YOLO

from config import (
    CLIP_MODEL_PATH, YOLO_MODEL_PATH, OUTPUT_DIR,
    INFERENCE_WEIGHTS, EMOTION_LABELS, NUM_CLASSES,
)
from models import Model


def parse_args():
    parser = argparse.ArgumentParser(description="CAFE 视频表情识别推理")
    parser.add_argument("--input", type=str,
                        default=os.path.join(OUTPUT_DIR, "..", "subject_8_video.mp4"),
                        help="输入视频路径")
    parser.add_argument("--output", type=str,
                        default=os.path.join(OUTPUT_DIR, "..", "clip_resnet_result.mp4"),
                        help="输出视频路径")
    parser.add_argument("--weights", type=str, default=INFERENCE_WEIGHTS,
                        help="训练好的模型权重路径")
    parser.add_argument("--clip_model", type=str, default=CLIP_MODEL_PATH,
                        help="CLIP 模型路径")
    parser.add_argument("--yolo", type=str, default=YOLO_MODEL_PATH,
                        help="YOLO 人脸检测模型路径")
    parser.add_argument("--device", type=str, default=None,
                        help="设备 (cuda / cpu)")
    parser.add_argument("--no_window", action="store_true", default=False,
                        help="不弹出实时显示窗口")
    parser.add_argument("--conf", type=float, default=0.4,
                        help="YOLO 人脸检测置信度阈值")
    return parser.parse_args()


def preprocess_face(face_crop, device):
    """将裁剪的人脸图像转为模型输入 tensor。"""
    rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (224, 224))
    img = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    img = (img - mean) / std
    return img.unsqueeze(0).to(device)


def main():
    args = parse_args()

    # 设备
    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🖥️  设备: {args.device}")

    # 加载 CLIP
    clip_model, _ = clip.load(args.clip_model, device=args.device)
    print(f"✅ CLIP 模型加载成功: {args.clip_model}")

    # 加载表情识别模型
    model = Model(
        clip_model=clip_model,
        num_classes=NUM_CLASSES,
        resnet_pretrained_path=None,
        device=args.device,
    ).to(args.device)
    state_dict = torch.load(args.weights, map_location=args.device)
    model.load_state_dict(state_dict)
    model.eval()
    print(f"✅ 表情识别模型加载成功: {args.weights}")

    # 加载人脸检测器
    try:
        face_detector = YOLO(args.yolo)
        print(f"✅ YOLO 人脸检测器加载成功: {args.yolo}")
    except Exception as e:
        print(f"❌ YOLO 加载失败: {e}")
        return

    # 视频初始化
    cap = cv2.VideoCapture(args.input)
    if not cap.isOpened():
        print(f"❌ 无法打开视频: {args.input}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_duration = frame_total / fps if fps > 0 else 0
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(args.output, fourcc, fps, (width, height))

    print(f"🎬 输入: {args.input}")
    print(f"📽️  输出: {args.output}")
    print(f"⏱️  视频时长: {video_duration:.2f}s | 总帧数: {frame_total}")

    # 时间统计
    frame_count = 0
    total_start = time.time()
    total_det = 0.0
    total_prep = 0.0
    total_infer = 0.0
    total_draw = 0.0

    # 推理主循环
    with torch.no_grad():
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # 1. YOLO 人脸检测
            t0 = time.time()
            det_results = face_detector(frame, conf=args.conf, verbose=False)
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

                    # 2. 预处理
                    t1 = time.time()
                    img = preprocess_face(face_crop, args.device)
                    curr_prep += time.time() - t1

                    # 3. 表情推理
                    t2 = time.time()
                    pred = model(img, phase='test')
                    emotion_idx = pred.argmax(1).item()
                    emotion_name = EMOTION_LABELS[emotion_idx]
                    curr_infer += time.time() - t2

                    # 4. 绘制
                    t3 = time.time()
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    cv2.putText(frame, emotion_name, (x1, max(y1 - 10, 20)),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
                    curr_draw += time.time() - t3

            total_prep += curr_prep
            total_infer += curr_infer
            total_draw += curr_draw
            frame_count += 1
            video_writer.write(frame)

    cap.release()
    video_writer.release()

    # 输出统计
    total_elapsed = time.time() - total_start
    avg_total = (total_det + total_prep + total_infer + total_draw) / frame_count

    print("\n" + "=" * 60)
    print("📊 推理速度统计")
    print(f"  视频时长:       {video_duration:.2f}s")
    print(f"  总推理时间:     {total_elapsed:.2f}s")
    print(f"  人脸检测:       {total_det/frame_count*1000:.1f} ms/帧")
    print(f"  图像预处理:     {total_prep/frame_count*1000:.1f} ms/帧")
    print(f"  表情推理:       {total_infer/frame_count*1000:.1f} ms/帧")
    print(f"  绘制:           {total_draw/frame_count*1000:.1f} ms/帧")
    print(f"  平均单帧:       {avg_total*1000:.1f} ms")
    print(f"  FPS:            {1/avg_total:.1f}")
    print("=" * 60)
    print(f"\n✅ 推理完成！输出: {args.output}")


if __name__ == "__main__":
    main()
