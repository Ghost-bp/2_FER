"""
将指定受试者的图片合成为 mp4 视频。
用法：
    python generate_video.py                                # 使用默认配置（受试者 08）
    python generate_video.py --subject 02 --fps 15          # 指定受试者和帧率
    python generate_video.py --data_dir ./KMU-FED --subject 03
"""

import argparse
import os
import cv2


def parse_args():
    parser = argparse.ArgumentParser(description="将受试者图片合成为视频")
    parser.add_argument("--data_dir", type=str,
                        default="../../KMU-FED/KMU-FED",
                        help="KMU-FED 数据集目录")
    parser.add_argument("--subject", type=str, default="08",
                        help="要提取的受试者 ID（如 01, 02, ..., 12）")
    parser.add_argument("--output", type=str, default=None,
                        help="输出视频路径（默认: KMU-FED/subject_{subject}_video.mp4）")
    parser.add_argument("--fps", type=int, default=10,
                        help="视频帧率")
    parser.add_argument("--frame_size", type=int, nargs=2, default=[480, 480],
                        metavar=("W", "H"),
                        help="输出视频尺寸（默认: 480 480）")
    return parser.parse_args()


def main():
    args = parse_args()

    data_dir = os.path.abspath(args.data_dir)
    if args.output is None:
        base_dir = os.path.join(os.path.dirname(__file__), "..", "..", "KMU-FED")
        os.makedirs(base_dir, exist_ok=True)
        output_path = os.path.join(base_dir, f"subject_{args.subject}_video.mp4")
    else:
        output_path = args.output

    frame_size = tuple(args.frame_size)

    # ---- 提取图片 ----
    all_frames = []
    for fname in sorted(os.listdir(data_dir)):
        if not fname.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue

        parts = fname.split("_")
        if len(parts) < 3:
            continue

        current_subject = parts[0]  # 如 "01", "02", ...

        if current_subject == args.subject:
            img_path = os.path.join(data_dir, fname)
            img = cv2.imread(img_path)
            if img is None:
                continue
            img = cv2.resize(img, frame_size)
            all_frames.append(img)

    if len(all_frames) == 0:
        print(f"❌ 未找到受试者 {args.subject} 的图片")
        return

    # ---- 生成视频 ----
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, args.fps, frame_size)

    for frame in all_frames:
        out.write(frame)
    out.release()

    print(f"\n✅ 受试者 {args.subject} 视频合成完成！")
    print(f"📽️  输出: {output_path}")
    print(f"🎞️  总帧数: {len(all_frames)}")
    print(f"📏 尺寸: {frame_size}")
    print(f"🎬 帧率: {args.fps}")


if __name__ == "__main__":
    main()
