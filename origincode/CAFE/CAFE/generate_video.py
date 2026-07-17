import os
import cv2

# ===================== 路径配置 =====================
IMG_DIR = "/data/chenruimin/KMU-FED"          # 数据集路径
OUTPUT_VIDEO = "KMU-FED/subject_8_video.mp4"   # 输出：第二个人的视频
FPS = 10                                      # 视频帧率（可调）
FRAME_SIZE = (480, 480)                       # 输出视频尺寸

# ===================== 只提取 第二个人（02_xxx）的所有图片 =====================
subject_id = "08"  # 第二个人
all_frames = []

for fname in sorted(os.listdir(IMG_DIR)):
    if not fname.lower().endswith(('.jpg', '.jpeg', '.png')):
        continue

    # 文件名格式：01_AN_mr_001.jpg → 取第一个部分作为人的ID
    parts = fname.split("_")
    if len(parts) < 3:
        continue

    current_subject = parts[0]  # 01, 02, 03...

    # 只保留第二个人
    if current_subject == subject_id:
        img_path = os.path.join(IMG_DIR, fname)
        img = cv2.imread(img_path)
        if img is None:
            continue
        img = cv2.resize(img, FRAME_SIZE)
        all_frames.append(img)

# ===================== 生成视频 =====================
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, FPS, FRAME_SIZE)

for frame in all_frames:
    out.write(frame)

out.release()

# ===================== 输出信息 =====================
print(f"\n✅ 第二个人（subject=02）视频合成完成！")
print(f"📽️ 输出视频：{OUTPUT_VIDEO}")
print(f"🎞️ 总帧数：{len(all_frames)}")
print(f"📏 视频尺寸：{FRAME_SIZE}")
print(f"🎬 帧率：{FPS}")