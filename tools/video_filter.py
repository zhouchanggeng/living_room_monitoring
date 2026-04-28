"""
视频筛选工具 - 筛选同时包含小孩和大人的视频
使用 YOLOv8 对视频逐帧推理, 当连续 N 帧同时检测到 Kid 和 Adult 时,
判定该视频命中并打印结果.

用法:
    python tools/video_filter.py --input /path/to/videos
    python tools/video_filter.py --input /path/to/video.mp4 \
        --weights runs/train/iter_20260425_yolov8s/weights/best.pt \
        --consecutive 5 --conf 0.4 --device 0
"""

import argparse
from pathlib import Path

import cv2
from ultralytics import YOLO

VIDEO_EXTS = {".mp4", ".avi", ".mkv", ".mov", ".flv", ".wmv", ".ts", ".m4v"}
KID_CLS = 0
ADULT_CLS = 1


def parse_args():
    p = argparse.ArgumentParser(description="筛选同时包含小孩和大人的视频")
    p.add_argument("--input", type=str, required=True, help="输入视频文件或目录")
    p.add_argument("--weights", type=str,
                   default="runs/train/iter_20260425_yolov8s/weights/best.pt",
                   help="模型权重路径")
    p.add_argument("--consecutive", type=int, default=3,
                   help="连续N帧同时检测到Kid+Adult才算命中 (默认3)")
    p.add_argument("--conf", type=float, default=0.3, help="置信度阈值")
    p.add_argument("--imgsz", type=int, nargs=2, default=[384, 640],
                   help="推理尺寸 h w (默认 384 640)")
    p.add_argument("--device", type=str, default="0", help="GPU id")
    p.add_argument("--sample-fps", type=float, default=0,
                   help="抽帧fps, 0表示使用原始帧率")
    return p.parse_args()


def check_video(model, video_path, consecutive_n, conf, imgsz, device, sample_fps):
    """检查单个视频是否连续N帧同时包含Kid和Adult."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"  [跳过] 无法打开: {video_path.name}")
        return False

    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    frame_interval = max(1, int(fps / sample_fps)) if sample_fps > 0 else 1

    hit_streak = 0
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_interval != 0:
            frame_idx += 1
            continue

        results = model.predict(
            source=frame, imgsz=imgsz, conf=conf,
            device=device, verbose=False,
        )

        det = results[0].boxes
        if det is not None and len(det) > 0:
            cls_set = set(det.cls.int().tolist())
            both_present = (KID_CLS in cls_set) and (ADULT_CLS in cls_set)
        else:
            both_present = False

        if both_present:
            hit_streak += 1
            if hit_streak >= consecutive_n:
                cap.release()
                return True
        else:
            hit_streak = 0

        frame_idx += 1

    cap.release()
    return False


def main():
    args = parse_args()

    input_path = Path(args.input)

    # 收集视频文件 (支持传入单个文件或目录)
    if input_path.is_file():
        videos = [input_path] if input_path.suffix.lower() in VIDEO_EXTS else []
    else:
        videos = sorted([f for f in input_path.iterdir()
                         if f.suffix.lower() in VIDEO_EXTS])
    if not videos:
        print(f"[错误] 未找到视频文件: {input_path}")
        return

    print("=" * 60)
    print(" 视频筛选: 同时包含 Kid + Adult")
    print(f"  模型:       {args.weights}")
    print(f"  连续帧阈值: {args.consecutive}")
    print(f"  置信度:     {args.conf}")
    print(f"  视频数量:   {len(videos)}")
    print("=" * 60)

    model = YOLO(args.weights)

    hit_count = 0
    hit_videos = []
    for i, vp in enumerate(videos, 1):
        print(f"[{i}/{len(videos)}] {vp.name} ...", end=" ", flush=True)
        if check_video(model, vp, args.consecutive, args.conf,
                       args.imgsz, args.device, args.sample_fps):
            hit_count += 1
            hit_videos.append(vp.name)
            print("✓ 命中")
        else:
            print("✗ 跳过")

    print(f"\n完成! {hit_count}/{len(videos)} 个视频命中")
    if hit_videos:
        print("命中列表:")
        for name in hit_videos:
            print(f"  {name}")


if __name__ == "__main__":
    main()
