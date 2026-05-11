"""
视频推理脚本 - 客厅小孩看护
用 distill_20260430_yolov8s best.pt 对视频做检测推理。
颜色规则:
  Kid   (class 0) -> 黄色 (0, 255, 255)  BGR
  Adult (class 1) -> 黑色 (0, 0, 0)      BGR

用法:
    python tools/video_inference.py \
        --weights runs/train/distill_20260430_yolov8s/weights/best.pt \
        --source /data/zcg/workspace/data/video/test_video \
        --output /data/zcg/workspace/data/video/test_video_result
"""

import argparse
from pathlib import Path

import cv2
from ultralytics import YOLO

# BGR 颜色
COLOR_KID = (0, 255, 255)    # 黄色
COLOR_ADULT = (0, 0, 0)      # 黑色
CLASS_COLORS = {0: COLOR_KID, 1: COLOR_ADULT}
CLASS_NAMES = {0: "Kid", 1: "Adult"}

VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".flv", ".wmv"}


def parse_args():
    p = argparse.ArgumentParser(description="视频推理 (YOLOv8s 客厅看护)")
    p.add_argument("--weights", type=str,
                   default="runs/train/distill_20260430_yolov8s/weights/best.pt")
    p.add_argument("--source", type=str,
                   default="/data/zcg/workspace/data/video/test_video")
    p.add_argument("--output", type=str,
                   default="/data/zcg/workspace/data/video/test_video_result")
    p.add_argument("--conf", type=float, default=0.25, help="置信度阈值")
    p.add_argument("--iou", type=float, default=0.45, help="NMS IoU阈值")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--device", type=str, default="0")
    return p.parse_args()


def draw_detections(frame, result, conf_th):
    """在帧上按类别绘制彩色检测框."""
    if result.boxes is None or len(result.boxes) == 0:
        return frame

    boxes = result.boxes.xyxy.cpu().numpy()
    confs = result.boxes.conf.cpu().numpy()
    clses = result.boxes.cls.cpu().numpy().astype(int)

    for (x1, y1, x2, y2), conf, cls in zip(boxes, confs, clses):
        if conf < conf_th:
            continue
        color = CLASS_COLORS.get(cls, (255, 255, 255))
        name = CLASS_NAMES.get(cls, str(cls))
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

        # 画框
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        # 画标签文字 (Adult用黑框白字, Kid用黄框黑字)
        label = f"{name} {conf:.2f}"
        (tw, th), bl = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        # 背景条
        cv2.rectangle(frame, (x1, y1 - th - bl - 4), (x1 + tw + 4, y1),
                      color, -1)
        # 文字颜色: Adult 框里的字用白色, Kid 框里的字用黑色
        text_color = (255, 255, 255) if cls == 1 else (0, 0, 0)
        cv2.putText(frame, label, (x1 + 2, y1 - bl - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2)
    return frame


def process_video(model, video_path, out_path, args):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"[跳过] 无法打开视频: {video_path}")
        return

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (w, h))
    if not writer.isOpened():
        print(f"[错误] 无法创建输出视频: {out_path}")
        cap.release()
        return

    print(f"\n[处理] {video_path.name}  {w}x{h}@{fps:.1f}fps  共{total}帧")

    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        idx += 1

        # YOLO 推理 (BGR 直接送入)
        results = model.predict(
            source=frame,
            imgsz=args.imgsz,
            conf=args.conf,
            iou=args.iou,
            device=args.device,
            verbose=False,
        )
        frame = draw_detections(frame, results[0], args.conf)
        writer.write(frame)

        if idx % 100 == 0 or idx == total:
            print(f"  进度: {idx}/{total} ({idx*100/max(total,1):.1f}%)")

    cap.release()
    writer.release()
    print(f"[完成] 输出: {out_path}")


def main():
    args = parse_args()

    src = Path(args.source)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not Path(args.weights).exists():
        raise FileNotFoundError(f"权重不存在: {args.weights}")

    videos = []
    if src.is_file() and src.suffix.lower() in VIDEO_EXTS:
        videos = [src]
    elif src.is_dir():
        for ext in VIDEO_EXTS:
            videos.extend(sorted(src.glob(f"*{ext}")))
            videos.extend(sorted(src.glob(f"*{ext.upper()}")))
    else:
        raise FileNotFoundError(f"source 不存在或不是视频: {src}")

    if not videos:
        print(f"[警告] 目录下未发现视频文件: {src}")
        return

    print(f"[加载模型] {args.weights}")
    model = YOLO(args.weights)

    print(f"[找到视频] {len(videos)} 个")
    for v in videos:
        print(f"  - {v}")

    for v in videos:
        out_path = out_dir / f"{v.stem}_det{v.suffix}"
        process_video(model, v, out_path, args)

    print(f"\n全部完成! 输出目录: {out_dir}")


if __name__ == "__main__":
    main()
