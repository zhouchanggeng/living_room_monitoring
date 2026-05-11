"""
YOLOv8 + ByteTrack 云台跟随离线 demo。

这个脚本不会控制真实云台，而是把每帧的跟随状态、模拟转动方向、
归一化误差和模拟坐标增量画到视频里，用于算法联调和仿真测试。

用法:
    python3 tools/ptz_follow_demo.py \
        --weights runs/train/distill_20260430_yolov8s/weights/best.pt \
        --source input.mp4 \
        --output output_follow.mp4 \
        --follow-mode child
"""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
from ultralytics import YOLO

from runtime.follow_controller import (
    FollowConfig,
    FollowController,
    FollowMode,
    FollowState,
    TrackedObject,
)


CLASS_NAMES = {0: "Kid", 1: "Adult"}
CLASS_COLORS = {
    "Kid": (0, 255, 255),
    "Adult": (0, 0, 0),
}


def parse_args():
    parser = argparse.ArgumentParser(description="YOLOv8 + ByteTrack 云台跟随 demo")
    parser.add_argument("--weights", required=True, help="YOLOv8 权重路径")
    parser.add_argument("--source", required=True, help="输入视频路径")
    parser.add_argument("--output", required=True, help="输出调试视频路径")
    parser.add_argument(
        "--follow-mode",
        choices=[mode.value for mode in FollowMode],
        default=FollowMode.AUTO.value,
        help="跟随目标: adult | child | auto",
    )
    parser.add_argument("--tracker", default="bytetrack.yaml", help="Ultralytics tracker 配置")
    parser.add_argument("--conf", type=float, default=0.5)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="0")
    parser.add_argument("--hold-seconds", type=float, default=0.8)
    parser.add_argument("--lost-seconds", type=float, default=1.2)
    return parser.parse_args()


def result_to_tracks(result) -> list[TrackedObject]:
    if result.boxes is None or len(result.boxes) == 0:
        return []
    if result.boxes.id is None:
        return []

    boxes = result.boxes.xyxy.cpu().numpy()
    confs = result.boxes.conf.cpu().numpy()
    clses = result.boxes.cls.cpu().numpy().astype(int)
    ids = result.boxes.id.cpu().numpy().astype(int)

    tracks = []
    for bbox, conf, cls_id, track_id in zip(boxes, confs, clses, ids):
        class_name = CLASS_NAMES.get(int(cls_id), str(cls_id))
        x1, y1, x2, y2 = [float(v) for v in bbox]
        tracks.append(
            TrackedObject(
                track_id=int(track_id),
                class_name=class_name,
                bbox_xyxy=(x1, y1, x2, y2),
                confidence=float(conf),
            )
        )
    return tracks


def draw_tracks(frame, tracks, command):
    for track in tracks:
        x1, y1, x2, y2 = [int(v) for v in track.bbox_xyxy]
        color = CLASS_COLORS.get(track.class_name, (255, 255, 255))
        is_target = track.track_id == command.target_track_id
        thickness = 3 if is_target else 1

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
        label = f"{track.class_name} id={track.track_id} {track.confidence:.2f}"
        if is_target:
            label = "LOCK " + label

        text_color = (255, 255, 255) if track.class_name == "Adult" else (0, 0, 0)
        (tw, th), base = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        cv2.rectangle(frame, (x1, max(0, y1 - th - base - 6)), (x1 + tw + 6, y1), color, -1)
        cv2.putText(
            frame,
            label,
            (x1 + 3, y1 - base - 3),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            text_color,
            2,
        )
    return frame


def draw_command(frame, command):
    state_color = (0, 180, 0) if command.state == FollowState.TRACK else (0, 140, 255)
    lines = [
        f"state={command.state.value}",
        f"target={command.target_class or '-'} id={command.target_track_id}",
        f"dir=({command.direction_x}, {command.direction_y})",
        f"delta=({command.sim_delta_xy[0]:.3f}, {command.sim_delta_xy[1]:.3f})",
        f"norm_err=({command.normalized_error_xy[0]:.3f}, {command.normalized_error_xy[1]:.3f})",
    ]
    y = 28
    for line in lines:
        cv2.putText(frame, line, (16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, state_color, 2)
        y += 28
    return frame


def main():
    args = parse_args()
    source = Path(args.source)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    if not Path(args.weights).exists():
        raise FileNotFoundError(f"权重不存在: {args.weights}")
    if not source.exists():
        raise FileNotFoundError(f"输入视频不存在: {source}")

    cap = cv2.VideoCapture(str(source))
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频: {source}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    cap.release()

    writer = cv2.VideoWriter(
        str(output),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"无法创建输出视频: {output}")

    model = YOLO(args.weights)
    controller = FollowController(
        FollowConfig(
            follow_mode=FollowMode(args.follow_mode),
            fps=fps,
            hold_seconds=args.hold_seconds,
            lost_seconds=args.lost_seconds,
        )
    )

    stream = model.track(
        source=str(source),
        stream=True,
        persist=True,
        tracker=args.tracker,
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        device=args.device,
        verbose=False,
    )

    for frame_idx, result in enumerate(stream, start=1):
        frame = result.orig_img.copy()
        tracks = result_to_tracks(result)
        command = controller.update(tracks, (width, height))

        draw_tracks(frame, tracks, command)
        draw_command(frame, command)
        writer.write(frame)

        if frame_idx % 100 == 0:
            print(
                f"frame={frame_idx} state={command.state.value} "
                f"target={command.target_track_id} "
                f"dir=({command.direction_x},{command.direction_y}) "
                f"delta=({command.sim_delta_xy[0]:.3f},{command.sim_delta_xy[1]:.3f})"
            )

    writer.release()
    print(f"输出完成: {output}")


if __name__ == "__main__":
    main()
