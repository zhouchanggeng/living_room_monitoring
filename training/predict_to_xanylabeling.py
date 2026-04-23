"""
使用训练好的 YOLOv8s 模型对图片进行推理,
输出 X-AnyLabeling (LabelMe) 兼容的 JSON 标注文件, 便于后续人工审核.

用法:
    python training/predict_to_xanylabeling.py \
        --weights runs/train/kids_care_yolov8s/weights/best.pt \
        --source all_20260416 \
        --conf 0.25 \
        --iou 0.45
"""

import argparse
import base64
import json
import os
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

from PIL import Image
from ultralytics import YOLO
from tqdm import tqdm

# 类别名映射
CLASS_NAMES = {0: "Kid", 1: "Adult"}
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args():
    p = argparse.ArgumentParser(description="YOLOv8 推理 -> X-AnyLabeling JSON")
    p.add_argument("--weights", type=str,
                   default="runs/train/kids_care_yolov8s/weights/best.pt")
    p.add_argument("--source", type=str, default="all_20260416",
                   help="图片目录")
    p.add_argument("--conf", type=float, default=0.25, help="置信度阈值")
    p.add_argument("--iou", type=float, default=0.45, help="NMS IoU阈值")
    p.add_argument("--imgsz", nargs=2, type=int, default=[384, 640],
                   help="推理尺寸 (h w)")
    p.add_argument("--batch", type=int, default=32, help="推理batch大小")
    p.add_argument("--device", type=str, default="0")
    p.add_argument("--save-img", action="store_true", default=False,
                   help="是否在JSON中嵌入base64图片(文件会很大)")
    return p.parse_args()


def build_xanylabeling_json(img_path: Path, results, conf_thr: float,
                            save_img_data: bool = False):
    """将单张图片的YOLO推理结果转为X-AnyLabeling JSON格式."""
    img = Image.open(img_path)
    w, h = img.size

    shapes = []
    if results.boxes is not None:
        for box in results.boxes:
            conf = float(box.conf[0])
            if conf < conf_thr:
                continue
            cls_id = int(box.cls[0])
            label = CLASS_NAMES.get(cls_id, str(cls_id))
            x1, y1, x2, y2 = box.xyxy[0].tolist()

            shapes.append({
                "label": label,
                "score": round(conf, 4),
                "points": [
                    [round(x1, 2), round(y1, 2)],
                    [round(x2, 2), round(y2, 2)],
                ],
                "group_id": None,
                "description": "",
                "difficult": False,
                "shape_type": "rectangle",
                "flags": {},
                "attributes": {},
            })

    image_data = None
    if save_img_data:
        with open(img_path, "rb") as f:
            image_data = base64.b64encode(f.read()).decode("utf-8")

    annotation = {
        "version": "2.4.0",
        "flags": {},
        "shapes": shapes,
        "imagePath": img_path.name,
        "imageData": image_data,
        "imageHeight": h,
        "imageWidth": w,
    }
    return annotation


def save_json(json_path: Path, data: dict):
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def main():
    args = parse_args()
    source_dir = Path(args.source)

    if not source_dir.is_dir():
        raise FileNotFoundError(f"图片目录不存在: {source_dir}")

    # 收集图片
    img_files = sorted(
        [f for f in source_dir.iterdir() if f.suffix.lower() in IMG_EXTS]
    )
    total = len(img_files)
    print(f"[信息] 共找到 {total} 张图片")
    print(f"[信息] 权重: {args.weights}")
    print(f"[信息] 置信度: {args.conf}, IoU: {args.iou}")

    # 加载模型
    model = YOLO(args.weights)

    # 分批推理
    batch_size = args.batch
    json_count = 0

    pbar = tqdm(total=total, desc="推理中")
    for i in range(0, total, batch_size):
        batch_paths = img_files[i:i + batch_size]
        batch_strs = [str(p) for p in batch_paths]

        # 批量推理
        results_list = model.predict(
            source=batch_strs,
            imgsz=args.imgsz,
            conf=args.conf,
            iou=args.iou,
            device=args.device,
            verbose=False,
        )

        # 并行写JSON
        for img_path, results in zip(batch_paths, results_list):
            annotation = build_xanylabeling_json(
                img_path, results, args.conf, args.save_img
            )
            json_path = img_path.with_suffix(".json")
            save_json(json_path, annotation)
            json_count += 1

        pbar.update(len(batch_paths))

    pbar.close()
    print(f"\n[完成] 共生成 {json_count} 个 JSON 标注文件")
    print(f"[信息] 标注保存在: {source_dir}/")
    print("[提示] 使用 X-AnyLabeling 打开该目录即可进行人工审核")


if __name__ == "__main__":
    main()
