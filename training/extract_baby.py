"""
从 baby_data 中提取包含 baby 相关标签的图片,
只保留 baby/baby head/baby nose 标签, 转为 X-AnyLabeling JSON 格式.

用法:
    python training/extract_baby.py --source baby_data/data --output baby_only
    extract_baby --source baby_data/data --output baby_only
"""

import argparse
import json
import shutil
from pathlib import Path

from PIL import Image
from tqdm import tqdm

# 原始标签映射
ORIG_CLASSES = {0: "adult head", 1: "baby", 2: "baby head", 3: "baby nose"}

# baby 相关的 class id
BABY_CLASS_IDS = {1, 2, 3}

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args():
    p = argparse.ArgumentParser(description="提取baby相关标注并转为X-AnyLabeling格式")
    p.add_argument("--source", "-s", type=str, default="baby_data/data",
                   help="数据目录 (含 images/ 和 labels/ 子目录)")
    p.add_argument("--output", "-o", type=str, default="baby_only",
                   help="输出目录")
    return p.parse_args()


def yolo_to_xyxy(cx, cy, w, h, img_w, img_h):
    """YOLO归一化中心坐标 -> 像素xyxy."""
    x1 = (cx - w / 2) * img_w
    y1 = (cy - h / 2) * img_h
    x2 = (cx + w / 2) * img_w
    y2 = (cy + h / 2) * img_h
    return round(x1, 2), round(y1, 2), round(x2, 2), round(y2, 2)


def build_xanylabeling_json(img_name, img_w, img_h, baby_boxes):
    """构建 X-AnyLabeling JSON."""
    shapes = []
    for cls_id, cx, cy, w, h in baby_boxes:
        x1, y1, x2, y2 = yolo_to_xyxy(cx, cy, w, h, img_w, img_h)
        shapes.append({
            "label": ORIG_CLASSES[cls_id],
            "points": [[x1, y1], [x2, y2]],
            "group_id": None,
            "description": "",
            "difficult": False,
            "shape_type": "rectangle",
            "flags": {},
            "attributes": {},
        })

    return {
        "version": "2.4.0",
        "flags": {},
        "shapes": shapes,
        "imagePath": img_name,
        "imageData": None,
        "imageHeight": img_h,
        "imageWidth": img_w,
    }


def main():
    args = parse_args()
    source = Path(args.source)
    img_dir = source / "images"
    lbl_dir = source / "labels"
    output = Path(args.output)

    if not img_dir.is_dir() or not lbl_dir.is_dir():
        raise FileNotFoundError(f"需要 {img_dir} 和 {lbl_dir} 目录")

    output.mkdir(parents=True, exist_ok=True)

    label_files = sorted(lbl_dir.glob("*.txt"))
    print(f"[信息] 共 {len(label_files)} 个标注文件")

    extracted = 0
    skipped = 0

    for lbl_path in tqdm(label_files, desc="处理中"):
        # 解析标注
        baby_boxes = []
        with open(lbl_path) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                cls_id = int(parts[0])
                if cls_id in BABY_CLASS_IDS:
                    cx, cy, w, h = map(float, parts[1:5])
                    baby_boxes.append((cls_id, cx, cy, w, h))

        # 跳过没有 baby 标签的
        if not baby_boxes:
            skipped += 1
            continue

        # 找对应图片
        stem = lbl_path.stem
        img_path = None
        for ext in IMG_EXTS:
            candidate = img_dir / f"{stem}{ext}"
            if candidate.exists():
                img_path = candidate
                break

        if img_path is None:
            skipped += 1
            continue

        # 读取图片尺寸
        img = Image.open(img_path)
        img_w, img_h = img.size

        # 复制图片
        shutil.copy2(img_path, output / img_path.name)

        # 生成 JSON
        annotation = build_xanylabeling_json(img_path.name, img_w, img_h, baby_boxes)
        json_path = output / f"{stem}.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(annotation, f, ensure_ascii=False, indent=2)

        extracted += 1

    print(f"\n[完成] 提取 {extracted} 张含baby标签的图片 -> {output}/")
    print(f"[信息] 跳过 {skipped} 张 (无baby标签或缺图片)")


if __name__ == "__main__":
    main()
