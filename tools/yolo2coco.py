"""
YOLO格式标注 -> COCO JSON 格式转换
适配 DEIMv2 训练所需的 COCO 格式数据集
"""

import json
import os
from pathlib import Path
from PIL import Image


def yolo_to_coco(img_dir: str, label_dir: str, class_names: list[str], output_json: str):
    """将 YOLO 格式标注转换为 COCO JSON 格式.

    Args:
        img_dir: 图片目录
        label_dir: YOLO 标注目录 (txt 文件, 每行: class_id cx cy w h)
        class_names: 类别名称列表, 如 ["Kid", "Adult"]
        output_json: 输出 COCO JSON 路径
    """
    img_dir = Path(img_dir)
    label_dir = Path(label_dir)

    categories = [{"id": i, "name": name} for i, name in enumerate(class_names)]
    images = []
    annotations = []
    ann_id = 0

    img_files = sorted(img_dir.glob("*.jpg"))
    print(f"  找到 {len(img_files)} 张图片, 标注目录: {label_dir}")

    for img_id, img_path in enumerate(img_files):
        img = Image.open(img_path)
        w, h = img.size

        images.append({
            "id": img_id,
            "file_name": img_path.name,
            "width": w,
            "height": h,
        })

        label_path = label_dir / (img_path.stem + ".txt")
        if not label_path.exists():
            continue

        with open(label_path) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                cls_id = int(parts[0])
                cx, cy, bw, bh = map(float, parts[1:5])

                # YOLO 归一化 -> COCO 像素坐标 (x, y, w, h)
                x1 = (cx - bw / 2) * w
                y1 = (cy - bh / 2) * h
                box_w = bw * w
                box_h = bh * h

                annotations.append({
                    "id": ann_id,
                    "image_id": img_id,
                    "category_id": cls_id,
                    "bbox": [round(x1, 2), round(y1, 2), round(box_w, 2), round(box_h, 2)],
                    "area": round(box_w * box_h, 2),
                    "iscrowd": 0,
                })
                ann_id += 1

    coco = {
        "images": images,
        "annotations": annotations,
        "categories": categories,
    }

    os.makedirs(os.path.dirname(output_json), exist_ok=True)
    with open(output_json, "w") as f:
        json.dump(coco, f, indent=2, ensure_ascii=False)

    print(f"  COCO JSON 已保存: {output_json}")
    print(f"  图片: {len(images)}, 标注: {len(annotations)}, 类别: {len(categories)}")
    return output_json


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="YOLO -> COCO 格式转换")
    p.add_argument("--data-dir", type=str, required=True, help="数据集根目录, 如 data/iter_20260425")
    p.add_argument("--classes", nargs="+", default=["Kid", "Adult"], help="类别名称")
    args = p.parse_args()

    data_dir = Path(args.data_dir)
    for split in ["train", "val"]:
        print(f"\n转换 {split} 集:")
        yolo_to_coco(
            img_dir=str(data_dir / split / "images"),
            label_dir=str(data_dir / split / "labels"),
            class_names=args.classes,
            output_json=str(data_dir / f"annotations/instances_{split}.json"),
        )
