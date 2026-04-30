"""
将 labeled_data 中修正后的 X-AnyLabeling JSON 标注转换为 YOLO 格式,
并更新到 data/ 目录中对应的 train/val 集.

用法:
    python tools/update_labels_from_xanylabeling.py \
        --src labeled_data/kidadult-20260424 \
        --dst data/iter_20260425
"""

import argparse
import json
import shutil
from pathlib import Path

CLASS_MAP = {"Kid": 0, "kid": 0, "Adult": 1, "adult": 1}
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def xanylabeling_to_yolo(json_path: Path) -> list[str]:
    """将一个 X-AnyLabeling JSON 转换为 YOLO 格式行列表."""
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    img_w = data["imageWidth"]
    img_h = data["imageHeight"]
    lines = []

    for shape in data.get("shapes", []):
        label = shape["label"]
        cls_id = CLASS_MAP.get(label)
        if cls_id is None:
            print(f"  [警告] 未知类别 '{label}', 跳过: {json_path.name}")
            continue

        pts = shape["points"]
        if shape["shape_type"] == "rectangle":
            if len(pts) == 2:
                x1, y1 = pts[0]
                x2, y2 = pts[1]
            elif len(pts) == 4:
                xs = [p[0] for p in pts]
                ys = [p[1] for p in pts]
                x1, y1 = min(xs), min(ys)
                x2, y2 = max(xs), max(ys)
            else:
                continue
        else:
            continue

        # 转换为 YOLO 归一化格式
        cx = ((x1 + x2) / 2) / img_w
        cy = ((y1 + y2) / 2) / img_h
        bw = abs(x2 - x1) / img_w
        bh = abs(y2 - y1) / img_h

        # 裁剪到 [0, 1]
        cx = max(0, min(1, cx))
        cy = max(0, min(1, cy))
        bw = max(0, min(1, bw))
        bh = max(0, min(1, bh))

        lines.append(f"{cls_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

    return lines


def main():
    p = argparse.ArgumentParser(description="X-AnyLabeling JSON -> YOLO 标注更新")
    p.add_argument("--src", type=str, required=True,
                   help="修正后的标注目录, 如 labeled_data/kidadult-20260424")
    p.add_argument("--dst", type=str, required=True,
                   help="YOLO 数据集目录, 如 data/iter_20260425")
    p.add_argument("--dry-run", action="store_true",
                   help="仅打印将要执行的操作, 不实际写入")
    args = p.parse_args()

    src_dir = Path(args.src)
    dst_dir = Path(args.dst)

    if not src_dir.is_dir():
        raise FileNotFoundError(f"源目录不存在: {src_dir}")
    if not dst_dir.is_dir():
        raise FileNotFoundError(f"目标目录不存在: {dst_dir}")

    # 收集所有 JSON 文件
    json_files = sorted(src_dir.glob("*.json"))
    print(f"[信息] 找到 {len(json_files)} 个 JSON 标注文件")

    # 构建目标集中已有图片的索引 (stem -> split)
    existing = {}
    for split in ["train", "val"]:
        img_dir = dst_dir / split / "images"
        if img_dir.is_dir():
            for f in img_dir.iterdir():
                if f.suffix.lower() in IMG_EXTS:
                    existing[f.stem] = split

    updated = 0
    added = 0
    skipped_no_img = 0
    skipped_no_shapes = 0

    for json_path in json_files:
        stem = json_path.stem

        # 找到对应的图片
        img_path = None
        for ext in IMG_EXTS:
            candidate = src_dir / (stem + ext)
            if candidate.exists():
                img_path = candidate
                break

        if img_path is None:
            skipped_no_img += 1
            continue

        # 转换标注
        yolo_lines = xanylabeling_to_yolo(json_path)

        # 确定目标 split
        if stem in existing:
            split = existing[stem]
        else:
            # 新图片默认放入 train
            split = "train"

        dst_img_dir = dst_dir / split / "images"
        dst_lbl_dir = dst_dir / split / "labels"

        if args.dry_run:
            action = "更新" if stem in existing else "新增"
            shapes_info = f"{len(yolo_lines)} 个标注" if yolo_lines else "空标注(无目标)"
            print(f"  [{action}] {stem} -> {split} ({shapes_info})")
        else:
            # 写入 YOLO 标注
            lbl_path = dst_lbl_dir / (stem + ".txt")
            if yolo_lines:
                lbl_path.write_text("\n".join(yolo_lines) + "\n")
            else:
                # 无目标的图片: 写空文件 (YOLO 支持)
                lbl_path.write_text("")

            # 复制图片 (如果不存在)
            dst_img_path = dst_img_dir / img_path.name
            if not dst_img_path.exists():
                shutil.copy2(img_path, dst_img_path)

        if stem in existing:
            updated += 1
        else:
            added += 1

    # 删除旧的 labels.cache (YOLO 会重新生成)
    if not args.dry_run:
        for split in ["train", "val"]:
            cache_file = dst_dir / split / "labels.cache"
            if cache_file.exists():
                cache_file.unlink()
                print(f"[信息] 已删除缓存: {cache_file}")

    print(f"\n[完成] 更新: {updated}, 新增: {added}")
    print(f"[信息] 跳过(无图片): {skipped_no_img}")
    if skipped_no_shapes > 0:
        print(f"[信息] 空标注: {skipped_no_shapes}")


if __name__ == "__main__":
    main()
