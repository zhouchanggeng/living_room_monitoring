"""
检测标注数据中同一目标被同时标为 Kid 和 Adult 的冲突标注.

原理: 在同一张图中, 找出类别不同但 IoU 很高的框对,
这些大概率是标注错误 (同一个人被标了两个类别).

用法:
    python tools/find_conflicting_labels.py \
        --labels-dir data/iter_20260425/train/labels \
        --images-dir data/iter_20260425/train/images \
        --iou-thresh 0.5

    # 也可以直接检查 X-AnyLabeling JSON 格式的标注
    python tools/find_conflicting_labels.py \
        --json-dir labeled_data/kidadult-20260423 \
        --iou-thresh 0.5
"""

import argparse
import json
from pathlib import Path

from PIL import Image


CLASS_NAMES = {0: "Kid", 1: "Adult"}


def parse_args():
    p = argparse.ArgumentParser(description="检测冲突标注")
    p.add_argument("--labels-dir", type=str, default=None,
                   help="YOLO 格式标注目录 (txt)")
    p.add_argument("--images-dir", type=str, default=None,
                   help="对应的图片目录 (用于将归一化坐标还原, 仅 YOLO 格式需要)")
    p.add_argument("--json-dir", type=str, default=None,
                   help="X-AnyLabeling JSON 标注目录")
    p.add_argument("--iou-thresh", type=float, default=0.5,
                   help="IoU 阈值, 超过此值且类别不同则视为冲突")
    p.add_argument("--output", type=str, default=None,
                   help="输出冲突列表到文件 (默认只打印)")
    p.add_argument("--export-dir", type=str, default=None,
                   help="将冲突图片和标注导出到此目录, 方便用 web-labeling 打开修正")
    return p.parse_args()


def compute_iou(box1, box2):
    """计算两个 [x1,y1,x2,y2] 框的 IoU."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter

    return inter / union if union > 0 else 0


def yolo_to_xyxy(cls_id, cx, cy, w, h, img_w, img_h):
    """YOLO 归一化 cxcywh -> 像素 xyxy."""
    cx, cy, w, h = cx * img_w, cy * img_h, w * img_w, h * img_h
    return int(cls_id), [cx - w/2, cy - h/2, cx + w/2, cy + h/2]


def load_yolo_labels(label_path, img_w, img_h):
    """读取一个 YOLO txt 标注文件, 返回 [(cls_id, [x1,y1,x2,y2]), ...]."""
    boxes = []
    with open(label_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            cls_id = int(parts[0])
            cx, cy, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
            _, xyxy = yolo_to_xyxy(cls_id, cx, cy, w, h, img_w, img_h)
            boxes.append((cls_id, xyxy))
    return boxes


def load_json_labels(json_path):
    """读取 X-AnyLabeling JSON, 返回 [(cls_id, [x1,y1,x2,y2]), ...]."""
    name_to_id = {"Kid": 0, "kid": 0, "Adult": 1, "adult": 1}
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    boxes = []
    for shape in data.get("shapes", []):
        label = shape.get("label", "")
        cls_id = name_to_id.get(label, -1)
        if cls_id < 0:
            continue
        pts = shape["points"]
        x1, y1 = pts[0]
        x2, y2 = pts[1]
        boxes.append((cls_id, [min(x1,x2), min(y1,y2), max(x1,x2), max(y1,y2)]))
    return boxes


def find_conflicts(boxes, iou_thresh):
    """在一张图的标注中找出类别不同但 IoU 高的框对."""
    conflicts = []
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            cls_i, box_i = boxes[i]
            cls_j, box_j = boxes[j]
            if cls_i == cls_j:
                continue
            iou = compute_iou(box_i, box_j)
            if iou >= iou_thresh:
                conflicts.append({
                    "box_a": {"class": CLASS_NAMES.get(cls_i, str(cls_i)), "bbox": box_i},
                    "box_b": {"class": CLASS_NAMES.get(cls_j, str(cls_j)), "bbox": box_j},
                    "iou": round(iou, 4),
                })
    return conflicts


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def main():
    args = parse_args()

    # all_conflicts: {display_name: {"conflicts": [...], "json_path": Path, "img_path": Path|None}}
    all_conflicts = {}
    total_conflict_images = 0
    total_conflict_pairs = 0

    if args.json_dir:
        json_dir = Path(args.json_dir)
        json_files = sorted(json_dir.rglob("*.json"))
        print(f"[信息] 扫描 JSON 标注: {len(json_files)} 个文件")
        for jf in json_files:
            boxes = load_json_labels(jf)
            conflicts = find_conflicts(boxes, args.iou_thresh)
            if conflicts:
                # 找对应图片
                img_path = None
                for ext in IMG_EXTS:
                    candidate = jf.with_suffix(ext)
                    if candidate.exists():
                        img_path = candidate
                        break
                all_conflicts[jf.name] = {
                    "conflicts": conflicts,
                    "json_path": jf,
                    "img_path": img_path,
                }
                total_conflict_images += 1
                total_conflict_pairs += len(conflicts)

    elif args.labels_dir:
        labels_dir = Path(args.labels_dir)
        images_dir = Path(args.images_dir) if args.images_dir else None
        label_files = sorted(labels_dir.glob("*.txt"))
        print(f"[信息] 扫描 YOLO 标注: {len(label_files)} 个文件")
        for lf in label_files:
            img_w, img_h = 640, 640
            img_path = None
            if images_dir:
                for ext in IMG_EXTS:
                    candidate = images_dir / (lf.stem + ext)
                    if candidate.exists():
                        img = Image.open(candidate)
                        img_w, img_h = img.size
                        img_path = candidate
                        break
            boxes = load_yolo_labels(lf, img_w, img_h)
            conflicts = find_conflicts(boxes, args.iou_thresh)
            if conflicts:
                all_conflicts[lf.stem] = {
                    "conflicts": conflicts,
                    "label_path": lf,
                    "img_path": img_path,
                }
                total_conflict_images += 1
                total_conflict_pairs += len(conflicts)
    else:
        print("[错误] 请指定 --labels-dir 或 --json-dir")
        return

    # 输出结果
    print(f"\n{'='*60}")
    print(f"冲突检测结果 (IoU 阈值: {args.iou_thresh})")
    print(f"{'='*60}")
    print(f"存在冲突的图片数: {total_conflict_images}")
    print(f"冲突框对总数: {total_conflict_pairs}")
    print(f"{'='*60}\n")

    for name, info in all_conflicts.items():
        print(f"📌 {name}:")
        for c in info["conflicts"]:
            a, b = c["box_a"], c["box_b"]
            print(f"   {a['class']} vs {b['class']}  IoU={c['iou']}")
        print()

    if args.output:
        out_path = Path(args.output)
        # 只保存冲突信息（不含 Path 对象）
        output_data = {}
        for name, info in all_conflicts.items():
            output_data[name] = info["conflicts"]
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        print(f"[信息] 冲突列表已保存到: {out_path}")

    # 导出冲突图片和标注到指定目录, 供 web-labeling 修正
    if args.export_dir and total_conflict_images > 0:
        export_dir = Path(args.export_dir)
        export_dir.mkdir(parents=True, exist_ok=True)
        exported = 0
        for name, info in all_conflicts.items():
            img_path = info.get("img_path")
            if img_path and img_path.exists():
                link = export_dir / img_path.name
                link.unlink(missing_ok=True)
                link.symlink_to(img_path.resolve())
            json_path = info.get("json_path")
            if json_path and json_path.exists():
                link = export_dir / json_path.name
                link.unlink(missing_ok=True)
                link.symlink_to(json_path.resolve())
            exported += 1
        print(f"[导出] 已将 {exported} 张冲突图片及标注软链接到: {export_dir}")
        print(f"[提示] 使用以下命令打开修正:")
        print(f"       web-labeling --data {export_dir}")
        print(f"[提示] 软链接直接指向原文件, 修改即生效, 无需复制回去")

    if total_conflict_images > 0 and not args.export_dir:
        print("[建议] 添加 --export-dir 参数可导出冲突图片, 配合 web-labeling 修正")


if __name__ == "__main__":
    main()
