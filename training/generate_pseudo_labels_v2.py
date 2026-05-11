"""
第二阶段: 双模型交叉验证伪标签生成

用 DEIMv2 和蒸馏后的 YOLOv8s 分别对未标注图片推理,
只保留两个模型都同意的检测结果 (IoU > 阈值 且 类别一致),
并对结果做同类别 NMS 去重, 生成高质量伪标签.

用法:
    python training/generate_pseudo_labels_v2.py --device 0
    python training/generate_pseudo_labels_v2.py --device 0 --match-iou 0.5 --conf-deimv2 0.5 --conf-yolo 0.4
    python training/generate_pseudo_labels_v2.py --device 0 --shard-id 0 --num-shards 4
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image
from tqdm import tqdm
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEIMV2_DIR = Path(__file__).resolve().parent / "DEIMv2"

# 默认路径
DEIMV2_CONFIG = str(DEIMV2_DIR / "configs/deimv2/deimv2_dinov3_x_kids_care.yml")
DEIMV2_CKPT = str(PROJECT_ROOT / "runs/train/iter_20260429_deimv2_x/best_stg2.pth")
YOLO_CKPT = str(PROJECT_ROOT / "runs/train/distill_20260430_yolov8s/weights/best.pt")
UNLABELED_DIR = str(PROJECT_ROOT / "pending_data/all_20260416")
OUTPUT_DIR = str(PROJECT_ROOT / "data/pseudo_labels_v2")

CLASS_NAMES = {0: "Kid", 1: "Adult"}
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args():
    p = argparse.ArgumentParser(description="双模型交叉验证伪标签生成")
    p.add_argument("--deimv2-config", type=str, default=DEIMV2_CONFIG)
    p.add_argument("--deimv2-ckpt", type=str, default=DEIMV2_CKPT)
    p.add_argument("--yolo-ckpt", type=str, default=YOLO_CKPT)
    p.add_argument("--input-dir", type=str, default=UNLABELED_DIR)
    p.add_argument("--output-dir", type=str, default=OUTPUT_DIR)
    p.add_argument("--conf-deimv2", type=float, default=0.5, help="DEIMv2 置信度阈值")
    p.add_argument("--conf-yolo", type=float, default=0.4, help="YOLOv8s 置信度阈值")
    p.add_argument("--match-iou", type=float, default=0.5, help="双模型匹配 IoU 阈值")
    p.add_argument("--nms-iou", type=float, default=0.6, help="同类别 NMS IoU 阈值")
    p.add_argument("--device", type=str, default="0")
    p.add_argument("--shard-id", type=int, default=0)
    p.add_argument("--num-shards", type=int, default=1)
    return p.parse_args()


# ==================== 模型加载 ====================

def load_deimv2(config_path, checkpoint_path, device):
    sys.path.insert(0, str(DEIMV2_DIR))
    from engine.core import YAMLConfig

    cfg = YAMLConfig(config_path, resume=checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state = checkpoint["ema"]["module"] if "ema" in checkpoint else checkpoint["model"]
    cfg.model.load_state_dict(state)

    class DeployModel(nn.Module):
        def __init__(self, model, postprocessor):
            super().__init__()
            self.model = model.deploy()
            self.postprocessor = postprocessor.deploy()

        def forward(self, images, orig_target_sizes):
            outputs = self.model(images)
            return self.postprocessor(outputs, orig_target_sizes)

    model = DeployModel(cfg.model, cfg.postprocessor).to(device).eval()
    img_size = cfg.yaml_cfg.get("eval_spatial_size", [640, 640])
    vit_backbone = bool(cfg.yaml_cfg.get("DINOv3STAs", False))
    return model, img_size, vit_backbone


def build_deimv2_transform(img_size, vit_backbone):
    ops = [T.Resize(img_size), T.ToTensor()]
    if vit_backbone:
        ops.append(T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
    return T.Compose(ops)


# ==================== 核心算法 ====================

def compute_iou_matrix(boxes_a, boxes_b):
    """计算两组 xyxy 框的 IoU 矩阵. [N,4] x [M,4] -> [N,M]"""
    x1 = np.maximum(boxes_a[:, 0:1], boxes_b[:, 0])
    y1 = np.maximum(boxes_a[:, 1:2], boxes_b[:, 1])
    x2 = np.minimum(boxes_a[:, 2:3], boxes_b[:, 2])
    y2 = np.minimum(boxes_a[:, 3:4], boxes_b[:, 3])
    inter = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    area_a = (boxes_a[:, 2] - boxes_a[:, 0]) * (boxes_a[:, 3] - boxes_a[:, 1])
    area_b = (boxes_b[:, 2] - boxes_b[:, 0]) * (boxes_b[:, 3] - boxes_b[:, 1])
    union = area_a[:, None] + area_b[None, :] - inter
    return inter / (union + 1e-6)


def cross_validate(dets_a, dets_b, match_iou=0.5):
    """
    双模型交叉验证: 只保留两个模型都同意的检测.
    dets_a, dets_b: list of (cls_id, x1, y1, x2, y2, score)
    返回: 匹配成功的检测 (取两者平均框, 较高置信度)
    """
    if not dets_a or not dets_b:
        return []

    a = np.array(dets_a)  # [N, 6]: cls, x1, y1, x2, y2, score
    b = np.array(dets_b)  # [M, 6]

    boxes_a = a[:, 1:5]
    boxes_b = b[:, 1:5]
    iou_mat = compute_iou_matrix(boxes_a, boxes_b)

    matched = []
    used_b = set()

    for i in range(len(a)):
        best_j = -1
        best_iou = match_iou
        for j in range(len(b)):
            if j in used_b:
                continue
            if int(a[i, 0]) != int(b[j, 0]):  # 类别必须一致
                continue
            if iou_mat[i, j] > best_iou:
                best_iou = iou_mat[i, j]
                best_j = j
        if best_j >= 0:
            used_b.add(best_j)
            cls_id = int(a[i, 0])
            # 取两个模型框的平均作为最终框
            avg_box = (a[i, 1:5] + b[best_j, 1:5]) / 2.0
            # 取较高的置信度
            score = max(a[i, 5], b[best_j, 5])
            matched.append((cls_id, *avg_box, score))

    return matched


def nms_per_class(detections, iou_thresh=0.6):
    """同类别 NMS 去重. detections: list of (cls, x1, y1, x2, y2, score)"""
    if not detections:
        return []

    dets = np.array(detections)
    results = []

    for cls_id in np.unique(dets[:, 0]):
        mask = dets[:, 0] == cls_id
        cls_dets = dets[mask]
        # 按 score 降序
        order = cls_dets[:, 5].argsort()[::-1]
        cls_dets = cls_dets[order]

        keep = []
        while len(cls_dets) > 0:
            keep.append(cls_dets[0])
            if len(cls_dets) == 1:
                break
            ious = compute_iou_matrix(cls_dets[0:1, 1:5], cls_dets[1:, 1:5])[0]
            cls_dets = cls_dets[1:][ious < iou_thresh]

        results.extend(keep)

    return results


def xyxy_to_yolo(x1, y1, x2, y2, w, h):
    return ((x1 + x2) / 2 / w, (y1 + y2) / 2 / h, (x2 - x1) / w, (y2 - y1) / h)


# ==================== 推理 ====================

def infer_deimv2(model, transform, img_path, device, conf):
    img = Image.open(img_path).convert("RGB")
    w, h = img.size
    im_data = transform(img).unsqueeze(0).to(device)
    orig_size = torch.tensor([[w, h]]).to(device)

    with torch.no_grad():
        labels, boxes, scores = model(im_data, orig_size)

    mask = scores[0] > conf
    dets = []
    for cls_id, box, sc in zip(labels[0][mask].cpu().numpy(),
                                boxes[0][mask].cpu().numpy(),
                                scores[0][mask].cpu().numpy()):
        x1, y1, x2, y2 = box
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        if x2 > x1 and y2 > y1:
            dets.append((int(cls_id), x1, y1, x2, y2, float(sc)))
    return dets, w, h


def infer_yolo(model, img_path, conf):
    results = model.predict(str(img_path), imgsz=[384, 640], conf=conf,
                            iou=0.45, verbose=False)[0]
    img = Image.open(img_path)
    w, h = img.size
    dets = []
    if results.boxes is not None:
        for box in results.boxes:
            cls_id = int(box.cls[0])
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            sc = float(box.conf[0])
            dets.append((cls_id, x1, y1, x2, y2, sc))
    return dets, w, h


# ==================== main ====================

def main():
    args = parse_args()
    device = f"cuda:{args.device}" if args.device.isdigit() else args.device

    shard_info = f" [shard {args.shard_id}/{args.num_shards}]" if args.num_shards > 1 else ""
    print("=" * 60)
    print(f" 双模型交叉验证伪标签生成{shard_info}")
    print(f"  DEIMv2 conf: {args.conf_deimv2}, YOLO conf: {args.conf_yolo}")
    print(f"  匹配 IoU:    {args.match_iou}")
    print(f"  NMS IoU:     {args.nms_iou}")
    print("=" * 60)

    # 加载模型
    print("\n加载 DEIMv2...")
    deimv2, img_size, vit_backbone = load_deimv2(args.deimv2_config, args.deimv2_ckpt, device)
    deimv2_transform = build_deimv2_transform(img_size, vit_backbone)

    print("加载 YOLOv8s...")
    yolo = YOLO(args.yolo_ckpt)

    # 收集图片
    input_path = Path(args.input_dir)
    images = sorted([f for f in input_path.iterdir() if f.suffix.lower() in IMG_EXTS])
    if args.num_shards > 1:
        images = images[args.shard_id::args.num_shards]
    print(f"待处理: {len(images)} 张")

    # 输出目录
    out_images = Path(args.output_dir) / "images"
    out_labels = Path(args.output_dir) / "labels"
    out_images.mkdir(parents=True, exist_ok=True)
    out_labels.mkdir(parents=True, exist_ok=True)

    # 推理 + 交叉验证
    stats = {"total": 0, "matched": 0, "boxes": 0,
             "deimv2_only": 0, "yolo_only": 0, "per_class": {0: 0, 1: 0}}

    for img_path in tqdm(images, desc="交叉验证"):
        try:
            dets_d, w, h = infer_deimv2(deimv2, deimv2_transform, img_path, device, args.conf_deimv2)
            dets_y, _, _ = infer_yolo(yolo, img_path, args.conf_yolo)
        except Exception as e:
            print(f"  [跳过] {img_path.name}: {e}")
            continue

        stats["total"] += 1
        stats["deimv2_only"] += len(dets_d)
        stats["yolo_only"] += len(dets_y)

        # 交叉验证
        matched = cross_validate(dets_d, dets_y, args.match_iou)
        # NMS 去重
        final = nms_per_class(matched, args.nms_iou)

        if not final:
            continue

        stats["matched"] += 1
        stats["boxes"] += len(final)

        # 写标签
        lines = []
        for det in final:
            cls_id = int(det[0])
            x1, y1, x2, y2 = det[1], det[2], det[3], det[4]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            cx, cy, bw, bh = xyxy_to_yolo(x1, y1, x2, y2, w, h)
            lines.append(f"{cls_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
            stats["per_class"][cls_id] = stats["per_class"].get(cls_id, 0) + 1

        if lines:
            dst_img = out_images / img_path.name
            if not dst_img.exists():
                os.symlink(str(img_path.resolve()), str(dst_img))
            with open(out_labels / (img_path.stem + ".txt"), "w") as f:
                f.write("\n".join(lines) + "\n")

    # 统计
    print("\n" + "=" * 60)
    print(" 交叉验证伪标签完成")
    print(f"  总图片:         {stats['total']}")
    print(f"  有匹配结果:     {stats['matched']}")
    print(f"  总框数:         {stats['boxes']}")
    print(f"  DEIMv2 原始框:  {stats['deimv2_only']}")
    print(f"  YOLOv8s 原始框: {stats['yolo_only']}")
    for cid, name in CLASS_NAMES.items():
        print(f"  {name}:           {stats['per_class'].get(cid, 0)}")
    if stats['deimv2_only'] > 0:
        print(f"  保留率:         {stats['boxes']}/{stats['deimv2_only']} = "
              f"{stats['boxes']/stats['deimv2_only']*100:.1f}%")
    print(f"  输出:           {args.output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
