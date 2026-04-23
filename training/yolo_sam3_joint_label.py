"""
YOLO + SAM3 联合标注: YOLO 提供候选框, SAM3 做验证/补充, 提高标签准确率.

策略:
  1. YOLO 检测候选框
  2. SAM3 文本 prompt 独立检测
  3. IoU 匹配融合:
     - 双方匹配(IoU>阈值): 保留, 置信度取均值 -> "confirmed"
     - 仅 YOLO 检测到(高置信): 保留但标记 -> "yolo_only"
     - 仅 SAM3 检测到(高置信): 保留但标记 -> "sam3_only"
     - 低置信且无匹配: 丢弃

用法 (在 conda 环境 x-anylabeling-server 中运行):
    python training/yolo_sam3_joint_label.py \
        --source origin_data/sample_1200_new \
        --yolo-weights runs/train/sample1000_yolov8s/weights/best.pt \
        --sam3-checkpoint models/sam3.pt \
        --device 0
"""

import argparse
import json
import os
import sys

SAM3_BASE = "/home/zcg/workspace/X-AnyLabeling-Server/app/models"
if SAM3_BASE not in sys.path:
    sys.path.insert(0, SAM3_BASE)

from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from ultralytics import YOLO

from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
YOLO_CLASSES = {0: "Kid", 1: "Adult"}
SAM3_PROMPTS = {"child": "Kid", "an adult": "Adult"}
BPE_PATH = "models/bpe_simple_vocab_16e6.txt.gz"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--source", type=str, default="origin_data/sample_1200_new")
    p.add_argument("--yolo-weights", type=str,
                   default="runs/train/sample1000_yolov8s/weights/best.pt")
    p.add_argument("--sam3-checkpoint", type=str, default="models/sam3.pt")
    p.add_argument("--bpe", type=str, default=BPE_PATH)
    p.add_argument("--yolo-conf", type=float, default=0.25)
    p.add_argument("--sam3-conf", type=float, default=0.3)
    p.add_argument("--iou-thresh", type=float, default=0.3,
                   help="YOLO-SAM3 匹配 IoU 阈值")
    p.add_argument("--yolo-only-conf", type=float, default=0.6,
                   help="仅YOLO检测到时, 需要的最低置信度")
    p.add_argument("--sam3-only-conf", type=float, default=0.5,
                   help="仅SAM3检测到时, 需要的最低置信度")
    p.add_argument("--imgsz", nargs=2, type=int, default=[384, 640])
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--device", type=str, default="0")
    return p.parse_args()


def compute_iou(box1, box2):
    """box: [x1, y1, x2, y2]"""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0


def clamp_box(x1, y1, x2, y2, w, h):
    return (round(max(0, x1), 2), round(max(0, y1), 2),
            round(min(w, x2), 2), round(min(h, y2), 2))


def run_sam3_on_image(processor, img_pil, sam3_conf):
    """SAM3 检测单张图片, 返回 [{label, score, box:[x1,y1,x2,y2]}]"""
    img_w, img_h = img_pil.size
    detections = []
    for prompt_text, label_name in SAM3_PROMPTS.items():
        state = processor.set_image(img_pil)
        state = processor.set_text_prompt(prompt_text, state)
        if "boxes" in state and len(state["boxes"]) > 0:
            boxes = state["boxes"].cpu()
            scores = state["scores"].cpu()
            for i in range(len(boxes)):
                score = float(scores[i])
                if score < sam3_conf:
                    continue
                x1, y1, x2, y2 = boxes[i].tolist()
                x1, y1, x2, y2 = clamp_box(x1, y1, x2, y2, img_w, img_h)
                detections.append({
                    "label": label_name, "score": score,
                    "box": [x1, y1, x2, y2],
                })
    return detections


def fuse_detections(yolo_dets, sam3_dets, iou_thresh, yolo_only_conf, sam3_only_conf):
    """融合 YOLO 和 SAM3 检测结果."""
    results = []
    sam3_matched = set()

    for yd in yolo_dets:
        best_iou, best_idx = 0, -1
        for j, sd in enumerate(sam3_dets):
            if sd["label"] != yd["label"] or j in sam3_matched:
                continue
            iou = compute_iou(yd["box"], sd["box"])
            if iou > best_iou:
                best_iou, best_idx = iou, j

        if best_iou >= iou_thresh and best_idx >= 0:
            # 双方匹配: 框取 YOLO 的(通常更精准), 置信度取均值
            sd = sam3_dets[best_idx]
            sam3_matched.add(best_idx)
            avg_score = (yd["score"] + sd["score"]) / 2
            results.append({
                "label": yd["label"],
                "score": round(avg_score, 4),
                "box": yd["box"],
                "source": "confirmed",
            })
        elif yd["score"] >= yolo_only_conf:
            # 仅 YOLO, 高置信才保留
            results.append({
                "label": yd["label"],
                "score": round(yd["score"], 4),
                "box": yd["box"],
                "source": "yolo_only",
            })

    # 仅 SAM3 检测到的
    for j, sd in enumerate(sam3_dets):
        if j not in sam3_matched and sd["score"] >= sam3_only_conf:
            results.append({
                "label": sd["label"],
                "score": round(sd["score"], 4),
                "box": sd["box"],
                "source": "sam3_only",
            })

    return results


def build_json(img_name, img_w, img_h, fused):
    shapes = []
    for det in fused:
        shapes.append({
            "label": det["label"],
            "score": det["score"],
            "points": [[det["box"][0], det["box"][1]],
                       [det["box"][2], det["box"][3]]],
            "group_id": None,
            "description": det["source"],
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
    source_dir = Path(args.source)
    img_files = sorted([f for f in source_dir.iterdir() if f.suffix.lower() in IMG_EXTS])
    total = len(img_files)
    print(f"图片: {total} 张")

    # 加载 YOLO
    print("加载 YOLO...")
    yolo_model = YOLO(args.yolo_weights)

    # 加载 SAM3
    print("加载 SAM3...")
    sam3_device = "cuda" if args.device.isdigit() else args.device
    sam3_model = build_sam3_image_model(
        bpe_path=os.path.abspath(args.bpe),
        device=sam3_device,
        eval_mode=True,
        checkpoint_path=os.path.abspath(args.sam3_checkpoint),
        load_from_HF=False,
        enable_segmentation=True,
    )
    sam3_processor = Sam3Processor(
        model=sam3_model,
        device=sam3_device,
        confidence_threshold=args.sam3_conf,
    )
    print("模型加载完成")

    # 统计
    stats = {"confirmed": 0, "yolo_only": 0, "sam3_only": 0, "total_imgs": 0}

    # 先批量 YOLO 推理, 缓存结果
    print("YOLO 批量推理...")
    yolo_results_map = {}  # img_name -> [dets]
    for i in tqdm(range(0, total, args.batch), desc="YOLO"):
        batch_paths = img_files[i:i + args.batch]
        results_list = yolo_model.predict(
            source=[str(p) for p in batch_paths],
            imgsz=args.imgsz, conf=args.yolo_conf, iou=0.45,
            device=args.device, verbose=False,
        )
        for img_path, results in zip(batch_paths, results_list):
            dets = []
            if results.boxes is not None:
                for box in results.boxes:
                    score = float(box.conf[0])
                    cls_id = int(box.cls[0])
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    dets.append({
                        "label": YOLO_CLASSES.get(cls_id, str(cls_id)),
                        "score": score,
                        "box": [round(x1, 2), round(y1, 2),
                                round(x2, 2), round(y2, 2)],
                    })
            yolo_results_map[img_path.name] = dets

    # SAM3 逐张推理 + 融合
    print("SAM3 推理 + 融合...")
    json_count = 0
    for img_path in tqdm(img_files, desc="SAM3+融合"):
        img_pil = Image.open(img_path).convert("RGB")
        img_w, img_h = img_pil.size

        yolo_dets = yolo_results_map.get(img_path.name, [])
        sam3_dets = run_sam3_on_image(sam3_processor, img_pil, args.sam3_conf)

        fused = fuse_detections(
            yolo_dets, sam3_dets,
            args.iou_thresh, args.yolo_only_conf, args.sam3_only_conf,
        )

        for det in fused:
            stats[det["source"]] += 1
        stats["total_imgs"] += 1

        annotation = build_json(img_path.name, img_w, img_h, fused)
        with open(img_path.with_suffix(".json"), "w", encoding="utf-8") as f:
            json.dump(annotation, f, ensure_ascii=False, indent=2)
        json_count += 1

    print(f"\n完成! 生成 {json_count} 个 JSON")
    print(f"统计: confirmed={stats['confirmed']}, "
          f"yolo_only={stats['yolo_only']}, sam3_only={stats['sam3_only']}")
    print("用 X-AnyLabeling 打开目录审核, description 字段标注了来源")


if __name__ == "__main__":
    main()
