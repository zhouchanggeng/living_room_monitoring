"""
Active Learning 采样: 挑选模型不确定性高的图片优先人工标注

不确定性度量策略:
  1. 双模型分歧: DEIMv2 和 YOLOv8s 检测结果不一致的图片
  2. 低置信度: 模型有检测但置信度处于中间地带的图片
  3. Kid 优先: 含 Kid 检测的图片优先级更高

输出: 按不确定性排序的图片列表, 复制到采样目录并生成预标注 JSON (X-AnyLabeling 格式)

用法:
    python tools/active_learning_sample.py --num 1000 --device 0
    python tools/active_learning_sample.py --num 500 --kid-boost 2.0 --device 0
"""

import argparse
import json
import os
import shutil
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
DEIMV2_DIR = PROJECT_ROOT / "training" / "DEIMv2"

DEIMV2_CONFIG = str(DEIMV2_DIR / "configs/deimv2/deimv2_dinov3_x_kids_care.yml")
DEIMV2_CKPT = str(PROJECT_ROOT / "runs/train/iter_20260429_deimv2_x/best_stg2.pth")
YOLO_CKPT = str(PROJECT_ROOT / "runs/train/distill_20260430_yolov8s/weights/best.pt")
UNLABELED_DIR = str(PROJECT_ROOT / "pending_data/all_20260416")

CLASS_NAMES = {0: "Kid", 1: "Adult"}
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args():
    p = argparse.ArgumentParser(description="Active Learning 采样")
    p.add_argument("--deimv2-config", type=str, default=DEIMV2_CONFIG)
    p.add_argument("--deimv2-ckpt", type=str, default=DEIMV2_CKPT)
    p.add_argument("--yolo-ckpt", type=str, default=YOLO_CKPT)
    p.add_argument("--input-dir", type=str, default=UNLABELED_DIR)
    p.add_argument("--output-dir", type=str, default=None,
                   help="输出目录 (默认: pending_data/active_sample_YYYYMMDD)")
    p.add_argument("--num", type=int, default=1000, help="采样数量")
    p.add_argument("--conf-low", type=float, default=0.3, help="低置信度阈值 (检测下限)")
    p.add_argument("--conf-uncertain", type=float, default=0.65,
                   help="不确定区间上限 (0.3-0.65 为不确定区间)")
    p.add_argument("--kid-boost", type=float, default=2.0,
                   help="含 Kid 检测的图片不确定性加权倍数")
    p.add_argument("--device", type=str, default="0")
    p.add_argument("--exclude-dirs", nargs="*", default=None,
                   help="排除已标注的图片目录")
    p.add_argument("--scan-limit", type=int, default=0,
                   help="扫描图片上限 (0=全部, 用于快速测试)")
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


# ==================== 不确定性计算 ====================

def compute_iou(box_a, box_b):
    """单对框 IoU, xyxy 格式."""
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    return inter / (area_a + area_b - inter + 1e-6)


def compute_uncertainty(dets_d, dets_y, conf_low, conf_uncertain, kid_boost):
    """
    计算单张图片的不确定性分数 (越高越应该优先标注).

    分歧度量:
      - 一个模型检测到而另一个没有 -> 高不确定性
      - 两个模型检测到但类别不同 -> 最高不确定性
      - 置信度在 [conf_low, conf_uncertain] 区间 -> 中等不确定性
    """
    score = 0.0
    has_kid = False

    # 1. 数量分歧
    n_d, n_y = len(dets_d), len(dets_y)
    count_diff = abs(n_d - n_y)
    score += count_diff * 1.0  # 每个数量差异 +1

    # 2. 逐框匹配分析
    matched_d = set()
    matched_y = set()

    for i, d in enumerate(dets_d):
        best_j, best_iou = -1, 0.3
        for j, y in enumerate(dets_y):
            if j in matched_y:
                continue
            iou = compute_iou(d[1:5], y[1:5])
            if iou > best_iou:
                best_iou = iou
                best_j = j
        if best_j >= 0:
            matched_d.add(i)
            matched_y.add(best_j)
            # 类别不一致 -> 高分歧
            if int(d[0]) != int(dets_y[best_j][0]):
                score += 3.0
            # 置信度在不确定区间
            for sc in [d[5], dets_y[best_j][5]]:
                if conf_low < sc < conf_uncertain:
                    score += 0.5

            if int(d[0]) == 0 or int(dets_y[best_j][0]) == 0:
                has_kid = True

    # 3. 未匹配框 (一个模型检测到另一个没有)
    unmatched_d = [d for i, d in enumerate(dets_d) if i not in matched_d]
    unmatched_y = [y for j, y in enumerate(dets_y) if j not in matched_y]

    for d in unmatched_d:
        score += 1.5  # DEIMv2 独有检测
        if conf_low < d[5] < conf_uncertain:
            score += 0.5
        if int(d[0]) == 0:
            has_kid = True

    for y in unmatched_y:
        score += 1.5  # YOLOv8s 独有检测
        if conf_low < y[5] < conf_uncertain:
            score += 0.5
        if int(y[0]) == 0:
            has_kid = True

    # 4. Kid 加权
    if has_kid:
        score *= kid_boost

    return score


# ==================== X-AnyLabeling JSON ====================

def make_xanylabeling_json(img_path, dets_d, dets_y, w, h):
    """生成预标注 JSON, 标记来源便于人工审核."""
    shapes = []
    for src, dets in [("DEIMv2", dets_d), ("YOLOv8s", dets_y)]:
        for d in dets:
            cls_id = int(d[0])
            x1, y1, x2, y2, sc = float(d[1]), float(d[2]), float(d[3]), float(d[4]), float(d[5])
            shapes.append({
                "label": CLASS_NAMES.get(cls_id, str(cls_id)),
                "score": round(sc, 4),
                "points": [[round(x1, 2), round(y1, 2)], [round(x2, 2), round(y2, 2)]],
                "group_id": None,
                "description": src,
                "difficult": False,
                "shape_type": "rectangle",
                "flags": {},
                "attributes": {"source": src},
            })
    return {
        "version": "2.4.0",
        "flags": {},
        "shapes": shapes,
        "imagePath": img_path.name,
        "imageData": None,
        "imageHeight": h,
        "imageWidth": w,
    }


# ==================== main ====================

def main():
    args = parse_args()
    device = f"cuda:{args.device}" if args.device.isdigit() else args.device

    if args.output_dir is None:
        from datetime import datetime
        date_str = datetime.now().strftime("%Y%m%d")
        args.output_dir = str(PROJECT_ROOT / "pending_data" / f"active_sample_{date_str}")

    print("=" * 60)
    print(" Active Learning 采样")
    print(f"  采样数量:   {args.num}")
    print(f"  不确定区间: [{args.conf_low}, {args.conf_uncertain}]")
    print(f"  Kid 加权:   {args.kid_boost}x")
    print(f"  输出:       {args.output_dir}")
    print("=" * 60)

    # 加载模型
    print("\n加载模型...")
    deimv2, img_size, vit_backbone = load_deimv2(args.deimv2_config, args.deimv2_ckpt, device)
    deimv2_tf = T.Compose([
        T.Resize(img_size), T.ToTensor(),
        *([ T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])] if vit_backbone else [])
    ])
    yolo = YOLO(args.yolo_ckpt)

    # 收集图片 (排除已标注)
    input_path = Path(args.input_dir)
    exclude_names = set()
    # 自动排除所有已用于训练的图片
    for data_dir in (PROJECT_ROOT / "data").iterdir():
        img_dir = data_dir / "train" / "images"
        if img_dir.exists():
            exclude_names |= {f.name for f in img_dir.iterdir() if f.suffix.lower() in IMG_EXTS}
    if args.exclude_dirs:
        for d in args.exclude_dirs:
            d = Path(d)
            if d.exists():
                exclude_names |= {f.name for f in d.iterdir() if f.suffix.lower() in IMG_EXTS}
    print(f"排除已标注: {len(exclude_names)} 张")

    images = sorted([f for f in input_path.iterdir()
                     if f.suffix.lower() in IMG_EXTS and f.name not in exclude_names])
    if args.scan_limit > 0:
        images = images[:args.scan_limit]
    print(f"待扫描: {len(images)} 张\n")

    # 扫描计算不确定性
    uncertainties = []  # (score, img_path, dets_d, dets_y, w, h)

    for img_path in tqdm(images, desc="计算不确定性"):
        try:
            img = Image.open(img_path).convert("RGB")
            w, h = img.size
        except Exception:
            continue

        # DEIMv2 推理
        with torch.no_grad():
            im_data = deimv2_tf(img).unsqueeze(0).to(device)
            orig_size = torch.tensor([[w, h]]).to(device)
            labels, boxes, scores = deimv2(im_data, orig_size)
            mask = scores[0] > args.conf_low
            dets_d = [(int(l), *b, float(s)) for l, b, s in
                      zip(labels[0][mask].cpu().numpy(),
                          boxes[0][mask].cpu().numpy(),
                          scores[0][mask].cpu().numpy())]

        # YOLOv8s 推理
        results = yolo.predict(str(img_path), imgsz=[384, 640],
                               conf=args.conf_low, iou=0.45, verbose=False)[0]
        dets_y = []
        if results.boxes is not None:
            for box in results.boxes:
                dets_y.append((int(box.cls[0]), *box.xyxy[0].tolist(), float(box.conf[0])))

        # 计算不确定性
        unc = compute_uncertainty(dets_d, dets_y, args.conf_low,
                                 args.conf_uncertain, args.kid_boost)

        if unc > 0:
            uncertainties.append((unc, img_path, dets_d, dets_y, w, h))

    # 按不确定性降序排序, 取 top-N
    uncertainties.sort(key=lambda x: x[0], reverse=True)
    selected = uncertainties[:args.num]

    print(f"\n不确定性 > 0 的图片: {len(uncertainties)}")
    print(f"选取 top {len(selected)} 张")

    if not selected:
        print("没有需要标注的图片!")
        return

    # 输出
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    kid_count = 0
    for unc, img_path, dets_d, dets_y, w, h in selected:
        # 复制图片
        shutil.copy2(img_path, out_dir / img_path.name)
        # 生成预标注 JSON
        ann = make_xanylabeling_json(img_path, dets_d, dets_y, w, h)
        with open(out_dir / (img_path.stem + ".json"), "w") as f:
            json.dump(ann, f, ensure_ascii=False, indent=2)
        # 统计 Kid
        if any(int(d[0]) == 0 for d in dets_d + dets_y):
            kid_count += 1

    # 保存不确定性排名
    ranking = [{"rank": i + 1, "image": s[1].name, "uncertainty": round(s[0], 3),
                "deimv2_dets": len(s[2]), "yolo_dets": len(s[3])}
               for i, s in enumerate(selected)]
    with open(out_dir / "uncertainty_ranking.json", "w") as f:
        json.dump(ranking, f, ensure_ascii=False, indent=2)

    print(f"\n" + "=" * 60)
    print(f" 采样完成")
    print(f"  采样数量:     {len(selected)}")
    print(f"  含 Kid 图片:  {kid_count} ({kid_count/len(selected)*100:.1f}%)")
    print(f"  不确定性范围: [{selected[-1][0]:.2f}, {selected[0][0]:.2f}]")
    print(f"  输出目录:     {out_dir}")
    print(f"  排名文件:     {out_dir / 'uncertainty_ranking.json'}")
    print(f"\n  用 X-AnyLabeling 打开 {out_dir} 即可开始标注")
    print(f"  预标注中 DEIMv2 和 YOLOv8s 的检测结果已标记来源 (description 字段)")
    print("=" * 60)


if __name__ == "__main__":
    main()
