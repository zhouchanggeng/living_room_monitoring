"""
第一阶段: DEIMv2 伪标签生成

使用训练好的 DEIMv2 (DINOv3-X) 对未标注图片集 all_20260416 进行推理,
以高置信度阈值过滤噪声, 生成 YOLO 格式伪标签, 供 YOLOv8s 半监督蒸馏训练使用.

用法:
    python training/generate_pseudo_labels.py
    python training/generate_pseudo_labels.py --conf 0.7 --device 0
"""

import argparse
import os
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEIMV2_DIR = Path(__file__).resolve().parent / "DEIMv2"

# 默认配置
DEIMV2_CONFIG = str(DEIMV2_DIR / "configs/deimv2/deimv2_dinov3_x_kids_care.yml")
DEIMV2_CKPT = str(PROJECT_ROOT / "runs/train/iter_20260429_deimv2_x/best_stg2.pth")
UNLABELED_DIR = str(PROJECT_ROOT / "pending_data/all_20260416")
OUTPUT_DIR = str(PROJECT_ROOT / "data/pseudo_labels_deimv2")

# 类别映射: DEIMv2 COCO 格式 category_id -> YOLO class_id
# DEIMv2 训练时 num_classes=2, 输出 label 0=Kid, 1=Adult
CLASS_NAMES = {0: "Kid", 1: "Adult"}
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args():
    p = argparse.ArgumentParser(description="DEIMv2 伪标签生成")
    p.add_argument("--config", type=str, default=DEIMV2_CONFIG)
    p.add_argument("--checkpoint", type=str, default=DEIMV2_CKPT)
    p.add_argument("--input-dir", type=str, default=UNLABELED_DIR)
    p.add_argument("--output-dir", type=str, default=OUTPUT_DIR)
    p.add_argument("--conf", type=float, default=0.65, help="置信度阈值 (高阈值过滤噪声)")
    p.add_argument("--device", type=str, default="0")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--exclude-dirs", nargs="*", default=None,
                   help="排除已用于训练的图片目录")
    # 多卡分片参数
    p.add_argument("--shard-id", type=int, default=0, help="当前分片 ID (多卡并行)")
    p.add_argument("--num-shards", type=int, default=1, help="总分片数 (多卡并行)")
    return p.parse_args()


def load_deimv2_model(config_path, checkpoint_path, device):
    """加载 DEIMv2 模型并切换到 deploy 模式."""
    sys.path.insert(0, str(DEIMV2_DIR))
    from engine.core import YAMLConfig

    cfg = YAMLConfig(config_path, resume=checkpoint_path)

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if "ema" in checkpoint:
        state = checkpoint["ema"]["module"]
    else:
        state = checkpoint["model"]

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
    vit_backbone = cfg.yaml_cfg.get("DINOv3STAs", False)

    return model, img_size, bool(vit_backbone)


def build_transform(img_size, vit_backbone):
    """构建图像预处理 transform."""
    ops = [T.Resize(img_size), T.ToTensor()]
    if vit_backbone:
        ops.append(T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
    return T.Compose(ops)


def collect_images(input_dir, exclude_dirs=None):
    """收集待推理图片, 排除已用图片."""
    exclude_names = set()
    if exclude_dirs:
        for d in exclude_dirs:
            d = Path(d)
            if d.exists():
                exclude_names |= {f.name for f in d.iterdir() if f.suffix.lower() in IMG_EXTS}
    print(f"排除已有图片: {len(exclude_names)} 张")

    input_path = Path(input_dir)
    candidates = sorted([
        f for f in input_path.iterdir()
        if f.suffix.lower() in IMG_EXTS and f.name not in exclude_names
    ])
    print(f"待推理图片: {len(candidates)} 张")
    return candidates


def xyxy_to_yolo(x1, y1, x2, y2, img_w, img_h):
    """xyxy 绝对坐标 -> YOLO 归一化 cx cy w h."""
    cx = (x1 + x2) / 2.0 / img_w
    cy = (y1 + y2) / 2.0 / img_h
    w = (x2 - x1) / img_w
    h = (y2 - y1) / img_h
    return cx, cy, w, h


def main():
    args = parse_args()
    device = f"cuda:{args.device}" if args.device.isdigit() else args.device

    shard_info = ""
    if args.num_shards > 1:
        shard_info = f" [shard {args.shard_id}/{args.num_shards}]"

    print("=" * 60)
    print(f" DEIMv2 伪标签生成{shard_info}")
    print(f"  模型:     {args.checkpoint}")
    print(f"  输入:     {args.input_dir}")
    print(f"  输出:     {args.output_dir}")
    print(f"  置信度:   {args.conf}")
    print(f"  设备:     {device}")
    print("=" * 60)

    # 1. 加载模型
    print("\n加载 DEIMv2 模型...")
    model, img_size, vit_backbone = load_deimv2_model(args.config, args.checkpoint, device)
    transform = build_transform(img_size, vit_backbone)
    print(f"  推理尺寸: {img_size}, ViT backbone: {vit_backbone}")

    # 2. 收集图片并按分片切分
    images = collect_images(args.input_dir, args.exclude_dirs)
    if args.num_shards > 1:
        images = images[args.shard_id::args.num_shards]
        print(f"  分片 {args.shard_id}: 处理 {len(images)} 张图片")
    if not images:
        print("没有待推理的图片!")
        return

    # 3. 创建输出目录
    out_images = Path(args.output_dir) / "images"
    out_labels = Path(args.output_dir) / "labels"
    out_images.mkdir(parents=True, exist_ok=True)
    out_labels.mkdir(parents=True, exist_ok=True)

    # 4. 批量推理
    stats = {"total": 0, "with_det": 0, "total_boxes": 0, "per_class": {0: 0, 1: 0}}

    with torch.no_grad():
        for i in tqdm(range(0, len(images), args.batch_size), desc="推理中"):
            batch_paths = images[i:i + args.batch_size]
            batch_imgs = []
            batch_sizes = []

            for img_path in batch_paths:
                try:
                    img = Image.open(img_path).convert("RGB")
                except Exception as e:
                    print(f"  [跳过] 无法读取: {img_path.name} ({e})")
                    continue
                w, h = img.size
                batch_imgs.append(transform(img))
                batch_sizes.append((img_path, w, h))

            if not batch_imgs:
                continue

            # 逐张推理 (DEIMv2 deploy 模式不支持 batch)
            for img_tensor, (img_path, w, h) in zip(batch_imgs, batch_sizes):
                im_data = img_tensor.unsqueeze(0).to(device)
                orig_size = torch.tensor([[w, h]]).to(device)

                labels, boxes, scores = model(im_data, orig_size)

                # 过滤低置信度
                mask = scores[0] > args.conf
                det_labels = labels[0][mask].cpu().numpy()
                det_boxes = boxes[0][mask].cpu().numpy()
                det_scores = scores[0][mask].cpu().numpy()

                stats["total"] += 1

                if len(det_labels) == 0:
                    continue

                stats["with_det"] += 1
                stats["total_boxes"] += len(det_labels)

                # 写 YOLO 格式标签
                label_lines = []
                for cls_id, box, score in zip(det_labels, det_boxes, det_scores):
                    cls_id = int(cls_id)
                    x1, y1, x2, y2 = box
                    # 边界裁剪
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(w, x2), min(h, y2)
                    if x2 <= x1 or y2 <= y1:
                        continue
                    cx, cy, bw, bh = xyxy_to_yolo(x1, y1, x2, y2, w, h)
                    label_lines.append(f"{cls_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
                    stats["per_class"][cls_id] = stats["per_class"].get(cls_id, 0) + 1

                if label_lines:
                    # 软链接图片到输出目录
                    dst_img = out_images / img_path.name
                    if not dst_img.exists():
                        os.symlink(str(img_path), str(dst_img))

                    # 写标签文件
                    label_file = out_labels / (img_path.stem + ".txt")
                    with open(label_file, "w") as f:
                        f.write("\n".join(label_lines) + "\n")

    # 5. 统计
    print("\n" + "=" * 60)
    print(" 伪标签生成完成")
    print(f"  总图片:       {stats['total']}")
    print(f"  有检测结果:   {stats['with_det']}")
    print(f"  总检测框:     {stats['total_boxes']}")
    for cls_id, name in CLASS_NAMES.items():
        print(f"  {name}:         {stats['per_class'].get(cls_id, 0)}")
    print(f"  输出目录:     {args.output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
