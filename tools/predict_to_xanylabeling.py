"""
使用训练好的模型对图片文件夹进行推理,
输出 X-AnyLabeling (LabelMe) 兼容的 JSON 标注文件, 便于后续人工审核.

支持模型: YOLOv8s, DEIMv2

用法 (YOLO):
    python tools/predict_to_xanylabeling.py \
        --weights runs/train/kids_care_yolov8s/weights/best.pt \
        --source all_20260416 \
        --conf 0.25

用法 (DEIMv2):
    python tools/predict_to_xanylabeling.py \
        --model-type deimv2 \
        --config training/DEIMv2/configs/deimv2/deimv2_dinov3_x_kids_care.yml \
        --weights runs/train/iter_20260425_deimv2_x/best.pth \
        --source all_20260416 \
        --conf 0.45
"""

import argparse
import base64
import json
import os
import sys
from pathlib import Path

from PIL import Image
from tqdm import tqdm

# 类别名映射
CLASS_NAMES = {0: "Kid", 1: "Adult"}
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args():
    p = argparse.ArgumentParser(description="模型推理 -> X-AnyLabeling JSON")
    p.add_argument("--model-type", type=str, default="yolo",
                   choices=["yolo", "deimv2"], help="模型类型")
    p.add_argument("--weights", type=str,
                   default="runs/train/kids_care_yolov8s/weights/best.pt")
    p.add_argument("--config", type=str, default=None,
                   help="DEIMv2 配置文件路径 (仅 deimv2 需要)")
    p.add_argument("--source", type=str, default="all_20260416",
                   help="图片目录")
    p.add_argument("--conf", type=float, default=0.25, help="置信度阈值")
    p.add_argument("--iou", type=float, default=0.45, help="NMS IoU阈值 (仅 yolo)")
    p.add_argument("--imgsz", nargs=2, type=int, default=[384, 640],
                   help="推理尺寸 (h w), 仅 yolo 使用; deimv2 从配置读取")
    p.add_argument("--batch", type=int, default=32, help="推理batch大小 (仅 yolo)")
    p.add_argument("--device", type=str, default="0")
    p.add_argument("--save-img", action="store_true", default=False,
                   help="是否在JSON中嵌入base64图片(文件会很大)")
    p.add_argument("--output", type=str, default=None,
                   help="输出目录 (默认保存在图片同目录)")
    return p.parse_args()


def build_xanylabeling_json(img_path: Path, shapes: list,
                            save_img_data: bool = False):
    """构建 X-AnyLabeling JSON 格式."""
    img = Image.open(img_path)
    w, h = img.size

    image_data = None
    if save_img_data:
        with open(img_path, "rb") as f:
            image_data = base64.b64encode(f.read()).decode("utf-8")

    return {
        "version": "2.4.0",
        "flags": {},
        "shapes": shapes,
        "imagePath": img_path.name,
        "imageData": image_data,
        "imageHeight": h,
        "imageWidth": w,
    }


def save_json(json_path: Path, data: dict):
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def make_shape(label, x1, y1, x2, y2, score):
    return {
        "label": label,
        "score": round(score, 4),
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
    }


# ==================== YOLO 推理 ====================

def run_yolo(args, img_files, out_dir):
    from ultralytics import YOLO

    model = YOLO(args.weights)
    batch_size = args.batch
    json_count = 0

    pbar = tqdm(total=len(img_files), desc="YOLO 推理中")
    for i in range(0, len(img_files), batch_size):
        batch_paths = img_files[i:i + batch_size]
        results_list = model.predict(
            source=[str(p) for p in batch_paths],
            imgsz=args.imgsz, conf=args.conf, iou=args.iou,
            device=args.device, verbose=False,
        )
        for img_path, results in zip(batch_paths, results_list):
            shapes = []
            if results.boxes is not None:
                for box in results.boxes:
                    conf = float(box.conf[0])
                    if conf < args.conf:
                        continue
                    cls_id = int(box.cls[0])
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    shapes.append(make_shape(
                        CLASS_NAMES.get(cls_id, str(cls_id)),
                        x1, y1, x2, y2, conf))
            annotation = build_xanylabeling_json(img_path, shapes, args.save_img)
            json_path = out_dir / (img_path.stem + ".json")
            save_json(json_path, annotation)
            json_count += 1
        pbar.update(len(batch_paths))
    pbar.close()
    return json_count


# ==================== DEIMv2 推理 ====================

def load_deimv2_model(config_path, weights_path, device):
    """加载 DEIMv2 模型, 返回 (model, img_size, vit_backbone)."""
    import torch
    import torch.nn as nn

    # 将 DEIMv2 engine 加入 sys.path
    deimv2_root = Path(__file__).resolve().parent.parent / "training" / "DEIMv2"
    if str(deimv2_root) not in sys.path:
        sys.path.insert(0, str(deimv2_root))
    from engine.core import YAMLConfig

    cfg = YAMLConfig(config_path, resume=weights_path)
    if 'HGNetv2' in cfg.yaml_cfg:
        cfg.yaml_cfg['HGNetv2']['pretrained'] = False

    checkpoint = torch.load(weights_path, map_location='cpu')
    state = checkpoint['ema']['module'] if 'ema' in checkpoint else checkpoint['model']
    cfg.model.load_state_dict(state)

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = cfg.model.deploy()
            self.postprocessor = cfg.postprocessor.deploy()

        def forward(self, images, orig_target_sizes):
            outputs = self.model(images)
            return self.postprocessor(outputs, orig_target_sizes)

    model = Model().to(device).eval()
    img_size = cfg.yaml_cfg["eval_spatial_size"]
    vit_backbone = bool(cfg.yaml_cfg.get('DINOv3STAs', False))
    return model, img_size, vit_backbone


def run_deimv2(args, img_files, out_dir):
    import torch
    import torchvision.transforms as T

    device = torch.device(f"cuda:{args.device}" if args.device.isdigit() else args.device)
    model, img_size, vit_backbone = load_deimv2_model(args.config, args.weights, device)

    transforms = T.Compose([
        T.Resize(img_size),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        if vit_backbone else T.Lambda(lambda x: x),
    ])

    json_count = 0
    pbar = tqdm(total=len(img_files), desc="DEIMv2 推理中")
    for img_path in img_files:
        im_pil = Image.open(img_path).convert('RGB')
        w, h = im_pil.size
        orig_size = torch.tensor([[w, h]]).to(device)
        im_data = transforms(im_pil).unsqueeze(0).to(device)

        with torch.no_grad():
            labels, boxes, scores = model(im_data, orig_size)

        shapes = []
        scr = scores[0]
        lab = labels[0]
        box = boxes[0]
        for j in range(len(scr)):
            conf = scr[j].item()
            if conf < args.conf:
                continue
            cls_id = lab[j].item()
            x1, y1, x2, y2 = box[j].tolist()
            shapes.append(make_shape(
                CLASS_NAMES.get(cls_id, str(cls_id)),
                x1, y1, x2, y2, conf))

        annotation = build_xanylabeling_json(img_path, shapes, args.save_img)
        json_path = out_dir / (img_path.stem + ".json")
        save_json(json_path, annotation)
        json_count += 1
        pbar.update(1)
    pbar.close()
    return json_count


# ==================== main ====================

def main():
    args = parse_args()
    source_dir = Path(args.source)

    if not source_dir.is_dir():
        raise FileNotFoundError(f"图片目录不存在: {source_dir}")

    if args.model_type == "deimv2" and not args.config:
        raise ValueError("DEIMv2 模型需要指定 --config 参数")

    out_dir = None
    if args.output:
        out_dir = Path(args.output)
    else:
        # 自动生成: {source_name}_{model_name}
        source_name = source_dir.name
        # 从权重路径提取模型名, 如 runs/train/iter_20260425_yolov8s/weights/best.pt -> iter_20260425_yolov8s
        weights_path = Path(args.weights)
        if weights_path.parent.name == "weights":
            model_name = weights_path.parent.parent.name
        else:
            # 直接在训练目录下, 如 runs/train/iter_20260425_deimv2_x/best_stg2.pth
            model_name = weights_path.parent.name
        out_dir = source_dir.parent / f"{source_name}_{model_name}"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[信息] 输出目录: {out_dir}")

    img_files = sorted(
        [f for f in source_dir.iterdir() if f.suffix.lower() in IMG_EXTS]
    )
    print(f"[信息] 模型类型: {args.model_type}")
    print(f"[信息] 共找到 {len(img_files)} 张图片")
    print(f"[信息] 权重: {args.weights}")
    print(f"[信息] 置信度: {args.conf}")

    if args.model_type == "yolo":
        json_count = run_yolo(args, img_files, out_dir)
    else:
        json_count = run_deimv2(args, img_files, out_dir)

    print(f"\n[完成] 共生成 {json_count} 个 JSON 标注文件")
    print(f"[信息] 标注保存在: {out_dir}/")
    print("[提示] 使用 X-AnyLabeling 打开该目录即可进行人工审核")


if __name__ == "__main__":
    main()
