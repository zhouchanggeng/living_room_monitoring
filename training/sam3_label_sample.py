"""
使用 SAM3 模型对 sample_1000 中的图片进行推理,
检测成人(Adult)和小孩(Kid), 输出 X-AnyLabeling 兼容的矩形框 JSON 标注文件.

用法 (在 conda 环境 x-anylabeling-server 中运行):
    python training/sam3_label_sample.py \
        --checkpoint models/sam3.pt \
        --source sample_1000 \
        --conf 0.15 \
        --device cuda

依赖: conda activate x-anylabeling-server
"""

import argparse
import json
import os
import sys

# SAM3 包在 X-AnyLabeling-Server 的 app/models 下
SAM3_BASE = "/home/zcg/workspace/X-AnyLabeling-Server/app/models"
if SAM3_BASE not in sys.path:
    sys.path.insert(0, SAM3_BASE)

from pathlib import Path

import torch
from PIL import Image
from tqdm import tqdm

from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

# 文本 prompt -> 标签名映射
# SAM3 是 grounding 模型, 用文本 prompt 检测对应目标
PROMPTS = {
    "child": "Kid",
    "an adult": "Adult",
}

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
BPE_PATH = "models/bpe_simple_vocab_16e6.txt.gz"


def parse_args():
    p = argparse.ArgumentParser(description="SAM3 推理 -> X-AnyLabeling JSON (矩形框)")
    p.add_argument("--checkpoint", type=str, default="models/sam3.pt")
    p.add_argument("--bpe", type=str, default=BPE_PATH)
    p.add_argument("--source", type=str, default="sample_1000", help="图片目录")
    p.add_argument("--conf", type=float, default=0.3, help="置信度阈值")
    p.add_argument("--device", type=str, default="cuda")
    return p.parse_args()


def build_xanylabeling_json(img_name: str, img_w: int, img_h: int,
                            detections: list) -> dict:
    """构建 X-AnyLabeling JSON, 只保存矩形框."""
    shapes = []
    for det in detections:
        shapes.append({
            "label": det["label"],
            "score": det["score"],
            "points": [
                [det["x1"], det["y1"]],
                [det["x2"], det["y2"]],
            ],
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


def clamp_box(x1, y1, x2, y2, img_w, img_h):
    """将框坐标限制在图片范围内."""
    return (
        round(max(0, x1), 2),
        round(max(0, y1), 2),
        round(min(img_w, x2), 2),
        round(min(img_h, y2), 2),
    )


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
    print(f"[信息] 模型: {args.checkpoint}")
    print(f"[信息] 置信度阈值: {args.conf}")

    # 加载 SAM3 模型
    print("[信息] 正在加载 SAM3 模型...")
    bpe_path = os.path.abspath(args.bpe)
    model = build_sam3_image_model(
        bpe_path=bpe_path,
        device=args.device,
        eval_mode=True,
        checkpoint_path=os.path.abspath(args.checkpoint),
        load_from_HF=False,
        enable_segmentation=True,
    )
    processor = Sam3Processor(
        model=model,
        device=args.device,
        confidence_threshold=args.conf,
    )
    print("[信息] 模型加载完成")

    json_count = 0
    for img_path in tqdm(img_files, desc="SAM3 推理中"):
        img = Image.open(img_path).convert("RGB")
        img_w, img_h = img.size

        all_detections = []

        # 对每个类别分别用文本 prompt 检测
        # 每次需要重新 set_image, 因为 set_text_prompt 会修改 state
        for prompt_text, label_name in PROMPTS.items():
            state = processor.set_image(img)
            state = processor.set_text_prompt(prompt_text, state)

            if "boxes" in state and len(state["boxes"]) > 0:
                boxes = state["boxes"].cpu()
                scores = state["scores"].cpu()

                for i in range(len(boxes)):
                    score = float(scores[i])
                    x1, y1, x2, y2 = boxes[i].tolist()
                    x1, y1, x2, y2 = clamp_box(x1, y1, x2, y2, img_w, img_h)
                    all_detections.append({
                        "label": label_name,
                        "score": round(score, 4),
                        "x1": x1, "y1": y1,
                        "x2": x2, "y2": y2,
                    })

        # 生成 JSON
        annotation = build_xanylabeling_json(
            img_path.name, img_w, img_h, all_detections
        )
        json_path = img_path.with_suffix(".json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(annotation, f, ensure_ascii=False, indent=2)
        json_count += 1

    print(f"\n[完成] 共生成 {json_count} 个 JSON 标注文件")
    print(f"[信息] 标注保存在: {source_dir}/")
    print("[提示] 使用 X-AnyLabeling 打开该目录即可进行人工审核")


if __name__ == "__main__":
    main()
