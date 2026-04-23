"""
从 origin_data/all_20260416 随机抽取图片(排除已用的sample_1000),
复制到新目录, 然后用训练好的模型推理生成 X-AnyLabeling JSON.
"""
import os
import random
import shutil
from pathlib import Path

# ---- 配置 ----
ALL_DIR = Path("origin_data/all_20260416")
EXCLUDE_DIR = Path("origin_data/sample_1000")
OUTPUT_DIR = Path("origin_data/sample_1200_new")
SAMPLE_N = 1200
WEIGHTS = "runs/train/sample1000_yolov8s/weights/best.pt"
CONF = 0.25
IOU = 0.45
IMGSZ = [384, 640]
BATCH = 32
DEVICE = "0"

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def main():
    # 1. 收集已用文件名
    exclude_names = set()
    if EXCLUDE_DIR.exists():
        exclude_names = {f.name for f in EXCLUDE_DIR.iterdir() if f.suffix.lower() in IMG_EXTS}
    print(f"排除已有图片: {len(exclude_names)} 张")

    # 2. 收集候选图片
    candidates = [f for f in ALL_DIR.iterdir()
                  if f.suffix.lower() in IMG_EXTS and f.name not in exclude_names]
    print(f"候选图片: {len(candidates)} 张")

    # 3. 随机抽样
    random.seed(42)
    sampled = random.sample(candidates, min(SAMPLE_N, len(candidates)))
    print(f"抽取: {len(sampled)} 张")

    # 4. 复制到输出目录
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for f in sampled:
        shutil.copy2(f, OUTPUT_DIR / f.name)
    print(f"已复制到: {OUTPUT_DIR}")

    # 5. 推理生成 X-AnyLabeling JSON
    import json
    import base64
    from PIL import Image
    from ultralytics import YOLO
    from tqdm import tqdm

    CLASS_NAMES = {0: "Kid", 1: "Adult"}
    model = YOLO(WEIGHTS)

    img_files = sorted([f for f in OUTPUT_DIR.iterdir() if f.suffix.lower() in IMG_EXTS])
    total = len(img_files)
    print(f"开始推理 {total} 张图片...")

    pbar = tqdm(total=total, desc="推理中")
    json_count = 0

    for i in range(0, total, BATCH):
        batch_paths = img_files[i:i + BATCH]
        results_list = model.predict(
            source=[str(p) for p in batch_paths],
            imgsz=IMGSZ, conf=CONF, iou=IOU,
            device=DEVICE, verbose=False,
        )

        for img_path, results in zip(batch_paths, results_list):
            img = Image.open(img_path)
            w, h = img.size
            shapes = []
            if results.boxes is not None:
                for box in results.boxes:
                    conf_val = float(box.conf[0])
                    if conf_val < CONF:
                        continue
                    cls_id = int(box.cls[0])
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    shapes.append({
                        "label": CLASS_NAMES.get(cls_id, str(cls_id)),
                        "score": round(conf_val, 4),
                        "points": [[round(x1, 2), round(y1, 2)],
                                   [round(x2, 2), round(y2, 2)]],
                        "group_id": None,
                        "description": "",
                        "difficult": False,
                        "shape_type": "rectangle",
                        "flags": {},
                        "attributes": {},
                    })

            annotation = {
                "version": "2.4.0",
                "flags": {},
                "shapes": shapes,
                "imagePath": img_path.name,
                "imageData": None,
                "imageHeight": h,
                "imageWidth": w,
            }
            json_path = img_path.with_suffix(".json")
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(annotation, f, ensure_ascii=False, indent=2)
            json_count += 1

        pbar.update(len(batch_paths))

    pbar.close()
    print(f"\n完成! 共生成 {json_count} 个 JSON 标注文件")
    print(f"输出目录: {OUTPUT_DIR}/")
    print("用 X-AnyLabeling 打开该目录即可审核")


if __name__ == "__main__":
    main()
