"""
从伪标签数据中抽取代理验证集，并对已训练模型做统一评估。

注意:
  这里的验证集真值来自 pseudo_labels_deimv2_20260508 的伪标签，
  因此结果表示“与伪标签的一致性”，不能等价视为人工标注真值上的真实精度。

用法:
  /data/zcg/miniconda3/bin/python training/evaluate_pseudo_val.py
  /data/zcg/miniconda3/bin/python training/evaluate_pseudo_val.py --sample-size 400 --device 0
  /data/zcg/miniconda3/bin/python training/evaluate_pseudo_val.py --skip-sample
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import shutil
import subprocess
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from PIL import Image
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RUNS_DIR = PROJECT_ROOT / "runs" / "train"
SOURCE_DATA_DIR = PROJECT_ROOT / "data" / "pseudo_labels_deimv2_20260508"
VAL_DATA_DIR = PROJECT_ROOT / "data" / "pseudo_val_20260508"
DEIMV2_DIR = PROJECT_ROOT / "training" / "DEIMv2"
DEIMV2_BASE_CONFIG = DEIMV2_DIR / "configs" / "deimv2" / "deimv2_dinov3_x_kids_care.yml"
PYTHON_BIN = Path("/data/zcg/miniconda3/bin/python")
TORCHRUN_BIN = Path("/data/zcg/miniconda3/bin/torchrun")
CLASS_NAMES = ["Kid", "Adult"]


def parse_args():
    p = argparse.ArgumentParser(description="代理验证集抽样 + 已训练模型评估")
    p.add_argument("--source-data", type=Path, default=SOURCE_DATA_DIR)
    p.add_argument("--output-data", type=Path, default=VAL_DATA_DIR)
    p.add_argument("--sample-size", type=int, default=400,
                   help="代理验证集图片数")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="0")
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--skip-sample", action="store_true",
                   help="跳过抽样，直接复用已有代理验证集")
    p.add_argument("--include-deimv2", action="store_true", default=True,
                   help="评估 DEIMv2 模型")
    p.add_argument("--include-yolo", action="store_true", default=True,
                   help="评估 YOLO 模型")
    p.add_argument("--skip-yolo", action="store_true",
                   help="跳过 YOLO 模型评估")
    p.add_argument("--skip-deimv2", action="store_true",
                   help="跳过 DEIMv2 模型评估")
    return p.parse_args()


def resolve_device(requested_device: str):
    if requested_device == "cpu":
        return "cpu"
    if torch.cuda.is_available():
        return requested_device
    print(f"[警告] 当前环境不可用 CUDA，device={requested_device} 将回退到 CPU")
    return "cpu"


def read_yolo_label(label_path: Path):
    records = []
    for line in label_path.read_text().splitlines():
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        cls_id = int(parts[0])
        box = tuple(float(x) for x in parts[1:5])
        records.append((cls_id, box))
    return records


def class_signature(records):
    return tuple(sorted({cls_id for cls_id, _ in records}))


def collect_candidates(source_data: Path):
    images_dir = source_data / "images"
    labels_dir = source_data / "labels"
    candidates = []
    for label_path in sorted(labels_dir.glob("*.txt")):
        image_path = images_dir / f"{label_path.stem}.jpg"
        if not image_path.exists():
            continue
        records = read_yolo_label(label_path)
        candidates.append({
            "stem": label_path.stem,
            "image_path": image_path,
            "label_path": label_path,
            "records": records,
            "signature": class_signature(records),
        })
    return candidates


def stratified_sample(candidates, sample_size: int, seed: int):
    if sample_size >= len(candidates):
        return candidates

    rng = random.Random(seed)
    buckets = {}
    for item in candidates:
        buckets.setdefault(item["signature"], []).append(item)

    for items in buckets.values():
        rng.shuffle(items)

    total = len(candidates)
    chosen = []
    remainders = []
    for signature, items in buckets.items():
        target = len(items) * sample_size / total
        count = min(len(items), int(target))
        chosen.extend(items[:count])
        remainders.append((target - count, signature, items[count:]))

    remaining = sample_size - len(chosen)
    remainders.sort(key=lambda x: x[0], reverse=True)
    for _, _, items in remainders:
        if remaining <= 0:
            break
        take = min(len(items), remaining)
        chosen.extend(items[:take])
        remaining -= take

    if remaining > 0:
        chosen_ids = {item["stem"] for item in chosen}
        leftovers = [item for item in candidates if item["stem"] not in chosen_ids]
        rng.shuffle(leftovers)
        chosen.extend(leftovers[:remaining])

    chosen.sort(key=lambda x: x["stem"])
    return chosen


def ensure_clean_dir(path: Path):
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def build_yolo_dataset(sampled, output_data: Path):
    images_dir = output_data / "val" / "images"
    labels_dir = output_data / "val" / "labels"
    annotations_dir = output_data / "annotations"
    ensure_clean_dir(images_dir)
    ensure_clean_dir(labels_dir)
    annotations_dir.mkdir(parents=True, exist_ok=True)

    manifest = []
    for item in sampled:
        dst_img = images_dir / item["image_path"].name
        dst_lbl = labels_dir / item["label_path"].name
        shutil.copy2(item["image_path"], dst_img)
        shutil.copy2(item["label_path"], dst_lbl)
        manifest.append({
            "stem": item["stem"],
            "signature": list(item["signature"]),
            "num_boxes": len(item["records"]),
        })

    yaml_text = (
        f"path: {output_data}\n"
        f"train: val/images\n"
        f"val: val/images\n"
        f"names:\n"
        f"  0: {CLASS_NAMES[0]}\n"
        f"  1: {CLASS_NAMES[1]}\n"
    )
    (output_data / "data.yaml").write_text(yaml_text)
    (output_data / "sample_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False)
    )


def yolo_to_coco(img_dir: Path, label_dir: Path, output_json: Path):
    categories = [{"id": idx, "name": name} for idx, name in enumerate(CLASS_NAMES)]
    images = []
    annotations = []
    ann_id = 0

    for img_id, img_path in enumerate(sorted(img_dir.glob("*.jpg"))):
        width, height = Image.open(img_path).size
        images.append({
            "id": img_id,
            "file_name": img_path.name,
            "width": width,
            "height": height,
        })
        label_path = label_dir / f"{img_path.stem}.txt"
        if not label_path.exists():
            continue
        for cls_id, (cx, cy, bw, bh) in read_yolo_label(label_path):
            x1 = (cx - bw / 2) * width
            y1 = (cy - bh / 2) * height
            box_w = bw * width
            box_h = bh * height
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
    output_json.write_text(json.dumps(coco, indent=2, ensure_ascii=False))


def summarize_sample(output_data: Path):
    labels_dir = output_data / "val" / "labels"
    signature_counter = Counter()
    box_counter = Counter()
    for label_path in labels_dir.glob("*.txt"):
        records = read_yolo_label(label_path)
        signature_counter[class_signature(records)] += 1
        for cls_id, _ in records:
            box_counter[cls_id] += 1
    return {
        "num_images": sum(signature_counter.values()),
        "image_signatures": {
            ",".join(CLASS_NAMES[i] for i in sig) if sig else "empty": count
            for sig, count in sorted(signature_counter.items(), key=lambda x: (len(x[0]), x[0]))
        },
        "box_counts": {CLASS_NAMES[k]: v for k, v in sorted(box_counter.items())},
    }


def discover_yolo_models():
    models = {}
    for run_dir in sorted(RUNS_DIR.iterdir()):
        best_pt = run_dir / "weights" / "best.pt"
        if best_pt.exists() and run_dir.name.startswith(("iter_", "distill_")):
            models[run_dir.name] = best_pt
    return models


def discover_deimv2_models():
    models = {}
    for run_dir in sorted(RUNS_DIR.iterdir()):
        if not run_dir.name.startswith("iter_") or "deimv2" not in run_dir.name:
            continue
        for candidate in ["best_stg2.pth", "checkpoint0029.pth", "last.pth"]:
            ckpt = run_dir / candidate
            if ckpt.exists():
                models[run_dir.name] = ckpt
                break
    return models


def evaluate_yolo_models(models, data_yaml: Path, device: str, batch: int, imgsz: int):
    results = {}
    for name, weights in models.items():
        print(f"[YOLO] 评估 {name}: {weights}")
        model = YOLO(str(weights))
        r = model.val(
            data=str(data_yaml),
            imgsz=imgsz,
            batch=batch,
            device=device,
            verbose=False,
        )
        results[name] = {
            "family": "yolo",
            "weights": str(weights),
            "mAP50": round(float(r.box.map50), 4),
            "mAP50-95": round(float(r.box.map), 4),
            "Precision": round(float(r.box.mp), 4),
            "Recall": round(float(r.box.mr), 4),
            "Kid_AP50": round(float(r.box.ap50[0]), 4),
            "Adult_AP50": round(float(r.box.ap50[1]), 4),
        }
    return results


def load_deimv2_config_text(base_config: Path, output_data: Path):
    cfg = yaml.safe_load(base_config.read_text())
    val_images = output_data / "val" / "images"
    val_ann = output_data / "annotations" / "instances_val.json"
    cfg["output_dir"] = str(output_data / "deimv2_eval_tmp")
    cfg["train_dataloader"]["dataset"]["img_folder"] = str(val_images)
    cfg["train_dataloader"]["dataset"]["ann_file"] = str(val_ann)
    cfg["val_dataloader"]["dataset"]["img_folder"] = str(val_images)
    cfg["val_dataloader"]["dataset"]["ann_file"] = str(val_ann)
    cfg["train_dataloader"]["total_batch_size"] = 1
    cfg["val_dataloader"]["total_batch_size"] = 1
    cfg["train_dataloader"]["num_workers"] = 0
    cfg["val_dataloader"]["num_workers"] = 0
    cfg["train_dataloader"]["shuffle"] = False
    cfg["val_dataloader"]["shuffle"] = False
    return yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False)


def parse_deimv2_metrics(output: str):
    metrics = {}
    pattern = re.compile(r"=\s*([0-9]*\.?[0-9]+)")
    for line in output.splitlines():
        if "Average Precision" in line and "IoU=0.50:0.95" in line and "area=   all" in line and "maxDets=100" in line:
            match = pattern.search(line)
            if match:
                metrics["mAP50-95"] = round(float(match.group(1)), 4)
        elif "Average Precision" in line and "IoU=0.50      " in line and "area=   all" in line and "maxDets=100" in line:
            match = pattern.search(line)
            if match:
                metrics["mAP50"] = round(float(match.group(1)), 4)
        elif "Average Recall" in line and "IoU=0.50:0.95" in line and "area=   all" in line and "maxDets=100" in line:
            match = pattern.search(line)
            if match:
                metrics["Recall"] = round(float(match.group(1)), 4)
    return metrics


def evaluate_deimv2_models(models, output_data: Path, device: str):
    results = {}
    config_path = DEIMV2_DIR / "configs" / "deimv2" / "deimv2_eval_pseudo_val_20260508.yml"
    config_path.write_text(load_deimv2_config_text(DEIMV2_BASE_CONFIG, output_data))
    deim_device = "cpu" if device == "cpu" else "cuda"
    for name, weights in models.items():
        print(f"[DEIMv2] 评估 {name}: {weights}")
        cmd = [
            str(TORCHRUN_BIN),
            "--master_port=7789",
            "--nproc_per_node=1",
            "train.py",
            "-c", str(config_path),
            "--test-only",
            "-r", str(weights),
            "--device", deim_device,
        ]
        env = dict(os.environ)
        if device != "cpu":
            env["CUDA_VISIBLE_DEVICES"] = device
        proc = subprocess.run(
            cmd,
            cwd=str(DEIMV2_DIR),
            env=env,
            capture_output=True,
            text=True,
        )
        output = proc.stdout + "\n" + proc.stderr
        metrics = parse_deimv2_metrics(output)
        if proc.returncode != 0:
            raise RuntimeError(f"DEIMv2 评估失败: {name}\n{output[-4000:]}")
        if not metrics:
            raise RuntimeError(f"未解析到 DEIMv2 指标: {name}\n{output[-4000:]}")
        results[name] = {
            "family": "deimv2",
            "weights": str(weights),
            "mAP50": metrics.get("mAP50"),
            "mAP50-95": metrics.get("mAP50-95"),
            "Precision": None,
            "Recall": metrics.get("Recall"),
            "Kid_AP50": None,
            "Adult_AP50": None,
        }
    return results


def plot_results(results, save_dir: Path):
    save_dir.mkdir(parents=True, exist_ok=True)
    sortable = sorted(
        results.items(),
        key=lambda item: (
            item[1].get("mAP50") is None,
            -(item[1].get("mAP50") or -1),
            item[0],
        ),
    )
    names = [name for name, _ in sortable]
    metrics = ["mAP50", "mAP50-95", "Recall"]
    values = {metric: [res.get(metric) or 0.0 for _, res in sortable] for metric in metrics}

    fig, axes = plt.subplots(1, 3, figsize=(max(12, len(names) * 2.2), 5))
    colors = plt.cm.Set2(np.linspace(0, 1, max(len(names), 3)))
    for ax, metric in zip(axes, metrics):
        bars = ax.bar(names, values[metric], color=colors[:len(names)])
        ax.set_title(metric)
        ax.set_ylim(0, 1.05)
        ax.grid(axis="y", alpha=0.3)
        ax.tick_params(axis="x", rotation=35)
        for bar, value in zip(bars, values[metric]):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"{value:.3f}", ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    fig.savefig(save_dir / "pseudo_val_comparison.png", dpi=150)
    plt.close(fig)


def format_value(value):
    if value is None:
        return "-"
    return f"{value:.4f}"


def main():
    args = parse_args()
    args.device = resolve_device(args.device)
    if args.skip_yolo:
        args.include_yolo = False
    if args.skip_deimv2:
        args.include_deimv2 = False

    if not PYTHON_BIN.exists() or not TORCHRUN_BIN.exists():
        raise FileNotFoundError("未找到 /data/zcg/miniconda3/bin 下的训练环境解释器")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = PROJECT_ROOT / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"evaluate_pseudo_val_{timestamp}.log"
    print(f"[日志] {log_path}")

    if not args.skip_sample:
        candidates = collect_candidates(args.source_data)
        sampled = stratified_sample(candidates, args.sample_size, args.seed)
        args.output_data.mkdir(parents=True, exist_ok=True)
        build_yolo_dataset(sampled, args.output_data)
        yolo_to_coco(
            args.output_data / "val" / "images",
            args.output_data / "val" / "labels",
            args.output_data / "annotations" / "instances_val.json",
        )
    elif not (args.output_data / "data.yaml").exists():
        raise FileNotFoundError(f"--skip-sample 指定复用数据集，但未找到: {args.output_data / 'data.yaml'}")

    summary = summarize_sample(args.output_data)
    print(f"[代理验证集] 图片: {summary['num_images']}")
    print(f"[代理验证集] 图片组成: {summary['image_signatures']}")
    print(f"[代理验证集] 框数量: {summary['box_counts']}")

    all_results = {}
    if args.include_yolo:
        yolo_models = discover_yolo_models()
        print(f"[发现 YOLO 模型] {len(yolo_models)} 个")
        all_results.update(
            evaluate_yolo_models(
                yolo_models,
                args.output_data / "data.yaml",
                args.device,
                args.batch,
                args.imgsz,
            )
        )

    if args.include_deimv2:
        deimv2_models = discover_deimv2_models()
        print(f"[发现 DEIMv2 模型] {len(deimv2_models)} 个")
        all_results.update(evaluate_deimv2_models(deimv2_models, args.output_data, args.device))

    comparison_dir = RUNS_DIR / "pseudo_val_20260508_comparison"
    comparison_dir.mkdir(parents=True, exist_ok=True)
    result_payload = {
        "source_data": str(args.source_data),
        "proxy_val_data": str(args.output_data),
        "warning": "Metrics are measured against pseudo labels, not human-verified ground truth.",
        "sample_summary": summary,
        "results": all_results,
    }
    (comparison_dir / "pseudo_val_results.json").write_text(
        json.dumps(result_payload, indent=2, ensure_ascii=False)
    )
    plot_results(all_results, comparison_dir)

    print("\n" + "=" * 110)
    print(f"{'模型':<32} {'类型':<8} {'mAP50':>8} {'mAP50-95':>10} {'Precision':>10} {'Recall':>8}")
    print("-" * 110)
    for name, metrics in sorted(
        all_results.items(),
        key=lambda item: (item[1].get("mAP50") is None, -(item[1].get("mAP50") or -1), item[0]),
    ):
        print(
            f"{name:<32} {metrics['family']:<8} "
            f"{format_value(metrics.get('mAP50')):>8} "
            f"{format_value(metrics.get('mAP50-95')):>10} "
            f"{format_value(metrics.get('Precision')):>10} "
            f"{format_value(metrics.get('Recall')):>8}"
        )
    print("=" * 110)
    print(f"[结果] {comparison_dir / 'pseudo_val_results.json'}")
    print(f"[图表] {comparison_dir / 'pseudo_val_comparison.png'}")


if __name__ == "__main__":
    main()
