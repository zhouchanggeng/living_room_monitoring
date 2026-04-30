"""
第一阶段: 伪标签半监督蒸馏训练 YOLOv8s

将 DEIMv2 生成的伪标签数据与人工标注数据合并,
训练 YOLOv8s 学生模型, 实现知识蒸馏.

流程:
  1. 合并真实标注 (iter_20260429) + 伪标签 (pseudo_labels_deimv2) 数据集
  2. 使用合并数据集训练 YOLOv8s
  3. 在原始验证集上评估, 与历史模型对比

用法:
    # 先生成伪标签
    python training/generate_pseudo_labels.py

    # 再进行蒸馏训练
    python training/distill_train.py
    python training/distill_train.py --pseudo-ratio 0.5 --conf-weight
    python training/distill_train.py --skip-train  # 仅对比评估
"""

import argparse
import json
import os
import random
import shutil
import sys
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RUNS_DIR = PROJECT_ROOT / "runs" / "train"

# 数据路径
REAL_DATA_DIR = PROJECT_ROOT / "data" / "iter_20260429"
PSEUDO_DATA_DIR = PROJECT_ROOT / "data" / "pseudo_labels_deimv2"
MERGED_DATA_DIR = PROJECT_ROOT / "data" / "distill_20260430"

# 训练配置
RUN_NAME = "distill_20260430_yolov8s"


def parse_args():
    p = argparse.ArgumentParser(description="伪标签半监督蒸馏训练 YOLOv8s")
    p.add_argument("--real-data", type=str, default=str(REAL_DATA_DIR))
    p.add_argument("--pseudo-data", type=str, default=str(PSEUDO_DATA_DIR))
    p.add_argument("--output-data", type=str, default=str(MERGED_DATA_DIR))
    p.add_argument("--pseudo-ratio", type=float, default=1.0,
                   help="使用伪标签数据的比例 (0-1), 1.0=全部使用")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--device", type=str, default="0")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--skip-merge", action="store_true", help="跳过数据合并 (已合并过)")
    p.add_argument("--skip-train", action="store_true", help="跳过训练, 仅做对比评估")
    return p.parse_args()


def merge_datasets(real_dir, pseudo_dir, output_dir, pseudo_ratio=1.0):
    """合并真实标注和伪标签数据集."""
    real_dir = Path(real_dir)
    pseudo_dir = Path(pseudo_dir)
    output_dir = Path(output_dir)

    # 输出目录
    out_train_img = output_dir / "train" / "images"
    out_train_lbl = output_dir / "train" / "labels"
    out_val_img = output_dir / "val" / "images"
    out_val_lbl = output_dir / "val" / "labels"

    for d in [out_train_img, out_train_lbl, out_val_img, out_val_lbl]:
        d.mkdir(parents=True, exist_ok=True)

    # 1. 复制真实训练数据 (软链接)
    real_train_imgs = sorted((real_dir / "train" / "images").glob("*"))
    real_train_lbls = sorted((real_dir / "train" / "labels").glob("*.txt"))
    print(f"真实训练数据: {len(real_train_imgs)} 图片, {len(real_train_lbls)} 标签")

    for img in real_train_imgs:
        dst = out_train_img / img.name
        if not dst.exists():
            os.symlink(str(img.resolve()), str(dst))
    for lbl in real_train_lbls:
        dst = out_train_lbl / lbl.name
        if not dst.exists():
            os.symlink(str(lbl.resolve()), str(dst))

    # 2. 添加伪标签数据
    pseudo_imgs_dir = pseudo_dir / "images"
    pseudo_lbls_dir = pseudo_dir / "labels"

    if not pseudo_lbls_dir.exists():
        print("[警告] 伪标签目录不存在, 仅使用真实数据")
        pseudo_count = 0
    else:
        pseudo_labels = sorted(pseudo_lbls_dir.glob("*.txt"))

        # 排除已在真实训练集中的图片
        real_names = {img.stem for img in real_train_imgs}
        pseudo_labels = [l for l in pseudo_labels if l.stem not in real_names]
        print(f"伪标签数据 (去重后): {len(pseudo_labels)} 张")

        # 按比例采样
        if pseudo_ratio < 1.0:
            random.seed(42)
            n = int(len(pseudo_labels) * pseudo_ratio)
            pseudo_labels = random.sample(pseudo_labels, n)
            print(f"采样 {pseudo_ratio*100:.0f}%: {len(pseudo_labels)} 张")

        pseudo_count = 0
        for lbl in pseudo_labels:
            img_name = lbl.stem + ".jpg"
            src_img = pseudo_imgs_dir / img_name
            if not src_img.exists():
                continue

            dst_img = out_train_img / img_name
            dst_lbl = out_train_lbl / lbl.name

            if not dst_img.exists():
                # 解析软链接获取真实路径
                real_src = src_img.resolve()
                os.symlink(str(real_src), str(dst_img))
            if not dst_lbl.exists():
                os.symlink(str(lbl.resolve()), str(dst_lbl))
            pseudo_count += 1

    # 3. 验证集直接链接原始验证集
    for img in (real_dir / "val" / "images").glob("*"):
        dst = out_val_img / img.name
        if not dst.exists():
            os.symlink(str(img.resolve()), str(dst))
    for lbl in (real_dir / "val" / "labels").glob("*.txt"):
        dst = out_val_lbl / lbl.name
        if not dst.exists():
            os.symlink(str(lbl.resolve()), str(dst))

    # 4. 生成 data.yaml
    data_yaml = output_dir / "data.yaml"
    yaml_content = (
        f"path: {output_dir}\n"
        f"train: train/images\n"
        f"val: val/images\n"
        f"names:\n"
        f"  0: Kid\n"
        f"  1: Adult\n"
    )
    data_yaml.write_text(yaml_content)

    total_train = len(list(out_train_img.iterdir()))
    total_val = len(list(out_val_img.iterdir()))
    print(f"\n合并完成:")
    print(f"  真实标注: {len(real_train_imgs)} 张")
    print(f"  伪标签:   {pseudo_count} 张")
    print(f"  训练集:   {total_train} 张")
    print(f"  验证集:   {total_val} 张")
    print(f"  data.yaml: {data_yaml}")

    return str(data_yaml)


def train(data_yaml, args):
    """训练 YOLOv8s."""
    run_dir = RUNS_DIR / RUN_NAME

    if args.resume and (run_dir / "weights" / "last.pt").exists():
        print(f"[恢复训练] {run_dir / 'weights' / 'last.pt'}")
        model = YOLO(str(run_dir / "weights" / "last.pt"))
    else:
        model = YOLO("yolov8s.pt")

    try:
        from swanlab.integration.ultralytics import add_swanlab_callback
        add_swanlab_callback(model, project="KidsCare-YOLOv8s",
                             experiment_name=RUN_NAME,
                             description="DEIMv2伪标签半监督蒸馏训练")
    except ImportError:
        pass

    model.train(
        data=data_yaml, imgsz=(384, 640), epochs=args.epochs,
        batch=args.batch, device=args.device,
        project=str(RUNS_DIR), name=RUN_NAME,
        pretrained=True, optimizer="SGD",
        lr0=0.01, lrf=0.01, momentum=0.937, weight_decay=0.0005,
        warmup_epochs=3, cos_lr=True,
        hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,
        translate=0.1, scale=0.5, fliplr=0.5,
        mosaic=1.0, mixup=0.1,
        save=True, save_period=10, val=True, plots=True, exist_ok=True,
    )

    best_pt = run_dir / "weights" / "best.pt"
    if not best_pt.exists():
        raise FileNotFoundError(f"训练完成但未找到 best.pt: {best_pt}")
    return best_pt


def evaluate_all(val_data_yaml, device="0"):
    """在验证集上评估所有迭代模型."""
    results = {}
    for d in sorted(RUNS_DIR.iterdir()):
        bp = d / "weights" / "best.pt"
        if bp.exists() and d.name.startswith(("iter_", "distill_")) and "yolov8s" in d.name:
            name = d.name.replace("_yolov8s", "")
            print(f"\n  评估 {name}: {bp}")
            model = YOLO(str(bp))
            r = model.val(data=val_data_yaml, imgsz=640, batch=32,
                          device=device, verbose=False)
            results[name] = {
                "mAP50": round(float(r.box.map50), 4),
                "mAP50-95": round(float(r.box.map), 4),
                "Precision": round(float(r.box.mp), 4),
                "Recall": round(float(r.box.mr), 4),
                "Kid_AP50": round(float(r.box.ap50[0]), 4),
                "Adult_AP50": round(float(r.box.ap50[1]), 4),
            }
            print(f"    mAP50={results[name]['mAP50']}, mAP50-95={results[name]['mAP50-95']}")
    return results


def plot_comparison(results, save_dir):
    """生成对比图表."""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    names = list(results.keys())
    n = len(names)
    colors = plt.cm.Set2(np.linspace(0, 1, max(n, 3)))

    metrics = ["mAP50", "mAP50-95", "Precision", "Recall"]
    fig, ax = plt.subplots(figsize=(max(10, n * 2.5), 6))
    x = np.arange(len(metrics))
    width = 0.7 / n

    for i, name in enumerate(names):
        vals = [results[name][m] for m in metrics]
        bars = ax.bar(x + i * width - 0.35 + width / 2, vals, width,
                      label=name, color=colors[i])
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.008,
                    f"{v:.3f}", ha="center", va="bottom", fontsize=8)

    ax.set_ylabel("Score")
    ax.set_title("Distillation Training - Model Comparison")
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_ylim(0, 1.05)
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_dir / "distill_comparison.png", dpi=150)
    plt.close(fig)

    # 保存 JSON
    with open(save_dir / "distill_results.json", "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # 打印表格
    print(f"\n{'=' * 100}")
    print(f"{'模型':<25} {'mAP50':>8} {'mAP50-95':>10} {'Precision':>10} {'Recall':>8} {'Kid_AP50':>10} {'Adult_AP50':>11}")
    print("-" * 100)
    for name, v in results.items():
        print(f"{name:<25} {v['mAP50']:>8.4f} {v['mAP50-95']:>10.4f} "
              f"{v['Precision']:>10.4f} {v['Recall']:>8.4f} "
              f"{v['Kid_AP50']:>10.4f} {v['Adult_AP50']:>11.4f}")
    print("=" * 100)


class Tee:
    def __init__(self, log_path):
        self.terminal = sys.stdout
        self.log = open(log_path, "w")
    def write(self, msg):
        self.terminal.write(msg)
        self.log.write(msg)
    def flush(self):
        self.terminal.flush()
        self.log.flush()


def main():
    args = parse_args()

    # 日志
    log_dir = PROJECT_ROOT / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"distill_train_{timestamp}.log"
    sys.stdout = sys.stderr = Tee(log_file)
    print(f"[日志] {log_file}")

    # 1. 合并数据集
    if not args.skip_merge and not args.skip_train:
        print("\n" + "=" * 60)
        print(" Step 1: 合并真实标注 + 伪标签数据集")
        print("=" * 60)
        data_yaml = merge_datasets(
            args.real_data, args.pseudo_data, args.output_data,
            pseudo_ratio=args.pseudo_ratio,
        )
    else:
        data_yaml = str(Path(args.output_data) / "data.yaml")
        if not Path(data_yaml).exists():
            data_yaml = str(REAL_DATA_DIR / "data.yaml")

    # 2. 训练
    if not args.skip_train:
        print("\n" + "=" * 60)
        print(" Step 2: YOLOv8s 蒸馏训练")
        print(f"  数据集: {data_yaml}")
        print("=" * 60)
        best_pt = train(data_yaml, args)
        print(f"\n训练完成! best.pt: {best_pt}")

    # 3. 对比评估 (使用原始验证集)
    print("\n" + "=" * 60)
    print(" Step 3: 模型对比评估")
    print("=" * 60)
    val_yaml = str(REAL_DATA_DIR / "data.yaml")
    results = evaluate_all(val_yaml, args.device)

    if results:
        save_dir = RUNS_DIR / RUN_NAME / "comparison"
        plot_comparison(results, save_dir)
        print(f"\n图表已保存到: {save_dir}/")


if __name__ == "__main__":
    main()
