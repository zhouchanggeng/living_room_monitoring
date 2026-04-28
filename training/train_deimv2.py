"""
客厅小孩看护 - DEIMv2 训练 & 与 YOLOv8s 效果对比脚本

流程:
  1. YOLO 标注 -> COCO JSON 格式转换
  2. 使用 DEIMv2 (DINOv3-X) 在 kids_care 数据集上 fine-tune
  3. 在验证集上评估 DEIMv2, 并与历史 YOLOv8s 模型做对比

用法:
    python training/train_deimv2.py
    python training/train_deimv2.py --skip-convert        # 已转换过标注, 跳过
    python training/train_deimv2.py --skip-train           # 跳过训练, 仅做对比
    python training/train_deimv2.py --device 0,1           # 多卡训练
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEIMV2_DIR = Path(__file__).resolve().parent / "DEIMv2"
RUNS_DIR = PROJECT_ROOT / "runs" / "train"
DATA_DIR = PROJECT_ROOT / "data" / "iter_20260425"

# DEIMv2 配置
DEIMV2_CONFIG = "configs/deimv2/deimv2_dinov3_x_kids_care.yml"
PRETRAINED_MODEL = PROJECT_ROOT / "pretrained" / "deimv2_dinov3_x_coco.pth"
DEIMV2_RUN_DIR = RUNS_DIR / "iter_20260425_deimv2_x"

# YOLOv8s 对比基线
YOLO_DATA_YAML = str(DATA_DIR / "data.yaml")


def parse_args():
    p = argparse.ArgumentParser(description="DEIMv2 训练 & 对比评估")
    p.add_argument("--device", type=str, default="0", help="GPU id, 多卡用 0,1")
    p.add_argument("--skip-convert", action="store_true", help="跳过 YOLO->COCO 标注转换")
    p.add_argument("--skip-train", action="store_true", help="跳过训练, 仅做对比评估")
    p.add_argument("--resume", type=str, default=None, help="从 checkpoint 恢复训练")
    return p.parse_args()


# ============================================================
# Step 1: YOLO -> COCO 标注转换
# ============================================================
def convert_annotations():
    print("\n" + "=" * 60)
    print(" Step 1: YOLO -> COCO 标注转换")
    print("=" * 60)

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from yolo2coco import yolo_to_coco

    class_names = ["Kid", "Adult"]
    for split in ["train", "val"]:
        out_json = str(DATA_DIR / f"annotations/instances_{split}.json")
        if Path(out_json).exists():
            print(f"  {split} 标注已存在, 跳过: {out_json}")
            continue
        print(f"\n  转换 {split} 集:")
        yolo_to_coco(
            img_dir=str(DATA_DIR / split / "images"),
            label_dir=str(DATA_DIR / split / "labels"),
            class_names=class_names,
            output_json=out_json,
        )


# ============================================================
# Step 2: DEIMv2 训练
# ============================================================
def train_deimv2(device: str, resume: str = None):
    print("\n" + "=" * 60)
    print(" Step 2: DEIMv2 (DINOv3-X) 训练")
    print(f"  配置:     {DEIMV2_CONFIG}")
    print(f"  预训练:   {PRETRAINED_MODEL}")
    print(f"  设备:     GPU {device}")
    print("=" * 60)

    if not PRETRAINED_MODEL.exists():
        raise FileNotFoundError(f"预训练模型不存在: {PRETRAINED_MODEL}")

    gpu_ids = device.split(",")
    nproc = len(gpu_ids)

    cmd = [
        "torchrun",
        f"--master_port=7778",
        f"--nproc_per_node={nproc}",
        "train.py",
        "-c", DEIMV2_CONFIG,
        "--use-amp",
        "--seed=0",
    ]

    if resume:
        cmd += ["-r", resume]
    else:
        cmd += ["-t", str(PRETRAINED_MODEL)]

    env_str = ",".join(gpu_ids)
    print(f"\n  运行命令: CUDA_VISIBLE_DEVICES={env_str} {' '.join(cmd)}")
    print(f"  工作目录: {DEIMV2_DIR}\n")

    import os
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = env_str

    result = subprocess.run(cmd, cwd=str(DEIMV2_DIR), env=env)
    if result.returncode != 0:
        raise RuntimeError(f"DEIMv2 训练失败, 返回码: {result.returncode}")

    print("\n  DEIMv2 训练完成!")


# ============================================================
# Step 3: 评估 DEIMv2
# ============================================================
def evaluate_deimv2(device: str):
    """在验证集上评估 DEIMv2, 返回 COCO 指标."""
    print("\n  评估 DEIMv2 模型...")

    # 找到最优 checkpoint
    ckpt_dir = DEIMV2_RUN_DIR
    best_ckpt = None

    # DEIMv2 保存格式: checkpoint{epoch}.pth
    ckpts = sorted(ckpt_dir.glob("checkpoint*.pth"))
    if ckpts:
        best_ckpt = str(ckpts[-1])  # 最后一个 epoch 通常最优

    # 也检查 best.pth
    if (ckpt_dir / "best.pth").exists():
        best_ckpt = str(ckpt_dir / "best.pth")

    if not best_ckpt:
        print("  [警告] 未找到 DEIMv2 checkpoint, 跳过评估")
        return None

    print(f"  使用 checkpoint: {best_ckpt}")

    gpu_ids = device.split(",")
    nproc = len(gpu_ids)

    cmd = [
        "torchrun",
        f"--master_port=7779",
        f"--nproc_per_node={nproc}",
        "train.py",
        "-c", DEIMV2_CONFIG,
        "--test-only",
        "-r", best_ckpt,
    ]

    import os
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ",".join(gpu_ids)

    result = subprocess.run(cmd, cwd=str(DEIMV2_DIR), env=env, capture_output=True, text=True)

    # 解析输出中的 COCO 指标
    metrics = parse_deimv2_metrics(result.stdout + result.stderr)
    return metrics


def parse_deimv2_metrics(output: str):
    """从 DEIMv2 评估输出中解析 COCO 指标."""
    metrics = {}
    for line in output.split("\n"):
        # COCO evaluator 输出格式: Average Precision (AP) @[ IoU=0.50:0.95 | ... ] = 0.xxx
        if "IoU=0.50:0.95" in line and "area=   all" in line and "maxDets=100" in line:
            try:
                metrics["mAP50-95"] = round(float(line.strip().split("=")[-1].strip()), 4)
            except ValueError:
                pass
        elif "IoU=0.50 " in line and "area=   all" in line and "maxDets=100" in line:
            try:
                metrics["mAP50"] = round(float(line.strip().split("=")[-1].strip()), 4)
            except ValueError:
                pass
        # Average Recall
        elif "IoU=0.50:0.95" in line and "area=   all" in line and "maxDets=100" in line and "Recall" in line:
            try:
                metrics["Recall"] = round(float(line.strip().split("=")[-1].strip()), 4)
            except ValueError:
                pass

    if not metrics:
        print("  [警告] 未能从输出中解析到 COCO 指标")
        print("  输出末尾 30 行:")
        for line in output.strip().split("\n")[-30:]:
            print(f"    {line}")

    return metrics if metrics else None


# ============================================================
# Step 4: 评估 YOLOv8s 模型
# ============================================================
def evaluate_yolo_models(device: str):
    """评估所有历史 YOLOv8s 模型, 返回指标字典."""
    from ultralytics import YOLO

    results = {}
    for d in sorted(RUNS_DIR.iterdir()):
        bp = d / "weights" / "best.pt"
        if bp.exists() and d.name.startswith("iter_") and "yolov8s" in d.name:
            iter_name = "_".join(d.name.split("_")[:2])
            model_label = f"{iter_name}_yolov8s"
            print(f"\n  评估 {model_label}: {bp}")
            model = YOLO(str(bp))
            r = model.val(data=YOLO_DATA_YAML, imgsz=640, batch=32,
                          device=device, verbose=False)
            results[model_label] = {
                "mAP50": round(float(r.box.map50), 4),
                "mAP50-95": round(float(r.box.map), 4),
                "Precision": round(float(r.box.mp), 4),
                "Recall": round(float(r.box.mr), 4),
                "Kid_AP50": round(float(r.box.ap50[0]), 4),
                "Adult_AP50": round(float(r.box.ap50[1]), 4),
            }
    return results


# ============================================================
# Step 5: 对比图表
# ============================================================
def plot_comparison(results: dict, save_dir: Path):
    """生成 DEIMv2 vs YOLOv8s 对比图表."""
    save_dir.mkdir(parents=True, exist_ok=True)

    names = list(results.keys())
    n = len(names)
    colors = plt.cm.Set2(np.linspace(0, 1, max(n, 3)))

    # 使用所有模型共有的指标
    common_metrics = ["mAP50", "mAP50-95"]
    # 检查哪些指标所有模型都有
    all_metrics = ["mAP50", "mAP50-95", "Precision", "Recall"]
    available_metrics = [m for m in all_metrics if all(m in results[name] for name in names)]

    # --- 图1: 主要指标柱状图 ---
    fig, ax = plt.subplots(figsize=(max(10, n * 3), 6))
    x = np.arange(len(available_metrics))
    width = 0.7 / n

    for i, name in enumerate(names):
        vals = [results[name].get(m, 0) for m in available_metrics]
        bars = ax.bar(x + i * width - 0.35 + width / 2, vals, width,
                      label=name, color=colors[i])
        for bar, v in zip(bars, vals):
            if v > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.008,
                        f"{v:.3f}", ha="center", va="bottom", fontsize=8)

    ax.set_ylabel("Score")
    ax.set_title("DEIMv2 vs YOLOv8s - Model Comparison (iter_20260425)")
    ax.set_xticks(x)
    ax.set_xticklabels(available_metrics)
    ax.set_ylim(0, 1.05)
    ax.legend(loc="lower right")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_dir / "deimv2_vs_yolov8s_metrics.png", dpi=150)
    plt.close(fig)

    # --- 图2: 分类别 AP50 (仅 YOLO 模型有此指标) ---
    yolo_names = [n for n in names if "yolov8s" in n]
    if yolo_names and all("Kid_AP50" in results[n] for n in yolo_names):
        class_metrics = ["Kid_AP50", "Adult_AP50"]
        fig, ax = plt.subplots(figsize=(max(6, len(yolo_names) * 2), 5))
        x = np.arange(len(class_metrics))
        width = 0.7 / len(yolo_names)

        for i, name in enumerate(yolo_names):
            vals = [results[name][m] for m in class_metrics]
            bars = ax.bar(x + i * width - 0.35 + width / 2, vals, width,
                          label=name, color=colors[i])
            for bar, v in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.008,
                        f"{v:.3f}", ha="center", va="bottom", fontsize=9)

        ax.set_ylabel("AP50")
        ax.set_title("YOLOv8s Per-Class AP50 (iter_20260425)")
        ax.set_xticks(x)
        ax.set_xticklabels(["Kid", "Adult"])
        ax.set_ylim(0, 1.05)
        ax.legend()
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        fig.savefig(save_dir / "yolov8s_per_class_ap50.png", dpi=150)
        plt.close(fig)

    # --- 保存 JSON ---
    with open(save_dir / "deimv2_vs_yolov8s_results.json", "w") as f:
        json.dump({"val_set": "iter_20260425", "results": results}, f, indent=2, ensure_ascii=False)

    # --- 打印表格 ---
    print(f"\n{'=' * 110}")
    header_metrics = available_metrics + [m for m in ["Kid_AP50", "Adult_AP50"]
                                          if any(m in results[n] for n in names)]
    header = f"{'模型':<30}"
    for m in header_metrics:
        header += f" {m:>10}"
    print(header)
    print("-" * 110)
    for name, v in results.items():
        row = f"{name:<30}"
        for m in header_metrics:
            val = v.get(m, None)
            row += f" {val:>10.4f}" if val is not None else f" {'N/A':>10}"
        print(row)
    print("=" * 110)

    print(f"\n对比图表已保存到: {save_dir}/")


def main():
    args = parse_args()

    # Step 1: 标注转换
    if not args.skip_convert and not args.skip_train:
        convert_annotations()
    else:
        # 即使跳过训练, 也确保标注存在 (对比评估可能需要)
        ann_train = DATA_DIR / "annotations/instances_train.json"
        if not ann_train.exists() and not args.skip_convert:
            convert_annotations()

    # Step 2: DEIMv2 训练
    if not args.skip_train:
        train_deimv2(device=args.device, resume=args.resume)

    # Step 3: 评估 DEIMv2
    print("\n" + "=" * 60)
    print(" Step 3: 模型评估 & 对比")
    print("=" * 60)

    deimv2_metrics = evaluate_deimv2(device=args.device)

    # Step 4: 评估 YOLOv8s
    print("\n  评估 YOLOv8s 历史模型...")
    yolo_results = evaluate_yolo_models(device=args.device)

    # Step 5: 合并结果 & 生成对比图表
    all_results = {}
    all_results.update(yolo_results)
    if deimv2_metrics:
        all_results["iter_20260425_deimv2_x"] = deimv2_metrics

    if all_results:
        save_dir = DEIMV2_RUN_DIR / "comparison"
        plot_comparison(all_results, save_dir)
    else:
        print("\n  [警告] 没有可用的评估结果")


if __name__ == "__main__":
    main()
