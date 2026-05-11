"""
客厅小孩看护 - YOLOv8x 训练脚本
模型: yolov8x | 输入尺寸: 640x384
数据: 真实标注 (iter_20260502) + 伪标签 (distill_20260502) 合并数据集

用法:
    # 使用最新蒸馏数据集训练 (真实标注+伪标签已合并)
    python training/train_yolov8x.py

    # 指定数据集和GPU
    python training/train_yolov8x.py --data data/distill_20260502/data.yaml --device 0,1

    # 恢复训练
    python training/train_yolov8x.py --resume

    # 跳过训练仅评估
    python training/train_yolov8x.py --skip-train
"""

import argparse
import json
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

# 默认使用最新的蒸馏数据集 (真实标注+伪标签已合并)
DEFAULT_DATA = str(PROJECT_ROOT / "data" / "distill_20260502" / "data.yaml")
MODEL = "yolov8x.pt"  # Ultralytics 自动下载预训练权重
IMGSZ = (384, 640)     # (h, w) - 非正方形输入


def parse_args():
    p = argparse.ArgumentParser(description="YOLOv8x 训练 (标注+伪标签)")
    p.add_argument("--data", type=str, default=DEFAULT_DATA,
                   help="data.yaml 路径")
    p.add_argument("--model", type=str, default=MODEL,
                   help="模型权重或配置, 如 yolov8x.pt")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch", type=int, default=8,
                   help="批大小 (yolov8x 较大, 默认8)")
    p.add_argument("--device", type=str, default="0",
                   help="GPU, 多卡用 '0,1'")
    p.add_argument("--resume", action="store_true",
                   help="从上次中断处继续训练")
    p.add_argument("--skip-train", action="store_true",
                   help="跳过训练, 仅做对比评估")
    p.add_argument("--name", type=str, default=None,
                   help="运行名称 (默认自动生成)")
    return p.parse_args()


def train(args, run_name):
    """训练 YOLOv8x, 返回 best.pt 路径."""
    run_dir = RUNS_DIR / run_name

    if args.resume and (run_dir / "weights" / "last.pt").exists():
        print(f"[恢复训练] {run_dir / 'weights' / 'last.pt'}")
        model = YOLO(str(run_dir / "weights" / "last.pt"))
    else:
        print(f"[加载模型] {args.model}")
        model = YOLO(args.model)

    # 尝试启用 SwanLab
    try:
        from swanlab.integration.ultralytics import add_swanlab_callback
        add_swanlab_callback(
            model,
            project="KidsCare-YOLOv8x",
            experiment_name=run_name,
            description="客厅小孩看护 YOLOv8x 640x384 训练 (标注+伪标签)",
        )
        print("[SwanLab] 已启用实验追踪")
    except ImportError:
        pass

    # TensorBoard
    try:
        from torch.utils.tensorboard import SummaryWriter  # noqa: F401
        print(f"[TensorBoard] 已启用, 启动命令: tensorboard --logdir {RUNS_DIR}")
    except ImportError:
        pass

    model.train(
        data=args.data,
        imgsz=IMGSZ,
        epochs=args.epochs,
        batch=args.batch,
        device=args.device,
        project=str(RUNS_DIR),
        name=run_name,
        pretrained=True,
        optimizer="SGD",
        lr0=0.01,
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=3,
        warmup_momentum=0.8,
        warmup_bias_lr=0.1,
        cos_lr=True,
        # 数据增强
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=0.0,
        translate=0.1,
        scale=0.5,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.1,
        # 保存与日志
        save=True,
        save_period=10,
        val=True,
        plots=True,
        exist_ok=True,
    )

    best_pt = run_dir / "weights" / "best.pt"
    if not best_pt.exists():
        raise FileNotFoundError(f"训练完成但未找到 best.pt: {best_pt}")
    return best_pt


def evaluate_all(val_data_yaml, device="0"):
    """在验证集上评估所有迭代模型 (包括 yolov8s 和 yolov8x)."""
    results = {}
    for d in sorted(RUNS_DIR.iterdir()):
        bp = d / "weights" / "best.pt"
        if bp.exists() and d.name.startswith(("iter_", "distill_")):
            name = d.name
            # 简化显示名
            for suffix in ["_yolov8s", "_yolov8x"]:
                if name.endswith(suffix):
                    name = name[:-len(suffix)] + suffix.replace("_", " ").strip()
                    break
            print(f"\n  评估 {name}: {bp}")
            model = YOLO(str(bp))
            r = model.val(data=val_data_yaml, imgsz=640, batch=16,
                          device=device, verbose=False)
            results[name] = {
                "mAP50": round(float(r.box.map50), 4),
                "mAP50-95": round(float(r.box.map), 4),
                "Precision": round(float(r.box.mp), 4),
                "Recall": round(float(r.box.mr), 4),
                "Kid_AP50": round(float(r.box.ap50[0]), 4),
                "Adult_AP50": round(float(r.box.ap50[1]), 4),
            }
            print(f"    mAP50={results[name]['mAP50']}, "
                  f"mAP50-95={results[name]['mAP50-95']}")
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
    ax.set_title("YOLOv8x Training - Model Comparison")
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_ylim(0, 1.05)
    ax.legend(loc="lower right", fontsize=7)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_dir / "yolov8x_comparison.png", dpi=150)
    plt.close(fig)

    # 保存 JSON
    with open(save_dir / "yolov8x_results.json", "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # 打印表格
    print(f"\n{'=' * 110}")
    print(f"{'模型':<35} {'mAP50':>8} {'mAP50-95':>10} {'Precision':>10} "
          f"{'Recall':>8} {'Kid_AP50':>10} {'Adult_AP50':>11}")
    print("-" * 110)
    for name, v in results.items():
        print(f"{name:<35} {v['mAP50']:>8.4f} {v['mAP50-95']:>10.4f} "
              f"{v['Precision']:>10.4f} {v['Recall']:>8.4f} "
              f"{v['Kid_AP50']:>10.4f} {v['Adult_AP50']:>11.4f}")
    print("=" * 110)


class Tee:
    """同时输出到终端和日志文件."""
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
    log_file = log_dir / f"train_yolov8x_{timestamp}.log"
    sys.stdout = sys.stderr = Tee(log_file)
    print(f"[日志] {log_file}")

    # 确认数据集
    data_path = Path(args.data)
    if not data_path.exists():
        # 尝试相对于项目根目录
        data_path = PROJECT_ROOT / args.data
        if not data_path.exists():
            raise FileNotFoundError(f"数据集配置不存在: {args.data}")
        args.data = str(data_path)

    # 运行名称
    data_name = Path(args.data).parent.name  # 如 distill_20260502
    run_name = args.name or f"{data_name}_yolov8x"

    print("\n" + "=" * 60)
    print("  客厅小孩看护 - YOLOv8x 训练")
    print(f"  模型:     {args.model}")
    print(f"  输入尺寸: {IMGSZ[1]}x{IMGSZ[0]} (WxH)")
    print(f"  数据集:   {args.data}")
    print(f"  运行名:   {run_name}")
    print(f"  Epochs:   {args.epochs}")
    print(f"  Batch:    {args.batch}")
    print(f"  Device:   {args.device}")
    print("=" * 60)

    # 1. 训练
    if not args.skip_train:
        best_pt = train(args, run_name)
        print(f"\n训练完成! best.pt: {best_pt}")
    else:
        print("[跳过训练, 仅做对比评估]")

    # 2. 对比评估 (使用iter_20260502验证集, 与历史模型对比)
    print("\n" + "=" * 60)
    print(" 模型对比评估")
    print("=" * 60)

    # 优先使用 iter_20260502 的验证集做对比
    val_yaml = str(PROJECT_ROOT / "data" / "iter_20260502" / "data.yaml")
    if not Path(val_yaml).exists():
        val_yaml = args.data
    print(f"  验证集: {val_yaml}")

    results = evaluate_all(val_yaml, args.device)
    if results:
        save_dir = RUNS_DIR / run_name / "comparison"
        plot_comparison(results, save_dir)
        print(f"\n图表已保存到: {save_dir}/")


if __name__ == "__main__":
    main()
