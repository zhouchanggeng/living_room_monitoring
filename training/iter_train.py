"""
客厅小孩看护 - 迭代训练脚本
每次迭代训练后, 自动收集所有历史 best.pt, 在当前验证集上做性能对比, 生成图表.

用法:
    python training/iter_train.py --data data/iter_20260425/data.yaml --name iter_20260425
    python training/iter_train.py --data data/iter_20260425/data.yaml --name iter_20260425 --resume
"""

import argparse
import json
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RUNS_DIR = PROJECT_ROOT / "runs" / "train"
MODEL_BASE = "yolov8s.pt"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=str, required=True, help="data.yaml 路径")
    p.add_argument("--name", type=str, required=True, help="迭代名称, 如 iter_20260425")
    p.add_argument("--model", type=str, default=MODEL_BASE, help="基础模型")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--device", type=str, default="0")
    p.add_argument("--resume", action="store_true", help="从上次中断处继续训练")
    p.add_argument("--skip-train", action="store_true", help="跳过训练, 仅做对比评估")
    return p.parse_args()


def train(args):
    """执行训练, 返回 best.pt 路径."""
    run_name = f"{args.name}_yolov8s"
    run_dir = RUNS_DIR / run_name

    if args.resume and (run_dir / "weights" / "last.pt").exists():
        print(f"[恢复训练] {run_dir / 'weights' / 'last.pt'}")
        model = YOLO(str(run_dir / "weights" / "last.pt"))
    else:
        model = YOLO(args.model)

    # 尝试加载 swanlab
    try:
        from swanlab.integration.ultralytics import add_swanlab_callback
        add_swanlab_callback(model, project="KidsCare-YOLOv8s",
                             experiment_name=run_name)
    except ImportError:
        pass

    model.train(
        data=args.data, imgsz=(384, 640), epochs=args.epochs,
        batch=args.batch, device=args.device,
        project=str(RUNS_DIR), name=run_name,
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


def collect_all_best_pts():
    """收集所有迭代的 best.pt, 按名称排序."""
    pts = {}
    for d in sorted(RUNS_DIR.iterdir()):
        bp = d / "weights" / "best.pt"
        if bp.exists() and d.name.startswith("iter_"):
            # 提取迭代名: iter_20260425_yolov8s -> iter_20260425
            iter_name = "_".join(d.name.split("_")[:2])
            pts[iter_name] = str(bp)
    return pts


def evaluate_all(all_pts, data_yaml, device="0"):
    """在同一验证集上评估所有模型, 返回 {name: {metric: value}}."""
    results = {}
    for name, pt_path in all_pts.items():
        print(f"\n[评估] {name}: {pt_path}")
        model = YOLO(pt_path)
        r = model.val(data=data_yaml, imgsz=640, batch=32, device=device, verbose=False)
        results[name] = {
            "mAP50": round(float(r.box.map50), 4),
            "mAP50-95": round(float(r.box.map), 4),
            "Precision": round(float(r.box.mp), 4),
            "Recall": round(float(r.box.mr), 4),
            "Kid_AP50": round(float(r.box.ap50[0]), 4),
            "Adult_AP50": round(float(r.box.ap50[1]), 4),
        }
        print(f"  mAP50={results[name]['mAP50']}, mAP50-95={results[name]['mAP50-95']}, "
              f"P={results[name]['Precision']}, R={results[name]['Recall']}")
    return results


def plot_comparison(results, val_name, save_dir):
    """生成对比图表并保存."""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    names = list(results.keys())
    n = len(names)
    colors = plt.cm.Set2(np.linspace(0, 1, max(n, 3)))

    # --- 图1: 主要指标柱状图 ---
    main_metrics = ["mAP50", "mAP50-95", "Precision", "Recall"]
    fig, ax = plt.subplots(figsize=(max(8, n * 2.5), 5))
    x = np.arange(len(main_metrics))
    width = 0.7 / n

    for i, name in enumerate(names):
        vals = [results[name][m] for m in main_metrics]
        bars = ax.bar(x + i * width - 0.35 + width / 2, vals, width,
                      label=name, color=colors[i])
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.008,
                    f"{v:.3f}", ha="center", va="bottom", fontsize=8)

    ax.set_ylabel("Score")
    ax.set_title(f"Model Comparison on {val_name}")
    ax.set_xticks(x)
    ax.set_xticklabels(main_metrics)
    ax.set_ylim(0, 1.05)
    ax.legend(loc="lower right")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_dir / "comparison_main_metrics.png", dpi=150)
    plt.close(fig)

    # --- 图2: 分类别 AP50 柱状图 ---
    class_metrics = ["Kid_AP50", "Adult_AP50"]
    fig, ax = plt.subplots(figsize=(max(6, n * 2), 5))
    x = np.arange(len(class_metrics))
    width = 0.7 / n

    for i, name in enumerate(names):
        vals = [results[name][m] for m in class_metrics]
        bars = ax.bar(x + i * width - 0.35 + width / 2, vals, width,
                      label=name, color=colors[i])
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.008,
                    f"{v:.3f}", ha="center", va="bottom", fontsize=9)

    ax.set_ylabel("AP50")
    ax.set_title(f"Per-Class AP50 on {val_name}")
    ax.set_xticks(x)
    ax.set_xticklabels(["Kid", "Adult"])
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_dir / "comparison_per_class.png", dpi=150)
    plt.close(fig)

    # --- 图3: 迭代趋势折线图 (>=2次迭代时) ---
    if n >= 2:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        for metric, ax in zip(["mAP50", "mAP50-95"], axes):
            vals = [results[name][metric] for name in names]
            ax.plot(names, vals, "o-", linewidth=2, markersize=8, color="#2196F3")
            for j, v in enumerate(vals):
                ax.annotate(f"{v:.4f}", (j, v), textcoords="offset points",
                            xytext=(0, 10), ha="center", fontsize=9)
            ax.set_title(metric)
            ax.set_ylabel("Score")
            ax.grid(alpha=0.3)
            ax.set_ylim(min(vals) - 0.05, max(vals) + 0.08)
        plt.suptitle(f"Iteration Trend on {val_name}", fontsize=13)
        fig.tight_layout()
        fig.savefig(save_dir / "comparison_trend.png", dpi=150)
        plt.close(fig)

    # --- 保存 JSON ---
    with open(save_dir / "comparison_results.json", "w") as f:
        json.dump({"val_set": val_name, "results": results}, f, indent=2, ensure_ascii=False)

    # --- 打印表格 ---
    print(f"\n{'=' * 100}")
    print(f"{'模型':<20} {'mAP50':>8} {'mAP50-95':>10} {'Precision':>10} {'Recall':>8} {'Kid_AP50':>10} {'Adult_AP50':>11}")
    print("-" * 100)
    for name, v in results.items():
        print(f"{name:<20} {v['mAP50']:>8.4f} {v['mAP50-95']:>10.4f} "
              f"{v['Precision']:>10.4f} {v['Recall']:>8.4f} "
              f"{v['Kid_AP50']:>10.4f} {v['Adult_AP50']:>11.4f}")
    print("=" * 100)

    print(f"\n图表已保存到: {save_dir}/")
    return save_dir


def main():
    args = parse_args()

    # 1. 训练
    if not args.skip_train:
        print(f"\n{'=' * 60}")
        print(f" 迭代训练: {args.name}")
        print(f" 数据集:   {args.data}")
        print(f"{'=' * 60}\n")
        best_pt = train(args)
        print(f"\n训练完成! best.pt: {best_pt}")
    else:
        print("[跳过训练, 仅做对比评估]")

    # 2. 收集所有历史 best.pt
    all_pts = collect_all_best_pts()
    print(f"\n找到 {len(all_pts)} 个迭代模型: {list(all_pts.keys())}")

    # 3. 在当前验证集上评估所有模型
    print(f"\n在 {args.data} 验证集上评估所有模型...")
    results = evaluate_all(all_pts, args.data, args.device)

    # 4. 生成对比图表
    save_dir = RUNS_DIR / f"{args.name}_yolov8s" / "comparison"
    plot_comparison(results, args.name, save_dir)


if __name__ == "__main__":
    main()
