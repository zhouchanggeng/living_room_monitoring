"""
客厅小孩看护 - YOLOv8s 训练脚本
模型: yolov8s | 输入尺寸: 640x384
目标部署平台: HiSilicon 3519DV500
"""

import os
import sys
from datetime import datetime
from pathlib import Path
from ultralytics import YOLO
from swanlab.integration.ultralytics import add_swanlab_callback

# ============ 训练配置 ============
DATA_YAML = "data/sample_1000/data.yaml"
MODEL = "yolov8s.pt"  # 预训练权重 (ultralytics 自动下载)
IMGSZ = (384, 640)            # (h, w) - 非正方形输入
EPOCHS = 100
BATCH = 16
DEVICE = "0"                  # GPU id, 多卡用 "0,1"
# 使用绝对路径, 避免受 ~/.config/Ultralytics/settings.yaml 中 runs_dir 影响
PROJECT = str(Path(__file__).resolve().parent.parent / "runs" / "train")
NAME = "sample1000_yolov8s"

# 针对海思3519DV500部署的训练优化
TRAIN_ARGS = dict(
    data=DATA_YAML,
    imgsz=IMGSZ,
    epochs=EPOCHS,
    batch=BATCH,
    device=DEVICE,
    project=PROJECT,
    name=NAME,
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
    # 日志自动保存到 logs/ (通过 shell 脚本调用时由 shell 负责日志, 跳过)
    if not os.environ.get("KIDSCARE_LOG_BY_SHELL"):
        log_dir = Path(__file__).resolve().parent.parent / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = log_dir / f"train_{NAME}_{timestamp}.log"
        sys.stdout = sys.stderr = Tee(log_file)
        print(f"[日志] 训练日志保存到: {log_file}")

    # 确认数据集路径
    data_path = Path(DATA_YAML)
    if not data_path.exists():
        raise FileNotFoundError(f"数据集配置不存在: {data_path}")

    print("=" * 60)
    print("客厅小孩看护 - YOLOv8s 训练")
    print(f"  输入尺寸: {IMGSZ[1]}x{IMGSZ[0]} (WxH)")
    print(f"  数据集:   {DATA_YAML}")
    print(f"  Epochs:   {EPOCHS}")
    print(f"  Batch:    {BATCH}")
    print("=" * 60)

    # 加载模型
    model = YOLO(MODEL)

    # 集成 SwanLab 实验追踪
    add_swanlab_callback(
        model,
        project="KidsCare-YOLOv8s",
        experiment_name=NAME,
        description="客厅小孩看护 YOLOv8s 640x384 训练, 部署目标: hisi3519dv500",
    )

    # 开始训练
    results = model.train(**TRAIN_ARGS)

    # 从训练结果中获取最优权重路径
    best_pt = Path(results.save_dir) / "weights" / "best.pt"
    if not best_pt.exists():
        print("[警告] 未找到 best.pt, 跳过验证")
        return None

    print(f"\n训练完成! 最优权重: {best_pt}")

    # 在验证集上评估
    model_best = YOLO(str(best_pt))
    metrics = model_best.val(data=DATA_YAML, imgsz=IMGSZ, device=DEVICE)
    print(f"mAP50:    {metrics.box.map50:.4f}")
    print(f"mAP50-95: {metrics.box.map:.4f}")

    return best_pt


if __name__ == "__main__":
    best_weight = main()
