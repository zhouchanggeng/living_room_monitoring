"""
客厅小孩看护 - YOLOv8s 训练脚本
模型: yolov8s | 输入尺寸: 640x384
目标部署平台: HiSilicon 3519DV500
"""

import os
from pathlib import Path
from ultralytics import YOLO
from swanlab.integration.ultralytics import add_swanlab_callback

# ============ 训练配置 ============
DATA_YAML = "data/children_and_adults/data.yaml"
MODEL = "yolov8s.pt"          # 预训练权重
IMGSZ = (384, 640)            # (h, w) - 非正方形输入
EPOCHS = 100
BATCH = 16
DEVICE = "0"                  # GPU id, 多卡用 "0,1"
PROJECT = "runs/train"
NAME = "kids_care_yolov8s"

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


def main():
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

    # 训练完成后验证
    best_pt = Path(PROJECT) / NAME / "weights" / "best.pt"
    print(f"\n训练完成! 最优权重: {best_pt}")

    # 在验证集上评估
    model_best = YOLO(str(best_pt))
    metrics = model_best.val(data=DATA_YAML, imgsz=IMGSZ, device=DEVICE)
    print(f"mAP50:    {metrics.box.map50:.4f}")
    print(f"mAP50-95: {metrics.box.map:.4f}")

    return best_pt


if __name__ == "__main__":
    best_weight = main()
