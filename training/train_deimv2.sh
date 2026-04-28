#!/bin/bash
# DEIMv2 (DINOv3-X) 训练 & 与 YOLOv8s 对比
# 用法:
#   bash training/train_deimv2.sh              # 完整流程: 转换标注 + 训练 + 对比
#   bash training/train_deimv2.sh --skip-train # 仅对比评估
#   bash training/train_deimv2.sh --device 0,1 # 多卡训练

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo "============================================================"
echo " DEIMv2 (DINOv3-X) 训练 - 客厅小孩看护"
echo " 预训练模型: deimv2_dinov3_x_coco.pth"
echo " 数据集:     data/iter_20260425"
echo "============================================================"

python "${SCRIPT_DIR}/train_deimv2.py" "$@" 2>&1 | tee "${PROJECT_ROOT}/train_deimv2.log"
