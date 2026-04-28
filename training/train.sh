#!/bin/bash
# ============================================================
# 客厅小孩看护 - YOLOv8s 训练启动脚本
# 用法: bash training/train.sh [--gpu 0] [--batch 16] [--epochs 100]
# ============================================================

set -e

# ---------- 默认参数 ----------
GPU_ID="0"
BATCH_SIZE=16
EPOCHS=100
RESUME=""

# ---------- 解析命令行参数 ----------
while [[ $# -gt 0 ]]; do
    case $1 in
        --gpu)      GPU_ID="$2";      shift 2 ;;
        --batch)    BATCH_SIZE="$2";   shift 2 ;;
        --epochs)   EPOCHS="$2";      shift 2 ;;
        --resume)   RESUME="--resume"; shift   ;;
        *)          echo "未知参数: $1"; exit 1 ;;
    esac
done

# ---------- 项目根目录 (脚本所在目录的上一级) ----------
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# ---------- 日志目录 ----------
LOG_DIR="${PROJECT_ROOT}/logs"
mkdir -p "$LOG_DIR"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="${LOG_DIR}/train_yolov8s_${TIMESTAMP}.log"

echo "============================================================"
echo " 客厅小孩看护 - YOLOv8s 训练"
echo " GPU:    ${GPU_ID}"
echo " Batch:  ${BATCH_SIZE}"
echo " Epochs: ${EPOCHS}"
echo " 项目:   ${PROJECT_ROOT}"
echo "============================================================"

# ---------- 检查数据集 ----------
DATA_YAML="${PROJECT_ROOT}/data/children_and_adults/data.yaml"
if [ ! -f "$DATA_YAML" ]; then
    echo "[错误] 数据集配置不存在: $DATA_YAML"
    exit 1
fi

TRAIN_IMG_DIR="${PROJECT_ROOT}/data/children_and_adults/images/train"
if [ ! -d "$TRAIN_IMG_DIR" ]; then
    echo "[错误] 训练图片目录不存在: $TRAIN_IMG_DIR"
    exit 1
fi

TRAIN_COUNT=$(ls "$TRAIN_IMG_DIR" | wc -l)
echo "[信息] 训练集图片数量: ${TRAIN_COUNT}"

# ---------- 检查GPU ----------
if command -v nvidia-smi &> /dev/null; then
    echo "[信息] GPU 状态:"
    nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv,noheader
else
    echo "[警告] 未检测到 nvidia-smi, 将使用CPU训练"
fi

# ---------- 设置环境变量 ----------
export CUDA_VISIBLE_DEVICES="${GPU_ID}"

# ---------- 开始训练 ----------
echo ""
echo "[开始] 训练启动..."
echo ""

cd "$PROJECT_ROOT"
export KIDSCARE_LOG_BY_SHELL=1
python training/train.py 2>&1 | tee "$LOG_FILE"

echo ""
echo "[完成] 训练结束"
echo "[信息] 最优权重: runs/train/kids_care_yolov8s/weights/best.pt"

# ---------- 自动导出ONNX ----------
BEST_PT=$(find runs/train -name "best.pt" -type f -newer training/train.py 2>/dev/null | head -1)
if [ -n "$BEST_PT" ]; then
    echo ""
    echo "[开始] 自动导出 ONNX..."
    python training/export_onnx.py --weights "$BEST_PT"
    echo "[完成] ONNX 导出结束"
else
    echo "[警告] 未找到 best.pt, 跳过 ONNX 导出"
fi

echo ""
echo "[日志] 训练日志已保存到: ${LOG_FILE}"
