#!/bin/bash
#
# 第二阶段: 交叉验证伪标签 + Active Learning 全流程
#
# 流程:
#   Step 1: 双模型交叉验证生成高质量伪标签
#   Step 2: Active Learning 采样不确定图片供人工标注
#   Step 3: (可选) 合并数据后蒸馏训练
#
# 用法:
#   bash training/run_distill_v2.sh                          # 全流程
#   bash training/run_distill_v2.sh --infer-gpus 0,1,2,3    # 4卡推理
#   bash training/run_distill_v2.sh --skip-pseudo --skip-al  # 仅训练
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

# ============ 默认参数 ============
INFER_GPUS="0,1"
TRAIN_GPUS="0,1"
CONF_DEIMV2=0.5
CONF_YOLO=0.4
MATCH_IOU=0.5
NMS_IOU=0.6
AL_NUM=1000
EPOCHS=100
TRAIN_BATCH=16
SKIP_PSEUDO=false
SKIP_AL=false
SKIP_TRAIN=false

# ============ 参数解析 ============
while [[ $# -gt 0 ]]; do
    case $1 in
        --infer-gpus)    INFER_GPUS="$2"; shift 2 ;;
        --train-gpus)    TRAIN_GPUS="$2"; shift 2 ;;
        --conf-deimv2)   CONF_DEIMV2="$2"; shift 2 ;;
        --conf-yolo)     CONF_YOLO="$2"; shift 2 ;;
        --match-iou)     MATCH_IOU="$2"; shift 2 ;;
        --al-num)        AL_NUM="$2"; shift 2 ;;
        --epochs)        EPOCHS="$2"; shift 2 ;;
        --train-batch)   TRAIN_BATCH="$2"; shift 2 ;;
        --skip-pseudo)   SKIP_PSEUDO=true; shift ;;
        --skip-al)       SKIP_AL=true; shift ;;
        --skip-train)    SKIP_TRAIN=true; shift ;;
        -h|--help)
            echo "用法: bash training/run_distill_v2.sh [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --infer-gpus GPU_IDS    推理 GPU (默认: 0,1)"
            echo "  --train-gpus GPU_IDS    训练 GPU (默认: 0,1)"
            echo "  --conf-deimv2 FLOAT     DEIMv2 置信度 (默认: 0.5)"
            echo "  --conf-yolo FLOAT       YOLOv8s 置信度 (默认: 0.4)"
            echo "  --match-iou FLOAT       匹配 IoU (默认: 0.5)"
            echo "  --al-num INT            Active Learning 采样数 (默认: 1000)"
            echo "  --epochs INT            训练 epochs (默认: 100)"
            echo "  --skip-pseudo           跳过伪标签生成"
            echo "  --skip-al               跳过 Active Learning 采样"
            echo "  --skip-train            跳过训练"
            exit 0 ;;
        *) echo "未知参数: $1"; exit 1 ;;
    esac
done

IFS=',' read -ra INFER_GPU_ARRAY <<< "$INFER_GPUS"
NUM_INFER_GPUS=${#INFER_GPU_ARRAY[@]}

echo "============================================================"
echo " 第二阶段: 交叉验证伪标签 + Active Learning"
echo "============================================================"
echo "  推理 GPU:       $INFER_GPUS ($NUM_INFER_GPUS 卡)"
echo "  训练 GPU:       $TRAIN_GPUS"
echo "  DEIMv2 conf:    $CONF_DEIMV2"
echo "  YOLOv8s conf:   $CONF_YOLO"
echo "  匹配 IoU:       $MATCH_IOU"
echo "  AL 采样数:      $AL_NUM"
echo "============================================================"
echo ""

# ============================================================
# Step 1: 多卡并行交叉验证伪标签
# ============================================================
if [ "$SKIP_PSEUDO" = false ]; then
    echo "=========================================="
    echo " Step 1: 交叉验证伪标签 ($NUM_INFER_GPUS GPUs)"
    echo "=========================================="

    PIDS=()
    LOG_DIR="$PROJECT_ROOT/logs"
    mkdir -p "$LOG_DIR"

    for i in "${!INFER_GPU_ARRAY[@]}"; do
        GPU_ID="${INFER_GPU_ARRAY[$i]}"
        LOG_FILE="$LOG_DIR/pseudo_v2_gpu${GPU_ID}.log"

        echo "  启动 GPU $GPU_ID (shard $i/$NUM_INFER_GPUS)"

        CUDA_VISIBLE_DEVICES="$GPU_ID" python training/generate_pseudo_labels_v2.py \
            --conf-deimv2 "$CONF_DEIMV2" \
            --conf-yolo "$CONF_YOLO" \
            --match-iou "$MATCH_IOU" \
            --nms-iou "$NMS_IOU" \
            --device 0 \
            --shard-id "$i" \
            --num-shards "$NUM_INFER_GPUS" \
            > "$LOG_FILE" 2>&1 &

        PIDS+=($!)
    done

    echo "  等待 $NUM_INFER_GPUS 个进程..."
    FAILED=0
    for i in "${!PIDS[@]}"; do
        if wait "${PIDS[$i]}"; then
            echo "  ✓ GPU ${INFER_GPU_ARRAY[$i]} 完成"
        else
            echo "  ✗ GPU ${INFER_GPU_ARRAY[$i]} 失败!"
            FAILED=$((FAILED + 1))
        fi
    done

    if [ "$FAILED" -gt 0 ]; then
        echo "  [错误] $FAILED 个进程失败"
        exit 1
    fi

    PSEUDO_DIR="$PROJECT_ROOT/data/pseudo_labels_v2"
    if [ -d "$PSEUDO_DIR/labels" ]; then
        echo "  伪标签: $(ls "$PSEUDO_DIR/labels/" | wc -l) 个文件"
    fi
    echo ""
fi

# ============================================================
# Step 2: Active Learning 采样
# ============================================================
if [ "$SKIP_AL" = false ]; then
    echo "=========================================="
    echo " Step 2: Active Learning 采样 ($AL_NUM 张)"
    echo "=========================================="

    CUDA_VISIBLE_DEVICES="${INFER_GPU_ARRAY[0]}" python tools/active_learning_sample.py \
        --num "$AL_NUM" \
        --device 0

    echo ""
fi

# ============================================================
# Step 3: 蒸馏训练
# ============================================================
if [ "$SKIP_TRAIN" = false ]; then
    echo "=========================================="
    echo " Step 3: YOLOv8s 蒸馏训练 v2 (GPU: $TRAIN_GPUS)"
    echo "=========================================="

    python training/distill_train.py \
        --pseudo-data data/pseudo_labels_v2 \
        --output-data data/distill_v2 \
        --device "$TRAIN_GPUS" \
        --epochs "$EPOCHS" \
        --batch "$TRAIN_BATCH"
else
    echo "=========================================="
    echo " Step 3: 跳过训练, 仅评估"
    echo "=========================================="
    python training/distill_train.py --skip-train --device "${INFER_GPU_ARRAY[0]}"
fi

echo ""
echo "============================================================"
echo " 全流程完成!"
echo "============================================================"
