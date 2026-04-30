#!/bin/bash
#
# 第一阶段: DEIMv2 伪标签蒸馏训练 YOLOv8s - 全流程编排脚本
#
# 流程:
#   Step 1: 多卡并行 DEIMv2 推理, 生成伪标签
#   Step 2: 合并真实标注 + 伪标签, 训练 YOLOv8s
#   Step 3: 在验证集上对比评估
#
# 用法:
#   bash run_distill.sh                    # 默认使用 GPU 0,1 (推理2卡, 训练2卡)
#   bash run_distill.sh --infer-gpus 0,1,2,3 --train-gpus 0,1  # 自定义
#   bash run_distill.sh --conf 0.7         # 调整伪标签置信度
#   bash run_distill.sh --skip-pseudo      # 跳过伪标签生成 (已生成过)
#   bash run_distill.sh --skip-train       # 跳过训练, 仅评估
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

# ============ 默认参数 ============
INFER_GPUS="0,1"          # 伪标签推理用的 GPU
TRAIN_GPUS="0,1"          # YOLOv8s 训练用的 GPU
CONF=0.65                 # 伪标签置信度阈值
BATCH_SIZE=8              # 推理 batch size (per GPU)
PSEUDO_RATIO=1.0          # 使用伪标签的比例
EPOCHS=100
TRAIN_BATCH=16            # 训练 batch size (total)
SKIP_PSEUDO=false
SKIP_TRAIN=false
RESUME=false

# ============ 参数解析 ============
while [[ $# -gt 0 ]]; do
    case $1 in
        --infer-gpus)   INFER_GPUS="$2"; shift 2 ;;
        --train-gpus)   TRAIN_GPUS="$2"; shift 2 ;;
        --conf)         CONF="$2"; shift 2 ;;
        --batch-size)   BATCH_SIZE="$2"; shift 2 ;;
        --pseudo-ratio) PSEUDO_RATIO="$2"; shift 2 ;;
        --epochs)       EPOCHS="$2"; shift 2 ;;
        --train-batch)  TRAIN_BATCH="$2"; shift 2 ;;
        --skip-pseudo)  SKIP_PSEUDO=true; shift ;;
        --skip-train)   SKIP_TRAIN=true; shift ;;
        --resume)       RESUME=true; shift ;;
        -h|--help)
            echo "用法: bash run_distill.sh [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --infer-gpus GPU_IDS   伪标签推理 GPU (默认: 0,1)"
            echo "  --train-gpus GPU_IDS   训练 GPU (默认: 0,1)"
            echo "  --conf FLOAT           伪标签置信度阈值 (默认: 0.65)"
            echo "  --batch-size INT       推理 batch size per GPU (默认: 8)"
            echo "  --pseudo-ratio FLOAT   伪标签使用比例 (默认: 1.0)"
            echo "  --epochs INT           训练 epochs (默认: 100)"
            echo "  --train-batch INT      训练 batch size (默认: 16)"
            echo "  --skip-pseudo          跳过伪标签生成"
            echo "  --skip-train           跳过训练, 仅评估"
            echo "  --resume               恢复训练"
            exit 0 ;;
        *) echo "未知参数: $1"; exit 1 ;;
    esac
done

# 解析 GPU 列表
IFS=',' read -ra INFER_GPU_ARRAY <<< "$INFER_GPUS"
NUM_INFER_GPUS=${#INFER_GPU_ARRAY[@]}

echo "============================================================"
echo " 第一阶段: DEIMv2 伪标签蒸馏训练 YOLOv8s"
echo "============================================================"
echo "  推理 GPU:     $INFER_GPUS ($NUM_INFER_GPUS 卡)"
echo "  训练 GPU:     $TRAIN_GPUS"
echo "  置信度阈值:   $CONF"
echo "  伪标签比例:   $PSEUDO_RATIO"
echo "  训练 Epochs:  $EPOCHS"
echo "  跳过伪标签:   $SKIP_PSEUDO"
echo "  跳过训练:     $SKIP_TRAIN"
echo "============================================================"
echo ""

# ============================================================
# Step 1: 多卡并行生成伪标签
# ============================================================
if [ "$SKIP_PSEUDO" = false ]; then
    echo "=========================================="
    echo " Step 1: 多卡并行生成伪标签 ($NUM_INFER_GPUS GPUs)"
    echo "=========================================="

    PIDS=()
    LOG_DIR="$PROJECT_ROOT/logs"
    mkdir -p "$LOG_DIR"

    for i in "${!INFER_GPU_ARRAY[@]}"; do
        GPU_ID="${INFER_GPU_ARRAY[$i]}"
        LOG_FILE="$LOG_DIR/pseudo_label_gpu${GPU_ID}.log"

        echo "  启动 GPU $GPU_ID (shard $i/$NUM_INFER_GPUS) -> $LOG_FILE"

        CUDA_VISIBLE_DEVICES="$GPU_ID" python training/generate_pseudo_labels.py \
            --conf "$CONF" \
            --device 0 \
            --batch-size "$BATCH_SIZE" \
            --shard-id "$i" \
            --num-shards "$NUM_INFER_GPUS" \
            > "$LOG_FILE" 2>&1 &

        PIDS+=($!)
    done

    # 等待所有推理进程完成
    echo ""
    echo "  等待 $NUM_INFER_GPUS 个推理进程完成..."
    FAILED=0
    for i in "${!PIDS[@]}"; do
        PID=${PIDS[$i]}
        GPU_ID="${INFER_GPU_ARRAY[$i]}"
        if wait "$PID"; then
            echo "  ✓ GPU $GPU_ID (PID $PID) 完成"
        else
            echo "  ✗ GPU $GPU_ID (PID $PID) 失败!"
            FAILED=$((FAILED + 1))
        fi
    done

    if [ "$FAILED" -gt 0 ]; then
        echo ""
        echo "  [错误] $FAILED 个推理进程失败, 查看日志:"
        for GPU_ID in "${INFER_GPU_ARRAY[@]}"; do
            echo "    tail $LOG_DIR/pseudo_label_gpu${GPU_ID}.log"
        done
        exit 1
    fi

    # 统计伪标签结果
    PSEUDO_DIR="$PROJECT_ROOT/data/pseudo_labels_deimv2"
    if [ -d "$PSEUDO_DIR/labels" ]; then
        NUM_LABELS=$(ls "$PSEUDO_DIR/labels/" | wc -l)
        NUM_IMAGES=$(ls "$PSEUDO_DIR/images/" | wc -l)
        echo ""
        echo "  伪标签生成完成: $NUM_LABELS 标签, $NUM_IMAGES 图片"
    fi
    echo ""
fi

# ============================================================
# Step 2: 蒸馏训练 YOLOv8s
# ============================================================
if [ "$SKIP_TRAIN" = false ]; then
    echo "=========================================="
    echo " Step 2: YOLOv8s 蒸馏训练 (GPU: $TRAIN_GPUS)"
    echo "=========================================="

    TRAIN_CMD="python training/distill_train.py \
        --device $TRAIN_GPUS \
        --pseudo-ratio $PSEUDO_RATIO \
        --epochs $EPOCHS \
        --batch $TRAIN_BATCH"

    if [ "$RESUME" = true ]; then
        TRAIN_CMD="$TRAIN_CMD --resume"
    fi

    if [ "$SKIP_PSEUDO" = true ]; then
        TRAIN_CMD="$TRAIN_CMD --skip-merge"
    fi

    echo "  $TRAIN_CMD"
    echo ""
    eval "$TRAIN_CMD"
else
    # 仅评估
    echo "=========================================="
    echo " Step 2: 跳过训练, 仅对比评估"
    echo "=========================================="
    python training/distill_train.py --skip-train --device "${INFER_GPU_ARRAY[0]}"
fi

echo ""
echo "============================================================"
echo " 全流程完成!"
echo "============================================================"
