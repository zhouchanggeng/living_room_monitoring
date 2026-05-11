# Training 统一入口使用说明

`training/` 顶层旧训练脚本已经归档到：

```text
/home/zcg/data/workspace/living_room_monitoring/archive/training_20260511_unified/
```

归档脚本只作为历史备份，不再作为当前训练入口使用。当前只保留：

```text
training/train_runner.py
training/train_runner.sh
training/__init__.py
```

## 基本用法

直接使用 Python 入口：

```bash
python3 training/train_runner.py --task basic
python3 training/train_runner.py --task iter --data data/iter_20260425/data.yaml --name iter_20260425
python3 training/train_runner.py --task distill --device 0,1 --epochs 100
```

使用 Bash 包装入口保存日志：

```bash
bash training/train_runner.sh --task distill-pipe --infer-gpus 0,1 --train-gpus 0,1
```

调试命令编排但不真正执行：

```bash
python3 training/train_runner.py --task yolov8x --dry-run
```

## Task 列表

| task | 用途 |
|---|---|
| `basic` | 基础 YOLOv8s 训练 |
| `iter` | 迭代训练 YOLOv8s，并评估历史迭代模型 |
| `yolov8x` | YOLOv8x 训练和历史模型评估 |
| `distill` | 合并真实标注和伪标签，训练 YOLOv8s |
| `distill-pipe` | DEIMv2 伪标签生成 + 蒸馏训练流水线 |
| `distill-v2` | 伪标签 v2 + Active Learning + 蒸馏训练流水线 |
| `pseudo` | DEIMv2 生成 YOLO 格式伪标签 |
| `pseudo-v2` | 伪标签 v2 生成入口 |
| `deimv2` | 训练 DEIMv2 |
| `deimv2-pipe` | DEIMv2 训练包装入口 |
| `eval-pseudo` | 从伪标签数据抽样代理验证集并评估 YOLO 模型 |

## 常用参数

| 参数 | 说明 | 默认值 |
|---|---|---|
| `--task` | 任务类型 | 必填 |
| `--data` | 数据集 `data.yaml` 或数据目录 | 视任务而定 |
| `--name` | 运行名称 | 自动推导 |
| `--model` | 模型权重或 yaml 配置 | `pretrained/yolov8s.pt` 或 `pretrained/yolov8x.pt` |
| `--pretrained-weights` | yaml 架构训练时迁移的预训练权重 | `None` |
| `--epochs` | 训练轮数 | `100` |
| `--batch` | 训练 batch size | `16` |
| `--device` | GPU，如 `0` 或 `0,1` | `0` |
| `--resume` | 恢复训练 | 关闭 |
| `--skip-train` | 跳过训练，仅评估或准备数据 | 关闭 |
| `--dry-run` | 只打印命令，不执行 | 关闭 |

## 伪标签和蒸馏参数

| 参数 | 说明 | 默认值 |
|---|---|---|
| `--real-data` | 真实标注数据目录 | `data/iter_20260429` |
| `--pseudo-data` | 伪标签数据目录 | `data/pseudo_labels_deimv2` |
| `--output-data` | 合并数据输出目录 | `data/distill_20260430` |
| `--pseudo-ratio` | 使用伪标签比例 | `1.0` |
| `--conf` | DEIMv2 伪标签置信度阈值 | `0.65` |
| `--conf-deimv2` | v2 中 DEIMv2 置信度阈值 | `0.5` |
| `--conf-yolo` | v2 中 YOLO 置信度阈值 | `0.4` |
| `--match-iou` | 双模型匹配 IoU 阈值 | `0.5` |
| `--nms-iou` | 同类别 NMS IoU 阈值 | `0.6` |
| `--infer-gpus` | 伪标签推理 GPU 列表 | `0,1` |
| `--train-gpus` | 蒸馏训练 GPU 列表 | `0,1` |
| `--batch-size` | 伪标签推理 batch size | `8` |
| `--train-batch` | 蒸馏训练 batch size | `16` |

## 示例

基础训练：

```bash
python3 training/train_runner.py \
  --task basic \
  --data data/sample_1000/data.yaml \
  --device 0
```

迭代训练：

```bash
python3 training/train_runner.py \
  --task iter \
  --data data/iter_20260425/data.yaml \
  --name iter_20260425 \
  --device 0,1
```

蒸馏训练：

```bash
python3 training/train_runner.py \
  --task distill \
  --real-data data/iter_20260429 \
  --pseudo-data data/pseudo_labels_deimv2 \
  --output-data data/distill_20260430 \
  --device 0,1
```

蒸馏流水线：

```bash
bash training/train_runner.sh \
  --task distill-pipe \
  --infer-gpus 0,1 \
  --train-gpus 0,1 \
  --conf 0.65
```

YOLOv8x 训练：

```bash
python3 training/train_runner.py \
  --task yolov8x \
  --data data/distill_20260502/data.yaml \
  --device 0,1
```

DEIMv2 训练：

```bash
python3 training/train_runner.py \
  --task deimv2 \
  --config training/DEIMv2/configs/deimv2/deimv2_dinov3_x_kids_care.yml \
  --device 0,1
```
