# Training 脚本用法统计

本文档整理 `training/` 目录下 Python 和 Shell 脚本的用途、常用命令、主要参数以及注意事项。

## 总体统计

| 范围 | `.py` 数量 | `.sh` 数量 | 说明 |
|---|---:|---:|---|
| `training/` 顶层 | 10 | 4 | 本项目主要训练、蒸馏、伪标签流水线 |
| `training/DEIMv2/` | 102 | 1 | DEIMv2 上游框架、模型、工具、推理部署脚本 |
| 合计 | 112 | 5 | 共 117 个 Python/Shell 文件 |

## 顶层主流程脚本

| 文件 | 用途 | 常用命令 |
|---|---|---|
| `training/train.py` | 基础 YOLOv8s 训练，写死使用 `data/sample_1000/data.yaml` | `python training/train.py` |
| `training/train.sh` | YOLOv8s 启动脚本，加日志、GPU 环境、训练后尝试导出 ONNX | `bash training/train.sh --gpu 0 --batch 16 --epochs 100` |
| `training/iter_train.py` | 迭代训练 YOLOv8s，并汇总历史 `iter_*` 模型评估图表 | `python training/iter_train.py --data data/iter_20260425/data.yaml --name iter_20260425` |
| `training/train_deimv2.py` | YOLO 标注转 COCO，训练 DEIMv2，并和 YOLOv8s 对比 | `python training/train_deimv2.py --device 0,1` |
| `training/train_deimv2.sh` | DEIMv2 训练包装脚本，保存日志 | `bash training/train_deimv2.sh --device 0,1` |
| `training/yolo2coco.py` | YOLO txt 标注转 COCO JSON | `python training/yolo2coco.py --data-dir data/iter_20260429 --classes Kid Adult` |

## 伪标签与蒸馏脚本

| 文件 | 用途 | 常用命令 |
|---|---|---|
| `training/generate_pseudo_labels.py` | 用 DEIMv2 对未标注图片生成 YOLO 格式伪标签 | `python training/generate_pseudo_labels.py --conf 0.65 --device 0` |
| `training/distill_train.py` | 合并真实标注和伪标签，训练 YOLOv8s 学生模型，并评估对比 | `python training/distill_train.py --pseudo-ratio 1.0 --device 0,1` |
| `training/run_distill.sh` | 第一阶段全流程：DEIMv2 多卡伪标签 -> YOLOv8s 蒸馏 -> 评估 | `bash training/run_distill.sh --infer-gpus 0,1 --train-gpus 0,1 --conf 0.65` |
| `training/generate_pseudo_labels_v2.py` | DEIMv2 + YOLOv8s 双模型交叉验证伪标签，只保留一致检测 | `python training/generate_pseudo_labels_v2.py --device 0 --match-iou 0.5` |
| `training/run_distill_v2.sh` | 第二阶段全流程：交叉验证伪标签 -> Active Learning -> 蒸馏训练 | `bash training/run_distill_v2.sh --infer-gpus 0,1 --skip-al` |
| `training/train_yolov8x.py` | 用合并数据训练 YOLOv8x，并和历史模型对比 | `python training/train_yolov8x.py --data data/distill_20260502/data.yaml --device 0,1` |
| `training/evaluate_pseudo_val.py` | 从伪标签数据抽代理验证集，统一评估已训练模型 | `python training/evaluate_pseudo_val.py --sample-size 400 --device 0` |

## 主要参数速查

### `training/iter_train.py`

| 参数 | 说明 | 默认值 |
|---|---|---|
| `--data` | `data.yaml` 路径 | 必填 |
| `--name` | 迭代名称，如 `iter_20260425` | 必填 |
| `--model` | 基础模型 | `yolov8s.pt` |
| `--epochs` | 训练轮数 | `100` |
| `--batch` | batch size | `16` |
| `--device` | GPU，如 `0` 或 `0,1` | `0` |
| `--resume` | 从上次中断处继续训练 | 关闭 |
| `--skip-train` | 跳过训练，仅做对比评估 | 关闭 |

### `training/train_deimv2.py`

| 参数 | 说明 | 默认值 |
|---|---|---|
| `--device` | GPU id，多卡用 `0,1` | `0` |
| `--skip-convert` | 跳过 YOLO -> COCO 标注转换 | 关闭 |
| `--skip-train` | 跳过训练，仅做对比评估 | 关闭 |
| `--resume` | 从指定 checkpoint 恢复训练 | `None` |

### `training/generate_pseudo_labels.py`

| 参数 | 说明 | 默认值 |
|---|---|---|
| `--config` | DEIMv2 配置文件 | `training/DEIMv2/configs/deimv2/deimv2_dinov3_x_kids_care.yml` |
| `--checkpoint` | DEIMv2 checkpoint | `runs/train/iter_20260429_deimv2_x/best_stg2.pth` |
| `--input-dir` | 未标注图片目录 | `pending_data/all_20260416` |
| `--output-dir` | 伪标签输出目录 | `data/pseudo_labels_deimv2` |
| `--conf` | 置信度阈值 | `0.65` |
| `--device` | 推理设备 | `0` |
| `--batch-size` | 推理 batch size | `8` |
| `--exclude-dirs` | 排除已用于训练的图片目录 | `None` |
| `--shard-id` | 多卡分片 ID | `0` |
| `--num-shards` | 多卡总分片数 | `1` |

### `training/distill_train.py`

| 参数 | 说明 | 默认值 |
|---|---|---|
| `--real-data` | 真实标注数据目录 | `data/iter_20260429` |
| `--pseudo-data` | 伪标签数据目录 | `data/pseudo_labels_deimv2` |
| `--output-data` | 合并数据输出目录 | `data/distill_20260430` |
| `--pseudo-ratio` | 使用伪标签比例，`0-1` | `1.0` |
| `--model` | 模型配置或权重 | `yolov8s.pt` |
| `--pretrained-weights` | 预训练权重 | `yolov8s.pt` |
| `--epochs` | 训练轮数 | `100` |
| `--batch` | batch size | `16` |
| `--device` | GPU，如 `0` 或 `0,1` | `0` |
| `--resume` | 恢复训练 | 关闭 |
| `--skip-merge` | 跳过数据合并 | 关闭 |
| `--skip-train` | 跳过训练，仅评估 | 关闭 |
| `--lr0` | 初始学习率 | `0.01` |
| `--run-suffix` | 运行名额外后缀 | 空 |

### `training/generate_pseudo_labels_v2.py`

| 参数 | 说明 | 默认值 |
|---|---|---|
| `--deimv2-config` | DEIMv2 配置文件 | `training/DEIMv2/configs/deimv2/deimv2_dinov3_x_kids_care.yml` |
| `--deimv2-ckpt` | DEIMv2 checkpoint | `runs/train/iter_20260429_deimv2_x/best_stg2.pth` |
| `--yolo-ckpt` | YOLOv8s checkpoint | `runs/train/distill_20260430_yolov8s/weights/best.pt` |
| `--input-dir` | 未标注图片目录 | `pending_data/all_20260416` |
| `--output-dir` | 伪标签输出目录 | `data/pseudo_labels_v2` |
| `--conf-deimv2` | DEIMv2 置信度阈值 | `0.5` |
| `--conf-yolo` | YOLOv8s 置信度阈值 | `0.4` |
| `--match-iou` | 双模型匹配 IoU 阈值 | `0.5` |
| `--nms-iou` | 同类别 NMS IoU 阈值 | `0.6` |
| `--device` | 推理设备 | `0` |
| `--shard-id` | 多卡分片 ID | `0` |
| `--num-shards` | 多卡总分片数 | `1` |

### `training/train_yolov8x.py`

| 参数 | 说明 | 默认值 |
|---|---|---|
| `--data` | `data.yaml` 路径 | `data/distill_20260502/data.yaml` |
| `--model` | 模型权重或配置 | `yolov8x.pt` |
| `--epochs` | 训练轮数 | `100` |
| `--batch` | batch size | `8` |
| `--device` | GPU，如 `0` 或 `0,1` | `0` |
| `--resume` | 恢复训练 | 关闭 |
| `--skip-train` | 跳过训练，仅评估 | 关闭 |
| `--name` | 运行名称 | 根据数据集名自动生成 |

### `training/evaluate_pseudo_val.py`

| 参数 | 说明 | 默认值 |
|---|---|---|
| `--source-data` | 伪标签源数据目录 | `data/pseudo_labels_deimv2_20260508` |
| `--output-data` | 代理验证集输出目录 | `data/pseudo_val_20260508` |
| `--sample-size` | 代理验证集图片数 | `400` |
| `--seed` | 随机种子 | `42` |
| `--device` | 评估设备 | `0` |
| `--batch` | batch size | `16` |
| `--imgsz` | 评估图片尺寸 | `640` |
| `--skip-sample` | 跳过抽样，复用已有代理验证集 | 关闭 |
| `--include-deimv2` | 评估 DEIMv2 模型 | 开启 |
| `--include-yolo` | 评估 YOLO 模型 | 开启 |
| `--skip-yolo` | 跳过 YOLO 模型评估 | 关闭 |
| `--skip-deimv2` | 跳过 DEIMv2 模型评估 | 关闭 |

## Shell 编排脚本

### `training/run_distill.sh`

第一阶段全流程脚本：

1. 多卡并行运行 `training/generate_pseudo_labels.py` 生成 DEIMv2 伪标签。
2. 调用 `training/distill_train.py` 合并真实标注和伪标签。
3. 训练 YOLOv8s，并在验证集上做对比评估。

常用命令：

```bash
bash training/run_distill.sh
bash training/run_distill.sh --infer-gpus 0,1,2,3 --train-gpus 0,1
bash training/run_distill.sh --conf 0.7
bash training/run_distill.sh --skip-pseudo
bash training/run_distill.sh --skip-train
```

主要参数：

| 参数 | 说明 | 默认值 |
|---|---|---|
| `--infer-gpus` | 伪标签推理 GPU | `0,1` |
| `--train-gpus` | 训练 GPU | `0,1` |
| `--conf` | 伪标签置信度阈值 | `0.65` |
| `--batch-size` | 推理 batch size per GPU | `8` |
| `--pseudo-ratio` | 伪标签使用比例 | `1.0` |
| `--epochs` | 训练轮数 | `100` |
| `--train-batch` | 训练 batch size | `16` |
| `--skip-pseudo` | 跳过伪标签生成 | 关闭 |
| `--skip-train` | 跳过训练，仅评估 | 关闭 |
| `--resume` | 恢复训练 | 关闭 |

### `training/run_distill_v2.sh`

第二阶段全流程脚本：

1. 多卡并行运行 `training/generate_pseudo_labels_v2.py` 生成双模型交叉验证伪标签。
2. 调用 `tools/active_learning_sample.py` 做 Active Learning 采样。
3. 调用 `training/distill_train.py` 基于 `data/pseudo_labels_v2` 训练。

常用命令：

```bash
bash training/run_distill_v2.sh
bash training/run_distill_v2.sh --infer-gpus 0,1,2,3
bash training/run_distill_v2.sh --skip-pseudo --skip-al
```

主要参数：

| 参数 | 说明 | 默认值 |
|---|---|---|
| `--infer-gpus` | 推理 GPU | `0,1` |
| `--train-gpus` | 训练 GPU | `0,1` |
| `--conf-deimv2` | DEIMv2 置信度阈值 | `0.5` |
| `--conf-yolo` | YOLOv8s 置信度阈值 | `0.4` |
| `--match-iou` | 双模型匹配 IoU 阈值 | `0.5` |
| `--al-num` | Active Learning 采样数 | `1000` |
| `--epochs` | 训练轮数 | `100` |
| `--train-batch` | 训练 batch size | `16` |
| `--skip-pseudo` | 跳过伪标签生成 | 关闭 |
| `--skip-al` | 跳过 Active Learning 采样 | 关闭 |
| `--skip-train` | 跳过训练 | 关闭 |

## DEIMv2 子目录

| 文件或目录 | 用途 |
|---|---|
| `training/DEIMv2/train.py` | DEIMv2 官方训练入口，通常由 `torchrun` 调用 |
| `training/DEIMv2/engine/` | 模型、backbone、data、solver、optimizer 等框架内部代码，不建议作为日常入口直接运行 |
| `training/DEIMv2/tools/` | 官方工具：ONNX/TensorRT 导出、推理、benchmark、可视化、数据处理 |
| `training/DEIMv2/tools/reference/safe_training.sh` | DEIMv2 官方安全训练/自动恢复示例脚本 |

DEIMv2 官方训练入口参数：

| 参数 | 说明 |
|---|---|
| `-c`, `--config` | 配置文件 |
| `-r`, `--resume` | 从 checkpoint 恢复 |
| `-t`, `--tuning` | 从 checkpoint fine-tune |
| `-d`, `--device` | 设备 |
| `--seed` | 随机种子 |
| `--use-amp` | 使用 AMP 混合精度 |
| `--output-dir` | 输出目录 |
| `--summary-dir` | TensorBoard summary 目录 |
| `--test-only` | 仅测试 |
| `-u`, `--update` | 更新 YAML 配置 |
| `--print-method` | 打印方法 |
| `--print-rank` | 打印 rank |
| `--local-rank` | 分布式 local rank |

常用命令示例：

```bash
cd training/DEIMv2
CUDA_VISIBLE_DEVICES=0,1 torchrun --master_port=7778 --nproc_per_node=2 train.py \
  -c configs/deimv2/deimv2_dinov3_x_kids_care.yml \
  --use-amp \
  --seed=0 \
  -t ../../pretrained/deimv2_dinov3_x_coco.pth
```

## 推荐使用顺序

1. 普通 YOLOv8s 训练：`training/iter_train.py`
2. 训练 DEIMv2 teacher：`training/train_deimv2.sh`
3. 生成第一版伪标签并蒸馏：`training/run_distill.sh`
4. 双模型交叉验证伪标签：`training/run_distill_v2.sh`
5. 更大 YOLOv8x 训练：`training/train_yolov8x.py`
6. 统一代理验证：`training/evaluate_pseudo_val.py`

## 注意事项

1. `training/train.sh` 检查的是 `data/children_and_adults/data.yaml`，但 `training/train.py` 实际写死使用 `data/sample_1000/data.yaml`，两者不一致。
2. `training/train.sh` 会调用 `training/export_onnx.py`，但当前 `training` 目录下没有这个文件。
3. `training/run_distill_v2.sh` 调用了 `tools/active_learning_sample.py`，它不在 `training` 目录内，需要确认项目根目录下是否存在。
4. `training/evaluate_pseudo_val.py` 写死了 `/data/zcg/miniconda3/bin/python` 和 `/data/zcg/miniconda3/bin/torchrun`，换环境时可能需要调整。
5. `training/DEIMv2/` 内部大部分 `.py` 是框架代码或官方工具，不建议和顶层业务流水线脚本混用统计口径。
