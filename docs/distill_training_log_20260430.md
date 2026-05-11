# 第一阶段: DEIMv2 伪标签蒸馏训练 YOLOv8s - 记录

日期: 2026-04-30

## 1. 任务目标

使用已训练好的 DEIMv2 (DINOv3-X) 教师模型，对未标注图片集 `all_20260416` (11.9万张) 生成高置信度伪标签，与人工标注数据合并后训练 YOLOv8s 学生模型，实现半监督知识蒸馏。

## 2. 创建的脚本

### 2.1 伪标签生成: `training/generate_pseudo_labels.py`

- 加载 DEIMv2 (`runs/train/iter_20260429_deimv2_x/best_stg2.pth`)
- 对 `pending_data/all_20260416` 中约 11.9 万张未标注图片推理
- 0.65 置信度阈值过滤低质量检测
- 支持多卡分片并行 (`--shard-id` / `--num-shards`)
- 输出 YOLO 格式标签到 `data/pseudo_labels_deimv2/`

### 2.2 蒸馏训练: `training/distill_train.py`

- 合并 `data/iter_20260429` (真实标注) + 伪标签数据到 `data/distill_20260430/`
- 自动去重，验证集不变
- 训练 YOLOv8s，训练完成后与所有历史模型对比评估
- 支持 `--pseudo-ratio` 控制伪标签使用比例

### 2.3 编排脚本: `training/run_distill.sh`

全流程一键运行，支持多卡并行：

```bash
# 默认: 2卡推理 + 2卡训练
bash training/run_distill.sh

# 4卡推理, 2卡训练, 高置信度
bash training/run_distill.sh --infer-gpus 0,1,2,3 --train-gpus 0,1 --conf 0.7

# 跳过伪标签生成, 仅训练
bash training/run_distill.sh --skip-pseudo --train-gpus 0,1,2,3
```

## 3. 实际运行配置

```bash
bash training/run_distill.sh \
    --infer-gpus 0,1,3,4 \
    --train-gpus 0,1 \
    --conf 0.65 \
    --batch-size 4 \
    --epochs 100
```

- 伪标签推理: 4 卡并行 (GPU 0,1,3,4)
- YOLOv8s 训练: 2 卡 DDP (GPU 0,1)
- 训练耗时: 13.3 小时

## 4. 数据统计

### 4.1 数据量对比

| | 图片数 | 标注框数 | Kid 框 | Adult 框 |
|---|---|---|---|---|
| 真实标注 (iter_20260429) | 3,393 | 4,406 | 830 (18.8%) | 3,576 (81.2%) |
| 伪标签 (pseudo_labels_deimv2) | 96,177 | 130,347 | 17,842 (13.7%) | 112,505 (86.3%) |
| 合并后训练集 | 96,645 | 134,753 | 18,672 | 116,081 |
| 验证集 | 542 | 712 | 117 (16.4%) | 595 (83.6%) |

- 数据放大倍数: **28.3x** (图片), **29.5x** (框)
- 伪标签覆盖率: 96,177 / 119,135 = 80.7% (19.3% 被高置信度阈值过滤)

### 4.2 Kid 框面积分布

| | 小 (<1%面积) | 中 (1-5%) | 大 (>5%) |
|---|---|---|---|
| 真实标注 | 122 (14.7%) | 542 (65.3%) | 166 (20.0%) |
| 伪标签 (采样) | 178 (9.4%) | 1,284 (68.1%) | 423 (22.4%) |
| 验证集 | 14 (12.0%) | 79 (67.5%) | 24 (20.5%) |

## 5. 训练结果

### 5.1 全模型对比 (iter_20260429 验证集)

| 模型 | mAP50 | mAP50-95 | Precision | Recall | Kid_AP50 | Adult_AP50 |
|---|---|---|---|---|---|---|
| **DEIMv2-DINOv3-X** (教师) | 0.8224 | 0.6345 | - | 0.8591* | - | - |
| **distill_20260430 YOLOv8s** (蒸馏) | **0.8209** | **0.6431** | 0.8232 | 0.7861 | **0.7123** | **0.9295** |
| iter_20260425 YOLOv8s | 0.7260 | 0.5719 | 0.7257 | 0.6510 | 0.5890 | 0.8630 |
| iter_20260423 YOLOv8s | 0.6714 | 0.5473 | 0.7462 | 0.6028 | 0.5152 | 0.8275 |

> *DEIMv2 使用 COCO evaluator (AR@100)，YOLOv8s 使用 ultralytics evaluator，评估方式略有差异

### 5.2 相对 iter_20260425 基线的提升

| 指标 | 基线 | 蒸馏后 | 提升 |
|---|---|---|---|
| mAP50 | 0.726 | 0.821 | **+9.5%** |
| mAP50-95 | 0.572 | 0.643 | **+7.1%** |
| Kid AP50 | 0.589 | 0.712 | **+12.3%** |
| Adult AP50 | 0.863 | 0.930 | **+6.7%** |
| Recall | 0.651 | 0.786 | **+13.5%** |

### 5.3 训练最终 epoch 指标

```
Epoch 100/100:
  box_loss: 0.3655  cls_loss: 0.2646  dfl_loss: 0.8295
  
Best model validation:
  all        542  712  P=0.825  R=0.783  mAP50=0.821  mAP50-95=0.640
  Kid        102  117  P=0.775  R=0.646  mAP50=0.712  mAP50-95=0.517
  Adult      457  595  P=0.876  R=0.919  mAP50=0.929  mAP50-95=0.763
```

## 6. 蒸馏生效原因分析

### 6.1 数据量爆炸式增长 (核心驱动力)

从 3,393 张扩展到 96,645 张 (28x)，模型见到的场景多样性大幅提升。覆盖了不同设备型号、时间段、家庭场景、人物姿态。

### 6.2 Kid 长尾类别的数据补充

Kid 真实标注只有 830 框，伪标签补充了 17,842 框 (21.5x)。直接反映在 Kid AP50 从 0.589 跳到 0.712 (+12.3%)，是所有指标中提升最大的。

### 6.3 教师模型足够强

DEIMv2-DINOv3-X 在验证集 mAP50=0.822，加上 0.65 高置信度阈值过滤，伪标签准确率高、噪声低。11.9 万张中 2.3 万张被过滤掉，只保留可靠检测结果。

### 6.4 学生超越教师

蒸馏后 YOLOv8s 的 mAP50-95 (0.643) 略超 DEIMv2 (0.635)，原因:
- YOLOv8s 同时学习了真实标注 (定位更精确) 和伪标签
- mosaic/mixup 等强数据增强带来额外泛化收益
- 评估方式差异 (COCO evaluator vs ultralytics evaluator)

## 7. 后续提升方向 (按优先级)

### 第一优先级: 补强 Kid 检测

1. **针对性补充 Kid 真实标注** — 从待标注数据中优先挑选含小孩的图片，新增 500-1000 张高质量标注
2. **Kid 专项数据增强** — copy-paste augmentation，小目标 scale jitter
3. **分类别置信度阈值** — Kid 用 0.5、Adult 用 0.7 重新生成伪标签，捞回更多 Kid 样本

### 第二优先级: 提升定位精度

4. **提高训练分辨率** — 640×640 或 960×576 训练，导出时适配部署
5. **SAM 精修伪标签框** — 用 DEIMv2 的框作为 SAM prompt，得到更精确的 bbox

### 第三优先级: 迭代蒸馏

6. **第二轮蒸馏** — 用蒸馏后 YOLOv8s 重新生成伪标签，与 DEIMv2 伪标签 ensemble
7. **多教师 ensemble** — 多个教师模型推理取交集

### 第四优先级: 模型和训练策略

8. **更大 backbone** — YOLOv8m 或 RepVGG 重参数化
9. **训练策略调优** — 200 epoch，更大 batch size，close mosaic last 10 epochs

## 8. 产出文件

```
training/generate_pseudo_labels.py    # 伪标签生成脚本 (支持多卡分片)
training/distill_train.py             # 蒸馏训练脚本
training/run_distill.sh               # 全流程编排脚本
data/pseudo_labels_deimv2/            # 伪标签数据 (96,177 张)
data/distill_20260430/                # 合并训练数据集
runs/train/distill_20260430_yolov8s/  # 训练结果
  ├── weights/best.pt                 # 最优权重
  ├── weights/last.pt
  └── comparison/                     # 对比图表和 JSON
logs/run_distill_20260430.log         # 运行日志
```
