# 客厅小孩看护 - 目标检测

基于 YOLOv8s 的儿童与成人检测模型，部署目标平台为海思 HiSilicon 3519DV500。

## 项目结构

```
├── data/
│   ├── iter_20260423/          # 第1轮迭代数据集
│   ├── iter_20260425/          # 第2轮迭代数据集 (含COCO格式标注)
│   ├── iter_20260429/          # 第3轮迭代数据集 (含COCO格式标注)
│   ├── pseudo_labels_deimv2/   # DEIMv2 生成的伪标签数据
│   └── distill_20260430/       # 蒸馏训练合并数据集 (真实标注+伪标签)
├── training/
│   ├── train_runner.py         # 统一训练/伪标签/蒸馏入口
│   ├── train_runner.sh         # 带日志保存的训练入口包装脚本
│   ├── __init__.py
│   └── DEIMv2/                 # DEIMv2 子模块
├── tools/
│   ├── crawl_baby_images.py           # 图片爬取
│   ├── sample_and_predict.py          # 采样与预测
│   ├── predict_to_xanylabeling.py     # 预测结果转 X-AnyLabeling 格式
│   ├── update_labels_from_xanylabeling.py  # X-AnyLabeling 标注回写 YOLO 格式
│   ├── active_learning_sample.py      # 主动学习样本筛选
│   ├── yolo_sam3_joint_label.py       # YOLO+SAM3 联合标注
│   ├── video_filter.py                # 视频过滤
│   ├── video_inference.py             # 视频推理
│   └── yolo2coco.py                   # YOLO 格式转 COCO 格式
├── demo/
│   └── ptz_follow_demo.py             # YOLO+ByteTrack 云台跟随仿真 Demo
├── runtime/
│   └── follow_controller.py     # 跟随目标选择、状态机、平滑控制与仿真指令输出
├── tests/
│   ├── test_follow_controller.py       # 跟随控制器单元测试
│   ├── test_pretrained_paths.py        # 预训练权重路径检查
│   └── test_training_runner_layout.py  # 训练入口布局检查
├── deployment/
│   └── export_onnx.py          # ONNX 导出脚本
└── pretrained/                 # 预训练权重
```

## 训练流程

### 1. 迭代训练 (数据标注驱动)

```bash
# YOLOv8s 迭代训练
python3 training/train_runner.py --task iter --data data/iter_20260425/data.yaml --name iter_20260425

# DEIMv2 教师模型训练
python3 training/train_runner.py --task deimv2 \
  --config training/DEIMv2/configs/deimv2/deimv2_dinov3_x_kids_care.yml \
  --device 0,1
```

### 2. 伪标签蒸馏训练

使用 DEIMv2 教师模型对未标注数据生成伪标签，与人工标注合并后训练 YOLOv8s：

```bash
# 全流程一键运行 (伪标签生成 → 合并数据 → 蒸馏训练 → 对比评估)
bash training/train_runner.sh --task distill-pipe --infer-gpus 0,1 --train-gpus 0,1

# 或分步执行:
python3 training/train_runner.py --task pseudo    # 生成伪标签
python3 training/train_runner.py --task distill   # 蒸馏训练
python3 training/train_runner.py --task distill --skip-train  # 仅对比评估
```

更多任务和参数见 `docs/training_scripts_usage.md`。

### 3. 导出部署

```bash
python deployment/export_onnx.py --weights runs/train/<run_name>/weights/best.pt
```

## 数据工具

```bash
# 将 X-AnyLabeling 修正后的标注更新到数据集
python tools/update_labels_from_xanylabeling.py --src labeled_data/kidadult-20260424 --dst data/iter_20260425

# YOLO 格式转 COCO 格式
python tools/yolo2coco.py --help
```

## 云台跟随算法骨架

核心逻辑在 `runtime/follow_controller.py`。YOLO 只负责输出检测框，ByteTrack 负责给目标分配稳定 `track_id`，跟随控制器负责锁定同一个目标、按 `adult | child | auto` 选择跟随对象、短时丢失保持、超时搜索恢复、死区防抖、低通滤波，并返回模拟云台应该往哪边转动的坐标值。

状态机：

```text
IDLE   : 没有目标，模拟云台不动
ACQUIRE: 发现目标，准备锁定
TRACK  : 稳定跟随当前 track_id
HOLD   : 目标短时丢失，保留最后锁定目标
SEARCH : 目标超时丢失，输出低速扫描方向
```

离线调试示例：

```bash
python3 demo/ptz_follow_demo.py \
  --weights runs/train/distill_20260508_yolov8s_lr001/weights/best.pt \
  --source /data/zcg/workspace/data/video/test_video/2714_time.mp4 \
  --output child.mp4 \
  --follow-mode child
```

`--follow-mode` 可选：

```text
adult: 只跟随大人
child: 只跟随小孩
auto : 大人/小孩都可跟随，优先保持当前锁定 track_id
```

输出视频会叠加以下调试信息：

```text
state   : 当前状态机状态
target  : 当前锁定类别和 track_id
dir     : 模拟云台转动方向，例如 (right, down)
delta   : 每帧模拟坐标增量，例如 (0.800, 0.300)
norm_err: 目标中心相对画面中心的归一化偏移
```

`FollowController.update()` 返回 `FollowCommand`，主要字段如下：

```text
direction_x/direction_y : left/right/up/down/stop，表示模拟云台应该往哪边转
normalized_error_xy     : 目标相对画面中心的归一化偏移，右/下为正，左/上为负
sim_delta_xy            : 经过低通滤波、死区、PD 控制和限幅后的每帧模拟坐标增量
target_track_id         : 当前锁定目标 ID
target_bbox_xyxy        : 当前锁定目标框
```

简单理解：`norm_err` 表示目标偏离画面中心多少，`delta` 表示模拟云台这一帧应该往哪边动、动多少。

运行跟随控制器测试：

```bash
python3 -m unittest tests.test_follow_controller
```

## 模型信息

- 模型: YOLOv8s / DEIMv2-DINOv3-X (教师模型)
- 输入尺寸: 640×384
- 类别: Kid, Adult
- 实验追踪: SwanLab
