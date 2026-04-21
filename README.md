# 客厅小孩看护 - 目标检测

基于 YOLOv8s 的儿童与成人检测模型，部署目标平台为海思 HiSilicon 3519DV500。

## 项目结构

```
├── data/children_and_adults/   # 数据集 (图片不纳入git)
│   ├── data.yaml               # 数据集配置
│   └── labels/                 # YOLO格式标注
├── training/
│   ├── train.py                # 训练脚本
│   └── export_onnx.py          # ONNX导出脚本
└── .gitignore
```

## 快速开始

```bash
# 训练
python training/train.py

# 导出ONNX (训练完成后)
python training/export_onnx.py --weights runs/train/kids_care_yolov8s/weights/best.pt
```

## 模型信息

- 模型: YOLOv8s
- 输入尺寸: 640×384
- 类别: Kid, Adult
- 实验追踪: SwanLab
