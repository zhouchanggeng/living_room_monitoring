"""
客厅小孩看护 - ONNX 导出脚本
目标平台: HiSilicon 3519DV500 (NNIE/SVP)
输入尺寸: 640x384
"""

import argparse
from pathlib import Path
from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(description="YOLOv8s ONNX导出 (适配海思3519DV500)")
    parser.add_argument(
        "--weights",
        type=str,
        default="runs/train/kids_care_yolov8s/weights/best.pt",
        help="训练好的 .pt 权重路径",
    )
    parser.add_argument(
        "--imgsz",
        nargs=2,
        type=int,
        default=[384, 640],
        help="输入尺寸 (h w), 默认 384 640",
    )
    parser.add_argument("--opset", type=int, default=11, help="ONNX opset版本, 海思推荐11")
    parser.add_argument("--simplify", action="store_true", default=True, help="onnx-simplifier简化")
    parser.add_argument("--half", action="store_true", default=False, help="FP16导出")
    return parser.parse_args()


def export_onnx(weights: str, imgsz: list, opset: int, simplify: bool, half: bool):
    weights_path = Path(weights)
    if not weights_path.exists():
        raise FileNotFoundError(f"权重文件不存在: {weights_path}")

    print("=" * 60)
    print("ONNX 导出 (适配海思3519DV500)")
    print(f"  权重:     {weights}")
    print(f"  输入尺寸: {imgsz[1]}x{imgsz[0]} (WxH)")
    print(f"  Opset:    {opset}")
    print(f"  Simplify: {simplify}")
    print(f"  FP16:     {half}")
    print("=" * 60)

    model = YOLO(weights)

    # 导出ONNX
    # 海思NNIE要求:
    #   - opset 11 兼容性最好
    #   - 静态batch=1, 固定输入尺寸
    #   - 简化图结构便于mapper转换
    export_path = model.export(
        format="onnx",
        imgsz=imgsz,
        opset=opset,
        simplify=simplify,
        half=half,
        dynamic=False,       # 海思需要静态shape
        batch=1,             # 固定batch=1
    )

    print(f"\nONNX导出完成: {export_path}")
    print("\n后续海思部署步骤:")
    print("  1. 使用 RuyiStudio 或 nnie_mapper 将 .onnx 转为 .wk 模型")
    print("  2. 配置 mapper cfg 文件, 指定输入尺寸 640x384, BGR格式")
    print("  3. 在3519DV500上使用 SVP_NNIE 接口加载 .wk 进行推理")

    return export_path


def main():
    args = parse_args()
    export_onnx(
        weights=args.weights,
        imgsz=args.imgsz,
        opset=args.opset,
        simplify=args.simplify,
        half=args.half,
    )


if __name__ == "__main__":
    main()
