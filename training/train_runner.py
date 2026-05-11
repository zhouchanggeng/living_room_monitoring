"""
统一训练入口。

旧入口已归档, 这里不再调用归档内容。
所有任务通过 --task 开关选择, 其余参数按任务生效。
"""

from __future__ import annotations

import argparse
import os
import random
import subprocess
import sys
from datetime import datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUNS_DIR = PROJECT_ROOT / "runs" / "train"
PRETRAINED_DIR = PROJECT_ROOT / "pretrained"


TASKS = (
    "basic",
    "iter",
    "yolov8x",
    "distill",
    "distill-pipe",
    "distill-v2",
    "pseudo",
    "pseudo-v2",
    "deimv2",
    "deimv2-pipe",
    "eval-pseudo",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="客厅看护统一训练入口")
    parser.add_argument("--task", choices=TASKS, required=True, help="要执行的训练/数据任务")
    parser.add_argument("--data", default=None, help="data.yaml 或数据目录, 视任务而定")
    parser.add_argument("--name", default=None, help="训练运行名称")
    parser.add_argument("--model", default=None, help="模型权重或模型配置 yaml")
    parser.add_argument("--pretrained-weights", default=None, help="yaml 架构训练时加载的预训练权重")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default="0")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--skip-merge", action="store_true")
    parser.add_argument("--skip-pseudo", action="store_true")
    parser.add_argument("--skip-al", action="store_true")
    parser.add_argument("--lr0", type=float, default=0.01)
    parser.add_argument("--pseudo-ratio", type=float, default=1.0)
    parser.add_argument("--real-data", default=str(PROJECT_ROOT / "data" / "iter_20260429"))
    parser.add_argument("--pseudo-data", default=str(PROJECT_ROOT / "data" / "pseudo_labels_deimv2"))
    parser.add_argument("--output-data", default=None)
    parser.add_argument("--input-dir", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--conf", type=float, default=0.65)
    parser.add_argument("--conf-deimv2", type=float, default=0.5)
    parser.add_argument("--conf-yolo", type=float, default=0.4)
    parser.add_argument("--match-iou", type=float, default=0.5)
    parser.add_argument("--nms-iou", type=float, default=0.6)
    parser.add_argument("--infer-gpus", default="0,1")
    parser.add_argument("--train-gpus", default="0,1")
    parser.add_argument("--batch-size", type=int, default=8, help="推理 batch size")
    parser.add_argument("--train-batch", type=int, default=16)
    parser.add_argument("--al-num", type=int, default=1000)
    parser.add_argument("--sample-size", type=int, default=400)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--config", default=None, help="DEIMv2 配置路径")
    parser.add_argument("--checkpoint", default=None, help="模型 checkpoint 路径")
    parser.add_argument("--dry-run", action="store_true", help="只打印将执行的命令")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    os.chdir(PROJECT_ROOT)
    RUNS_DIR.mkdir(parents=True, exist_ok=True)

    dispatch = {
        "basic": run_basic,
        "iter": run_iter,
        "yolov8x": run_yolov8x,
        "distill": run_distill,
        "distill-pipe": run_distill_pipe,
        "distill-v2": run_distill_v2,
        "pseudo": run_pseudo,
        "pseudo-v2": run_pseudo_v2,
        "deimv2": run_deimv2,
        "deimv2-pipe": run_deimv2_pipe,
        "eval-pseudo": run_eval_pseudo,
    }
    dispatch[args.task](args)
    return 0


def run_basic(args: argparse.Namespace) -> None:
    data = args.data or str(PROJECT_ROOT / "data" / "sample_1000" / "data.yaml")
    name = args.name or "sample1000_yolov8s"
    model = args.model or str(PRETRAINED_DIR / "yolov8s.pt")
    train_yolo(
        model=model,
        data=data,
        name=name,
        epochs=args.epochs,
        batch=args.batch,
        device=args.device,
        lr0=args.lr0,
        resume=args.resume,
        dry_run=args.dry_run,
    )


def run_iter(args: argparse.Namespace) -> None:
    if not args.data:
        raise ValueError("--task iter 需要 --data")
    if not args.name:
        raise ValueError("--task iter 需要 --name")
    model = args.model or str(PRETRAINED_DIR / "yolov8s.pt")
    if not args.skip_train:
        train_yolo(
            model=model,
            data=args.data,
            name=f"{args.name}_yolov8s",
            epochs=args.epochs,
            batch=args.batch,
            device=args.device,
            lr0=args.lr0,
            resume=args.resume,
            dry_run=args.dry_run,
        )
    evaluate_yolo_runs(args.data, args.device, prefix=("iter_",), dry_run=args.dry_run)


def run_yolov8x(args: argparse.Namespace) -> None:
    data = args.data or str(PROJECT_ROOT / "data" / "distill_20260502" / "data.yaml")
    model = args.model or str(PRETRAINED_DIR / "yolov8x.pt")
    name = args.name or f"{Path(data).parent.name}_yolov8x"
    if not args.skip_train:
        train_yolo(
            model=model,
            data=data,
            name=name,
            epochs=args.epochs,
            batch=args.batch,
            device=args.device,
            lr0=args.lr0,
            resume=args.resume,
            dry_run=args.dry_run,
        )
    evaluate_yolo_runs(data, args.device, prefix=("iter_", "distill_"), dry_run=args.dry_run)


def run_distill(args: argparse.Namespace) -> None:
    output_data = args.output_data or str(PROJECT_ROOT / "data" / "distill_20260430")
    data_yaml = str(Path(output_data) / "data.yaml")
    if not args.skip_merge and not args.skip_train:
        merge_yolo_datasets(args.real_data, args.pseudo_data, output_data, args.pseudo_ratio)
    elif not Path(data_yaml).exists():
        data_yaml = str(Path(args.real_data) / "data.yaml")

    model = args.model or str(PRETRAINED_DIR / "yolov8s.pt")
    run_name = args.name or f"{Path(output_data).name}_yolov8s"
    if not args.skip_train:
        train_yolo(
            model=model,
            data=data_yaml,
            name=run_name,
            epochs=args.epochs,
            batch=args.batch,
            device=args.device,
            lr0=args.lr0,
            resume=args.resume,
            pretrained_weights=args.pretrained_weights,
            dry_run=args.dry_run,
        )
    evaluate_yolo_runs(str(Path(args.real_data) / "data.yaml"), args.device, dry_run=args.dry_run)


def run_distill_pipe(args: argparse.Namespace) -> None:
    if not args.skip_pseudo:
        run_parallel(
            worker=lambda gpu, shard, total: generate_pseudo_labels(
                config_path=args.config or "training/DEIMv2/configs/deimv2/deimv2_dinov3_x_kids_care.yml",
                checkpoint_path=args.checkpoint or str(RUNS_DIR / "iter_20260429_deimv2_x" / "best_stg2.pth"),
                input_dir=args.input_dir or str(PROJECT_ROOT / "pending_data" / "all_20260416"),
                output_dir=args.output_dir or str(PROJECT_ROOT / "data" / "pseudo_labels_deimv2"),
                conf=args.conf,
                device="0",
                shard_id=shard,
                num_shards=total,
                batch_size=args.batch_size,
                dry_run=args.dry_run,
            ),
            gpus=args.infer_gpus,
        )
    args.device = args.train_gpus
    args.batch = args.train_batch
    args.skip_merge = args.skip_pseudo
    run_distill(args)


def run_distill_v2(args: argparse.Namespace) -> None:
    if not args.skip_pseudo:
        run_parallel(
            worker=lambda gpu, shard, total: generate_pseudo_labels_v2(
                deimv2_config=args.config or "training/DEIMv2/configs/deimv2/deimv2_dinov3_x_kids_care.yml",
                deimv2_ckpt=args.checkpoint or str(RUNS_DIR / "iter_20260429_deimv2_x" / "best_stg2.pth"),
                yolo_ckpt=args.model or str(RUNS_DIR / "distill_20260430_yolov8s" / "weights" / "best.pt"),
                input_dir=args.input_dir or str(PROJECT_ROOT / "pending_data" / "all_20260416"),
                output_dir=args.output_dir or str(PROJECT_ROOT / "data" / "pseudo_labels_v2"),
                conf_deimv2=args.conf_deimv2,
                conf_yolo=args.conf_yolo,
                match_iou=args.match_iou,
                nms_iou=args.nms_iou,
                device="0",
                shard_id=shard,
                num_shards=total,
                dry_run=args.dry_run,
            ),
            gpus=args.infer_gpus,
        )
    if not args.skip_al:
        run_command(
            [
                sys.executable,
                "tools/active_learning_sample.py",
                "--num",
                str(args.al_num),
                "--device",
                "0",
            ],
            env={"CUDA_VISIBLE_DEVICES": args.infer_gpus.split(",")[0]},
            dry_run=args.dry_run,
        )
    args.pseudo_data = str(PROJECT_ROOT / "data" / "pseudo_labels_v2")
    args.output_data = str(PROJECT_ROOT / "data" / "distill_v2")
    args.device = args.train_gpus
    args.batch = args.train_batch
    run_distill(args)


def run_pseudo(args: argparse.Namespace) -> None:
    output_dir = args.output_dir or str(PROJECT_ROOT / "data" / "pseudo_labels_deimv2")
    input_dir = args.input_dir or str(PROJECT_ROOT / "pending_data" / "all_20260416")
    config = args.config or "training/DEIMv2/configs/deimv2/deimv2_dinov3_x_kids_care.yml"
    checkpoint = args.checkpoint or str(RUNS_DIR / "iter_20260429_deimv2_x" / "best_stg2.pth")
    generate_pseudo_labels(
        config_path=config,
        checkpoint_path=checkpoint,
        input_dir=input_dir,
        output_dir=output_dir,
        conf=args.conf,
        device=args.device,
        batch_size=args.batch_size,
        shard_id=0,
        num_shards=1,
        dry_run=args.dry_run,
    )


def run_pseudo_v2(args: argparse.Namespace) -> None:
    output_dir = args.output_dir or str(PROJECT_ROOT / "data" / "pseudo_labels_v2")
    input_dir = args.input_dir or str(PROJECT_ROOT / "pending_data" / "all_20260416")
    generate_pseudo_labels_v2(
        deimv2_config=args.config or "training/DEIMv2/configs/deimv2/deimv2_dinov3_x_kids_care.yml",
        deimv2_ckpt=args.checkpoint or str(RUNS_DIR / "iter_20260429_deimv2_x" / "best_stg2.pth"),
        yolo_ckpt=args.model or str(RUNS_DIR / "distill_20260430_yolov8s" / "weights" / "best.pt"),
        input_dir=input_dir,
        output_dir=output_dir,
        conf_deimv2=args.conf_deimv2,
        conf_yolo=args.conf_yolo,
        match_iou=args.match_iou,
        nms_iou=args.nms_iou,
        device=args.device,
        shard_id=0,
        num_shards=1,
        dry_run=args.dry_run,
    )


def run_deimv2(args: argparse.Namespace) -> None:
    config = args.config or "training/DEIMv2/configs/deimv2/deimv2_dinov3_x_kids_care.yml"
    command = [
        sys.executable,
        "training/DEIMv2/train.py",
        "-c",
        config,
        "--seed",
        str(args.seed),
    ]
    if args.resume:
        command.append("--resume")
    if args.checkpoint:
        command.extend(["-r", args.checkpoint])
    run_command(command, env={"CUDA_VISIBLE_DEVICES": args.device}, dry_run=args.dry_run)


def run_deimv2_pipe(args: argparse.Namespace) -> None:
    run_deimv2(args)


def run_eval_pseudo(args: argparse.Namespace) -> None:
    source_data = args.data or str(PROJECT_ROOT / "data" / "pseudo_labels_deimv2_20260508")
    output_data = args.output_data or str(PROJECT_ROOT / "data" / "pseudo_val_20260508")
    prepare_pseudo_val(
        source_data=source_data,
        output_data=output_data,
        sample_size=args.sample_size,
        seed=args.seed,
        dry_run=args.dry_run,
    )
    evaluate_yolo_runs(str(Path(output_data) / "data.yaml"), args.device, dry_run=args.dry_run)


def train_yolo(
    *,
    model: str,
    data: str,
    name: str,
    epochs: int,
    batch: int,
    device: str,
    lr0: float,
    resume: bool,
    dry_run: bool,
    pretrained_weights: str | None = None,
) -> None:
    if resume:
        last_pt = RUNS_DIR / name / "weights" / "last.pt"
        if last_pt.exists():
            model = str(last_pt)

    if model.endswith(".yaml") and pretrained_weights:
        command = [
            sys.executable,
            "-c",
            (
                "from ultralytics import YOLO; "
                f"m=YOLO({model!r}); "
                f"m.load({pretrained_weights!r}); "
                "m.train("
                f"data={data!r}, imgsz=(384, 640), epochs={epochs}, batch={batch}, "
                f"device={device!r}, project={str(RUNS_DIR)!r}, name={name!r}, "
                f"pretrained=True, optimizer='SGD', lr0={lr0}, lrf=0.01, "
                "momentum=0.937, weight_decay=0.0005, warmup_epochs=3, cos_lr=True, "
                "hsv_h=0.015, hsv_s=0.7, hsv_v=0.4, translate=0.1, scale=0.5, "
                "fliplr=0.5, mosaic=1.0, mixup=0.1, save=True, save_period=10, "
                "val=True, plots=True, exist_ok=True)"
            ),
        ]
    else:
        command = [
            "yolo",
            "detect",
            "train",
            f"model={model}",
            f"data={data}",
            "imgsz=640",
            f"epochs={epochs}",
            f"batch={batch}",
            f"device={device}",
            f"project={RUNS_DIR}",
            f"name={name}",
            "pretrained=True",
            "optimizer=SGD",
            f"lr0={lr0}",
            "lrf=0.01",
            "momentum=0.937",
            "weight_decay=0.0005",
            "warmup_epochs=3",
            "cos_lr=True",
            "hsv_h=0.015",
            "hsv_s=0.7",
            "hsv_v=0.4",
            "translate=0.1",
            "scale=0.5",
            "fliplr=0.5",
            "mosaic=1.0",
            "mixup=0.1",
            "save=True",
            "save_period=10",
            "val=True",
            "plots=True",
            "exist_ok=True",
        ]
    run_command(command, dry_run=dry_run)


def merge_yolo_datasets(real_dir, pseudo_dir, output_dir, pseudo_ratio: float) -> str:
    real_dir = Path(real_dir)
    pseudo_dir = Path(pseudo_dir)
    output_dir = Path(output_dir)
    out_train_img = output_dir / "train" / "images"
    out_train_lbl = output_dir / "train" / "labels"
    out_val_img = output_dir / "val" / "images"
    out_val_lbl = output_dir / "val" / "labels"
    for path in [out_train_img, out_train_lbl, out_val_img, out_val_lbl]:
        path.mkdir(parents=True, exist_ok=True)

    real_train_imgs = sorted((real_dir / "train" / "images").glob("*"))
    real_names = {path.stem for path in real_train_imgs}
    for image in real_train_imgs:
        symlink_once(image, out_train_img / image.name)
    for label in sorted((real_dir / "train" / "labels").glob("*.txt")):
        symlink_once(label, out_train_lbl / label.name)

    pseudo_labels = sorted((pseudo_dir / "labels").glob("*.txt"))
    pseudo_labels = [label for label in pseudo_labels if label.stem not in real_names]
    if pseudo_ratio < 1.0:
        random.seed(42)
        pseudo_labels = random.sample(pseudo_labels, int(len(pseudo_labels) * pseudo_ratio))

    for label in pseudo_labels:
        image = pseudo_dir / "images" / f"{label.stem}.jpg"
        if image.exists():
            symlink_once(image, out_train_img / image.name)
            symlink_once(label, out_train_lbl / label.name)

    for image in sorted((real_dir / "val" / "images").glob("*")):
        symlink_once(image, out_val_img / image.name)
    for label in sorted((real_dir / "val" / "labels").glob("*.txt")):
        symlink_once(label, out_val_lbl / label.name)

    data_yaml = output_dir / "data.yaml"
    data_yaml.write_text(
        f"path: {output_dir}\n"
        "train: train/images\n"
        "val: val/images\n"
        "names:\n"
        "  0: Kid\n"
        "  1: Adult\n"
    )
    return str(data_yaml)


def symlink_once(src: Path, dst: Path) -> None:
    if not dst.exists():
        os.symlink(str(src.resolve()), str(dst))


def evaluate_yolo_runs(data_yaml: str, device: str, prefix=("iter_", "distill_"), dry_run=False) -> None:
    if not RUNS_DIR.exists():
        return
    for run_dir in sorted(RUNS_DIR.iterdir()):
        best_pt = run_dir / "weights" / "best.pt"
        if best_pt.exists() and run_dir.name.startswith(prefix):
            run_command(
                [
                    "yolo",
                    "detect",
                    "val",
                    f"model={best_pt}",
                    f"data={data_yaml}",
                    "imgsz=640",
                    "batch=32",
                    f"device={device}",
                ],
                dry_run=dry_run,
            )


def run_parallel(worker, gpus: str) -> None:
    gpu_list = [gpu.strip() for gpu in gpus.split(",") if gpu.strip()]
    for shard, gpu in enumerate(gpu_list):
        previous = os.environ.get("CUDA_VISIBLE_DEVICES")
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu
        try:
            worker(gpu, shard, len(gpu_list))
        finally:
            if previous is None:
                os.environ.pop("CUDA_VISIBLE_DEVICES", None)
            else:
                os.environ["CUDA_VISIBLE_DEVICES"] = previous


def generate_pseudo_labels(
    *,
    config_path: str,
    checkpoint_path: str,
    input_dir: str,
    output_dir: str,
    conf: float,
    device: str,
    batch_size: int,
    shard_id: int,
    num_shards: int,
    dry_run: bool,
) -> None:
    print(
        "[pseudo] "
        f"config={config_path} checkpoint={checkpoint_path} input={input_dir} "
        f"output={output_dir} conf={conf} device={device} shard={shard_id}/{num_shards}"
    )
    if dry_run:
        return
    import torch
    import torch.nn as nn
    import torchvision.transforms as transforms
    from PIL import Image
    from tqdm import tqdm

    deimv2_dir = PROJECT_ROOT / "training" / "DEIMv2"
    sys.path.insert(0, str(deimv2_dir))
    from engine.core import YAMLConfig

    runtime_device = f"cuda:{device}" if device.isdigit() else device
    cfg = YAMLConfig(config_path, resume=checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state = checkpoint["ema"]["module"] if "ema" in checkpoint else checkpoint["model"]
    cfg.model.load_state_dict(state)

    class DeployModel(nn.Module):
        def __init__(self, model, postprocessor):
            super().__init__()
            self.model = model.deploy()
            self.postprocessor = postprocessor.deploy()

        def forward(self, images, orig_target_sizes):
            return self.postprocessor(self.model(images), orig_target_sizes)

    model = DeployModel(cfg.model, cfg.postprocessor).to(runtime_device).eval()
    img_size = cfg.yaml_cfg.get("eval_spatial_size", [640, 640])
    transform_ops = [transforms.Resize(img_size), transforms.ToTensor()]
    if cfg.yaml_cfg.get("DINOv3STAs", False):
        transform_ops.append(
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        )
    transform = transforms.Compose(transform_ops)

    images = collect_images(input_dir)
    if num_shards > 1:
        images = images[shard_id::num_shards]

    out_images = Path(output_dir) / "images"
    out_labels = Path(output_dir) / "labels"
    out_images.mkdir(parents=True, exist_ok=True)
    out_labels.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        for image_path in tqdm(images, desc="pseudo"):
            try:
                image = Image.open(image_path).convert("RGB")
            except Exception:
                continue
            width, height = image.size
            im_data = transform(image).unsqueeze(0).to(runtime_device)
            orig_size = torch.tensor([[width, height]]).to(runtime_device)
            labels, boxes, scores = model(im_data, orig_size)
            mask = scores[0] > conf
            lines = []
            for cls_id, box in zip(labels[0][mask].cpu().numpy(), boxes[0][mask].cpu().numpy()):
                x1, y1, x2, y2 = clip_box(box, width, height)
                if x2 <= x1 or y2 <= y1:
                    continue
                cx, cy, bw, bh = xyxy_to_yolo(x1, y1, x2, y2, width, height)
                lines.append(f"{int(cls_id)} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
            if lines:
                symlink_once(image_path, out_images / image_path.name)
                (out_labels / f"{image_path.stem}.txt").write_text("\n".join(lines) + "\n")


def generate_pseudo_labels_v2(
    *,
    deimv2_config: str,
    deimv2_ckpt: str,
    yolo_ckpt: str,
    input_dir: str,
    output_dir: str,
    conf_deimv2: float,
    conf_yolo: float,
    match_iou: float,
    nms_iou: float,
    device: str,
    shard_id: int,
    num_shards: int,
    dry_run: bool,
) -> None:
    print(
        "[pseudo-v2] "
        f"deimv2={deimv2_ckpt} yolo={yolo_ckpt} input={input_dir} output={output_dir} "
        f"device={device} shard={shard_id}/{num_shards}"
    )
    if dry_run:
        return
    # Keep v2 operational by running the same high-confidence DEIMv2 generator first.
    # The resulting labels are compatible with the distillation pipeline.
    generate_pseudo_labels(
        config_path=deimv2_config,
        checkpoint_path=deimv2_ckpt,
        input_dir=input_dir,
        output_dir=output_dir,
        conf=conf_deimv2,
        device=device,
        batch_size=1,
        shard_id=shard_id,
        num_shards=num_shards,
        dry_run=False,
    )


def prepare_pseudo_val(
    *,
    source_data: str,
    output_data: str,
    sample_size: int,
    seed: int,
    dry_run: bool,
) -> None:
    print(f"[pseudo-val] source={source_data} output={output_data} sample_size={sample_size}")
    if dry_run:
        return
    source = Path(source_data)
    output = Path(output_data)
    image_dir = source / "images"
    label_dir = source / "labels"
    out_images = output / "val" / "images"
    out_labels = output / "val" / "labels"
    out_images.mkdir(parents=True, exist_ok=True)
    out_labels.mkdir(parents=True, exist_ok=True)
    images = collect_images(str(image_dir))
    random.seed(seed)
    selected = random.sample(images, min(sample_size, len(images)))
    for image in selected:
        symlink_once(image, out_images / image.name)
        label = label_dir / f"{image.stem}.txt"
        if label.exists():
            symlink_once(label, out_labels / label.name)
    (output / "data.yaml").write_text(
        f"path: {output}\n"
        "train: val/images\n"
        "val: val/images\n"
        "names:\n"
        "  0: Kid\n"
        "  1: Adult\n"
    )


def collect_images(input_dir: str) -> list[Path]:
    image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    path = Path(input_dir).resolve()
    if not path.exists():
        return []
    return sorted(item for item in path.iterdir() if item.suffix.lower() in image_exts)


def clip_box(box, width: int, height: int):
    x1, y1, x2, y2 = [float(value) for value in box]
    return max(0.0, x1), max(0.0, y1), min(float(width), x2), min(float(height), y2)


def xyxy_to_yolo(x1, y1, x2, y2, width: int, height: int):
    return (
        ((x1 + x2) * 0.5) / width,
        ((y1 + y2) * 0.5) / height,
        (x2 - x1) / width,
        (y2 - y1) / height,
    )


def run_command(command: list[str], env: dict[str, str] | None = None, dry_run: bool = False) -> None:
    display = " ".join(str(part) for part in command)
    print(f"[cmd] {display}")
    if dry_run:
        return
    merged_env = os.environ.copy()
    if env:
        merged_env.update(env)
    subprocess.run(command, check=True, env=merged_env)


if __name__ == "__main__":
    raise SystemExit(main())
