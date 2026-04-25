"""
从指定目录随机抽取 n 张图片及其对应 JSON 标注, 打包为 tar.gz.

用法:
    python training/sample_pack.py --source all_20260416 --num 100
    python training/sample_pack.py --source all_20260416 --num 500 --output sample_500.tar.gz
"""

import argparse
import random
import tarfile
from pathlib import Path

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args():
    p = argparse.ArgumentParser(description="随机抽样图片+JSON并打包")
    p.add_argument("--source", type=str, default="all_20260416", help="图片目录")
    p.add_argument("--num", "-n", type=int, required=True, help="抽取数量")
    p.add_argument("--output", "-o", type=str, default=None,
                   help="输出文件名, 默认 sample_<n>.tar.gz")
    p.add_argument("--seed", type=int, default=None, help="随机种子(可复现)")
    return p.parse_args()


def main():
    args = parse_args()
    source = Path(args.source)

    if not source.is_dir():
        raise FileNotFoundError(f"目录不存在: {source}")

    # 找出同时有 JSON 的图片
    img_files = []
    for f in source.iterdir():
        if f.suffix.lower() in IMG_EXTS and f.with_suffix(".json").exists():
            img_files.append(f)

    print(f"[信息] 目录中有标注的图片: {len(img_files)} 张")

    n = min(args.num, len(img_files))
    if n < args.num:
        print(f"[警告] 可用数量不足, 实际抽取 {n} 张")

    if args.seed is not None:
        random.seed(args.seed)
    sampled = random.sample(img_files, n)

    output = args.output or f"sample_{n}.tar.gz"
    with tarfile.open(output, "w:gz") as tar:
        for img_path in sampled:
            json_path = img_path.with_suffix(".json")
            tar.add(str(img_path), arcname=img_path.name)
            tar.add(str(json_path), arcname=json_path.name)

    print(f"[完成] 已打包 {n} 张图片 + JSON -> {output}")


if __name__ == "__main__":
    main()
