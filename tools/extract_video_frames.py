"""
批量解压 pending_data/video 下的 zip, 并对解压出的视频按 1 fps 抽帧.

用法:
    # 默认: 解压 + 抽帧 (1fps)
    python tools/extract_video_frames.py

    # 自定义路径与采样帧率
    python tools/extract_video_frames.py \
        --zip-dir pending_data/video \
        --extract-dir pending_data/video/extracted \
        --frames-dir pending_data/video/frames \
        --fps 1 --workers 8

    # 仅解压, 不抽帧
    python tools/extract_video_frames.py --unzip-only

    # 跳过解压, 只对已解压的视频抽帧
    python tools/extract_video_frames.py --skip-unzip
"""

import argparse
import multiprocessing as mp
import zipfile
from functools import partial
from pathlib import Path

import cv2
from tqdm import tqdm

VIDEO_EXTS = {".mp4", ".avi", ".mkv", ".mov", ".flv", ".wmv", ".ts", ".m4v"}


def parse_args():
    p = argparse.ArgumentParser(description="解压视频 zip 并按指定 fps 抽帧")
    p.add_argument("--zip-dir", type=str,
                   default="pending_data/video",
                   help="存放 zip 文件的目录")
    p.add_argument("--extract-dir", type=str,
                   default="pending_data/video/extracted",
                   help="zip 解压输出目录")
    p.add_argument("--frames-dir", type=str,
                   default="pending_data/video/frames",
                   help="抽帧图片输出目录")
    p.add_argument("--fps", type=float, default=1.0,
                   help="抽帧采样率 (每秒抽几帧, 默认 1)")
    p.add_argument("--workers", type=int, default=max(1, mp.cpu_count() // 2),
                   help="并行进程数 (默认 CPU/2)")
    p.add_argument("--jpg-quality", type=int, default=90,
                   help="JPG 压缩质量 1-100 (默认 90)")
    p.add_argument("--unzip-only", action="store_true", help="仅解压, 不抽帧")
    p.add_argument("--skip-unzip", action="store_true", help="跳过解压, 只抽帧")
    p.add_argument("--overwrite", action="store_true",
                   help="覆盖已存在的帧输出 (默认跳过已完成的视频)")
    return p.parse_args()


# ---------------- 解压 ----------------

def unzip_one(zip_path: Path, extract_dir: Path):
    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(extract_dir)
        return zip_path.name, True, ""
    except Exception as e:
        return zip_path.name, False, str(e)


def do_unzip(zip_dir: Path, extract_dir: Path, workers: int):
    zip_files = sorted(zip_dir.glob("*.zip"))
    if not zip_files:
        print(f"[解压] 未在 {zip_dir} 找到 zip 文件")
        return
    extract_dir.mkdir(parents=True, exist_ok=True)
    print(f"[解压] 共 {len(zip_files)} 个 zip -> {extract_dir}")

    func = partial(unzip_one, extract_dir=extract_dir)
    with mp.Pool(processes=workers) as pool:
        fail = 0
        for name, ok, err in tqdm(pool.imap_unordered(func, zip_files),
                                  total=len(zip_files), desc="unzip"):
            if not ok:
                fail += 1
                tqdm.write(f"  [失败] {name}: {err}")
    print(f"[解压] 完成, 失败 {fail} 个")


# ---------------- 抽帧 ----------------

def extract_frames_one(video_path: Path, frames_root: Path, extract_root: Path,
                       fps: float, jpg_quality: int, overwrite: bool):
    """对单个视频按 fps 抽帧, 输出到 frames_root/<相对目录>/<视频名>_fXXXXXX.jpg"""
    try:
        rel = video_path.relative_to(extract_root).parent
    except ValueError:
        rel = Path(video_path.parent.name)
    out_dir = frames_root / rel
    out_dir.mkdir(parents=True, exist_ok=True)

    stem = video_path.stem
    # 若已存在同前缀的帧且不覆盖, 则跳过
    if not overwrite and any(out_dir.glob(f"{stem}_f*.jpg")):
        return video_path.name, 0, True, "skip"

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return video_path.name, 0, False, "cannot open"

    src_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    if src_fps <= 0:
        src_fps = 25.0
    interval = max(1, int(round(src_fps / max(fps, 1e-6))))

    saved = 0
    frame_idx = 0
    encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), int(jpg_quality)]
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % interval == 0:
            out_path = out_dir / f"{stem}_f{saved:06d}.jpg"
            cv2.imwrite(str(out_path), frame, encode_params)
            saved += 1
        frame_idx += 1
    cap.release()
    return video_path.name, saved, True, ""


def do_extract(extract_dir: Path, frames_dir: Path, fps: float,
               workers: int, jpg_quality: int, overwrite: bool):
    videos = [p for p in extract_dir.rglob("*") if p.suffix.lower() in VIDEO_EXTS]
    if not videos:
        print(f"[抽帧] 未在 {extract_dir} 找到视频")
        return
    frames_dir.mkdir(parents=True, exist_ok=True)
    print(f"[抽帧] 共 {len(videos)} 个视频, fps={fps} -> {frames_dir}")

    func = partial(extract_frames_one,
                   frames_root=frames_dir, extract_root=extract_dir,
                   fps=fps, jpg_quality=jpg_quality, overwrite=overwrite)
    total_frames = 0
    fail = 0
    skipped = 0
    with mp.Pool(processes=workers) as pool:
        for name, saved, ok, info in tqdm(pool.imap_unordered(func, videos),
                                          total=len(videos), desc="extract"):
            if not ok:
                fail += 1
                tqdm.write(f"  [失败] {name}: {info}")
            else:
                if info == "skip":
                    skipped += 1
                total_frames += saved
    print(f"[抽帧] 完成, 共输出 {total_frames} 帧, 跳过 {skipped} 个, 失败 {fail} 个")


# ---------------- main ----------------

def main():
    args = parse_args()
    zip_dir = Path(args.zip_dir).resolve()
    extract_dir = Path(args.extract_dir).resolve()
    frames_dir = Path(args.frames_dir).resolve()

    if not args.skip_unzip:
        do_unzip(zip_dir, extract_dir, args.workers)

    if not args.unzip_only:
        do_extract(extract_dir, frames_dir, args.fps,
                   args.workers, args.jpg_quality, args.overwrite)


if __name__ == "__main__":
    main()
