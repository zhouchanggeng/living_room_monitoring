"""
客厅宝宝图片爬虫工具

支持从 Bing / Baidu 图片搜索引擎批量爬取客厅场景下的宝宝图片，
用于补充 kids-care 训练数据集。

用法:
    python -m tools.crawl_baby_images --engine bing --keywords "客厅宝宝" --num 200 --output ./crawled_images
"""

import argparse
import hashlib
import logging
import os
import re
import time
import json
import urllib.parse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from bs4 import BeautifulSoup

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 通用常量 / 请求头
# ---------------------------------------------------------------------------
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
    "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
}

# 图片最小尺寸（字节），过滤掉过小的缩略图
MIN_IMAGE_SIZE = 5 * 1024  # 5 KB

# 默认搜索关键词组合
DEFAULT_KEYWORDS = [
    "客厅宝宝",
    "客厅婴儿爬行",
    "宝宝在客厅玩耍",
    "亚洲宝宝客厅",
    "中国宝宝客厅玩耍",
    "亚洲婴儿爬行客厅",
    "asian baby in living room",
    "asian toddler playing living room",
    "asian infant living room floor",
    "chinese baby living room",
]


# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------
def _md5(data: bytes) -> str:
    return hashlib.md5(data).hexdigest()


def _is_valid_image_url(url: str) -> bool:
    """简单判断 URL 是否指向图片资源"""
    lower = url.lower()
    if any(lower.endswith(ext) for ext in (".jpg", ".jpeg", ".png", ".webp", ".bmp")):
        return True
    if re.search(r"\.(jpg|jpeg|png|webp|bmp)", lower):
        return True
    return False


def _download_image(url: str, save_dir: Path, seen_hashes: set, timeout: int = 15) -> str | None:
    """
    下载单张图片并保存。
    返回保存路径，若跳过则返回 None。
    """
    try:
        resp = requests.get(url, headers=HEADERS, timeout=timeout, stream=True)
        resp.raise_for_status()

        content_type = resp.headers.get("Content-Type", "")
        if "image" not in content_type and not _is_valid_image_url(url):
            return None

        data = resp.content
        if len(data) < MIN_IMAGE_SIZE:
            return None

        # 去重
        h = _md5(data)
        if h in seen_hashes:
            return None
        seen_hashes.add(h)

        # 推断扩展名
        ext = ".jpg"
        if "png" in content_type:
            ext = ".png"
        elif "webp" in content_type:
            ext = ".webp"
        elif "bmp" in content_type:
            ext = ".bmp"

        save_path = save_dir / f"{h}{ext}"
        save_path.write_bytes(data)
        return str(save_path)
    except Exception as e:
        logger.debug("下载失败 %s: %s", url, e)
        return None


# ---------------------------------------------------------------------------
# Bing 图片搜索
# ---------------------------------------------------------------------------
class BingImageCrawler:
    """通过 Bing 图片搜索爬取图片 URL"""

    BASE_URL = "https://www.bing.com/images/search"

    def collect_urls(self, keyword: str, num: int = 100) -> list[str]:
        urls: list[str] = []
        first = 0
        while len(urls) < num:
            params = {
                "q": keyword,
                "form": "HDRSC2",
                "first": first,
                "count": 35,
                "qft": "+filterui:photo-photo",  # 仅照片
            }
            try:
                resp = requests.get(
                    self.BASE_URL, params=params, headers=HEADERS, timeout=20
                )
                resp.raise_for_status()
                soup = BeautifulSoup(resp.text, "html.parser")

                # 方法 1: 从 <a class="iusc"> 的 m 属性中提取
                found = 0
                for a_tag in soup.find_all("a", class_="iusc"):
                    m = a_tag.get("m")
                    if m:
                        try:
                            m_json = json.loads(m)
                            murl = m_json.get("murl")
                            if murl and murl not in urls:
                                urls.append(murl)
                                found += 1
                        except json.JSONDecodeError:
                            pass

                # 方法 2: 从 img 标签的 src/data-src 提取
                if found == 0:
                    for img in soup.find_all("img"):
                        for attr in ("src", "data-src", "src2"):
                            src = img.get(attr, "")
                            if src and src.startswith("http") and _is_valid_image_url(src):
                                if src not in urls:
                                    urls.append(src)
                                    found += 1

                if found == 0:
                    break  # 没有更多结果

                first += 35
                time.sleep(1)  # 避免被封
            except Exception as e:
                logger.warning("Bing 请求失败 (first=%d): %s", first, e)
                break

        return urls[:num]


# ---------------------------------------------------------------------------
# 百度图片搜索
# ---------------------------------------------------------------------------
class BaiduImageCrawler:
    """通过百度图片搜索爬取图片 URL"""

    BASE_URL = "https://image.baidu.com/search/acjson"

    def collect_urls(self, keyword: str, num: int = 100) -> list[str]:
        urls: list[str] = []
        pn = 0
        rn = 60
        while len(urls) < num:
            params = {
                "tn": "resultjson_com",
                "logid": "",
                "ipn": "rj",
                "ct": 201326592,
                "is": "",
                "fp": "result",
                "fr": "",
                "word": keyword,
                "queryWord": keyword,
                "cl": 2,
                "lm": -1,
                "ie": "utf-8",
                "oe": "utf-8",
                "adpicid": "",
                "st": -1,
                "z": "",
                "ic": 0,
                "hd": "",
                "latest": "",
                "copyright": "",
                "s": "",
                "se": "",
                "tab": "",
                "width": "",
                "height": "",
                "face": 0,
                "istype": 2,
                "qc": "",
                "nc": 1,
                "expermode": "",
                "nojc": "",
                "pn": pn,
                "rn": rn,
                "gsm": "78",
            }
            try:
                resp = requests.get(
                    self.BASE_URL, params=params, headers=HEADERS, timeout=20
                )
                resp.raise_for_status()
                data = resp.json()
                items = data.get("data", [])
                found = 0
                for item in items:
                    obj_url = item.get("objURL") or item.get("middleURL") or item.get("thumbURL")
                    if obj_url and obj_url.startswith("http") and obj_url not in urls:
                        urls.append(obj_url)
                        found += 1

                if found == 0:
                    break

                pn += rn
                time.sleep(1)
            except Exception as e:
                logger.warning("百度请求失败 (pn=%d): %s", pn, e)
                break

        return urls[:num]


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
ENGINES = {
    "bing": BingImageCrawler,
    "baidu": BaiduImageCrawler,
}


def crawl(
    keywords: list[str],
    engine_name: str = "bing",
    num_per_keyword: int = 100,
    output_dir: str = "./crawled_images",
    max_workers: int = 8,
) -> int:
    """
    执行爬取任务。

    Args:
        keywords: 搜索关键词列表
        engine_name: 搜索引擎名称 (bing / baidu)
        num_per_keyword: 每个关键词爬取的图片数量
        output_dir: 图片保存目录
        max_workers: 并发下载线程数

    Returns:
        成功下载的图片总数
    """
    save_dir = Path(output_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    engine_cls = ENGINES.get(engine_name.lower())
    if engine_cls is None:
        raise ValueError(f"不支持的搜索引擎: {engine_name}，可选: {list(ENGINES.keys())}")

    engine = engine_cls()
    seen_hashes: set = set()
    total_saved = 0

    # 1. 收集所有图片 URL
    all_urls: list[str] = []
    for kw in keywords:
        logger.info("正在收集关键词 [%s] 的图片 URL (目标 %d 张) ...", kw, num_per_keyword)
        urls = engine.collect_urls(kw, num_per_keyword)
        logger.info("  -> 获取到 %d 个 URL", len(urls))
        all_urls.extend(urls)

    # URL 去重
    all_urls = list(dict.fromkeys(all_urls))
    logger.info("去重后共 %d 个待下载 URL", len(all_urls))

    # 2. 多线程下载
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(_download_image, url, save_dir, seen_hashes): url
            for url in all_urls
        }
        for future in as_completed(futures):
            result = future.result()
            if result:
                total_saved += 1
                if total_saved % 20 == 0:
                    logger.info("已下载 %d 张图片 ...", total_saved)

    logger.info("爬取完成！共保存 %d 张图片到 %s", total_saved, save_dir)
    return total_saved


# ---------------------------------------------------------------------------
# CLI 入口
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="客厅宝宝图片爬虫 — 从搜索引擎批量爬取客厅场景宝宝图片",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "示例:\n"
            "  # 使用 Bing 爬取默认关键词，每个关键词 200 张\n"
            '  python -m tools.crawl_baby_images --engine bing --num 200\n\n'
            "  # 使用百度，自定义关键词\n"
            '  python -m tools.crawl_baby_images --engine baidu --keywords "客厅宝宝" "婴儿爬行" --num 100\n\n'
            "  # 指定输出目录和并发数\n"
            '  python -m tools.crawl_baby_images --output ./baby_images --workers 16\n'
        ),
    )
    parser.add_argument(
        "--engine",
        type=str,
        default="bing",
        choices=list(ENGINES.keys()),
        help="搜索引擎 (默认: bing)",
    )
    parser.add_argument(
        "--keywords",
        type=str,
        nargs="+",
        default=None,
        help="搜索关键词列表 (默认使用内置关键词组合)",
    )
    parser.add_argument(
        "--num",
        type=int,
        default=100,
        help="每个关键词爬取的图片数量 (默认: 100)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./crawled_images",
        help="图片保存目录 (默认: ./crawled_images)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="并发下载线程数 (默认: 8)",
    )
    args = parser.parse_args()

    keywords = args.keywords if args.keywords else DEFAULT_KEYWORDS
    crawl(
        keywords=keywords,
        engine_name=args.engine,
        num_per_keyword=args.num,
        output_dir=args.output,
        max_workers=args.workers,
    )


if __name__ == "__main__":
    main()
