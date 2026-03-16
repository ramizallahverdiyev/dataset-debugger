"""
download.py — Download and extract Tiny ImageNet dataset.
"""

import os
import zipfile
import requests
from pathlib import Path
from tqdm import tqdm

TINY_IMAGENET_URL = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"
DEFAULT_DATA_DIR = Path("data/raw")


def download_tiny_imagenet(data_dir: Path = DEFAULT_DATA_DIR, force: bool = False) -> Path:
    """Download and extract Tiny ImageNet dataset."""
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    zip_path = data_dir / "tiny-imagenet-200.zip"
    extract_path = data_dir / "tiny-imagenet-200"

    if extract_path.exists() and not force:
        print(f"[✓] Tiny ImageNet already exists at {extract_path}")
        return extract_path

    print(f"[↓] Downloading Tiny ImageNet ...")
    response = requests.get(TINY_IMAGENET_URL, stream=True)
    total = int(response.headers.get("content-length", 0))

    with open(zip_path, "wb") as f, tqdm(total=total, unit="B", unit_scale=True) as bar:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
            bar.update(len(chunk))

    print(f"[↗] Extracting to {data_dir} ...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(data_dir)

    zip_path.unlink()
    print(f"[✓] Tiny ImageNet ready at {extract_path}")
    return extract_path


def get_class_labels(data_dir: Path = DEFAULT_DATA_DIR) -> dict:
    """Parse wnids.txt → {class_id: class_index}."""
    wnids_path = Path(data_dir) / "tiny-imagenet-200" / "wnids.txt"
    with open(wnids_path) as f:
        wnids = [line.strip() for line in f.readlines()]
    return {wnid: idx for idx, wnid in enumerate(sorted(wnids))}


def get_val_annotations(data_dir: Path = DEFAULT_DATA_DIR) -> dict:
    """Parse val/val_annotations.txt → {filename: class_id}."""
    val_ann_path = Path(data_dir) / "tiny-imagenet-200" / "val" / "val_annotations.txt"
    annotations = {}
    with open(val_ann_path) as f:
        for line in f:
            parts = line.strip().split("\t")
            filename, class_id = parts[0], parts[1]
            annotations[filename] = class_id
    return annotations


if __name__ == "__main__":
    download_tiny_imagenet()
