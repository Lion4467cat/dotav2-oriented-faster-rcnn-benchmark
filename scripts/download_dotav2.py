"""Download, cache, extract, and validate DOTA v2.0 train/val data for Colab.

Design:
- Keeps heavy raw archives in persistent Google Drive (--cache-dir) so you only
  download once across Colab sessions.
- Extracts images and labelTxt files onto fast local NVMe (--out-dir) for high
  DataLoader throughput.
- Supports --source ultralytics (direct GitHub Release DOTAv2.zip), gdrive,
  kaggle, or local.
- Auto-detects normalized YOLO-OBB label format (class_idx x1 y1 ... x4 y4) and
  converts it to canonical pixel-space DOTA labelTxt for MMRotate DOTADataset.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tarfile
import zipfile
from collections import Counter
from pathlib import Path

from PIL import Image
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from configs.dota_v2_classes import DOTA_V2_CLASSES, DOTA_V2_SPACE_TO_HYPHEN

OFFICIAL_GDRIVE_FOLDERS = {
    "train_images": "https://drive.google.com/drive/folders/1gmeE3D7R62UAtuIFOB9j2M5cUPTwtsxK",
    "val_images": "https://drive.google.com/drive/folders/1n5w45suVOyaqY84hltJhIZdtVFD9B224",
}

ULTRALYTICS_DOTAV2_URL = (
    "https://github.com/ultralytics/yolov5/releases/download/v1.0/DOTAv2.zip"
)

HUGGINGFACE_DOTAV2_URLS = {
    "train_images": "https://huggingface.co/datasets/isaaccorley/dota/resolve/main/dotav2.0_images_train.tar.gz",
    "val_images": "https://huggingface.co/datasets/isaaccorley/dota/resolve/main/dotav2.0_images_val.tar.gz",
    "train_ann": "https://huggingface.co/datasets/isaaccorley/dota/resolve/main/dotav2.0_annotations_train.tar.gz",
    "val_ann": "https://huggingface.co/datasets/isaaccorley/dota/resolve/main/dotav2.0_annotations_val.tar.gz",
}

# Ultralytics dota.yaml class ID mapping (0..17 -> canonical hyphenated names).
ULTRALYTICS_ID_TO_CLASS = {
    0: "plane",
    1: "ship",
    2: "storage-tank",
    3: "baseball-diamond",
    4: "tennis-court",
    5: "basketball-court",
    6: "ground-track-field",
    7: "harbor",
    8: "bridge",
    9: "large-vehicle",
    10: "small-vehicle",
    11: "helicopter",
    12: "roundabout",
    13: "soccer-ball-field",
    14: "swimming-pool",
    15: "container-crane",
    16: "airport",
    17: "helipad",
}


def download_url(url: str, dest_file: Path) -> None:
    """Download a direct HTTP/HTTPS archive with streaming progress if not cached."""
    if dest_file.is_file() and dest_file.stat().st_size > 1024 * 1024:
        print(f"[download] Using cached archive: {dest_file}")
        return
    import requests

    dest_file.parent.mkdir(parents=True, exist_ok=True)
    tmp_file = dest_file.with_suffix(".tmp")
    print(f"[download] Streaming {url} -> {dest_file}")
    with requests.get(url, stream=True, timeout=120) as resp:
        resp.raise_for_status()
        total = int(resp.headers.get("content-length", 0))
        with (
            open(tmp_file, "wb") as fh,
            tqdm(total=total, unit="B", unit_scale=True, desc=dest_file.name) as pbar,
        ):
            for chunk in resp.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    fh.write(chunk)
                    pbar.update(len(chunk))
    tmp_file.replace(dest_file)


def download_gdrive_folder(url: str, dest: Path) -> None:
    """Download a public Google Drive folder using gdown."""
    import gdown

    dest.mkdir(parents=True, exist_ok=True)
    print(f"[download] Fetching folder {url} -> {dest}")
    gdown.download_folder(url=url, output=str(dest), quiet=False, use_cookies=False)


def download_gdrive_file(file_id_or_url: str, dest_file: Path) -> None:
    """Download a single Google Drive file by ID or URL using gdown."""
    import gdown

    dest_file.parent.mkdir(parents=True, exist_ok=True)
    url = (
        file_id_or_url
        if file_id_or_url.startswith("http")
        else f"https://drive.google.com/uc?id={file_id_or_url}"
    )
    print(f"[download] Fetching file {url} -> {dest_file}")
    gdown.download(url=url, output=str(dest_file), quiet=False, fuzzy=True)


def download_kaggle_dataset(dataset_slug: str, dest: Path) -> None:
    """Download a Kaggle dataset archive using the kaggle CLI."""
    dest.mkdir(parents=True, exist_ok=True)
    cmd = ["kaggle", "datasets", "download", "-d", dataset_slug, "-p", str(dest)]
    print(f"[download] Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def extract_archives(src_dir: Path, target_dir: Path) -> None:
    """Extract all .zip and .tar.gz/.tgz/.tar archives in src_dir into target_dir."""
    target_dir.mkdir(parents=True, exist_ok=True)
    archives = sorted(
        [
            *src_dir.rglob("*.zip"),
            *src_dir.rglob("*.tar.gz"),
            *src_dir.rglob("*.tgz"),
            *src_dir.rglob("*.tar"),
        ]
    )
    if not archives:
        print(f"[extract] No archives found under {src_dir}")
        return
    for apath in archives:
        print(f"[extract] {apath.name} -> {target_dir}")
        if apath.suffix == ".zip":
            with zipfile.ZipFile(apath, "r") as zf:
                zf.extractall(target_dir)
        else:
            with tarfile.open(apath, "r:*") as tf:
                tf.extractall(target_dir)


def _locate_split_dirs(root: Path, split: str) -> tuple[Path, Path]:
    """Find (images_dir, label_dir) for a split under extracted directories."""
    candidates = [
        # Canonical DOTA layout
        (root / split / "images", root / split / "labelTxt"),
        (root / split / "images", root / split / "labelTxt-v2.0"),
        # Ultralytics DOTAv2.zip layout (images/{split}, labels/{split})
        (root / "images" / split, root / "labels" / f"{split}_original"),
        (root / "images" / split, root / "labels" / split),
        (root / "DOTAv2" / "images" / split, root / "DOTAv2" / "labels" / f"{split}_original"),
        (root / "DOTAv2" / "images" / split, root / "DOTAv2" / "labels" / split),
    ]
    for img_dir, lbl_dir in candidates:
        if img_dir.is_dir() and lbl_dir.is_dir():
            return img_dir, lbl_dir

    # Fallback recursive search.
    img_dirs = [p for p in root.rglob(split) if p.is_dir() and p.parent.name == "images"]
    lbl_dirs = [
        p
        for p in root.rglob(split)
        if p.is_dir() and p.parent.name in ("labels", "labelTxt")
    ]
    if img_dirs and lbl_dirs:
        return img_dirs[0], lbl_dirs[0]

    split_root = root / split
    if split_root.is_dir():
        img_dirs = [p for p in split_root.rglob("images") if p.is_dir()]
        lbl_dirs = [
            p for p in split_root.rglob("*") if p.is_dir() and "label" in p.name.lower()
        ]
        if not img_dirs and any(
            p.suffix.lower() in {".png", ".jpg", ".jpeg"} for p in split_root.rglob("*")
        ):
            img_dirs = [
                p
                for p in [split_root, *split_root.rglob("*")]
                if p.is_dir()
                and any(
                    f.suffix.lower() in {".png", ".jpg", ".jpeg"}
                    for f in p.iterdir()
                    if f.is_file()
                )
            ]
        if not lbl_dirs and any(split_root.rglob("*.txt")):
            lbl_dirs = [
                p
                for p in [split_root, *split_root.rglob("*")]
                if p.is_dir() and any(f.suffix == ".txt" for f in p.iterdir() if f.is_file())
            ]
        if img_dirs and lbl_dirs:
            return img_dirs[0], lbl_dirs[0]

    raise FileNotFoundError(
        f"Could not find images/ and labels for split '{split}' under {root}."
    )


def _index_images(img_dir: Path) -> dict[str, Path]:
    return {
        p.stem: p
        for p in img_dir.iterdir()
        if p.is_file() and p.suffix.lower() in {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
    }


def _is_yolo_obb_line(parts: list[str]) -> bool:
    if len(parts) != 9 or not parts[0].isdigit():
        return False
    try:
        [float(x) for x in parts[1:]]
    except ValueError:
        return False
    return True


def _sanitize_polygon_coords(coords: list[float]) -> list[float] | None:
    """Repair self-intersecting polygons or drop zero-area degenerate boxes."""
    try:
        from shapely.geometry import Polygon
    except ImportError:
        return coords

    pts = [(coords[i], coords[i + 1]) for i in range(0, 8, 2)]
    poly = Polygon(pts)
    if not poly.is_valid:
        poly = poly.buffer(0)
    if poly.is_empty or poly.area < 1.0:
        return None
    if poly.geom_type == "MultiPolygon":
        poly = max(poly.geoms, key=lambda g: g.area)
    if len(poly.exterior.coords) == 5:
        flat: list[float] = []
        for x, y in list(poly.exterior.coords)[:4]:
            flat.extend([float(x), float(y)])
        return flat
    rect = poly.minimum_rotated_rectangle
    if rect.is_empty or rect.area < 1.0 or len(rect.exterior.coords) < 5:
        return None
    flat = []
    for x, y in list(rect.exterior.coords)[:4]:
        flat.extend([float(x), float(y)])
    return flat


def normalize_and_validate_labels(
    label_dir: Path, img_dir: Path | None = None, normalize_spaces: bool = True
) -> Counter[str]:
    """Validate/convert annotations to canonical DOTA labelTxt and count instances."""
    counts: Counter[str] = Counter()
    unknown: Counter[str] = Counter()
    valid_set = set(DOTA_V2_CLASSES)
    img_index = _index_images(img_dir) if img_dir else {}
    size_cache: dict[str, tuple[int, int]] = {}
    txt_files = sorted(label_dir.glob("*.txt"))

    for txt_path in txt_files:
        lines = txt_path.read_text(encoding="utf-8", errors="replace").splitlines()
        updated_lines: list[str] = []
        changed = False
        for line in lines:
            stripped = line.strip()
            if not stripped or stripped.startswith(("imagesource", "gsd")):
                updated_lines.append(line)
                continue
            parts = stripped.split()

            if _is_yolo_obb_line(parts):
                cls_id = int(parts[0])
                cls_name = ULTRALYTICS_ID_TO_CLASS.get(cls_id, f"unknown_{cls_id}")
                if txt_path.stem not in size_cache:
                    img_path = img_index.get(txt_path.stem)
                    if img_path is None:
                        raise FileNotFoundError(
                            f"Image for YOLO label {txt_path.name} not found in {img_dir}"
                        )
                    with Image.open(img_path) as im:
                        size_cache[txt_path.stem] = im.size
                w, h = size_cache[txt_path.stem]
                norm_coords = [float(x) for x in parts[1:]]
                raw_px = [
                    val * (w if idx % 2 == 0 else h) for idx, val in enumerate(norm_coords)
                ]
                sanitized = _sanitize_polygon_coords(raw_px)
                changed = True
                if sanitized is None:
                    continue
                px_coords = [f"{val:.1f}" for val in sanitized]
                if cls_name in valid_set:
                    counts[cls_name] += 1
                else:
                    unknown[cls_name] += 1
                updated_lines.append(" ".join([*px_coords, cls_name, "0"]))
                continue

            if len(parts) < 10:
                updated_lines.append(line)
                continue

            try:
                raw_px = [float(x) for x in parts[:8]]
            except ValueError:
                updated_lines.append(line)
                continue
            sanitized = _sanitize_polygon_coords(raw_px)
            if sanitized is None:
                changed = True
                continue
            if sanitized != raw_px:
                changed = True
            coords = [f"{val:.1f}" for val in sanitized]
            difficult = parts[-1]
            raw_cls = " ".join(parts[8:-1])
            cls_name = (
                DOTA_V2_SPACE_TO_HYPHEN.get(raw_cls, raw_cls)
                if normalize_spaces
                else raw_cls
            )
            if cls_name != raw_cls:
                changed = True
            if cls_name in valid_set:
                counts[cls_name] += 1
            else:
                unknown[cls_name] += 1
            updated_lines.append(" ".join([*coords, cls_name, difficult]))

        if changed:
            txt_path.write_text("\n".join(updated_lines) + "\n", encoding="utf-8")

    # Ensure every image has a corresponding .txt file for BBoxToolkit img_split.
    for stem in img_index:
        expected_txt = label_dir / f"{stem}.txt"
        if not expected_txt.exists():
            expected_txt.write_text("", encoding="utf-8")

    if unknown:
        print(f"[validate] WARNING: Unknown labels found in {label_dir}: {dict(unknown)}")
    return counts


def stage_split(
    extracted_root: Path, out_root: Path, split: str, normalize_spaces: bool
) -> None:
    """Copy split into canonical out_root/{split}/{images,labelTxt} layout."""
    img_src, lbl_src = _locate_split_dirs(extracted_root, split)
    img_dst = out_root / split / "images"
    lbl_dst = out_root / split / "labelTxt"
    img_dst.parent.mkdir(parents=True, exist_ok=True)

    if img_src.resolve() != img_dst.resolve():
        if img_dst.exists():
            shutil.rmtree(img_dst)
        shutil.copytree(img_src, img_dst)

    if lbl_src.resolve() != lbl_dst.resolve():
        if lbl_dst.exists():
            shutil.rmtree(lbl_dst)
        shutil.copytree(lbl_src, lbl_dst)

    n_imgs = len(_index_images(img_dst))
    n_lbls = len(list(lbl_dst.glob("*.txt")))
    counts = normalize_and_validate_labels(
        lbl_dst, img_dir=img_dst, normalize_spaces=normalize_spaces
    )
    print(f"[stage:{split}] images={n_imgs}, label_files={n_lbls}, instances={sum(counts.values())}")
    missing = [c for c in DOTA_V2_CLASSES if counts[c] == 0]
    if missing:
        print(f"[stage:{split}] WARNING: classes with 0 instances: {missing}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download & stage DOTA v2.0 train/val.")
    parser.add_argument(
        "--source",
        choices=["huggingface", "ultralytics", "local", "gdrive", "kaggle"],
        default="huggingface",
        help="Where to fetch raw archives if not already cached.",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("/content/drive/MyDrive/dotav2_benchmark/raw_archives"),
        help="Persistent directory holding raw .zip archives (e.g. Google Drive).",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("/content/data/dotav2_raw"),
        help="Fast local NVMe directory for extracted train/val images & labelTxt.",
    )
    parser.add_argument("--kaggle-dataset", type=str, default="")
    parser.add_argument("--v2-train-ann", type=str, default="")
    parser.add_argument("--v2-val-ann", type=str, default="")
    parser.add_argument("--v2-extra-images", type=str, default="")
    parser.add_argument("--skip-extract", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cache_dir: Path = args.cache_dir
    out_dir: Path = args.out_dir

    if args.source == "huggingface":
        download_url(
            HUGGINGFACE_DOTAV2_URLS["train_images"],
            cache_dir / "train" / "dotav2.0_images_train.tar.gz",
        )
        download_url(
            HUGGINGFACE_DOTAV2_URLS["train_ann"],
            cache_dir / "train" / "dotav2.0_annotations_train.tar.gz",
        )
        download_url(
            HUGGINGFACE_DOTAV2_URLS["val_images"],
            cache_dir / "val" / "dotav2.0_images_val.tar.gz",
        )
        download_url(
            HUGGINGFACE_DOTAV2_URLS["val_ann"],
            cache_dir / "val" / "dotav2.0_annotations_val.tar.gz",
        )
    elif args.source == "ultralytics":
        download_url(ULTRALYTICS_DOTAV2_URL, cache_dir / "DOTAv2.zip")
    elif args.source == "gdrive":
        download_gdrive_folder(OFFICIAL_GDRIVE_FOLDERS["train_images"], cache_dir / "train")
        download_gdrive_folder(OFFICIAL_GDRIVE_FOLDERS["val_images"], cache_dir / "val")
        if args.v2_train_ann:
            download_gdrive_file(args.v2_train_ann, cache_dir / "train" / "dota_v2_train_ann.zip")
        if args.v2_val_ann:
            download_gdrive_file(args.v2_val_ann, cache_dir / "val" / "dota_v2_val_ann.zip")
        if args.v2_extra_images:
            download_gdrive_file(
                args.v2_extra_images, cache_dir / "train" / "dota_v2_extra_images.zip"
            )
    elif args.source == "kaggle":
        if not args.kaggle_dataset:
            raise ValueError("--kaggle-dataset is required when --source kaggle")
        download_kaggle_dataset(args.kaggle_dataset, cache_dir)

    if not args.skip_extract:
        extract_dir = out_dir / "_extracted"
        for split in ("train", "val"):
            split_cache = cache_dir / split
            if split_cache.is_dir():
                extract_archives(split_cache, extract_dir / split)
        # Extract top-level zips (e.g. DOTAv2.zip) once.
        for zpath in sorted(cache_dir.glob("*.zip")):
            print(f"[extract] {zpath.name} -> {extract_dir}")
            with zipfile.ZipFile(zpath, "r") as zf:
                zf.extractall(extract_dir)
        for split in ("train", "val"):
            stage_split(extract_dir, out_dir, split, normalize_spaces=True)
    else:
        for split in ("train", "val"):
            stage_split(out_dir, out_dir, split, normalize_spaces=True)

    print(f"[done] DOTA v2.0 ready at {out_dir}")


if __name__ == "__main__":
    main()
