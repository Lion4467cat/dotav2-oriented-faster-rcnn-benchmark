"""Download, cache, extract, and validate DOTA v2.0 train/val data for Colab.

Design:
- Keeps heavy raw archives in persistent Google Drive (--cache-dir) so you only
  download once across Colab sessions.
- Extracts images and labelTxt files onto fast local NVMe (--out-dir) for high
  DataLoader throughput.
- Validates and optionally normalizes space-separated class names (e.g.
  'small vehicle' -> 'small-vehicle') to match MMRotate DOTADataset.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import zipfile
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from configs.dota_v2_classes import DOTA_V2_CLASSES, DOTA_V2_SPACE_TO_HYPHEN

# Official CAPTAIN-WHU DOTA base image Google Drive folders (train & val).
OFFICIAL_GDRIVE_FOLDERS = {
    "train_images": "https://drive.google.com/drive/folders/1gmeE3D7R62UAtuIFOB9j2M5cUPTwtsxK",
    "val_images": "https://drive.google.com/drive/folders/1n5w45suVOyaqY84hltJhIZdtVFD9B224",
}


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
    """Extract all .zip archives in src_dir into target_dir."""
    target_dir.mkdir(parents=True, exist_ok=True)
    zips = sorted(src_dir.rglob("*.zip"))
    if not zips:
        print(f"[extract] No .zip files found under {src_dir}")
        return
    for zpath in zips:
        print(f"[extract] {zpath.name} -> {target_dir}")
        with zipfile.ZipFile(zpath, "r") as zf:
            zf.extractall(target_dir)


def _locate_split_dirs(root: Path, split: str) -> tuple[Path, Path]:
    """Find (images_dir, label_dir) for a split under extracted directories."""
    candidates = [
        (root / split / "images", root / split / "labelTxt"),
        (root / split / "images", root / split / "labelTxt-v2.0"),
        (root / split / "images", root / split / "DOTA-v2.0_train_hbb_obb" / "labelTxt"),
    ]
    for img_dir, lbl_dir in candidates:
        if img_dir.is_dir() and lbl_dir.is_dir():
            return img_dir, lbl_dir

    # Fallback recursive search by directory name.
    split_root = root / split
    if split_root.is_dir():
        img_dirs = [p for p in split_root.rglob("images") if p.is_dir()]
        lbl_dirs = [
            p for p in split_root.rglob("*") if p.is_dir() and "label" in p.name.lower()
        ]
        if img_dirs and lbl_dirs:
            return img_dirs[0], lbl_dirs[0]

    raise FileNotFoundError(
        f"Could not find images/ and labelTxt/ for split '{split}' under {root}. "
        "Ensure both DOTA images and DOTA-v2.0 labelTxt archives are extracted."
    )


def normalize_and_validate_labels(
    label_dir: Path, normalize_spaces: bool = True
) -> Counter[str]:
    """Validate DOTA v2.0 annotation files and return per-class instance counts."""
    counts: Counter[str] = Counter()
    unknown: Counter[str] = Counter()
    valid_set = set(DOTA_V2_CLASSES)
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
            if len(parts) < 10:
                updated_lines.append(line)
                continue
            coords = parts[:8]
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

        if changed and normalize_spaces:
            txt_path.write_text("\n".join(updated_lines) + "\n", encoding="utf-8")

    if unknown:
        print(f"[validate] WARNING: Unknown labels found in {label_dir}: {dict(unknown)}")
    return counts


def stage_split(
    extracted_root: Path, out_root: Path, split: str, normalize_spaces: bool
) -> None:
    """Copy/link split into canonical out_root/{split}/{images,labelTxt} layout."""
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

    n_imgs = len(list(img_dst.glob("*.*")))
    n_lbls = len(list(lbl_dst.glob("*.txt")))
    counts = normalize_and_validate_labels(lbl_dst, normalize_spaces=normalize_spaces)
    print(f"[stage:{split}] images={n_imgs}, label_files={n_lbls}, instances={sum(counts.values())}")
    missing = [c for c in DOTA_V2_CLASSES if counts[c] == 0]
    if missing:
        print(f"[stage:{split}] WARNING: classes with 0 instances: {missing}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download & stage DOTA v2.0 train/val.")
    parser.add_argument(
        "--source",
        choices=["local", "gdrive", "kaggle"],
        default="local",
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
    parser.add_argument(
        "--kaggle-dataset",
        type=str,
        default="",
        help="Kaggle dataset slug (used when --source kaggle).",
    )
    parser.add_argument(
        "--v2-train-ann",
        type=str,
        default="",
        help="Optional Google Drive file ID/URL for DOTA-v2.0 train labelTxt zip.",
    )
    parser.add_argument(
        "--v2-val-ann",
        type=str,
        default="",
        help="Optional Google Drive file ID/URL for DOTA-v2.0 val labelTxt zip.",
    )
    parser.add_argument(
        "--v2-extra-images",
        type=str,
        default="",
        help="Optional Google Drive file ID/URL for DOTA-v2.0 extra images zip.",
    )
    parser.add_argument(
        "--skip-extract",
        action="store_true",
        help="Only validate/stage already-extracted files under --out-dir.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cache_dir: Path = args.cache_dir
    out_dir: Path = args.out_dir

    if args.source == "gdrive":
        download_gdrive_folder(OFFICIAL_GDRIVE_FOLDERS["train_images"], cache_dir / "train")
        download_gdrive_folder(OFFICIAL_GDRIVE_FOLDERS["val_images"], cache_dir / "val")
        if args.v2_train_ann:
            download_gdrive_file(args.v2_train_ann, cache_dir / "train" / "dota_v2_train_ann.zip")
        if args.v2_val_ann:
            download_gdrive_file(args.v2_val_ann, cache_dir / "val" / "dota_v2_val_ann.zip")
        if args.v2_extra_images:
            download_gdrive_file(args.v2_extra_images, cache_dir / "train" / "dota_v2_extra_images.zip")
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
        extract_archives(cache_dir, extract_dir)
        for split in ("train", "val"):
            stage_split(extract_dir, out_dir, split, normalize_spaces=True)
    else:
        for split in ("train", "val"):
            stage_split(out_dir, out_dir, split, normalize_spaces=True)

    print(f"[done] DOTA v2.0 ready at {out_dir}")


if __name__ == "__main__":
    main()
