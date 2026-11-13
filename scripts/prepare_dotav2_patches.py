"""Tile DOTA v2.0 high-res aerial imagery into 1024x1024 patches (200px overlap).

Uses MMRotate's img_split tool (BBoxToolkit) on Colab NVMe storage for fast
DataLoader throughput, and supports --smoke-test-limit for rapid end-to-end
pipeline verification before running full DOTA v2.0 training.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path


def build_split_config(
    img_dirs: list[str],
    ann_dirs: list[str],
    save_dir: Path,
    patch_size: int,
    overlap: int,
    nproc: int,
) -> Path:
    """Write a temporary JSON config compatible with MMRotate's img_split.py."""
    cfg = {
        "nproc": nproc,
        "img_dirs": img_dirs,
        "ann_dirs": ann_dirs,
        "sizes": [patch_size],
        "gaps": [overlap],
        "rates": [1.0],
        "img_rate_thr": 0.6,
        "iof_thr": 0.7,
        "no_padding": False,
        "padding_value": [104, 116, 124],
        "save_dir": str(save_dir),
        "save_ext": ".png",
    }
    cfg_path = save_dir.parent / f"_split_cfg_{save_dir.name}.json"
    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    cfg_path.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    return cfg_path


def create_subset_view(src_split: Path, dst_split: Path, limit: int) -> Path:
    """Create a symlinked subset of images & labelTxt for rapid smoke tests."""
    if dst_split.exists():
        shutil.rmtree(dst_split)
    (dst_split / "images").mkdir(parents=True, exist_ok=True)
    (dst_split / "labelTxt").mkdir(parents=True, exist_ok=True)

    imgs = sorted((src_split / "images").glob("*.*"))[:limit]
    for img_path in imgs:
        (dst_split / "images" / img_path.name).symlink_to(img_path.resolve())
        lbl_path = src_split / "labelTxt" / f"{img_path.stem}.txt"
        if lbl_path.exists():
            (dst_split / "labelTxt" / lbl_path.name).symlink_to(lbl_path.resolve())
    return dst_split


def run_img_split(mmrotate_dir: Path, split_cfg: Path) -> None:
    """Invoke MMRotate's img_split.py with the generated JSON config."""
    img_split_py = mmrotate_dir / "tools" / "data" / "dota" / "split" / "img_split.py"
    if not img_split_py.is_file():
        raise FileNotFoundError(
            f"MMRotate img_split.py not found at {img_split_py}. "
            "Verify mmrotate (dev-1.x) is cloned correctly."
        )
    cmd = [sys.executable, str(img_split_py), "--base-json", str(split_cfg)]
    print(f"[tile] Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tile DOTA v2.0 into 1024x1024 patches.")
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=Path("/content/data/dotav2_raw"),
        help="Raw DOTA v2.0 directory containing train/ and val/.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("/content/data/split_ss_dotav2"),
        help="Output directory for patched train/ and val/ splits.",
    )
    parser.add_argument(
        "--mmrotate-dir",
        type=Path,
        default=Path("mmrotate"),
        help="Path to cloned mmrotate (dev-1.x) repository.",
    )
    parser.add_argument("--patch-size", type=int, default=1024)
    parser.add_argument("--overlap", type=int, default=200)
    parser.add_argument("--nproc", type=int, default=8)
    parser.add_argument(
        "--smoke-test-limit",
        type=int,
        default=0,
        help="If > 0, only tile this many images per split for smoke testing.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    for split in ("train", "val"):
        split_src = args.raw_dir / split
        if not (split_src / "images").is_dir():
            raise FileNotFoundError(f"Missing {split_src / 'images'}")

        if args.smoke_test_limit > 0:
            split_src = create_subset_view(
                split_src,
                args.out_dir / "_smoke_raw" / split,
                args.smoke_test_limit,
            )

        save_dir = args.out_dir / split
        if save_dir.exists():
            shutil.rmtree(save_dir)
        cfg_path = build_split_config(
            img_dirs=[str(split_src / "images")],
            ann_dirs=[str(split_src / "labelTxt")],
            save_dir=save_dir,
            patch_size=args.patch_size,
            overlap=args.overlap,
            nproc=args.nproc,
        )
        run_img_split(args.mmrotate_dir, cfg_path)

        n_patches = len(list((save_dir / "images").glob("*.png")))
        n_anns = len(list((save_dir / "annfiles").glob("*.txt")))
        print(f"[tile:{split}] patches={n_patches}, annfiles={n_anns} -> {save_dir}")


if __name__ == "__main__":
    main()
