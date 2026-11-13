"""Pre-flight verification script for Google Colab Pro/Pro+ GPU & MMRotate env."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

CONFIG_FILES = [
    "configs/baseline_rotated_faster_rcnn_r50_fpn_1x_dotav2.py",
    "configs/benchmark_oriented_faster_rcnn_convnext_s_pafpn_kld_2x_dotav2.py",
]


def recommend_batch_profile(gpu_name: str, vram_gb: float) -> dict[str, int | str]:
    """Recommend batch_size and grad accumulation for Colab GPU tier."""
    if vram_gb >= 38:
        return {"tier": "A100-40GB+", "batch_size": 4, "accumulative_counts": 2}
    if vram_gb >= 20:
        return {"tier": "L4/A10-24GB", "batch_size": 2, "accumulative_counts": 4}
    return {"tier": f"T4/LowVRAM ({gpu_name})", "batch_size": 1, "accumulative_counts": 8}


def check_packages() -> dict[str, str]:
    """Import core OpenMMLab + PyTorch packages and return version strings."""
    versions: dict[str, str] = {}
    for pkg in ("torch", "mmengine", "mmcv", "mmdet", "mmrotate"):
        mod = __import__(pkg)
        versions[pkg] = getattr(mod, "__version__", "unknown")
    return versions


def check_configs(project_root: Path) -> None:
    """Parse project configs using MMEngine to ensure _base_ paths resolve."""
    from mmengine.config import Config

    for rel in CONFIG_FILES:
        cfg_path = project_root / rel
        if not cfg_path.is_file():
            raise FileNotFoundError(f"Missing config: {cfg_path}")
        cfg = Config.fromfile(str(cfg_path))
        num_classes = cfg.model.roi_head.bbox_head.num_classes
        if num_classes != 18:
            raise ValueError(f"{rel}: expected num_classes=18, got {num_classes}")
        print(f"[config:ok] {rel} (model={cfg.model.type}, num_classes={num_classes})")


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify Colab GPU & MMRotate environment.")
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
    )
    parser.add_argument("--json-out", type=Path, default=None)
    args = parser.parse_args()

    versions = check_packages()
    print("[versions]", json.dumps(versions, indent=2))

    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU not available. Enable GPU runtime in Colab.")

    gpu_name = torch.cuda.get_device_name(0)
    vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    profile = recommend_batch_profile(gpu_name, vram_gb)

    disk_total, _, disk_free = shutil.disk_usage("/content" if Path("/content").exists() else ".")
    summary = {
        "gpu_name": gpu_name,
        "vram_gb": round(vram_gb, 2),
        "disk_free_gb": round(disk_free / (1024**3), 2),
        "disk_total_gb": round(disk_total / (1024**3), 2),
        "recommended_profile": profile,
        "versions": versions,
    }
    print("[hardware]", json.dumps(summary, indent=2))

    check_configs(args.project_root)

    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # noqa: BLE001
        print(f"[verify:FAILED] {exc}", file=sys.stderr)
        sys.exit(1)
