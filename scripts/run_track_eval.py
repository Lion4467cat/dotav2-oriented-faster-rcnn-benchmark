"""Run MMRotate test.py on best checkpoint and compute extended OBB metrics."""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import time
from pathlib import Path

_MEM_RE = re.compile(r"memory:\s*([0-9]+)")


def find_best_checkpoint(work_dir: Path) -> Path:
    """Locate best_dota_mAP_epoch_*.pth (or newest epoch_*.pth) inside work_dir."""
    bests = sorted(work_dir.glob("best_dota_mAP_epoch_*.pth"))
    if bests:
        return bests[-1]
    epochs = sorted(
        work_dir.glob("epoch_*.pth"),
        key=lambda p: int(p.stem.split("_")[-1]) if p.stem.split("_")[-1].isdigit() else -1,
    )
    if epochs:
        return epochs[-1]
    raise FileNotFoundError(f"No checkpoint found in {work_dir}")


def extract_peak_vram_gb(work_dir: Path) -> float | None:
    """Parse max reported MMEngine MiB memory usage from training logs."""
    peak_mib = 0
    for log_path in work_dir.rglob("*.log"):
        for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
            m = _MEM_RE.search(line)
            if m:
                peak_mib = max(peak_mib, int(m.group(1)))
    return round(peak_mib / 1024.0, 2) if peak_mib > 0 else None


def count_model_params_m(config_path: Path) -> float | None:
    """Instantiate MMRotate config via mmengine to count parameters in millions."""
    try:
        from mmengine.config import Config
        from mmengine.registry import MODELS

        cfg = Config.fromfile(str(config_path))
        model = MODELS.build(cfg.model)
        total_params = sum(p.numel() for p in model.parameters())
        return round(total_params / 1e6, 2)
    except (ImportError, OSError, RuntimeError, ValueError, KeyError):
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Run test.py + extended OBB metric evaluation.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument(
        "--ann-dir",
        type=Path,
        default=Path("/content/data/split_ss_dotav2/val/annfiles"),
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--skip-test", action="store_true", help="Reuse existing preds.pkl")
    args = parser.parse_args()

    ckpt = args.checkpoint or find_best_checkpoint(args.work_dir)
    eval_dir = args.work_dir / "eval_test"
    eval_dir.mkdir(parents=True, exist_ok=True)
    preds_pkl = eval_dir / "preds.pkl"

    env = dict(os.environ)
    env.setdefault("MPLBACKEND", "Agg")

    fps: float | None = None
    if not args.skip_test or not preds_pkl.is_file():
        cmd = [
            sys.executable,
            "mmrotate/tools/test.py",
            str(args.config),
            str(ckpt),
            "--work-dir",
            str(eval_dir),
            "--out",
            str(preds_pkl),
            "--cfg-options",
            f"test_dataloader.batch_size={args.batch_size}",
        ]
        print(f"[run-eval] Running: {' '.join(cmd)}")
        t0 = time.perf_counter()
        subprocess.run(cmd, check=True, env=env)
        elapsed = max(time.perf_counter() - t0, 1e-6)
        n_imgs = len(list(args.ann_dir.glob("*.txt"))) if args.ann_dir.is_dir() else 0
        if n_imgs > 0:
            fps = n_imgs / elapsed

    params_m = count_model_params_m(args.config)
    peak_vram_gb = extract_peak_vram_gb(args.work_dir)

    ext_cmd = [
        sys.executable,
        "scripts/compute_extended_metrics.py",
        "--preds-pkl",
        str(preds_pkl),
        "--ann-dir",
        str(args.ann_dir),
        "--out-json",
        str(eval_dir / "extended_metrics.json"),
    ]
    if fps is not None:
        ext_cmd += ["--fps", f"{fps:.2f}"]
    if params_m is not None:
        ext_cmd += ["--params-m", f"{params_m:.2f}"]
    if peak_vram_gb is not None:
        ext_cmd += ["--peak-vram-gb", f"{peak_vram_gb:.2f}"]

    subprocess.run(ext_cmd, check=True, env=env)


if __name__ == "__main__":
    main()
