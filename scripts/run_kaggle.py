#!/usr/bin/env python3
"""Master automated execution script for Kaggle 2x Tesla T4 GPU benchmark run.

Completes Track B training from Epoch 17 to Epoch 24 with multi-GPU batching,
then runs extended evaluation and compiles final metrics.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import time
from pathlib import Path


def log(msg: str) -> None:
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {msg}", flush=True)


def run_cmd(cmd: list[str], cwd: Path | None = None, env: dict[str, str] | None = None, check: bool = True) -> int:
    cmd_str = " ".join(str(c) for c in cmd)
    log(f"Executing: {cmd_str}")
    p = subprocess.Popen(
        cmd,
        cwd=str(cwd) if cwd else None,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert p.stdout is not None
    for line in p.stdout:
        print(line, end="", flush=True)
    p.wait()
    if check and p.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {p.returncode}: {cmd_str}")
    return p.returncode


def main() -> None:
    log("=== Starting Kaggle DOTA v2.0 Track B Pipeline ===")

    # 1. Hardware & Environment Check
    log("Step 1/6: Verifying GPUs...")
    run_cmd(["nvidia-smi"], check=False)

    repo_dir = Path("/kaggle/working/ADA")
    venv_dir = Path("/tmp/mmrotate_venv")
    venv_py = venv_dir / "bin" / "python"
    torchrun_bin = venv_dir / "bin" / "torchrun"

    # Use /tmp/data (962 GB free) instead of /kaggle/working (20 GB limit)
    old_data_dir = Path("/kaggle/working/data")
    if old_data_dir.exists() and not old_data_dir.is_symlink():
        log("Removing leftover /kaggle/working/data to reclaim working disk...")
        shutil.rmtree(str(old_data_dir), ignore_errors=True)

    tmp_data_dir = Path("/tmp/data")
    tmp_data_dir.mkdir(parents=True, exist_ok=True)
    if not old_data_dir.exists():
        try:
            old_data_dir.symlink_to(tmp_data_dir)
            log("Created symlink /kaggle/working/data -> /tmp/data")
        except Exception as e:
            log(f"Notice: /kaggle/working/data symlink: {e}")

    # Ensure /content/data links directly to /tmp/data
    content_dir = Path("/content")
    content_dir.mkdir(parents=True, exist_ok=True)
    content_data = Path("/content/data")
    if content_data.is_symlink() or content_data.is_file():
        content_data.unlink()
    elif content_data.is_dir():
        shutil.rmtree(str(content_data), ignore_errors=True)
    try:
        content_data.symlink_to(tmp_data_dir)
        log("Created symlink /content/data -> /tmp/data")
    except Exception as e:
        log(f"Notice: /content/data symlink: {e}")

    # 2. Environment Provisioning
    if not venv_py.is_file():
        log("Step 2/6: Setting up Python 3.11 + OpenMMLab venv in /tmp/mmrotate_venv...")
        setup_script = repo_dir / "scripts" / "setup_colab.sh"
        run_cmd(["bash", str(setup_script), str(repo_dir), str(venv_dir)])
    else:
        log("Step 2/6: Python 3.11 venv already provisioned. Ensuring mmpretrain is installed...")
        run_cmd([str(venv_py), "-m", "pip", "install", "-q", "mmpretrain>=1.0.0rc8"])

    # 3. Checkpoint Staging (epoch_16.pth from Google Drive)
    log("Step 3/6: Staging Track B Epoch 16 checkpoint...")
    work_dir = Path("/kaggle/working/work_dirs/track_b_convnext_s_pafpn_kld_2x")
    work_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = work_dir / "epoch_16.pth"

    if not ckpt_path.is_file() or ckpt_path.stat().st_size < 100 * 1024 * 1024:
        log("Installing gdown and downloading epoch_16.pth (File ID: 12I5f7mNGUGJpMCIVKBLh1IekMhDkrpPs)...")
        run_cmd([str(venv_py), "-m", "pip", "install", "-q", "gdown"])
        gdown_bin = venv_dir / "bin" / "gdown"
        run_cmd([str(gdown_bin), "12I5f7mNGUGJpMCIVKBLh1IekMhDkrpPs", "-O", str(ckpt_path)])

    size_mb = ckpt_path.stat().st_size / (1024 * 1024)
    log(f"Checkpoint ready: {ckpt_path} ({size_mb:.1f} MB)")

    # Set last_checkpoint pointer for MMEngine
    last_ckpt_file = work_dir / "last_checkpoint"
    last_ckpt_file.write_text(str(ckpt_path), encoding="utf-8")

    # 4. Dataset Acquisition & Patch Tiling (in /tmp/data where 960+ GB is available)
    log("Step 4/6: Checking DOTA v2.0 patched dataset in /tmp/data...")
    split_dir = Path("/tmp/data/split_ss_dotav2")
    train_patch_imgs = split_dir / "train" / "images"
    val_patch_imgs = split_dir / "val" / "images"

    need_tiling = (
        not train_patch_imgs.is_dir()
        or len(list(train_patch_imgs.glob("*.png"))) == 0
        or not val_patch_imgs.is_dir()
        or len(list(val_patch_imgs.glob("*.png"))) == 0
    )

    if need_tiling:
        raw_dir = Path("/tmp/data/dotav2_raw")
        cache_dir = Path("/tmp/data/raw_archives")
        cache_dir.mkdir(parents=True, exist_ok=True)

        log("Downloading DOTA v2.0 via Ultralytics release...")
        run_cmd([
            str(venv_py),
            str(repo_dir / "scripts" / "download_dotav2.py"),
            "--source", "ultralytics",
            "--cache-dir", str(cache_dir),
            "--out-dir", str(raw_dir),
        ])

        # Remove raw zip archive immediately after extraction to free 2 GB
        raw_zip = cache_dir / "DOTAv2.zip"
        if raw_zip.exists():
            raw_zip.unlink()
            log("Removed raw DOTAv2.zip to conserve disk space.")

        log("Tiling 1024x1024 patches with 200px overlap...")
        run_cmd([
            str(venv_py),
            str(repo_dir / "scripts" / "prepare_dotav2_patches.py"),
            "--raw-dir", str(raw_dir),
            "--out-dir", str(split_dir),
            "--mmrotate-dir", str(repo_dir / "mmrotate"),
            "--patch-size", "1024",
            "--overlap", "200",
            "--nproc", "4",
        ])
    else:
        n_train = len(list(train_patch_imgs.glob("*.png")))
        n_val = len(list(val_patch_imgs.glob("*.png")))
        log(f"Patched dataset ready: {n_train} train patches, {n_val} val patches.")

    # 5. Distributed Multi-GPU Training (Epochs 17 to 24)
    log("Step 5/6: Launching Track B 2x T4 multi-GPU distributed training (Epochs 17 -> 24)...")
    config_file = repo_dir / "configs" / "benchmark_oriented_faster_rcnn_convnext_s_pafpn_kld_2x_dotav2.py"

    env = dict(os.environ)
    env["MPLBACKEND"] = "Agg"
    env["PYTHONPATH"] = f"{repo_dir}:{repo_dir / 'mmrotate'}:{env.get('PYTHONPATH', '')}"

    train_cmd = [
        str(torchrun_bin),
        "--nproc_per_node=2",
        str(repo_dir / "mmrotate" / "tools" / "train.py"),
        str(config_file),
        "--work-dir", str(work_dir),
        "--resume",
        "--launcher", "pytorch",
        "--cfg-options",
        "train_dataloader.batch_size=2",
        "train_dataloader.num_workers=2",
        "val_dataloader.batch_size=2",
        "val_dataloader.num_workers=2",
        "optim_wrapper.accumulative_counts=2",
        "train_cfg.val_interval=4",
        f"data_root={split_dir}/",
        f"raw_train_dataset.data_root={split_dir}/",
        f"train_dataloader.dataset.dataset.data_root={split_dir}/",
        f"val_dataloader.dataset.data_root={split_dir}/",
        f"test_dataloader.dataset.data_root={split_dir}/",
        "resume=True",
        f"load_from={ckpt_path}",
    ]

    t0 = time.time()
    run_cmd(train_cmd, cwd=repo_dir, env=env)
    elapsed_min = (time.time() - t0) / 60.0
    log(f"Training completed in {elapsed_min:.1f} minutes!")

    # 6. Final Evaluation & Report Generation
    log("Step 6/6: Running extended evaluation and compiling final benchmark report...")
    eval_cmd = [
        str(venv_py),
        str(repo_dir / "scripts" / "run_track_eval.py"),
        "--config", str(config_file),
        "--work-dir", str(work_dir),
        "--ann-dir", str(split_dir / "val" / "annfiles"),
        "--batch-size", "2",
    ]
    run_cmd(eval_cmd, cwd=repo_dir, env=env)

    report_out = work_dir / "final_benchmark_report.md"
    report_cmd = [
        str(venv_py),
        str(repo_dir / "scripts" / "evaluate_benchmark.py"),
        "--flagship-dir", str(work_dir),
        "--out", str(report_out),
    ]
    run_cmd(report_cmd, cwd=repo_dir, env=env)

    log("=" * 60)
    log("FINAL BENCHMARK REPORT COMPLETED:")
    log("=" * 60)
    if report_out.exists():
        print(report_out.read_text(encoding="utf-8"))
    log("=== All steps completed successfully! ===")


if __name__ == "__main__":
    main()
