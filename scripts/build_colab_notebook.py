"""Generate turnkey colab_runner.ipynb using nbformat following ML best practices."""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf


def build_notebook(out_path: Path) -> None:
    nb = nbf.v4.new_notebook()
    nb.metadata["colab"] = {"provenance": [], "gpuType": "A100"}
    nb.metadata["accelerator"] = "GPU"

    cells = [
        nbf.v4.new_markdown_cell(
            "# DOTA v2.0 Oriented Faster R-CNN Benchmark (Google Colab Pro/Pro+)\n\n"
            "This notebook establishes a two-track **OBB (`le90`)** benchmark on **DOTA v2.0** (18 classes):\n"
            "- **Track A (Baseline)**: Rotated Faster R-CNN (`ResNet-50 + FPN`, `1x`, `SmoothL1Loss`)\n"
            "- **Track B (Flagship)**: Context-Aware Oriented Faster R-CNN (`ConvNeXt-S + PAFPN + Oriented RPN + KLD Loss + ClassBalancedDataset`, `2x`)\n\n"
            "Before running:\n"
            "1. Go to **Runtime -> Change runtime type** -> select **A100 GPU** (or **L4**) + **High-RAM**.\n"
            "2. Review **Step 2** (`DATA_SOURCE = 'ultralytics'` downloads `DOTAv2.zip` automatically and caches it on Drive).\n"
            "3. Click **Runtime -> Run all**.\n"
        ),
        nbf.v4.new_code_cell(
            "# Step 1: Mount Google Drive (persistent checkpoints, logs, and raw archive cache)\n"
            "from google.colab import drive\n"
            "drive.mount('/content/drive')\n\n"
            "!nvidia-smi\n"
        ),
        nbf.v4.new_markdown_cell(
            "**Analysis (Step 1):** Verify above that `nvidia-smi` reports an **A100 (40GB/80GB)** or **L4 (24GB)** GPU and that Google Drive is mounted at `/content/drive` for checkpoint persistence."
        ),
        nbf.v4.new_code_cell(
            "# Step 2: Configure Repo, Dataset Source & Run Mode\n"
            "GITHUB_REPO_URL = 'https://github.com/Lion4467cat/dotav2-oriented-faster-rcnn-benchmark.git'\n"
            "WORK_ROOT = '/content/drive/MyDrive/dotav2_benchmark'\n"
            "REPO_DIR = '/content/ADA'\n"
            "VENV_PY = '/content/mmrotate_venv/bin/python'\n\n"
            "# Dataset source: 'huggingface' (full 18-class DOTA-v2.0 1830/593), 'ultralytics', 'local', 'gdrive', or 'kaggle'\n"
            "DATA_SOURCE = 'huggingface'\n"
            "KAGGLE_DATASET_SLUG = ''  # Optional if DATA_SOURCE == 'kaggle'\n"
            "V2_TRAIN_ANN_ID = ''      # Optional if DATA_SOURCE == 'gdrive'\n"
            "V2_VAL_ANN_ID = ''        # Optional if DATA_SOURCE == 'gdrive'\n\n"
            "# Set SMOKE_TEST_LIMIT = 50 for a fast pipeline sanity check, or 0 for full DOTA v2.0\n"
            "SMOKE_TEST_LIMIT = 0\n"
            "RUN_BASELINE_TRACK_A = True\n"
            "RUN_FLAGSHIP_TRACK_B = True\n"
        ),
        nbf.v4.new_markdown_cell(
            "**Analysis (Step 2):** Confirm `DATA_SOURCE` (`'ultralytics'` caches `DOTAv2.zip` in `WORK_ROOT/raw_archives`), and use `SMOKE_TEST_LIMIT = 50` if running an initial pipeline check before full training."
        ),
        nbf.v4.new_code_cell(
            "# Step 3: Clone/Update Repo & Provision Python 3.11 + MMRotate (dev-1.x) via uv\n"
            "import os, subprocess\n"
            "os.makedirs(f'{WORK_ROOT}/raw_archives', exist_ok=True)\n"
            "os.makedirs(f'{WORK_ROOT}/work_dirs', exist_ok=True)\n\n"
            "if not os.path.exists(REPO_DIR):\n"
            "    subprocess.run(['git', 'clone', GITHUB_REPO_URL, REPO_DIR], check=True)\n"
            "else:\n"
            "    subprocess.run(['git', '-C', REPO_DIR, 'pull', '--ff-only'], check=True)\n\n"
            "%cd /content/ADA\n"
            "!bash scripts/setup_colab.sh /content/ADA /content/mmrotate_venv\n"
        ),
        nbf.v4.new_markdown_cell(
            "**Analysis (Step 3):** `setup_colab.sh` creates an isolated Python 3.11 venv via `uv` and installs prebuilt PyTorch 2.1.2 (`cu121`) + MMCV 2.1.0 + MMDet 3.x + MMRotate `dev-1.x`, avoiding Colab's system Python 3.13 incompatibility."
        ),
        nbf.v4.new_code_cell(
            "# Step 4: Download / Extract DOTA v2.0 & Tile 1024x1024 Patches onto Fast Local NVMe\n"
            "cmd = [\n"
            "    VENV_PY, 'scripts/download_dotav2.py',\n"
            "    '--source', DATA_SOURCE,\n"
            "    '--cache-dir', f'{WORK_ROOT}/raw_archives',\n"
            "    '--out-dir', '/content/data/dotav2_raw',\n"
            "]\n"
            "if KAGGLE_DATASET_SLUG:\n"
            "    cmd += ['--kaggle-dataset', KAGGLE_DATASET_SLUG]\n"
            "if V2_TRAIN_ANN_ID:\n"
            "    cmd += ['--v2-train-ann', V2_TRAIN_ANN_ID]\n"
            "if V2_VAL_ANN_ID:\n"
            "    cmd += ['--v2-val-ann', V2_VAL_ANN_ID]\n"
            "subprocess.run(cmd, check=True)\n\n"
            "tile_cmd = [\n"
            "    VENV_PY, 'scripts/prepare_dotav2_patches.py',\n"
            "    '--raw-dir', '/content/data/dotav2_raw',\n"
            "    '--out-dir', '/content/data/split_ss_dotav2',\n"
            "    '--mmrotate-dir', '/content/ADA/mmrotate',\n"
            "    '--patch-size', '1024',\n"
            "    '--overlap', '200',\n"
            "    '--nproc', '8',\n"
            "]\n"
            "if SMOKE_TEST_LIMIT > 0:\n"
            "    tile_cmd += ['--smoke-test-limit', str(SMOKE_TEST_LIMIT)]\n"
            "subprocess.run(tile_cmd, check=True)\n"
        ),
        nbf.v4.new_markdown_cell(
            "**Analysis (Step 4):** Inspect the staged instance counts and patch counts printed above. `train` and `val` are tiled independently onto `/content/data/split_ss_dotav2/` to prevent leakage and avoid slow Google Drive FUSE I/O."
        ),
        nbf.v4.new_code_cell(
            "# Step 5: Auto-Tune Batch Size for Detected Colab GPU (A100-80GB / A100-40GB / L4 / T4)\n"
            "import json\n"
            "env_info = json.loads(open('/content/ADA/colab_env_summary.json').read())\n"
            "prof = env_info['recommended_profile']\n"
            "BS = prof['batch_size']\n"
            "ACC = prof['accumulative_counts']\n"
            "print(f\"Detected {env_info['gpu_name']} ({env_info['vram_gb']} GB) -> batch_size={BS}, accum={ACC}\")\n"
        ),
        nbf.v4.new_markdown_cell(
            "**Analysis (Step 5):** Effective batch size (`BS * ACC = 8`) stays invariant across A100-80GB, A100-40GB, L4, and T4 runtimes so learning rate schedules remain calibrated."
        ),
        nbf.v4.new_code_cell(
            "# Step 6: Train Track A — Vanilla Rotated Faster R-CNN (R50-FPN, 1x Baseline)\n"
            "if RUN_BASELINE_TRACK_A:\n"
            "    track_a_dir = f'{WORK_ROOT}/work_dirs/track_a_baseline_r50_fpn_1x'\n"
            "    !MPLBACKEND=Agg {VENV_PY} mmrotate/tools/train.py \\\n"
            "        configs/baseline_rotated_faster_rcnn_r50_fpn_1x_dotav2.py \\\n"
            "        --work-dir {track_a_dir} \\\n"
            "        --resume \\\n"
            "        --cfg-options train_dataloader.batch_size={BS} optim_wrapper.accumulative_counts={ACC}\n"
        ),
        nbf.v4.new_markdown_cell(
            "**Analysis (Step 6):** Check Track A's validation `dota/mAP` progression and per-class APs in the logs. Checkpoints are persisted to Google Drive every epoch with `--resume` support."
        ),
        nbf.v4.new_code_cell(
            "# Step 7: Train Track B — Context-Aware Oriented Faster R-CNN (ConvNeXt-S + PAFPN + KLD, 2x)\n"
            "if RUN_FLAGSHIP_TRACK_B:\n"
            "    track_b_dir = f'{WORK_ROOT}/work_dirs/track_b_convnext_s_pafpn_kld_2x'\n"
            "    !MPLBACKEND=Agg {VENV_PY} mmrotate/tools/train.py \\\n"
            "        configs/benchmark_oriented_faster_rcnn_convnext_s_pafpn_kld_2x_dotav2.py \\\n"
            "        --work-dir {track_b_dir} \\\n"
            "        --resume \\\n"
            "        --cfg-options train_dataloader.batch_size={BS} optim_wrapper.accumulative_counts={ACC}\n"
        ),
        nbf.v4.new_markdown_cell(
            "**Analysis (Step 7):** Compare Track B's convergence against Track A, paying close attention to rare classes (`helipad`, `airport`, `container-crane`) and high-aspect-ratio classes (`ship`, `bridge`, `large-vehicle`)."
        ),
        nbf.v4.new_code_cell(
            "# Step 8: Generate Final Benchmark Report (Saved to Google Drive)\n"
            "report_cmd = [VENV_PY, 'scripts/evaluate_benchmark.py', '--out', f'{WORK_ROOT}/benchmark_report.md']\n"
            "if RUN_BASELINE_TRACK_A:\n"
            "    report_cmd += ['--baseline-dir', f'{WORK_ROOT}/work_dirs/track_a_baseline_r50_fpn_1x']\n"
            "if RUN_FLAGSHIP_TRACK_B:\n"
            "    report_cmd += ['--flagship-dir', f'{WORK_ROOT}/work_dirs/track_b_convnext_s_pafpn_kld_2x']\n"
            "subprocess.run(report_cmd, check=True)\n"
        ),
        nbf.v4.new_markdown_cell(
            "## Conclusion & Benchmark Summary\n\n"
            "Review the generated `benchmark_report.md` in Google Drive (`WORK_ROOT/benchmark_report.md`) to inspect:\n"
            "- Overall `dota/mAP` lift of **Track B (ConvNeXt-S + PAFPN + Oriented RPN + KLD + ClassBalanced)** over **Track A (Rotated Faster R-CNN R50-FPN)**.\n"
            "- Per-class AP@0.5 across all 18 DOTA v2.0 categories.\n"
            "- Best checkpoints saved under `WORK_ROOT/work_dirs/*/best_dota_mAP_epoch_*.pth` for downstream inference or `test-dev` submission.\n"
        ),
    ]

    nb["cells"] = cells
    out_path.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(nb, str(out_path))
    print(f"[notebook] Generated {out_path}")


if __name__ == "__main__":
    build_notebook(Path(__file__).resolve().parents[1] / "colab_runner.ipynb")
