# DOTA v2.0 Oriented Faster R-CNN Benchmark (Google Colab Pro/Pro+)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Lion4467cat/dotav2-oriented-faster-rcnn-benchmark/blob/main/colab_runner.ipynb)

This repository implements a two-track **Oriented Bounding Box (`le90`)** benchmark on **DOTA v2.0** (18 classes) built on **MMRotate (`dev-1.x`)**, designed so your local 16GB RAM / 8GB VRAM machine only handles lightweight code & validation (via `uv`), while all dataset tiling, training, and evaluation execute on **Google Colab Pro/Pro+ (A100 / L4)**.

See [SPEC_PLAN.md](SPEC_PLAN.md) for the full architecture rationale.

## Local Setup (`uv`)

```bash
uv venv --python 3.11
uv sync
uv run pytest
```

## Running on Google Colab Pro / Pro+

1. **Open `colab_runner.ipynb` directly in Google Colab:**
   - Click the badge above or open:
     `https://colab.research.google.com/github/Lion4467cat/dotav2-oriented-faster-rcnn-benchmark/blob/main/colab_runner.ipynb`
2. **Select GPU Runtime:**
   - Go to **Runtime -> Change runtime type** -> select **A100 GPU** (or **L4**) + **High-RAM**.
3. **Prepare DOTA v2.0 archives:**
   - Upload your DOTA v2.0 `train` & `val` `.zip` archives once into Google Drive at `MyDrive/dotav2_benchmark/raw_archives/` (with `DATA_SOURCE = 'local'` in Step 2), **or** set `DATA_SOURCE = 'gdrive'` / `'kaggle'` in Step 2.
4. **Click `Runtime -> Run all`:**
   - First-run recommendation: set `SMOKE_TEST_LIMIT = 50` in Step 2 for a fast ~5-minute pipeline sanity check, then set `SMOKE_TEST_LIMIT = 0` for the full DOTA v2.0 benchmark.
   - Checkpoints, logs, and `benchmark_report.md` are automatically persisted to Google Drive (`MyDrive/dotav2_benchmark/`).
