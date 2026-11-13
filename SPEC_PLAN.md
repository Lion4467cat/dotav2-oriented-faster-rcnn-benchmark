# DOTA v2.0 Oriented Faster R-CNN Benchmark — Architecture & Spec Plan

## 1. Benchmark Objective & Scope
- **Dataset**: DOTA v2.0 (`11,268` aerial images total across splits, `1,793,658` oriented instances, **18 categories** including `container-crane`, `airport`, and `helipad`).
- **Task**: Oriented Object Detection (**OBB**, `le90` angle representation: $\theta \in [-90^\circ, 90^\circ)$).
- **Evaluation Protocol**: Train on DOTA v2.0 `train` split, evaluate locally on DOTA v2.0 `val` split using rotated IoU **mAP@0.5** (`DOTAMetric`, per-class AP for all 18 classes + overall mAP).
- **Framework**: OpenMMLab **MMRotate (`dev-1.x`)** + **MMDetection 3.x** + **MMEngine** + **MMCV 2.x** on PyTorch 2.x (CUDA).
- **Local vs. Cloud Split**:
  - **Local PC (16GB RAM / 8GB VRAM)**: Managed via `uv` (`.venv`, Python 3.11) for lightweight config validation, unit tests, and GitHub sync. **Zero heavy training locally.**
  - **Google Colab Pro / Pro+ (L4 24GB or A100 40GB, High-RAM)**: Executes dataset download/Drive caching, `1024×1024` NVMe patching, baseline & flagship training, and benchmark reporting.

---

## 2. Two-Track Benchmark Architecture Stack

To establish a rigorous, defensible benchmark number (rather than an uncalibrated single run), we run **two comparable two-stage Faster R-CNN OBB experiments** on the exact same `1024×1024` DOTA v2.0 `train`/`val` splits:

### Track A — Official-Style Baseline (`baseline_rotated_faster_rcnn_r50_fpn_1x_dotav2.py`)
- **Detector**: `RotatedFasterRCNN` (Horizontal RPN $\rightarrow$ Rotated RoIAlign $\rightarrow$ `RotatedShared2FCBBoxHead`).
- **Backbone / Neck**: `ResNet-50` + `FPN` (ImageNet-1k pretrained).
- **Regression Loss**: `SmoothL1Loss` on `DeltaXYWHAOBBoxCoder`.
- **Schedule**: `1x` (12 epochs), `SGD` (`lr=0.005`, momentum `0.9`), AMP `fp16`.
- **Role**: Establishes our apples-to-apples DOTA v2.0 `val` baseline (~47% range on DOTA v2.0 in literature due to horizontal proposal misalignment on dense/oblique targets).

### Track B — Unique Strong Benchmark Stack (`benchmark_oriented_faster_rcnn_convnext_s_pafpn_kld_2x_dotav2.py`)
**"Context-Aware Oriented Faster R-CNN"**: stays strictly within the two-stage Faster R-CNN paradigm while targeting DOTA v2.0's three hardest failure modes (tiny dense objects, extreme aspect ratios / angle boundary discontinuity, and severe 18-class imbalance):

| Component | Baseline (Track A) | Flagship Stack (Track B) | Why It Matters on DOTA v2.0 |
| :--- | :--- | :--- | :--- |
| **Backbone** | `ResNet-50` | **`mmdet.ConvNeXt-Small`** (IN-1k/22k pretrained, `drop_path_rate=0.4`) | 7×7 depthwise kernels capture wide aerial context (`airport`, `harbor`, `bridge`) without Swin's quadratic memory spikes on `1024×1024` patches. |
| **Neck** | `FPN` (top-down) | **`mmdet.PAFPN`** (top-down + bottom-up path augmentation) | Preserves high-resolution localization signals for dense tiny instances (`small-vehicle`, `ship`, `storage-tank`) while strengthening semantic flow for `airport` / `ground-track-field`. |
| **RPN Stage** | Horizontal RPN | **`OrientedRPNHead` (`MidpointOffsetCoder`)** | Directly emits oriented proposals in 6 parameters, eliminating horizontal proposal clutter and feature misalignment in dense ship/vehicle clusters. |
| **RoI Extractor** | `RoIAlignRotated` (`7×7`) | `RoIAlignRotated` (`7×7`, `sample_num=2`) | Extracts rotation-aligned RoI features into `RotatedShared2FCBBoxHead` (18 classes). |
| **BBox Regression Loss** | `SmoothL1Loss` | **`GDLoss` (`loss_type='kld'`, `fun='log1p'`, `reg_decoded_bbox=True`)** | Models OBBs as 2D Gaussian distributions via Kullback-Leibler Divergence — eliminates angle boundary discontinuity and stabilizes high-aspect-ratio (`bridge`, `large-vehicle`, `ship`) + square-like (`storage-tank`, `roundabout`) regression. |
| **Imbalance Handling** | Uniform sampling | **`ClassBalancedDataset` (`oversample_thr=1e-3`)** | Boosts rare DOTA v2.0 classes (`helipad`, `airport`, `container-crane`, `helicopter`) against `small-vehicle` dominance. |
| **Augmentation** | H/V/D Flip (`0.75`) | H/V/D Flip (`0.75`) + **`RandomRotate` (`prob=0.5`, `±180°`)** | Exploits aerial rotational invariance for stronger orientation generalization. |
| **Optimizer & Schedule** | `SGD` `1x` (12 ep) | **`AdamW` (`lr=1e-4`, `wd=0.05`) + `CosineAnnealingLR` `2x` (24 ep) + AMP** | Matches ConvNeXt optimization dynamics; 2× schedule gives rare/oversampled classes convergence room. |

---

## 3. Data & Storage Pipeline (Colab Pro / Pro+ Spec)
- **Patching Spec**: Single-scale `1024×1024` patches with `200px` overlap (`stride=824`), keeping non-empty patches (`drop_empty=True` for `train` to keep epoch wall-time fast on Colab; keeping all `val` patches for exact unskewed evaluation).
- **I/O Architecture (Critical for Colab Speed)**:
  1. **Google Drive (`/content/drive/MyDrive/dotav2_benchmark/`)**: Stores **raw dataset archives** (downloaded once) + **training checkpoints / logs / final benchmark JSON & Markdown reports**.
  2. **Colab Local NVMe (`/content/data/split_ss_dotav2/`)**: Extracts & tiles `1024×1024` PNG patches onto local SSD/NVMe at runtime startup (~10–30× faster DataLoader throughput than reading tens of thousands of small files from FUSE-mounted Drive).
- **Auto-Checkpoint & Resume**: `CheckpointHook(interval=1, max_keep_ckpts=3, save_best='dota/mAP', rule='greater')` writing directly to Google Drive, so even if a Colab Pro session disconnects mid-run, resuming with `--resume` picks up from the last epoch.

---

## 4. GPU Scaling Matrix (Colab Pro / Pro+)

| Colab GPU | VRAM | Batch Size / GPU | `accumulative_counts` | Effective Batch | Estimated Speed |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **A100** | 40 GB | `4` (or `8` w/ AMP) | `1` (or `2`) | `8` | Fastest (~1x baseline ~1–2h, 2x ConvNeXt-S ~4–6h) |
| **L4** | 24 GB | `2` | `4` | `8` | Moderate (~2× A100 wall time) |
| **T4** (Fallback) | 16 GB | `1` or `2` | `4` | `4–8` | Slower; use `ResNet-50` or `ConvNeXt-Tiny` if OOM |

---

## 5. Bridging Local Workspace to Google Colab
1. Local edits & validation happen here in `/home/tinkerer/Desktop/ADA` inside the `uv` `.venv`.
2. Push this repository to GitHub (`gh repo create ... --push`).
3. Open `colab_runner.ipynb` directly in Google Colab (`https://colab.research.google.com/github/<YOUR_USER>/<YOUR_REPO>/blob/main/colab_runner.ipynb`), select **A100** (or **L4**) + **High-RAM**, and run all cells sequentially.
4. Pull/inspect the resulting `benchmark_report.md` and `val` metrics back locally or in Drive.
