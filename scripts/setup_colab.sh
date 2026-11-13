#!/usr/bin/env bash
set -euo pipefail

# Installs PyTorch 2.1.2 (cu121) + prebuilt MMCV 2.1.0 + MMDet 3.x + MMRotate (dev-1.x)
# on Google Colab without slow from-source CUDA compilation.
REPO_DIR="${1:-$(pwd)}"
cd "$REPO_DIR"

echo "[setup] Installing PyTorch 2.1.2 + cu121 (matches prebuilt MMCV 2.1.0 wheels)..."
pip install -q torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu121

echo "[setup] Installing OpenMMLab 2.x prebuilt wheels..."
pip install -q -U openmim gdown kaggle
pip install -q "mmengine>=0.8.0,<0.11.0"
pip install -q "mmcv==2.1.0" -f https://download.openmmlab.com/mmcv/dist/cu121/torch2.1/index.html
pip install -q "mmdet>=3.2.0,<3.4.0"

if [ ! -d "$REPO_DIR/mmrotate" ]; then
  echo "[setup] Cloning mmrotate (dev-1.x)..."
  git clone --depth 1 -b dev-1.x https://github.com/open-mmlab/mmrotate.git "$REPO_DIR/mmrotate"
fi

echo "[setup] Installing mmrotate (editable)..."
pip install -q -e "$REPO_DIR/mmrotate"
pip install -q shapely terminaltables einops

echo "[setup] Done. Running environment verification..."
python "$REPO_DIR/scripts/verify_env.py" --project-root "$REPO_DIR" --json-out "$REPO_DIR/colab_env_summary.json"
