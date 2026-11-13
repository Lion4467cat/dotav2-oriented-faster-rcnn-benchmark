#!/usr/bin/env bash
set -euo pipefail

# Provisions an isolated Python 3.11 venv via `uv` on Google Colab (whose system
# Python may be 3.12/3.13+), then installs PyTorch 2.1.2 (cu121) + prebuilt
# MMCV 2.1.0 + MMDet 3.x + MMRotate (dev-1.x) without compiling CUDA ops.
REPO_DIR="${1:-$(pwd)}"
VENV_DIR="${2:-/content/mmrotate_venv}"
cd "$REPO_DIR"

UV_BIN="$(command -v uv || true)"
if [ -z "$UV_BIN" ] && [ -x "$HOME/.local/bin/uv" ]; then
  UV_BIN="$HOME/.local/bin/uv"
fi
if [ -z "$UV_BIN" ]; then
  echo "[setup] Installing uv..."
  curl -LsSf https://astral.sh/uv/install.sh | sh
  UV_BIN="$HOME/.local/bin/uv"
fi

echo "[setup] Creating Python 3.11 venv at $VENV_DIR using $UV_BIN..."
"$UV_BIN" venv "$VENV_DIR" --python 3.11 --seed --clear
VENV_PY="$VENV_DIR/bin/python"

cat > /tmp/mm_constraints.txt << 'EOF'
numpy<2.0
setuptools<70
torch==2.1.2
torchvision==0.16.2
EOF

echo "[setup] Installing PyTorch 2.1.2 + cu121..."
"$UV_BIN" pip install --python "$VENV_PY" -q \
  torch==2.1.2 torchvision==0.16.2 \
  --index-url https://download.pytorch.org/whl/cu121

echo "[setup] Installing NumPy < 2.0, OpenMMLab 2.x, and helper dependencies..."
"$UV_BIN" pip install --python "$VENV_PY" -q -c /tmp/mm_constraints.txt \
  "numpy<2.0" "setuptools<70" \
  "mmengine>=0.8.0,<0.11.0" \
  "gdown>=5.2.0" "kaggle" "shapely" "terminaltables" "einops" "pillow" "tqdm" "requests"

"$UV_BIN" pip install --python "$VENV_PY" -q -c /tmp/mm_constraints.txt \
  "mmcv==2.1.0" -f https://download.openmmlab.com/mmcv/dist/cu121/torch2.1/index.html

"$UV_BIN" pip install --python "$VENV_PY" -q -c /tmp/mm_constraints.txt \
  "mmdet>=3.0.0rc6,<3.2.0"

if [ ! -d "$REPO_DIR/mmrotate" ]; then
  echo "[setup] Cloning mmrotate (dev-1.x)..."
  git clone --depth 1 -b dev-1.x https://github.com/open-mmlab/mmrotate.git "$REPO_DIR/mmrotate"
fi

echo "[setup] Installing mmrotate (editable, no build isolation)..."
"$UV_BIN" pip install --python "$VENV_PY" -q -c /tmp/mm_constraints.txt \
  --no-build-isolation -e "$REPO_DIR/mmrotate"

echo "[setup] Done. Running environment verification..."
"$VENV_PY" "$REPO_DIR/scripts/verify_env.py" --project-root "$REPO_DIR" --json-out "$REPO_DIR/colab_env_summary.json"
