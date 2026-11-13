"""Local validation tests for DOTA v2.0 benchmark configs, helpers, and notebook."""

from __future__ import annotations

import ast
import json
from pathlib import Path

import nbformat

from configs.dota_v2_classes import DOTA_V2_CLASSES, DOTA_V2_PALETTE
from scripts.download_dotav2 import normalize_and_validate_labels
from scripts.evaluate_benchmark import extract_best_metrics, render_markdown_report
from scripts.verify_env import recommend_batch_profile

ROOT = Path(__file__).resolve().parents[1]


def test_dota_v2_classes_count_and_uniqueness() -> None:
    assert len(DOTA_V2_CLASSES) == 18
    assert len(set(DOTA_V2_CLASSES)) == 18
    assert len(DOTA_V2_PALETTE) == 18
    assert "airport" in DOTA_V2_CLASSES
    assert "helipad" in DOTA_V2_CLASSES
    assert "container-crane" in DOTA_V2_CLASSES


def test_mmrotate_configs_parse_and_target_18_classes() -> None:
    cfg_paths = [
        ROOT / "configs/baseline_rotated_faster_rcnn_r50_fpn_1x_dotav2.py",
        ROOT / "configs/benchmark_oriented_faster_rcnn_convnext_s_pafpn_kld_2x_dotav2.py",
    ]
    for path in cfg_paths:
        src = path.read_text(encoding="utf-8")
        tree = ast.parse(src, filename=str(path))
        assert tree is not None
        assert "num_classes=len(DOTA_V2_CLASSES)" in src
        assert '"DOTAMetric"' in src


def test_normalize_and_validate_labels(tmp_path: Path) -> None:
    lbl_file = tmp_path / "P0001.txt"
    lbl_file.write_text(
        "imagesource:GoogleEarth\n"
        "gsd:0.5\n"
        "10 20 30 20 30 40 10 40 small vehicle 0\n"
        "50 60 70 60 70 80 50 80 helipad 0\n",
        encoding="utf-8",
    )
    counts = normalize_and_validate_labels(tmp_path, normalize_spaces=True)
    assert counts["small-vehicle"] == 1
    assert counts["helipad"] == 1
    assert "small-vehicle 0" in lbl_file.read_text(encoding="utf-8")


def test_convert_yolo_obb_to_dota_labeltxt(tmp_path: Path) -> None:
    from PIL import Image

    img_dir = tmp_path / "images"
    lbl_dir = tmp_path / "labels"
    img_dir.mkdir()
    lbl_dir.mkdir()
    Image.new("RGB", (1000, 500)).save(img_dir / "P0002.png")
    (lbl_dir / "P0002.txt").write_text(
        "10 0.1 0.2 0.2 0.2 0.2 0.4 0.1 0.4\n"
        "17 2.2 0.5 2.3 0.5 2.3 0.6 2.2 0.6\n",
        encoding="utf-8",
    )
    counts = normalize_and_validate_labels(lbl_dir, img_dir=img_dir)
    assert counts["small-vehicle"] == 1
    assert counts["helipad"] == 1
    text = (lbl_dir / "P0002.txt").read_text(encoding="utf-8")
    assert "100.0 100.0 200.0 100.0 200.0 200.0 100.0 200.0 small-vehicle 0" in text
    assert "2200.0 250.0 2300.0 250.0 2300.0 300.0 2200.0 300.0 helipad 0" in text


def test_gpu_batch_profiles() -> None:
    assert recommend_batch_profile("NVIDIA A100-SXM4-80GB", 80.0)["batch_size"] == 8
    assert recommend_batch_profile("NVIDIA A100-SXM4-40GB", 40.0)["batch_size"] == 4
    assert recommend_batch_profile("NVIDIA L4", 22.5)["batch_size"] == 2
    assert recommend_batch_profile("Tesla T4", 15.0)["batch_size"] == 1


def test_evaluate_benchmark_report(tmp_path: Path) -> None:
    base_dir = tmp_path / "base"
    flag_dir = tmp_path / "flag"
    base_dir.mkdir()
    flag_dir.mkdir()

    (base_dir / "scalars.json").write_text(
        json.dumps({"step": 12, "dota/mAP": 0.4712, "dota/AP50": 0.471}) + "\n",
        encoding="utf-8",
    )
    (base_dir / "train.log").write_text(
        "| plane              | 4859  | 8707  | 0.842  | 0.788 |\n"
        "| airport            | 102   | 14    | 0.049  | 0.091 |\n"
        "11/13 12:17:26 - mmengine - INFO - Epoch(val) [12][1781/1781]    dota/mAP: 0.4712  dota/AP50: 0.4710\n",
        encoding="utf-8",
    )
    (flag_dir / "scalars.json").write_text(
        json.dumps({"step": 24, "dota/mAP": 0.5689, "dota/AP50": 0.569}) + "\n",
        encoding="utf-8",
    )
    (flag_dir / "train.log").write_text(
        "| plane              | 4859  | 8707  | 0.890  | 0.860 |\n"
        "| airport            | 102   | 140   | 0.400  | 0.350 |\n"
        "11/13 15:17:26 - mmengine - INFO - Epoch(val) [24][1781/1781]    dota/mAP: 0.5689  dota/AP50: 0.5690\n",
        encoding="utf-8",
    )

    b_metrics = extract_best_metrics(base_dir)
    f_metrics = extract_best_metrics(flag_dir)
    md = render_markdown_report(b_metrics, f_metrics)
    assert "0.4712" in md
    assert "0.5689" in md
    assert "+0.0977" in md
    assert "0.7880" in md
    assert "0.3500" in md


def test_colab_notebook_valid() -> None:
    nb_path = ROOT / "colab_runner.ipynb"
    assert nb_path.is_file()
    nb = nbformat.read(str(nb_path), as_version=4)
    assert len(nb.cells) >= 10
