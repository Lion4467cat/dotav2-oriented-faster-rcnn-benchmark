"""Local validation tests for DOTA v2.0 benchmark configs, helpers, and notebook."""

from __future__ import annotations

import ast
import json
import math
from pathlib import Path

import nbformat
import numpy as np

from configs.dota_v2_classes import DOTA_V2_CLASSES, DOTA_V2_PALETTE
from scripts.compute_extended_metrics import (
    compute_rotated_iou_matrix,
    evaluate_predictions,
    long_edge_angle_deg,
)
from scripts.download_dotav2 import normalize_and_validate_labels
from scripts.evaluate_benchmark import (
    extract_best_metrics,
    load_extended_metrics,
    render_markdown_report,
)
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


def test_rotated_iou_and_long_edge_invariance() -> None:
    b1 = np.array([[100.0, 100.0, 80.0, 20.0, 0.3]], dtype=np.float32)
    b1_swapped = np.array([[100.0, 100.0, 20.0, 80.0, 0.3 + math.pi / 2.0]], dtype=np.float32)
    b_far = np.array([[500.0, 500.0, 40.0, 10.0, 0.0]], dtype=np.float32)

    iou = compute_rotated_iou_matrix(b1, np.vstack([b1_swapped, b_far]))
    assert math.isclose(float(iou[0, 0]), 1.0, abs_tol=1e-4)
    assert float(iou[0, 1]) == 0.0
    ang1 = float(long_edge_angle_deg(b1)[0])
    ang2 = float(long_edge_angle_deg(b1_swapped)[0])
    assert math.isclose(ang1, ang2, abs_tol=1e-3)


def test_extended_metrics_suite_and_report_integration(tmp_path: Path) -> None:
    ann_dir = tmp_path / "annfiles"
    ann_dir.mkdir()
    # One small-vehicle (easy), one large ship (easy), one difficult harbor
    (ann_dir / "P0001.txt").write_text(
        "90 90 110 90 110 110 90 110 small-vehicle 0\n"
        "200 260 400 260 400 340 200 340 ship 0\n"
        "500 500 600 500 600 550 500 550 harbor 1\n",
        encoding="utf-8",
    )

    samples = [
        {
            "img_path": str(tmp_path / "images" / "P0001.png"),
            "pred_instances": {
                "bboxes": np.array(
                    [
                        [100.0, 100.0, 20.0, 20.0, 0.0],  # exact TP on small-vehicle
                        [300.0, 300.0, 200.0, 80.0, 0.05],  # TP on ship (~2.86 deg error)
                        [550.0, 525.0, 100.0, 50.0, 0.0],  # hits difficult=1 harbor (ignored for mAP)
                        [700.0, 700.0, 40.0, 40.0, 0.0],  # low-conf FP
                    ],
                    dtype=np.float32,
                ),
                "scores": np.array([0.95, 0.90, 0.92, 0.20], dtype=np.float32),
                "labels": np.array(
                    [
                        DOTA_V2_CLASSES.index("small-vehicle"),
                        DOTA_V2_CLASSES.index("ship"),
                        DOTA_V2_CLASSES.index("harbor"),
                        DOTA_V2_CLASSES.index("plane"),
                    ],
                    dtype=np.int64,
                ),
            },
        }
    ]

    ext = evaluate_predictions(samples, ann_dir=ann_dir)
    assert ext["num_gt_non_difficult"] == 2
    assert ext["map50"] == 1.0
    assert ext["map50_95"] >= 0.8
    assert ext["micro_prf"]["f1_optimal"]["precision"] == 1.0
    assert ext["micro_prf"]["f1_optimal"]["recall"] == 1.0
    assert ext["micro_prf"]["at_conf"]["0.25"]["precision"] == 1.0
    assert ext["scale_ap50"]["small"]["ap50"] == 1.0
    assert ext["scale_ap50"]["large"]["ap50"] == 1.0
    assert ext["difficulty_recall"]["difficult_total"] == 1
    assert ext["difficulty_recall"]["difficult_recall"] == 1.0
    assert ext["orientation_error_deg"]["mean_elongated_tp"] is not None

    work_dir = tmp_path / "work"
    (work_dir / "eval_test").mkdir(parents=True)
    (work_dir / "scalars.json").write_text(
        json.dumps({"step": 12, "dota/mAP": 0.59}) + "\n", encoding="utf-8"
    )
    (work_dir / "eval_test" / "extended_metrics.json").write_text(
        json.dumps(ext), encoding="utf-8"
    )
    loaded_ext = load_extended_metrics(work_dir)
    assert loaded_ext is not None
    md = render_markdown_report(
        extract_best_metrics(work_dir), None, baseline_ext=loaded_ext
    )
    assert "Reference Target (User Provided)" in md
    assert "Mean Angle Error - Elongated" in md
