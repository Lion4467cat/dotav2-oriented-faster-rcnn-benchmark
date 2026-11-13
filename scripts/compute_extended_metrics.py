"""Compute comprehensive OBB evaluation metrics from MMRotate test predictions.

Measures user-requested and DOTA v2.0 OBB-specific metrics on the validation split:
- mAP50, AP75, and mAP50-95 (IoU 0.50:0.05:0.95)
- Global & Macro Precision, Recall, F1-score (at F1-optimal and fixed conf thresholds)
- Scale-bucketed AP50 & Recall (Small < 32^2, Medium 32^2..96^2, Large > 96^2 px^2)
- Class-agnostic localization AP50 & label confusion rate
- Mean / Median Absolute Angle Error (degrees) on matched True Positives
- Easy (difficult=0) vs Difficult (difficult=1) GT Recall
- Per-class AP50, AP50-95, Precision, Recall, and F1
"""

from __future__ import annotations

import argparse
import json
import math
import pickle
import sys
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from configs.dota_v2_classes import DOTA_V2_CLASSES

try:
    import cv2  # type: ignore
except ImportError:  # pragma: no cover - optional locally
    cv2 = None

try:
    import torch  # type: ignore
    from mmcv.ops import box_iou_rotated  # type: ignore
except ImportError:  # pragma: no cover - optional locally
    torch = None
    box_iou_rotated = None


SMALL_AREA_MAX = 32.0 * 32.0
MEDIUM_AREA_MAX = 96.0 * 96.0
IOU_THRESHOLDS = np.round(np.linspace(0.50, 0.95, 10), 2)
FIXED_CONF_THRESHOLDS = (0.25, 0.30, 0.50)


def _to_numpy(arr: Any) -> np.ndarray:
    if arr is None:
        return np.empty((0,), dtype=np.float32)
    if hasattr(arr, "tensor"):
        arr = arr.tensor
    if hasattr(arr, "detach"):
        arr = arr.detach().cpu().numpy()
    return np.asarray(arr)


def quad_to_rbox_np(poly: np.ndarray) -> tuple[float, float, float, float, float]:
    """Convert a 4-vertex polygon [8] into minimal enclosing (cx, cy, w, h, theta_rad)."""
    pts = np.asarray(poly, dtype=np.float64).reshape(4, 2)
    if cv2 is not None:
        (cx, cy), (w, h), angle_deg = cv2.minAreaRect(pts.astype(np.float32))
        theta = math.radians(float(angle_deg))
        w_f, h_f = float(w), float(h)
        if theta >= math.pi / 2.0:
            theta -= math.pi / 2.0
            w_f, h_f = h_f, w_f
        elif theta < -math.pi / 2.0:
            theta += math.pi / 2.0
            w_f, h_f = h_f, w_f
        return float(cx), float(cy), max(w_f, 1e-3), max(h_f, 1e-3), theta

    best_area = float("inf")
    best_rbox = (float(pts[:, 0].mean()), float(pts[:, 1].mean()), 1.0, 1.0, 0.0)
    center = pts.mean(axis=0)
    centered = pts - center
    for i in range(4):
        edge = pts[(i + 1) % 4] - pts[i]
        norm = float(np.hypot(edge[0], edge[1]))
        if norm < 1e-8:
            continue
        theta = math.atan2(float(edge[1]), float(edge[0]))
        c, s = math.cos(theta), math.sin(theta)
        rot = np.array([[c, s], [-s, c]], dtype=np.float64)
        proj = centered @ rot.T
        mins = proj.min(axis=0)
        maxs = proj.max(axis=0)
        w = float(maxs[0] - mins[0])
        h = float(maxs[1] - mins[1])
        area = w * h
        if area < best_area:
            best_area = area
            mid_local = 0.5 * (mins + maxs)
            mid_world = center + np.array(
                [c * mid_local[0] - s * mid_local[1], s * mid_local[0] + c * mid_local[1]]
            )
            # Canonicalize angle into [-pi/2, pi/2)
            t = theta
            wf, hf = w, h
            while t >= math.pi / 2.0:
                t -= math.pi / 2.0
                wf, hf = hf, wf
            while t < -math.pi / 2.0:
                t += math.pi / 2.0
                wf, hf = hf, wf
            best_rbox = (
                float(mid_world[0]),
                float(mid_world[1]),
                max(wf, 1e-3),
                max(hf, 1e-3),
                t,
            )
    return best_rbox


def _rbox_to_corners(rbox: np.ndarray) -> np.ndarray:
    """Convert single (cx, cy, w, h, theta_rad) box into [4, 2] CCW/CW corners."""
    cx, cy, w, h, theta = [float(x) for x in rbox]
    dx = 0.5 * w
    dy = 0.5 * h
    c, s = math.cos(theta), math.sin(theta)
    local = np.array([[-dx, -dy], [dx, -dy], [dx, dy], [-dx, dy]], dtype=np.float64)
    rot = np.array([[c, -s], [s, c]], dtype=np.float64)
    return local @ rot.T + np.array([cx, cy], dtype=np.float64)


def _poly_area(pts: np.ndarray) -> float:
    if len(pts) < 3:
        return 0.0
    x = pts[:, 0]
    y = pts[:, 1]
    return 0.5 * abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def _ensure_ccw(pts: np.ndarray) -> np.ndarray:
    x = pts[:, 0]
    y = pts[:, 1]
    signed_area = float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))
    return pts[::-1] if signed_area < 0 else pts


def _clip_convex_polygon(subject: np.ndarray, clipper: np.ndarray) -> np.ndarray:
    """Sutherland-Hodgman polygon clipping between two convex polygons."""
    output = _ensure_ccw(subject)
    clip = _ensure_ccw(clipper)
    for i in range(len(clip)):
        a = clip[i]
        b = clip[(i + 1) % len(clip)]
        edge = b - a
        inp = output
        if len(inp) == 0:
            break
        pts: list[np.ndarray] = []
        s_pt = inp[-1]
        for e_pt in inp:
            s_in = edge[0] * (s_pt[1] - a[1]) - edge[1] * (s_pt[0] - a[0]) >= -1e-9
            e_in = edge[0] * (e_pt[1] - a[1]) - edge[1] * (e_pt[0] - a[0]) >= -1e-9
            if e_in:
                if not s_in:
                    pts.append(_line_intersection(s_pt, e_pt, a, b))
                pts.append(e_pt)
            elif s_in:
                pts.append(_line_intersection(s_pt, e_pt, a, b))
            s_pt = e_pt
        output = np.array(pts, dtype=np.float64) if pts else np.empty((0, 2), dtype=np.float64)
    return output


def _line_intersection(p1: np.ndarray, p2: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    dc = a - b
    dp = p1 - p2
    n1 = a[0] * b[1] - a[1] * b[0]
    n2 = p1[0] * p2[1] - p1[1] * p2[0]
    denom = dc[0] * dp[1] - dc[1] * dp[0]
    if abs(denom) < 1e-12:
        return p2
    return np.array([(n1 * dp[0] - n2 * dc[0]) / denom, (n1 * dp[1] - n2 * dc[1]) / denom])


def compute_rotated_iou_matrix(boxes1: np.ndarray, boxes2: np.ndarray) -> np.ndarray:
    """Compute exact pairwise rotated IoU [N, M] for (cx, cy, w, h, theta_rad) boxes."""
    if len(boxes1) == 0 or len(boxes2) == 0:
        return np.zeros((len(boxes1), len(boxes2)), dtype=np.float32)

    if box_iou_rotated is not None and torch is not None:
        t1 = torch.as_tensor(boxes1, dtype=torch.float32)
        t2 = torch.as_tensor(boxes2, dtype=torch.float32)
        if torch.cuda.is_available() and len(boxes1) * len(boxes2) > 512:
            return box_iou_rotated(t1.cuda(), t2.cuda()).cpu().numpy()
        return box_iou_rotated(t1, t2).numpy()

    out = np.zeros((len(boxes1), len(boxes2)), dtype=np.float32)
    corners1 = [_rbox_to_corners(b) for b in boxes1]
    corners2 = [_rbox_to_corners(b) for b in boxes2]
    areas1 = boxes1[:, 2] * boxes1[:, 3]
    areas2 = boxes2[:, 2] * boxes2[:, 3]
    radii1 = 0.5 * np.hypot(boxes1[:, 2], boxes1[:, 3])
    radii2 = 0.5 * np.hypot(boxes2[:, 2], boxes2[:, 3])

    for i in range(len(boxes1)):
        for j in range(len(boxes2)):
            dist = math.hypot(
                float(boxes1[i, 0] - boxes2[j, 0]), float(boxes1[i, 1] - boxes2[j, 1])
            )
            if dist > float(radii1[i] + radii2[j]):
                continue
            inter_poly = _clip_convex_polygon(corners1[i], corners2[j])
            inter = _poly_area(inter_poly)
            union = float(areas1[i] + areas2[j]) - inter
            out[i, j] = inter / union if union > 1e-8 else 0.0
    return out


def long_edge_angle_deg(rboxes: np.ndarray) -> np.ndarray:
    """Return orientation angle of the longer side in [0, 180) degrees (convention-invariant)."""
    if len(rboxes) == 0:
        return np.empty((0,), dtype=np.float32)
    w = rboxes[:, 2]
    h = rboxes[:, 3]
    theta_deg = np.degrees(rboxes[:, 4].astype(np.float64))
    phi = np.where(w >= h, theta_deg, theta_deg + 90.0)
    return (phi % 180.0).astype(np.float32)


def acute_angle_diff_deg(a_deg: np.ndarray, b_deg: np.ndarray) -> np.ndarray:
    """Acute axial difference between orientations modulo 180 deg, in [0, 90]."""
    diff = np.abs(a_deg - b_deg) % 180.0
    return np.minimum(diff, 180.0 - diff)


def compute_voc_allpoint_ap(recalls: np.ndarray, precisions: np.ndarray) -> float:
    """Compute exact area under the interpolated precision-recall curve (COCO/DOTA standard)."""
    if len(recalls) == 0:
        return 0.0
    mrec = np.concatenate(([0.0], recalls, [1.0]))
    mpre = np.concatenate(([1.0], precisions, [0.0]))
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))


def parse_dota_annfile(ann_path: Path, class_to_idx: dict[str, int]) -> dict[str, np.ndarray]:
    """Parse DOTA patch annfile into rboxes [G, 5], labels [G], difficult [G]."""
    rboxes: list[tuple[float, float, float, float, float]] = []
    labels: list[int] = []
    difficults: list[int] = []
    if ann_path.is_file():
        for raw in ann_path.read_text(encoding="utf-8", errors="replace").splitlines():
            parts = raw.strip().split()
            if len(parts) < 9:
                continue
            try:
                coords = np.array([float(x) for x in parts[:8]], dtype=np.float64)
            except ValueError:
                continue
            cls_name = parts[8]
            if cls_name not in class_to_idx:
                continue
            diff = int(parts[9]) if len(parts) >= 10 else 0
            rboxes.append(quad_to_rbox_np(coords))
            labels.append(class_to_idx[cls_name])
            difficults.append(diff)

    if not rboxes:
        return {
            "rboxes": np.empty((0, 5), dtype=np.float32),
            "labels": np.empty((0,), dtype=np.int64),
            "difficult": np.empty((0,), dtype=np.int32),
        }
    return {
        "rboxes": np.array(rboxes, dtype=np.float32),
        "labels": np.array(labels, dtype=np.int64),
        "difficult": np.array(difficults, dtype=np.int32),
    }


def _extract_gt_for_sample(
    sample: dict[str, Any],
    ann_dir: Path | None,
    class_to_idx: dict[str, int],
) -> dict[str, np.ndarray]:
    if ann_dir is not None and sample.get("img_path"):
        stem = Path(str(sample["img_path"])).stem
        ann_path = ann_dir / f"{stem}.txt"
        return parse_dota_annfile(ann_path, class_to_idx)

    gt = sample.get("gt_instances") or {}
    bboxes = _to_numpy(gt.get("bboxes")).astype(np.float32)
    labels = _to_numpy(gt.get("labels")).astype(np.int64)
    if len(bboxes) == 0:
        bboxes = np.empty((0, 5), dtype=np.float32)
        labels = np.empty((0,), dtype=np.int64)
    diff = _to_numpy(gt.get("difficult")).astype(np.int32)
    if len(diff) != len(bboxes):
        diff = np.zeros((len(bboxes),), dtype=np.int32)
    return {"rboxes": bboxes, "labels": labels, "difficult": diff}


def _match_predictions(
    preds: list[dict[str, np.ndarray]],
    gts: list[dict[str, np.ndarray]],
    iou_mats: list[np.ndarray],
    iou_thr: float,
    target_class: int | None = None,
    area_range: tuple[float, float] | None = None,
) -> dict[str, Any]:
    """Greedy score-sorted DOTA/COCO matching for one class (or class-agnostic if None)."""
    cand_scores: list[float] = []
    cand_img: list[int] = []
    cand_pidx: list[int] = []
    cand_area: list[float] = []

    gt_matched: list[np.ndarray] = []
    num_valid_gt = 0

    for img_i, (p, g) in enumerate(zip(preds, gts)):
        g_labels = g["labels"]
        g_diff = g["difficult"]
        g_areas = g["areas"]
        if target_class is None:
            cls_mask = np.ones(len(g_labels), dtype=bool)
        else:
            cls_mask = g_labels == target_class

        if area_range is None:
            in_area = np.ones(len(g_labels), dtype=bool)
        else:
            in_area = (g_areas >= area_range[0]) & (g_areas < area_range[1])

        valid_gt = cls_mask & (g_diff == 0) & in_area
        num_valid_gt += int(np.sum(valid_gt))
        gt_matched.append(np.zeros(len(g_labels), dtype=bool))

        p_labels = p["labels"]
        p_mask = np.ones(len(p_labels), dtype=bool) if target_class is None else (p_labels == target_class)
        idxs = np.where(p_mask)[0]
        for pidx in idxs:
            cand_scores.append(float(p["scores"][pidx]))
            cand_img.append(img_i)
            cand_pidx.append(int(pidx))
            cand_area.append(float(p["areas"][pidx]))

    n_cand = len(cand_scores)
    if n_cand == 0:
        return {
            "ap": 0.0,
            "num_gt": num_valid_gt,
            "scores": np.empty((0,), dtype=np.float32),
            "tp": np.empty((0,), dtype=np.float32),
            "fp": np.empty((0,), dtype=np.float32),
            "matched_angle_errs": np.empty((0,), dtype=np.float32),
            "matched_tp_scores": np.empty((0,), dtype=np.float32),
            "matched_is_elongated": np.empty((0,), dtype=bool),
        }

    order = np.argsort(-np.array(cand_scores, dtype=np.float64), kind="mergesort")
    tp = np.zeros(n_cand, dtype=np.float32)
    fp = np.zeros(n_cand, dtype=np.float32)
    matched_angle_errs: list[float] = []
    matched_tp_scores: list[float] = []
    matched_is_elongated: list[bool] = []

    for out_k, orig_k in enumerate(order):
        img_i = cand_img[orig_k]
        pidx = cand_pidx[orig_k]
        g = gts[img_i]
        g_labels = g["labels"]
        g_diff = g["difficult"]
        g_areas = g["areas"]

        cls_mask = np.ones(len(g_labels), dtype=bool) if target_class is None else (g_labels == target_class)
        cand_gt_idxs = np.where(cls_mask)[0]
        if len(cand_gt_idxs) == 0:
            if area_range is None or (area_range[0] <= cand_area[orig_k] < area_range[1]):
                fp[out_k] = 1.0
            continue

        ious = iou_mats[img_i][pidx, cand_gt_idxs]
        best_pos = int(np.argmax(ious))
        best_iou = float(ious[best_pos])
        best_gt = int(cand_gt_idxs[best_pos])

        if best_iou < iou_thr:
            if area_range is None or (area_range[0] <= cand_area[orig_k] < area_range[1]):
                fp[out_k] = 1.0
            continue

        is_ignored_gt = bool(g_diff[best_gt] != 0)
        if area_range is not None:
            gt_in_area = bool(area_range[0] <= g_areas[best_gt] < area_range[1])
            is_ignored_gt = is_ignored_gt or (not gt_in_area)

        if is_ignored_gt:
            continue

        if not gt_matched[img_i][best_gt]:
            gt_matched[img_i][best_gt] = True
            tp[out_k] = 1.0
            err = float(
                acute_angle_diff_deg(
                    np.array([preds[img_i]["angles"][pidx]]),
                    np.array([g["angles"][best_gt]]),
                )[0]
            )
            matched_angle_errs.append(err)
            matched_tp_scores.append(float(cand_scores[orig_k]))
            matched_is_elongated.append(bool(g["aspect_ratios"][best_gt] >= 1.5))
        else:
            if area_range is None or (area_range[0] <= cand_area[orig_k] < area_range[1]):
                fp[out_k] = 1.0

    sorted_scores = np.array([cand_scores[i] for i in order], dtype=np.float32)
    tp_cum = np.cumsum(tp)
    fp_cum = np.cumsum(fp)
    if num_valid_gt > 0:
        recalls = tp_cum / float(num_valid_gt)
        precisions = tp_cum / np.maximum(tp_cum + fp_cum, 1e-12)
        ap = compute_voc_allpoint_ap(recalls, precisions)
    else:
        ap = 0.0

    return {
        "ap": float(ap),
        "num_gt": num_valid_gt,
        "scores": sorted_scores,
        "tp": tp,
        "fp": fp,
        "matched_angle_errs": np.array(matched_angle_errs, dtype=np.float32),
        "matched_tp_scores": np.array(matched_tp_scores, dtype=np.float32),
        "matched_is_elongated": np.array(matched_is_elongated, dtype=bool),
    }


def _prf_from_sorted(
    scores: np.ndarray, tp: np.ndarray, fp: np.ndarray, total_gt: int
) -> dict[str, Any]:
    """Compute F1-optimal and fixed-threshold Precision, Recall, and F1."""
    if total_gt == 0 or len(scores) == 0:
        empty = {"precision": 0.0, "recall": 0.0, "f1": 0.0, "tp": 0, "fp": 0, "fn": total_gt}
        return {
            "f1_optimal": {**empty, "conf_threshold": 0.0},
            "at_conf": {f"{c:.2f}": dict(empty) for c in FIXED_CONF_THRESHOLDS},
        }

    tp_cum = np.cumsum(tp)
    fp_cum = np.cumsum(fp)
    precisions = tp_cum / np.maximum(tp_cum + fp_cum, 1e-12)
    recalls = tp_cum / float(total_gt)
    f1s = (2.0 * precisions * recalls) / np.maximum(precisions + recalls, 1e-12)

    best_idx = int(np.argmax(f1s))
    f1_opt = {
        "conf_threshold": float(scores[best_idx]),
        "precision": round(float(precisions[best_idx]), 4),
        "recall": round(float(recalls[best_idx]), 4),
        "f1": round(float(f1s[best_idx]), 4),
        "tp": int(tp_cum[best_idx]),
        "fp": int(fp_cum[best_idx]),
        "fn": int(total_gt - tp_cum[best_idx]),
    }

    at_conf: dict[str, dict[str, Any]] = {}
    for thr in FIXED_CONF_THRESHOLDS:
        keep = scores >= thr
        tp_c = int(np.sum(tp[keep]))
        fp_c = int(np.sum(fp[keep]))
        p_c = tp_c / max(tp_c + fp_c, 1)
        r_c = tp_c / total_gt
        f1_c = (2.0 * p_c * r_c) / max(p_c + r_c, 1e-12)
        at_conf[f"{thr:.2f}"] = {
            "precision": round(float(p_c), 4),
            "recall": round(float(r_c), 4),
            "f1": round(float(f1_c), 4),
            "tp": tp_c,
            "fp": fp_c,
            "fn": int(total_gt - tp_c),
        }

    return {"f1_optimal": f1_opt, "at_conf": at_conf}


def _compute_difficult_recall(
    preds: list[dict[str, np.ndarray]],
    gts: list[dict[str, np.ndarray]],
    iou_mats: list[np.ndarray],
    conf_thr: float,
    iou_thr: float = 0.50,
) -> dict[str, Any]:
    """Compare recall on easy (difficult=0) vs hard (difficult=1) GT instances at conf_thr."""
    easy_total, easy_hit = 0, 0
    diff_total, diff_hit = 0, 0
    for p, g, iou_mat in zip(preds, gts, iou_mats):
        keep_p = p["scores"] >= conf_thr
        p_labels = p["labels"][keep_p]
        sub_iou = iou_mat[keep_p] if len(iou_mat) > 0 else iou_mat

        for g_idx in range(len(g["labels"])):
            is_diff = bool(g["difficult"][g_idx] != 0)
            if is_diff:
                diff_total += 1
            else:
                easy_total += 1
            if len(p_labels) == 0:
                continue
            same_cls = p_labels == g["labels"][g_idx]
            if np.any(same_cls) and float(np.max(sub_iou[same_cls, g_idx])) >= iou_thr:
                if is_diff:
                    diff_hit += 1
                else:
                    easy_hit += 1

    return {
        "conf_threshold": round(conf_thr, 4),
        "easy_total": easy_total,
        "easy_recalled": easy_hit,
        "easy_recall": round(easy_hit / max(easy_total, 1), 4),
        "difficult_total": diff_total,
        "difficult_recalled": diff_hit,
        "difficult_recall": round(diff_hit / max(diff_total, 1), 4) if diff_total > 0 else None,
    }


def evaluate_predictions(
    samples: list[dict[str, Any]],
    ann_dir: Path | None = None,
    classes: tuple[str, ...] = DOTA_V2_CLASSES,
) -> dict[str, Any]:
    """Run complete OBB metric suite over a list of MMRotate prediction sample dicts."""
    class_to_idx = {name: i for i, name in enumerate(classes)}
    preds: list[dict[str, np.ndarray]] = []
    gts: list[dict[str, np.ndarray]] = []
    iou_mats: list[np.ndarray] = []

    for sample in samples:
        pi = sample.get("pred_instances") or {}
        p_boxes = _to_numpy(pi.get("bboxes")).astype(np.float32).reshape(-1, 5)
        p_scores = _to_numpy(pi.get("scores")).astype(np.float32).reshape(-1)
        p_labels = _to_numpy(pi.get("labels")).astype(np.int64).reshape(-1)
        p_areas = (p_boxes[:, 2] * p_boxes[:, 3]) if len(p_boxes) else np.empty((0,), dtype=np.float32)
        preds.append(
            {
                "rboxes": p_boxes,
                "scores": p_scores,
                "labels": p_labels,
                "areas": p_areas,
                "angles": long_edge_angle_deg(p_boxes),
            }
        )

        g_raw = _extract_gt_for_sample(sample, ann_dir, class_to_idx)
        g_boxes = g_raw["rboxes"]
        if len(g_boxes):
            g_areas = g_boxes[:, 2] * g_boxes[:, 3]
            ar = np.maximum(g_boxes[:, 2], g_boxes[:, 3]) / np.maximum(
                np.minimum(g_boxes[:, 2], g_boxes[:, 3]), 1e-3
            )
        else:
            g_areas = np.empty((0,), dtype=np.float32)
            ar = np.empty((0,), dtype=np.float32)

        gts.append(
            {
                "rboxes": g_boxes,
                "labels": g_raw["labels"],
                "difficult": g_raw["difficult"],
                "areas": g_areas,
                "aspect_ratios": ar,
                "angles": long_edge_angle_deg(g_boxes),
            }
        )
        iou_mats.append(compute_rotated_iou_matrix(p_boxes, g_boxes))

    # 1. Per-class AP across IoU 0.50:0.05:0.95
    num_classes = len(classes)
    ap_matrix = np.zeros((num_classes, len(IOU_THRESHOLDS)), dtype=np.float64)
    gt_counts = np.zeros(num_classes, dtype=np.int64)
    global_scores_50: list[np.ndarray] = []
    global_tp_50: list[np.ndarray] = []
    global_fp_50: list[np.ndarray] = []
    all_angle_errs: list[np.ndarray] = []
    all_tp_scores: list[np.ndarray] = []
    all_tp_elongated: list[np.ndarray] = []
    per_class_summary: dict[str, dict[str, Any]] = {}

    for c_idx, cls_name in enumerate(classes):
        res_50 = _match_predictions(preds, gts, iou_mats, iou_thr=0.50, target_class=c_idx)
        gt_counts[c_idx] = res_50["num_gt"]
        ap_matrix[c_idx, 0] = res_50["ap"]
        global_scores_50.append(res_50["scores"])
        global_tp_50.append(res_50["tp"])
        global_fp_50.append(res_50["fp"])
        if len(res_50["matched_angle_errs"]):
            all_angle_errs.append(res_50["matched_angle_errs"])
            all_tp_scores.append(res_50["matched_tp_scores"])
            all_tp_elongated.append(res_50["matched_is_elongated"])

        for t_idx, thr in enumerate(IOU_THRESHOLDS[1:], start=1):
            res_t = _match_predictions(preds, gts, iou_mats, iou_thr=float(thr), target_class=c_idx)
            ap_matrix[c_idx, t_idx] = res_t["ap"]

        cls_prf = _prf_from_sorted(res_50["scores"], res_50["tp"], res_50["fp"], res_50["num_gt"])
        per_class_summary[cls_name] = {
            "num_gt": int(gt_counts[c_idx]),
            "ap50": round(float(ap_matrix[c_idx, 0]), 4),
            "ap75": round(float(ap_matrix[c_idx, 5]), 4),
            "ap50_95": round(float(np.mean(ap_matrix[c_idx])), 4),
            "precision_f1_opt": cls_prf["f1_optimal"]["precision"],
            "recall_f1_opt": cls_prf["f1_optimal"]["recall"],
            "f1_opt": cls_prf["f1_optimal"]["f1"],
        }

    non_empty = gt_counts > 0
    map50 = float(np.mean(ap_matrix[non_empty, 0])) if np.any(non_empty) else 0.0
    ap75 = float(np.mean(ap_matrix[non_empty, 5])) if np.any(non_empty) else 0.0
    map50_95 = float(np.mean(ap_matrix[non_empty, :])) if np.any(non_empty) else 0.0

    # 2. Global (micro) Precision, Recall, F1 across all predictions at IoU=0.50
    cat_scores = np.concatenate(global_scores_50) if global_scores_50 else np.empty((0,), dtype=np.float32)
    cat_tp = np.concatenate(global_tp_50) if global_tp_50 else np.empty((0,), dtype=np.float32)
    cat_fp = np.concatenate(global_fp_50) if global_fp_50 else np.empty((0,), dtype=np.float32)
    if len(cat_scores):
        g_order = np.argsort(-cat_scores.astype(np.float64), kind="mergesort")
        cat_scores = cat_scores[g_order]
        cat_tp = cat_tp[g_order]
        cat_fp = cat_fp[g_order]
    total_gt = int(np.sum(gt_counts))
    micro_prf = _prf_from_sorted(cat_scores, cat_tp, cat_fp, total_gt)

    macro_p = float(np.mean([per_class_summary[c]["precision_f1_opt"] for i, c in enumerate(classes) if non_empty[i]])) if np.any(non_empty) else 0.0
    macro_r = float(np.mean([per_class_summary[c]["recall_f1_opt"] for i, c in enumerate(classes) if non_empty[i]])) if np.any(non_empty) else 0.0
    macro_f1 = float(np.mean([per_class_summary[c]["f1_opt"] for i, c in enumerate(classes) if non_empty[i]])) if np.any(non_empty) else 0.0

    # 3. Scale-bucketed AP50 (small, medium, large)
    scale_buckets = {
        "small": (0.0, SMALL_AREA_MAX),
        "medium": (SMALL_AREA_MAX, MEDIUM_AREA_MAX),
        "large": (MEDIUM_AREA_MAX, float("inf")),
    }
    scale_metrics: dict[str, dict[str, Any]] = {}
    for bucket_name, a_rng in scale_buckets.items():
        b_aps: list[float] = []
        b_gt = 0
        for c_idx in range(num_classes):
            r_b = _match_predictions(
                preds, gts, iou_mats, iou_thr=0.50, target_class=c_idx, area_range=a_rng
            )
            if r_b["num_gt"] > 0:
                b_aps.append(r_b["ap"])
                b_gt += r_b["num_gt"]
        scale_metrics[bucket_name] = {
            "num_gt": b_gt,
            "ap50": round(float(np.mean(b_aps)) if b_aps else 0.0, 4),
        }

    # 4. Class-agnostic localization AP50
    loc_res = _match_predictions(preds, gts, iou_mats, iou_thr=0.50, target_class=None)
    loc_ap50 = float(loc_res["ap"])

    # 5. Orientation angle error on TPs at >= F1-optimal confidence
    f1_conf = float(micro_prf["f1_optimal"]["conf_threshold"])
    micro_prf["f1_optimal"]["conf_threshold"] = round(f1_conf, 4)
    if all_angle_errs:
        raw_ang = np.concatenate(all_angle_errs)
        raw_scores = np.concatenate(all_tp_scores)
        raw_elong = np.concatenate(all_tp_elongated)
        conf_mask = raw_scores >= f1_conf
        flat_ang = raw_ang[conf_mask]
        flat_elong = raw_ang[conf_mask & raw_elong]
    else:
        flat_ang = np.empty((0,), dtype=np.float32)
        flat_elong = np.empty((0,), dtype=np.float32)

    # 6. Easy vs Difficult GT Recall at micro F1-optimal confidence
    diff_stats = _compute_difficult_recall(preds, gts, iou_mats, conf_thr=f1_conf, iou_thr=0.50)

    return {
        "num_images": len(samples),
        "num_gt_non_difficult": total_gt,
        "num_predictions": len(cat_scores),
        "map50": round(map50, 4),
        "ap75": round(ap75, 4),
        "map50_95": round(map50_95, 4),
        "class_agnostic_ap50": round(loc_ap50, 4),
        "classification_gap_ap50": round(max(0.0, loc_ap50 - map50), 4),
        "micro_prf": micro_prf,
        "macro_f1_opt": {
            "precision": round(macro_p, 4),
            "recall": round(macro_r, 4),
            "f1": round(macro_f1, 4),
        },
        "scale_ap50": scale_metrics,
        "orientation_error_deg": {
            "mean_all_tp": round(float(np.mean(flat_ang)), 2) if len(flat_ang) else None,
            "median_all_tp": round(float(np.median(flat_ang)), 2) if len(flat_ang) else None,
            "mean_elongated_tp": round(float(np.mean(flat_elong)), 2) if len(flat_elong) else None,
            "median_elongated_tp": round(float(np.median(flat_elong)), 2) if len(flat_elong) else None,
        },
        "difficulty_recall": diff_stats,
        "per_class": per_class_summary,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute extended DOTA v2.0 OBB metrics from preds.pkl.")
    parser.add_argument("--preds-pkl", type=Path, required=True, help="Output pkl from mmrotate tools/test.py --out")
    parser.add_argument("--ann-dir", type=Path, default=None, help="Directory of validation DOTA txt annfiles")
    parser.add_argument("--out-json", type=Path, required=True, help="Path to write extended metrics JSON")
    parser.add_argument("--fps", type=float, default=None, help="Measured validation throughput (img/s)")
    parser.add_argument("--params-m", type=float, default=None, help="Model parameter count in millions")
    parser.add_argument("--peak-vram-gb", type=float, default=None, help="Peak GPU memory in GiB")
    args = parser.parse_args()

    with args.preds_pkl.open("rb") as f:
        samples = pickle.load(f)

    metrics = evaluate_predictions(samples, ann_dir=args.ann_dir)
    if args.fps is not None or args.params_m is not None or args.peak_vram_gb is not None:
        metrics["operational"] = {
            "fps": round(args.fps, 2) if args.fps is not None else None,
            "params_m": round(args.params_m, 2) if args.params_m is not None else None,
            "peak_vram_gb": round(args.peak_vram_gb, 2) if args.peak_vram_gb is not None else None,
        }

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    f1o = metrics["micro_prf"]["f1_optimal"]
    print(
        f"[extended-eval] mAP50={metrics['map50']:.4f} | mAP50-95={metrics['map50_95']:.4f} | "
        f"AP75={metrics['ap75']:.4f} | P={f1o['precision']:.4f} R={f1o['recall']:.4f} "
        f"F1={f1o['f1']:.4f} (@conf={f1o['conf_threshold']})"
    )
    print(f"[extended-eval] Saved JSON to {args.out_json}")


if __name__ == "__main__":
    main()
