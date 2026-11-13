"""Parse MMEngine/MMRotate evaluation metrics and generate a benchmark report."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from configs.dota_v2_classes import DOTA_V2_CLASSES

_TABLE_ROW_RE = re.compile(
    r"^\|\s*([a-z0-9-]+)\s*\|\s*\d+\s*\|\s*\d+\s*\|\s*([0-9.]+)\s*\|\s*([0-9.]+)\s*\|$"
)
_VAL_SUMMARY_RE = re.compile(r"Epoch\(val\)\s+\[(\d+)\].*?dota/mAP:\s*([0-9.]+)")


def _find_map_key(rec: dict) -> str | None:
    if "dota/mAP" in rec:
        return "dota/mAP"
    for k in rec:
        if k.endswith("/mAP") or k == "mAP":
            return k
    return None


def _parse_log_per_class_aps(work_dir: Path) -> tuple[float, int, dict[str, float]]:
    """Parse MMRotate DOTAMetric PrettyTable rows from MMEngine .log files."""
    log_files = sorted(work_dir.rglob("*.log"))
    best_map = -1.0
    best_epoch = 0
    best_aps: dict[str, float] = {}
    valid_classes = set(DOTA_V2_CLASSES)

    for lf in log_files:
        current_aps: dict[str, float] = {}
        for raw_line in lf.read_text(encoding="utf-8", errors="replace").splitlines():
            line = raw_line.strip()
            row_match = _TABLE_ROW_RE.match(line)
            if row_match:
                cls_name, _recall, ap = row_match.groups()
                if cls_name in valid_classes:
                    current_aps[f"dota/{cls_name}_AP"] = float(ap)
                continue

            val_match = _VAL_SUMMARY_RE.search(line)
            if val_match:
                epoch = int(val_match.group(1))
                m_ap = float(val_match.group(2))
                if m_ap > best_map:
                    best_map = m_ap
                    best_epoch = epoch
                    best_aps = dict(current_aps)
                current_aps.clear()

    return best_map, best_epoch, best_aps


def extract_best_metrics(work_dir: Path) -> dict[str, float]:
    """Find best validation record (mAP + per-class APs) from scalars.json and .log."""
    scalar_files = sorted(work_dir.rglob("scalars.json"))
    best_record: dict[str, float] = {}
    best_map = -1.0

    for sf in scalar_files:
        for line in sf.read_text(encoding="utf-8", errors="replace").splitlines():
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            map_key = _find_map_key(rec)
            if map_key and float(rec[map_key]) > best_map:
                best_map = float(rec[map_key])
                best_record = {
                    k: float(v) for k, v in rec.items() if isinstance(v, (int, float))
                }
                best_record["dota/mAP"] = best_map

    log_map, log_epoch, log_aps = _parse_log_per_class_aps(work_dir)
    if log_aps:
        for k, v in log_aps.items():
            best_record.setdefault(k, v)
    if log_map > best_map:
        best_record["dota/mAP"] = log_map
        best_record["step"] = float(log_epoch)

    if "dota/mAP" not in best_record:
        raise ValueError(f"No mAP records found in {work_dir}")
    return best_record


def render_markdown_report(
    baseline: dict[str, float] | None,
    flagship: dict[str, float] | None,
) -> str:
    """Format a Markdown comparison report for DOTA v2.0 val mAP and per-class AP."""
    lines = [
        "# DOTA v2.0 Benchmark Report (`val` split, OBB `le90`)",
        "",
        "| Model Track | Overall `dota/mAP` | Best Step / Epoch |",
        "| :--- | :---: | :---: |",
    ]
    if baseline:
        step = int(baseline.get("step", 0))
        lines.append(
            f"| Track A: Rotated Faster R-CNN (R50-FPN, 1x) | **{baseline['dota/mAP']:.4f}** | {step} |"
        )
    if flagship:
        step = int(flagship.get("step", 0))
        lines.append(
            f"| Track B: Oriented Faster R-CNN (ConvNeXt-S + PAFPN + KLD, 2x) | **{flagship['dota/mAP']:.4f}** | {step} |"
        )
    if baseline and flagship:
        delta = flagship["dota/mAP"] - baseline["dota/mAP"]
        lines.append(f"| **Delta (Track B - Track A)** | **`{delta:+.4f}`** | — |")

    lines.extend(["", "## Per-Class AP@0.5 Breakdown", ""])
    lines.append("| Class | Track A (Baseline) | Track B (Flagship) | Delta |")
    lines.append("| :--- | :---: | :---: | :---: |")

    for cls_name in DOTA_V2_CLASSES:
        key = f"dota/{cls_name}_AP"
        a_val = baseline.get(key) if baseline else None
        b_val = flagship.get(key) if flagship else None
        a_str = f"{a_val:.4f}" if a_val is not None else "—"
        b_str = f"{b_val:.4f}" if b_val is not None else "—"
        d_str = f"{(b_val - a_val):+.4f}" if (a_val is not None and b_val is not None) else "—"
        lines.append(f"| `{cls_name}` | {a_str} | {b_str} | {d_str} |")

    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate DOTA v2.0 benchmark Markdown report.")
    parser.add_argument("--baseline-dir", type=Path, default=None)
    parser.add_argument("--flagship-dir", type=Path, default=None)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("benchmark_report.md"),
    )
    args = parser.parse_args()

    baseline = extract_best_metrics(args.baseline_dir) if args.baseline_dir else None
    flagship = extract_best_metrics(args.flagship_dir) if args.flagship_dir else None

    if baseline is None and flagship is None:
        raise ValueError("Provide at least one of --baseline-dir or --flagship-dir")

    report = render_markdown_report(baseline, flagship)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(report, encoding="utf-8")
    print(report)
    print(f"[report] Saved to {args.out}")


if __name__ == "__main__":
    main()
