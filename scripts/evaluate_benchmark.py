"""Parse MMEngine/MMRotate evaluation metrics and generate a benchmark report."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from configs.dota_v2_classes import DOTA_V2_CLASSES


def _find_map_key(rec: dict) -> str | None:
    if "dota/mAP" in rec:
        return "dota/mAP"
    for k in rec:
        if k.endswith("/mAP") or k == "mAP":
            return k
    return None


def extract_best_metrics(work_dir: Path) -> dict[str, float]:
    """Find the best validation record (highest mAP) from MMEngine scalars.json."""
    scalar_files = sorted(work_dir.rglob("scalars.json"))
    if not scalar_files:
        raise FileNotFoundError(f"No scalars.json found under {work_dir}")

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
    if not best_record:
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
        lines.append(f"| Track A: Rotated Faster R-CNN (R50-FPN, 1x) | **{baseline['dota/mAP']:.4f}** | {step} |")
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
