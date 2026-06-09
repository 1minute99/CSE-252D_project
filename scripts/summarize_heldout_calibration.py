"""
Summarize held-out threshold calibration reports.

Reads one or more JSON files produced by heldout_calibrate_thresholds.py and
prints mean/std test metrics for the default CriticConfig versus thresholds
selected on the dev fold.

Usage:
    python scripts/summarize_heldout_calibration.py results/heldout_seed*.json
"""

from __future__ import annotations

import argparse
import glob
import json
import statistics
from pathlib import Path
from typing import Any


def _mean(values: list[float]) -> float:
    return statistics.mean(values) if values else 0.0


def _std(values: list[float]) -> float:
    return statistics.stdev(values) if len(values) > 1 else 0.0


def _load_reports(patterns: list[str]) -> list[dict[str, Any]]:
    paths: list[Path] = []
    for pattern in patterns:
        matches = sorted(Path(p) for p in glob.glob(pattern))
        if matches:
            paths.extend(matches)
        else:
            path = Path(pattern)
            if path.exists():
                paths.append(path)

    unique_paths = sorted(set(paths))
    if not unique_paths:
        raise FileNotFoundError("No held-out calibration reports matched the input patterns.")

    reports = []
    for path in unique_paths:
        report = json.loads(path.read_text(encoding="utf-8"))
        report["_path"] = str(path)
        reports.append(report)
    return reports


def _row(report: dict[str, Any]) -> dict[str, Any]:
    default_test = report["default"]["test"]
    selected_test = report["selected"]["test"]
    return {
        "path": report["_path"],
        "seed": report.get("seed"),
        "default_accuracy": float(default_test["accuracy"]),
        "selected_accuracy": float(selected_test["accuracy"]),
        "delta_accuracy": float(selected_test["accuracy"]) - float(default_test["accuracy"]),
        "default_macro_f1": float(default_test["macro_f1"]),
        "selected_macro_f1": float(selected_test["macro_f1"]),
        "delta_macro_f1": float(selected_test["macro_f1"]) - float(default_test["macro_f1"]),
        "selected_config": report["selected"]["config"],
    }


def summarize(reports: list[dict[str, Any]]) -> dict[str, Any]:
    rows = [_row(report) for report in reports]
    delta_acc = [row["delta_accuracy"] for row in rows]
    delta_f1 = [row["delta_macro_f1"] for row in rows]

    return {
        "n": len(rows),
        "rows": rows,
        "metrics": {
            "default_accuracy": {
                "mean": _mean([row["default_accuracy"] for row in rows]),
                "std": _std([row["default_accuracy"] for row in rows]),
            },
            "selected_accuracy": {
                "mean": _mean([row["selected_accuracy"] for row in rows]),
                "std": _std([row["selected_accuracy"] for row in rows]),
            },
            "delta_accuracy": {
                "mean": _mean(delta_acc),
                "std": _std(delta_acc),
                "wins": sum(1 for value in delta_acc if value > 0),
                "losses": sum(1 for value in delta_acc if value < 0),
                "ties": sum(1 for value in delta_acc if value == 0),
            },
            "default_macro_f1": {
                "mean": _mean([row["default_macro_f1"] for row in rows]),
                "std": _std([row["default_macro_f1"] for row in rows]),
            },
            "selected_macro_f1": {
                "mean": _mean([row["selected_macro_f1"] for row in rows]),
                "std": _std([row["selected_macro_f1"] for row in rows]),
            },
            "delta_macro_f1": {
                "mean": _mean(delta_f1),
                "std": _std(delta_f1),
                "wins": sum(1 for value in delta_f1 if value > 0),
                "losses": sum(1 for value in delta_f1 if value < 0),
                "ties": sum(1 for value in delta_f1 if value == 0),
            },
        },
    }


def _fmt(value: float) -> str:
    return f"{value:.4f}"


def print_summary(summary: dict[str, Any]) -> None:
    print(f"Held-out calibration reports: {summary['n']}")
    print()
    print("seed  default_acc  selected_acc  delta_acc  default_f1  selected_f1  delta_f1")
    for row in summary["rows"]:
        seed = row["seed"] if row["seed"] is not None else "-"
        print(
            f"{seed!s:>4}  "
            f"{_fmt(row['default_accuracy']):>11}  "
            f"{_fmt(row['selected_accuracy']):>12}  "
            f"{_fmt(row['delta_accuracy']):>9}  "
            f"{_fmt(row['default_macro_f1']):>10}  "
            f"{_fmt(row['selected_macro_f1']):>11}  "
            f"{_fmt(row['delta_macro_f1']):>8}"
        )

    metrics = summary["metrics"]
    print()
    print("Aggregate test metrics")
    print(
        "  accuracy: "
        f"default={_fmt(metrics['default_accuracy']['mean'])}±{_fmt(metrics['default_accuracy']['std'])}, "
        f"selected={_fmt(metrics['selected_accuracy']['mean'])}±{_fmt(metrics['selected_accuracy']['std'])}, "
        f"delta={_fmt(metrics['delta_accuracy']['mean'])}±{_fmt(metrics['delta_accuracy']['std'])} "
        f"(wins={metrics['delta_accuracy']['wins']}, losses={metrics['delta_accuracy']['losses']})"
    )
    print(
        "  macro-F1: "
        f"default={_fmt(metrics['default_macro_f1']['mean'])}±{_fmt(metrics['default_macro_f1']['std'])}, "
        f"selected={_fmt(metrics['selected_macro_f1']['mean'])}±{_fmt(metrics['selected_macro_f1']['std'])}, "
        f"delta={_fmt(metrics['delta_macro_f1']['mean'])}±{_fmt(metrics['delta_macro_f1']['std'])} "
        f"(wins={metrics['delta_macro_f1']['wins']}, losses={metrics['delta_macro_f1']['losses']})"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize held-out calibration reports")
    parser.add_argument(
        "reports",
        nargs="*",
        default=["results/heldout_seed*.json"],
        help="Report JSON files or glob patterns.",
    )
    parser.add_argument("--json_out", default="", help="Optional path to save the aggregate summary JSON.")
    args = parser.parse_args()

    reports = _load_reports(args.reports)
    summary = summarize(reports)
    print_summary(summary)

    if args.json_out:
        out = Path(args.json_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"\nSummary saved -> {out}")


if __name__ == "__main__":
    main()
