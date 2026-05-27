"""
Offline threshold calibration.

Replays Critic._verify_relation over a saved evidence dump (produced by
`evaluate.py --dump_evidence ...`) and grid-searches CriticConfig values
for the best per-relation F1. No model calls happen here — this is pure
replay, so a 200-cell grid over 200 items is essentially free.

Usage:
    python scripts/calibrate_thresholds.py \\
        --evidence results/whatsup200/evidence.json \\
        --report   results/whatsup200/calibration.json
"""

from __future__ import annotations

import argparse
import itertools
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path

# Make src/ importable.
_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from config import CriticConfig  # noqa: E402
from critic import _verify_relation  # noqa: E402
from state import BoundingBox  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger("calibrate")


DEFAULT_GRID = {
    "margin": [0.01, 0.02, 0.05, 0.10],
    "on_iou_threshold": [0.01, 0.05, 0.15, 0.30],
    "contains_coverage_threshold": [0.40, 0.55, 0.70, 0.85],
    "area_ratio_threshold": [0.30, 0.50, 0.70, 0.90],
}


def load_evidence(path: str) -> list[dict]:
    items = json.loads(Path(path).read_text())
    for it in items:
        for key in ("b1", "b2"):
            it[key] = BoundingBox(**it[key])
    return items


def score(items: list[dict], cfg: CriticConfig) -> dict:
    """
    Per-relation TP/FP/FN/TN counts plus overall accuracy + macro-F1.

    GT is "yes"/"no"; we evaluate the verifier's binary answer. Items missing
    depth are still scored (depth-only relations behave correctly because d1=d2
    means the relation cannot pass at any margin > 0 — that's a true negative).
    """
    per_rel: dict[str, dict[str, int]] = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0, "tn": 0})
    for it in items:
        d1 = it["d1"] if it["d1"] is not None else 0.5
        d2 = it["d2"] if it["d2"] is not None else 0.5
        passed, _ = _verify_relation(it["relation"], it["b1"], it["b2"], d1, d2, cfg)
        gt = it["gt"] == "yes"
        rel = it["relation"]
        bucket = per_rel[rel]
        if passed and gt:
            bucket["tp"] += 1
        elif passed and not gt:
            bucket["fp"] += 1
        elif not passed and gt:
            bucket["fn"] += 1
        else:
            bucket["tn"] += 1

    f1_per_rel: dict[str, float] = {}
    correct_total = 0
    total = 0
    for rel, b in per_rel.items():
        n = b["tp"] + b["fp"] + b["fn"] + b["tn"]
        correct = b["tp"] + b["tn"]
        precision = b["tp"] / max(b["tp"] + b["fp"], 1)
        recall = b["tp"] / max(b["tp"] + b["fn"], 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-9)
        f1_per_rel[rel] = round(f1, 4)
        correct_total += correct
        total += n

    macro_f1 = sum(f1_per_rel.values()) / max(len(f1_per_rel), 1)
    return {
        "accuracy": round(correct_total / max(total, 1), 4),
        "macro_f1": round(macro_f1, 4),
        "per_relation_f1": f1_per_rel,
        "per_relation_counts": {k: dict(v) for k, v in per_rel.items()},
    }


def grid_search(items: list[dict], grid: dict[str, list[float]], top_k: int = 10) -> list[dict]:
    keys = list(grid.keys())
    cells = []
    for values in itertools.product(*[grid[k] for k in keys]):
        overrides = dict(zip(keys, values))
        cfg = CriticConfig(**overrides)
        result = score(items, cfg)
        cells.append({"config": overrides, **result})
    cells.sort(key=lambda c: (c["macro_f1"], c["accuracy"]), reverse=True)
    return cells[:top_k]


def main():
    parser = argparse.ArgumentParser(description="Offline threshold calibration")
    parser.add_argument("--evidence", required=True, help="Evidence dump from evaluate.py --dump_evidence")
    parser.add_argument("--report", default="", help="Optional JSON path to save the top-k table")
    parser.add_argument("--top_k", type=int, default=10)
    args = parser.parse_args()

    items = load_evidence(args.evidence)
    logger.info(f"Loaded {len(items)} evidence items from {args.evidence}")

    baseline = score(items, CriticConfig())
    logger.info(f"Baseline (defaults): accuracy={baseline['accuracy']} macro_f1={baseline['macro_f1']}")
    for rel, f1 in baseline["per_relation_f1"].items():
        counts = baseline["per_relation_counts"][rel]
        logger.info(f"  {rel:12s} F1={f1:.3f}  TP={counts['tp']} FP={counts['fp']} FN={counts['fn']} TN={counts['tn']}")

    top = grid_search(items, DEFAULT_GRID, top_k=args.top_k)
    logger.info(f"\nTop {len(top)} configs by macro-F1:")
    for i, cell in enumerate(top, 1):
        logger.info(f"  #{i}  macro_f1={cell['macro_f1']:.4f}  acc={cell['accuracy']:.4f}  {cell['config']}")

    if args.report:
        Path(args.report).parent.mkdir(parents=True, exist_ok=True)
        Path(args.report).write_text(
            json.dumps(
                {"baseline": baseline, "top_k": top, "grid": DEFAULT_GRID},
                indent=2,
            )
        )
        logger.info(f"Report saved -> {args.report}")


if __name__ == "__main__":
    main()
