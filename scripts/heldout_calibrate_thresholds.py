"""
Held-out threshold calibration.

Splits a saved geometric evidence dump into deterministic dev/test folds,
selects CriticConfig thresholds on the dev fold, and reports the selected
configuration on the held-out test fold. This avoids tuning and reporting on
the same VSR-200 evidence items.

Usage:
    python scripts/heldout_calibrate_thresholds.py \
        --evidence results/vsr200_evidence_recal_k2.json \
        --report results/vsr200_heldout_calibration.json
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from collections import defaultdict
from pathlib import Path

# Make src/ and scripts/ importable when run from the repo root.
ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
SCRIPTS = ROOT / "scripts"
for path in (SRC, SCRIPTS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from calibrate_thresholds import DEFAULT_GRID, grid_search, load_evidence, score  # noqa: E402
from config import CriticConfig  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger("heldout_calibrate")


def stratified_split(items: list[dict], dev_fraction: float, seed: int) -> tuple[list[dict], list[dict]]:
    """Split by relation so dev/test keep a similar relation distribution."""
    grouped: dict[str, list[dict]] = defaultdict(list)
    for item in items:
        grouped[item["relation"]].append(item)

    rng = random.Random(seed)
    dev: list[dict] = []
    test: list[dict] = []
    for relation, group in sorted(grouped.items()):
        group = list(group)
        rng.shuffle(group)
        n_dev = max(1, min(len(group) - 1, round(len(group) * dev_fraction)))
        dev.extend(group[:n_dev])
        test.extend(group[n_dev:])
        logger.info(
            "%-12s dev=%2d test=%2d total=%2d",
            relation,
            n_dev,
            len(group) - n_dev,
            len(group),
        )

    rng.shuffle(dev)
    rng.shuffle(test)
    return dev, test


def main() -> None:
    parser = argparse.ArgumentParser(description="Held-out Critic threshold calibration")
    parser.add_argument("--evidence", required=True, help="Evidence dump from evaluate.py --dump_evidence")
    parser.add_argument("--report", default="", help="Optional JSON path to save calibration report")
    parser.add_argument("--dev_fraction", type=float, default=0.5, help="Fraction of each relation used for calibration")
    parser.add_argument("--seed", type=int, default=252, help="Deterministic split seed")
    parser.add_argument("--top_k", type=int, default=10, help="Number of dev configs to keep in the report")
    args = parser.parse_args()

    if not 0.0 < args.dev_fraction < 1.0:
        raise ValueError("--dev_fraction must be between 0 and 1")

    items = load_evidence(args.evidence)
    logger.info("Loaded %d evidence items from %s", len(items), args.evidence)
    dev, test = stratified_split(items, args.dev_fraction, args.seed)
    logger.info("Split summary: dev=%d test=%d seed=%d", len(dev), len(test), args.seed)

    default_cfg = CriticConfig()
    default_dev = score(dev, default_cfg)
    default_test = score(test, default_cfg)

    top_dev = grid_search(dev, DEFAULT_GRID, top_k=args.top_k)
    selected = top_dev[0]
    selected_cfg = CriticConfig(**selected["config"])
    selected_test = score(test, selected_cfg)

    logger.info(
        "Default dev : accuracy=%.4f macro_f1=%.4f",
        default_dev["accuracy"],
        default_dev["macro_f1"],
    )
    logger.info(
        "Default test: accuracy=%.4f macro_f1=%.4f",
        default_test["accuracy"],
        default_test["macro_f1"],
    )
    logger.info("Selected config from dev: %s", selected["config"])
    logger.info(
        "Selected dev : accuracy=%.4f macro_f1=%.4f",
        selected["accuracy"],
        selected["macro_f1"],
    )
    logger.info(
        "Selected test: accuracy=%.4f macro_f1=%.4f",
        selected_test["accuracy"],
        selected_test["macro_f1"],
    )

    report = {
        "evidence": args.evidence,
        "seed": args.seed,
        "dev_fraction": args.dev_fraction,
        "n_total": len(items),
        "n_dev": len(dev),
        "n_test": len(test),
        "grid": DEFAULT_GRID,
        "default": {
            "config": default_cfg.__dict__,
            "dev": default_dev,
            "test": default_test,
        },
        "selected": {
            "config": selected["config"],
            "dev": {k: v for k, v in selected.items() if k != "config"},
            "test": selected_test,
        },
        "top_dev": top_dev,
    }

    if args.report:
        out = Path(args.report)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2), encoding="utf-8")
        logger.info("Report saved -> %s", out)


if __name__ == "__main__":
    main()
