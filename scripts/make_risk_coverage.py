"""
Risk-coverage analysis: is our abstention actually selective?

A good selective predictor, when it abstains, removes the items it would most
likely get wrong — so accuracy on the retained items rises as coverage drops.
We test several per-item confidence signals by sorting items most-confident
first, sweeping the abstain threshold, and plotting selective accuracy vs
coverage. A signal that ranks errors well produces a curve that climbs as
coverage falls; a useless signal stays flat (= random abstention).

Input: results/vsr200_risk_k2.json (full-coverage run with per-item
executor_confidence, geo_confidence, executor_agreed, correct).

Usage:
  python scripts/make_risk_coverage.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
OUT = RES / "figures" / "paper"

plt.rcParams.update({
    "font.size": 12, "axes.titlesize": 13, "axes.labelsize": 12,
    "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 200,
})


def load_items(path: Path) -> list[dict]:
    data = json.loads(path.read_text())
    return [r for r in data["results"] if "error" not in r]


def signal_executor(r: dict) -> float:
    return float(r.get("executor_confidence") or 0.0)


def signal_geo(r: dict) -> float:
    return float(r.get("geo_confidence") or 0.0)


def signal_combined(r: dict) -> float:
    """Agreement is the strongest correctness cue: when the VLM and geometry
    agree, trust it most; otherwise fall back to the VLM's own confidence,
    discounted by how strongly geometry contradicted it."""
    ec = signal_executor(r)
    gc = signal_geo(r)
    agreed = r.get("executor_agreed")
    if agreed is True:
        return 0.5 + 0.5 * ec            # high band: agreement
    if agreed is False:
        return 0.5 * ec * (1.0 - gc)     # low band: contested
    return 0.4 * ec                       # no geometry (detector miss)


SIGNALS = {
    "executor confidence": signal_executor,
    "geometric confidence": signal_geo,
    "combined (agreement-aware)": signal_combined,
}


def risk_coverage_curve(items: list[dict], signal_fn) -> tuple[list[float], list[float]]:
    """Return (coverage, selective_accuracy) at a fixed coverage grid.

    Items are sorted most-confident first; at each target coverage we keep that
    top fraction and report its accuracy. Evaluating on a 5%-step grid (rather
    than per-item) gives smooth, readable curves.
    """
    scored = sorted(items, key=signal_fn, reverse=True)
    n = len(scored)
    # cumulative correct
    cum = []
    c = 0
    for r in scored:
        c += 1 if r.get("correct") else 0
        cum.append(c)
    covs, accs = [], []
    cov = 1.0
    while cov >= 0.30 - 1e-9:
        k = max(1, round(cov * n))
        covs.append(k / n)
        accs.append(cum[k - 1] / k)
        cov -= 0.05
    return covs, accs


def aurc(covs: list[float], accs: list[float]) -> float:
    """Mean selective accuracy across the swept coverage range (higher=better)."""
    return sum(accs) / max(len(accs), 1)


def main():
    items = load_items(RES / "vsr200_fullcoverage_recal_k2.json")
    full_acc = sum(1 for r in items if r.get("correct")) / len(items)
    print(f"Loaded {len(items)} items; full-coverage accuracy = {full_acc:.3f}")

    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    colors = {"executor confidence": "#3b82f6", "geometric confidence": "#f59e0b",
              "combined (agreement-aware)": "#10b981"}

    for name, fn in SIGNALS.items():
        covs, accs = risk_coverage_curve(items, fn)
        score = aurc(covs, accs)
        ax.plot(covs, accs, "-o", lw=2.2, ms=4, color=colors[name],
                label=f"{name} (mean={score:.3f})")
        print(f"  {name:28s} mean selective acc over sweep = {score:.3f}")

    # Random-abstention reference: removing a random subset leaves accuracy
    # unchanged in expectation.
    ax.axhline(full_acc, color="#6b7280", ls="--", lw=1.5,
               label=f"random abstention ({full_acc:.3f})")

    ax.set_xlabel("Coverage (fraction of items answered)")
    ax.set_ylabel("Selective accuracy (on answered items)")
    ax.set_title("Risk–coverage: does confidence-based abstention help?")
    ax.set_xlim(1.02, 0.18)  # reversed: full coverage on the left, more abstain to the right
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper right", fontsize=9, framealpha=0.92)
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "paper_fig5_risk_coverage.png", bbox_inches="tight")
    fig.savefig(OUT / "paper_fig5_risk_coverage.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT / 'paper_fig5_risk_coverage.png'} / .pdf")


if __name__ == "__main__":
    main()
