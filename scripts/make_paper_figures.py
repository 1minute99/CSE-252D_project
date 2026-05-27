"""
Publication-quality figures for the CSE 252D final paper.

Reads the real result JSONs (no hardcoded numbers) and writes PNG + PDF to
results/figures/paper/. Tells the "with vs without abstain" story using both
operating points plus the improvement journey.

Inputs (all under results/):
  ablation/executor_only.json            VLM baseline (full coverage)
  ablation/full_k2.json                  original abstain-on-disagree (k=2)
  vsr200_arbitration_depthcap_k2.json    selective (arbitration + depth-cap)
  vsr200_fullcoverage_k2.json            full coverage (+ VLM fallback)

Usage:
  python scripts/make_paper_figures.py
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
OUT = RES / "figures" / "paper"

# Consistent, print-friendly palette.
C_BASELINE = "#6b7280"   # gray
C_OLD = "#fb7185"        # red  (broken / original)
C_SELECTIVE = "#3b82f6"  # blue (with abstain)
C_FULL = "#10b981"       # green (without abstain / full coverage)
C_ACCENT = "#f59e0b"     # amber

plt.rcParams.update({
    "font.size": 12,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 200,
})


def load(rel: str) -> dict:
    return json.loads((RES / rel).read_text())


def summary(d: dict) -> dict:
    return d["summary"]


def source_breakdown(d: dict) -> dict[str, dict]:
    b = defaultdict(lambda: {"correct": 0, "n": 0})
    for r in d["results"]:
        if r.get("abstain"):
            key = "abstain"
        else:
            key = r.get("answer_source", "?")
        b[key]["n"] += 1
        if r.get("correct"):
            b[key]["correct"] += 1
    return {k: {"n": v["n"], "acc": v["correct"] / v["n"] if v["n"] else 0.0} for k, v in b.items()}


def _annotate_bars(ax, bars, fmt="{:.3f}", dy=0.012):
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + dy, fmt.format(h),
                ha="center", va="bottom", fontsize=10)


def _save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{name}.png", bbox_inches="tight")
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT / name}.png / .pdf")


# ── Figure 1: improvement journey ────────────────────────────────────────────

def fig_journey(baseline, old, arbitration, selective, full):
    labels = [
        "Original\n(abstain on\ndisagree)",
        "+ Confidence\narbitration",
        "+ Depth\ndefers to VLM\n(selective)",
        "+ VLM fallback\n(full coverage)",
    ]
    # accuracy at each stage (raw accuracy, comparable across all)
    accs = [
        summary(old)["accuracy"],            # 0.49
        summary(arbitration)["accuracy"],     # 0.655 (arbitration, no depth-cap)
        summary(selective)["accuracy"],       # 0.69
        summary(full)["accuracy"],            # 0.77
    ]
    colors = [C_OLD, C_ACCENT, C_SELECTIVE, C_FULL]

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    bars = ax.bar(range(len(labels)), accs, color=colors, width=0.62)
    _annotate_bars(ax, bars)
    base = summary(baseline)["accuracy"]
    ax.axhline(base, color=C_BASELINE, ls="--", lw=1.6)
    ax.text(len(labels) - 0.5, base + 0.008, f"GPT-4o baseline {base:.3f}",
            ha="right", va="bottom", color=C_BASELINE, fontsize=10)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_ylabel("Accuracy (VSR-200, full denominator)")
    ax.set_ylim(0, 0.85)
    ax.set_title("Fixing the pipeline: accuracy across policy changes")
    _save(fig, "paper_fig1_journey")


# ── Figure 2: with vs without abstain, vs baseline ───────────────────────────

def fig_with_without(baseline, selective, full):
    groups = ["GPT-4o\nbaseline", "Selective\n(with abstain)", "Full coverage\n(no abstain)"]
    acc = [summary(baseline)["accuracy"], summary(selective)["accuracy"], summary(full)["accuracy"]]
    sel = [summary(baseline)["selective_accuracy"], summary(selective)["selective_accuracy"], summary(full)["selective_accuracy"]]
    cov = [summary(baseline)["coverage"], summary(selective)["coverage"], summary(full)["coverage"]]

    x = range(len(groups))
    w = 0.26
    fig, ax = plt.subplots(figsize=(7.6, 4.4))
    b1 = ax.bar([i - w for i in x], acc, w, label="Accuracy (all items)", color=C_SELECTIVE)
    b2 = ax.bar([i for i in x], sel, w, label="Selective accuracy (answered only)", color=C_FULL)
    b3 = ax.bar([i + w for i in x], cov, w, label="Coverage", color=C_BASELINE, alpha=0.7)
    for b in (b1, b2, b3):
        _annotate_bars(ax, b, dy=0.008)
    ax.axhline(summary(baseline)["accuracy"], color=C_OLD, ls=":", lw=1.4)
    ax.set_xticks(list(x))
    ax.set_xticklabels(groups)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Rate")
    ax.set_title("With vs. without abstain — both beat the VLM on their metric")
    ax.legend(loc="lower center", fontsize=9, ncol=1, framealpha=0.9)
    _save(fig, "paper_fig2_with_without_abstain")


# ── Figure 3: answer-source attribution ──────────────────────────────────────

def fig_answer_source(selective, full):
    sel_b = source_breakdown(selective)
    full_b = source_breakdown(full)
    order = ["agreement", "vlm_deferred", "geometry_override", "vlm_fallback", "abstain"]
    pretty = {
        "agreement": "agreement",
        "vlm_deferred": "VLM\ndeferred",
        "geometry_override": "geometry\noverride",
        "vlm_fallback": "VLM fallback\n(det. miss)",
        "abstain": "abstain",
    }

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.3), sharey=True)
    for ax, b, title, color in [
        (axes[0], sel_b, "Selective (with abstain)", C_SELECTIVE),
        (axes[1], full_b, "Full coverage (no abstain)", C_FULL),
    ]:
        keys = [k for k in order if k in b]
        accs = [b[k]["acc"] for k in keys]
        ns = [b[k]["n"] for k in keys]
        bars = ax.bar(range(len(keys)), accs, color=color, width=0.62)
        for i, (bar, n) in enumerate(zip(bars, ns)):
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.012, f"{h:.2f}\n(n={n})",
                    ha="center", va="bottom", fontsize=9)
        ax.set_xticks(range(len(keys)))
        ax.set_xticklabels([pretty[k] for k in keys], fontsize=9)
        ax.set_title(title)
        ax.set_ylim(0, 1.0)
    axes[0].set_ylabel("Accuracy within bucket")
    fig.suptitle("How answers are produced: the Critic acts as a verifier + router", y=1.02)
    _save(fig, "paper_fig3_answer_source")


# ── Figure 4: accuracy vs coverage (operating points) ────────────────────────

def fig_risk_coverage(baseline, old, selective, full):
    # (coverage, selective_accuracy, label, color)
    pts = [
        (summary(baseline)["coverage"], summary(baseline)["selective_accuracy"], "GPT-4o baseline", C_BASELINE),
        (summary(old)["coverage"], summary(old)["selective_accuracy"], "Original (abstain)", C_OLD),
        (summary(selective)["coverage"], summary(selective)["selective_accuracy"], "Selective", C_SELECTIVE),
        (summary(full)["coverage"], summary(full)["selective_accuracy"], "Full coverage", C_FULL),
    ]
    fig, ax = plt.subplots(figsize=(6.6, 4.4))
    for cov, acc, label, color in pts:
        ax.scatter(cov, acc, s=110, color=color, zorder=3, edgecolor="white", linewidth=1)
        ax.annotate(f"{label}\n({cov:.2f}, {acc:.3f})", (cov, acc),
                    textcoords="offset points", xytext=(8, 8), fontsize=9)
    ax.axhline(summary(baseline)["selective_accuracy"], color=C_BASELINE, ls="--", lw=1.2, alpha=0.7)
    ax.set_xlabel("Coverage (fraction of items answered)")
    ax.set_ylabel("Selective accuracy (accuracy on answered items)")
    ax.set_xlim(0.55, 1.05)
    ax.set_ylim(0.70, 0.80)
    ax.set_title("Accuracy vs. coverage operating points")
    ax.grid(True, alpha=0.25)
    _save(fig, "paper_fig4_accuracy_vs_coverage")


def main():
    baseline = load("ablation/executor_only.json")
    old = load("ablation/full_k2.json")
    arbitration = load("vsr200_arbitration_k2.json")
    selective = load("vsr200_arbitration_depthcap_k2.json")
    full = load("vsr200_fullcoverage_k2.json")

    fig_journey(baseline, old, arbitration, selective, full)
    fig_with_without(baseline, selective, full)
    fig_answer_source(selective, full)
    fig_risk_coverage(baseline, old, selective, full)


if __name__ == "__main__":
    main()
