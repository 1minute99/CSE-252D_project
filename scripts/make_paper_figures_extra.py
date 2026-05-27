"""
Additional publication figures for the CSE 252D final paper.

All computed from data already on disk (no new evals):
  results/vsr200_risk_k2.json        full-coverage run, per-item signals
  results/ablation/executor_only.json VLM baseline (per-item gt/pred/relation)
  data/vsr_strat200.json             relation per idx

Figures (PNG + PDF to results/figures/paper/):
  fig6_confidence_informativeness  accuracy vs confidence bin, geo vs VLM signal
  fig7_per_relation_final          full-coverage system vs VLM baseline, per relation
  fig8_geoconf_by_relation         mean geometric confidence per relation (why depth defers)
  fig9_confusion                   yes/no confusion matrices, system vs baseline
  fig10_source_composition         answer-source composition (correct vs wrong)
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
OUT = RES / "figures" / "paper"

C_SYS = "#10b981"     # system (green)
C_BASE = "#6b7280"    # baseline (gray)
C_GEO = "#f59e0b"     # geometric (amber)
C_EXE = "#3b82f6"     # executor/VLM (blue)
DEPTH_RELS = {"behind", "in_front"}

plt.rcParams.update({
    "font.size": 12, "axes.titlesize": 13, "axes.labelsize": 12,
    "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 200,
})


def load(p):
    return json.loads((RES / p).read_text())["results"]


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{name}.png", bbox_inches="tight")
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT / name}.png / .pdf")


def relation_map():
    split = json.loads((ROOT / "data/vsr_strat200.json").read_text())
    return {i: split[i]["relation"] for i in range(len(split))}


# ── fig6: confidence informativeness (does confidence predict correctness?) ──

def equal_count_bins(items, key, nbins=4):
    vals = sorted(items, key=lambda r: r.get(key) or 0.0)
    n = len(vals)
    out = []
    for b in range(nbins):
        lo = b * n // nbins
        hi = (b + 1) * n // nbins
        chunk = vals[lo:hi]
        if not chunk:
            continue
        mean_conf = np.mean([(r.get(key) or 0.0) for r in chunk])
        acc = np.mean([1 if r.get("correct") else 0 for r in chunk])
        out.append((mean_conf, acc, len(chunk)))
    return out


def fig6_informativeness(risk):
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.3), sharey=True)
    for ax, key, color, title in [
        (axes[0], "geo_confidence", C_GEO, "Geometric confidence"),
        (axes[1], "executor_confidence", C_EXE, "VLM self-confidence"),
    ]:
        bins = equal_count_bins(risk, key, nbins=4)
        xs = [b[0] for b in bins]
        ys = [b[1] for b in bins]
        ns = [b[2] for b in bins]
        ax.plot(xs, ys, "-o", color=color, lw=2.2, ms=7)
        for x, y, n in zip(xs, ys, ns):
            ax.annotate(f"n={n}", (x, y), textcoords="offset points", xytext=(0, 8),
                        ha="center", fontsize=9)
        ax.set_title(title)
        ax.set_xlabel("mean confidence in bin")
        ax.grid(True, alpha=0.25)
        ax.set_ylim(0.55, 0.95)
    axes[0].set_ylabel("Empirical accuracy")
    fig.suptitle("Does confidence predict correctness? (equal-count bins)", y=1.02)
    save(fig, "paper_fig6_confidence_informativeness")


# ── fig7: per-relation, full-coverage system vs VLM baseline ─────────────────

def per_relation_acc(items, relmap):
    b = defaultdict(lambda: {"c": 0, "n": 0})
    for r in items:
        if r.get("error"):
            continue
        rel = r.get("relation") or relmap.get(r["idx"])
        if not rel:
            continue
        b[rel]["n"] += 1
        b[rel]["c"] += 1 if r.get("correct") else 0
    return {k: v["c"] / v["n"] for k, v in b.items()}


def fig7_per_relation(risk, baseline, relmap):
    sys_acc = per_relation_acc(risk, relmap)
    base_acc = per_relation_acc(baseline, relmap)
    rels = sorted(set(sys_acc) | set(base_acc))
    x = np.arange(len(rels))
    w = 0.38
    fig, ax = plt.subplots(figsize=(8.2, 4.3))
    b1 = ax.bar(x - w / 2, [sys_acc.get(r, 0) for r in rels], w, label="Full pipeline (full coverage)", color=C_SYS)
    b2 = ax.bar(x + w / 2, [base_acc.get(r, 0) for r in rels], w, label="GPT-4o baseline", color=C_BASE)
    ax.set_xticks(x)
    ax.set_xticklabels(rels, rotation=20, ha="right")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1.0)
    ax.axhline(0.5, color="k", ls=":", lw=0.8, alpha=0.4)
    ax.set_title("Per-relation accuracy: full-coverage system vs. VLM baseline")
    ax.legend(loc="lower right", fontsize=9)
    save(fig, "paper_fig7_per_relation_final")


# ── fig8: mean geometric confidence per relation ─────────────────────────────

def fig8_geoconf_by_relation(risk, relmap):
    b = defaultdict(list)
    for r in risk:
        rel = relmap.get(r["idx"])
        if rel:
            b[rel].append(r.get("geo_confidence") or 0.0)
    rels = sorted(b, key=lambda r: np.mean(b[r]))
    means = [np.mean(b[r]) for r in rels]
    colors = [C_EXE if r in DEPTH_RELS else C_GEO for r in rels]
    fig, ax = plt.subplots(figsize=(8.2, 4.3))
    bars = ax.bar(range(len(rels)), means, color=colors, width=0.62)
    ax.axhline(0.40, color="k", ls="--", lw=1.3)
    ax.text(len(rels) - 0.5, 0.41, "arbitration threshold τ=0.40", ha="right", va="bottom", fontsize=9)
    ax.set_xticks(range(len(rels)))
    ax.set_xticklabels(rels, rotation=20, ha="right")
    ax.set_ylabel("Mean geometric confidence")
    ax.set_title("Geometric confidence by relation (depth relations in blue)")
    save(fig, "paper_fig8_geoconf_by_relation")


# ── fig9: confusion matrices ─────────────────────────────────────────────────

def confusion(items):
    # rows = actual (yes,no), cols = predicted (yes,no)
    m = np.zeros((2, 2), dtype=int)
    idx = {"yes": 0, "no": 1}
    for r in items:
        g = r.get("gt"); p = r.get("pred")
        if g in idx and p in idx:
            m[idx[g], idx[p]] += 1
    return m


def _draw_cm(ax, m, title):
    ax.imshow(m, cmap="Blues")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["pred yes", "pred no"])
    ax.set_yticks([0, 1]); ax.set_yticklabels(["actual yes", "actual no"])
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(m[i, j]), ha="center", va="center",
                    color="white" if m[i, j] > m.max() / 2 else "black", fontsize=14)
    ax.set_title(title)


def fig9_confusion(risk, baseline):
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 4.0))
    _draw_cm(axes[0], confusion(risk), "Full pipeline (full coverage)")
    _draw_cm(axes[1], confusion(baseline), "GPT-4o baseline")
    fig.suptitle("Yes/No confusion: error structure", y=1.0)
    save(fig, "paper_fig9_confusion")


# ── fig10: answer-source composition ─────────────────────────────────────────

def fig10_source_composition(risk):
    order = ["agreement", "vlm_deferred", "geometry_override", "vlm_fallback"]
    pretty = {"agreement": "agreement", "vlm_deferred": "VLM deferred",
              "geometry_override": "geometry override", "vlm_fallback": "VLM fallback"}
    corr = defaultdict(int); wrong = defaultdict(int)
    for r in risk:
        s = r.get("answer_source", "?")
        if r.get("correct"):
            corr[s] += 1
        else:
            wrong[s] += 1
    keys = [k for k in order if (corr[k] + wrong[k]) > 0]
    c = [corr[k] for k in keys]
    w = [wrong[k] for k in keys]
    fig, ax = plt.subplots(figsize=(8.0, 4.3))
    ax.bar(range(len(keys)), c, color=C_SYS, label="correct")
    ax.bar(range(len(keys)), w, bottom=c, color="#fb7185", label="incorrect")
    for i, k in enumerate(keys):
        tot = corr[k] + wrong[k]
        ax.text(i, tot + 1.5, f"n={tot}\n{corr[k]/tot:.0%}", ha="center", va="bottom", fontsize=9)
    ax.set_xticks(range(len(keys)))
    ax.set_xticklabels([pretty[k] for k in keys])
    ax.set_ylabel("Number of items")
    ax.set_ylim(0, max(corr[k] + wrong[k] for k in keys) * 1.25)
    ax.set_title("How the 200 answers are produced (and where errors fall)")
    ax.legend(loc="upper right", fontsize=10)
    save(fig, "paper_fig10_source_composition")


def main():
    risk = load("vsr200_fullcoverage_recal_k2.json")
    baseline = load("ablation/executor_only.json")
    relmap = relation_map()

    fig6_informativeness(risk)
    fig7_per_relation(risk, baseline, relmap)
    fig8_geoconf_by_relation(risk, relmap)
    fig9_confusion(risk, baseline)
    fig10_source_composition(risk)


if __name__ == "__main__":
    main()
