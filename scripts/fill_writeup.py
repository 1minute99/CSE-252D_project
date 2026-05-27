"""
Post-ablation: substitute real numbers into results/writeup_results.md.

Reads results/ablation/*.json, computes per-relation accuracy by joining
results to the split JSON, and writes results/writeup_results_filled.md
with the [FILL] placeholders replaced.

Usage:
  python scripts/fill_writeup.py \\
      --ablation_dir results/ablation \\
      --split data/vsr_strat200.json \\
      --writeup results/writeup_results.md \\
      --out results/writeup_results_filled.md
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


def load_cell(d: Path, name: str) -> dict | None:
    p = d / f"{name}.json"
    if not p.exists():
        return None
    return json.loads(p.read_text())


def fmt(x, prec=3) -> str:
    if isinstance(x, (int, float)):
        if isinstance(x, float):
            return f"{x:.{prec}f}"
        return str(x)
    return str(x)


def per_relation_accuracy(results: list[dict], rel_by_idx: dict[int, str]) -> dict[str, float]:
    buckets: dict[str, list[bool]] = defaultdict(list)
    for r in results:
        if r.get("error"):
            continue
        rel = r.get("relation") or rel_by_idx.get(r.get("idx"))
        if not rel:
            continue
        buckets[rel].append(bool(r.get("correct")))
    return {rel: sum(v) / max(len(v), 1) for rel, v in buckets.items()}


def build_substitutions(ablation_dir: Path, split: list[dict]) -> dict:
    rel_by_idx = {i: it.get("relation", "") for i, it in enumerate(split)}

    full = {}
    for k in (1, 2, 3):
        c = load_cell(ablation_dir, f"full_k{k}")
        if c is None:
            continue
        s = c["summary"]
        full[k] = {
            "summary": s,
            "per_relation": per_relation_accuracy(c["results"], rel_by_idx),
        }

    exec_only = load_cell(ablation_dir, "executor_only")
    if exec_only is not None:
        exec_only = {
            "summary": exec_only["summary"],
            "per_relation": per_relation_accuracy(exec_only["results"], rel_by_idx),
        }

    return {"full": full, "exec_only": exec_only}


def render_md(template: str, sub: dict) -> str:
    full = sub["full"]
    exec_only = sub["exec_only"]
    best_k = max(full.keys()) if full else None
    best = full.get(best_k) if best_k else None

    out = template

    # §3.1 — full pipeline at best-available k
    if best is not None:
        s = best["summary"]
        row = (
            f"| {s.get('n', '')} | {fmt(s.get('accuracy', 0))} | "
            f"{fmt(s.get('selective_accuracy', 0))} | {fmt(s.get('coverage', 0))} | "
            f"{fmt(s.get('abstain_rate', 0))} | {fmt(s.get('verification_rate', 0))} |"
        )
        out = out.replace(
            "| 200 | `[FILL]` | `[FILL]`       | `[FILL]` | `[FILL]`| `[FILL]`     |",
            row,
        )

    # §3.2 — k ablation rows
    for k in (1, 2, 3):
        if k not in full:
            continue
        s = full[k]["summary"]
        old = f"| {k} | `[FILL]` | `[FILL]`       | `[FILL]` | `[FILL]`  |"
        new = (
            f"| {k} | {fmt(s.get('accuracy', 0))} | {fmt(s.get('selective_accuracy', 0))} | "
            f"{fmt(s.get('coverage', 0))} | {fmt(s.get('avg_iterations', 0), 2)} |"
        )
        out = out.replace(old, new)

    # §4 — critic vs no-critic
    if best is not None and exec_only is not None:
        e = exec_only["summary"]
        f = best["summary"]
        rows = [
            ("Accuracy", "accuracy"),
            ("Selective acc.", "selective_accuracy"),
            ("Abstain rate", "abstain_rate"),
        ]
        for label, key in rows:
            ev = e.get(key, 0)
            fv = f.get(key, 0)
            delta = (fv - ev) if isinstance(ev, (int, float)) and isinstance(fv, (int, float)) else 0
            old_placeholder = f"| {label}"
            # Replace the matching row in the markdown table.
            placeholder_line = (
                f"| {label}"
                + " " * (len("| Selective acc.   ") - len(f"| {label}"))
                + "| `[FILL]`      | `[FILL]`            | `[FILL]` |"
            )
            new_line = (
                f"| {label}{' ' * max(1, 16 - len(label))}| "
                f"{fmt(ev)}         | {fmt(fv)}                | "
                f"{('+' if delta >= 0 else '')}{fmt(delta)} |"
            )
            # Loose match: replace the line with [FILL]s starting with the label.
            for line in out.split("\n"):
                if line.startswith(f"| {label}") and "[FILL]" in line:
                    out = out.replace(line, new_line, 1)
                    break

        # Verification rate is full-only
        vr = f.get("verification_rate", 0)
        for line in out.split("\n"):
            if line.startswith("| Verification") and "[FILL]" in line:
                new = f"| Verification     | n/a           | {fmt(vr)}            | — |"
                out = out.replace(line, new, 1)
                break

    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ablation_dir", default="results/ablation")
    parser.add_argument("--split", default="data/vsr_strat200.json")
    parser.add_argument("--writeup", default="results/writeup_results.md")
    parser.add_argument("--out", default="results/writeup_results_filled.md")
    args = parser.parse_args()

    split = json.loads(Path(args.split).read_text())
    sub = build_substitutions(Path(args.ablation_dir), split)
    template = Path(args.writeup).read_text()
    rendered = render_md(template, sub)
    Path(args.out).write_text(rendered)

    print(f"Wrote {args.out}")
    # Print a quick summary so the user sees the headline numbers immediately.
    if sub["full"]:
        best_k = max(sub["full"].keys())
        s = sub["full"][best_k]["summary"]
        print(f"\nFull pipeline (k={best_k}): "
              f"acc={s.get('accuracy')} sel={s.get('selective_accuracy')} "
              f"abstain={s.get('abstain_rate')} verif={s.get('verification_rate')}")
        for rel, acc in sorted(sub["full"][best_k]["per_relation"].items()):
            print(f"  {rel:10s}  {acc:.3f}")
    if sub["exec_only"]:
        s = sub["exec_only"]["summary"]
        print(f"\nExecutor-only baseline: "
              f"acc={s.get('accuracy')} sel={s.get('selective_accuracy')} "
              f"abstain={s.get('abstain_rate')}")


if __name__ == "__main__":
    main()
