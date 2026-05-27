"""
Ablation harness.

Sweeps two axes over a dataset split:
  * `k` (max_iterations) ∈ {1, 2, 3} for the full pipeline.
  * Pipeline mode ∈ {full, executor_only} — the latter bypasses the Critic so
    you get a head-to-head against the bare VLM ("does verification help?").

Writes one JSON per cell plus a tidy summary.csv into --output_dir. Use
--dry-run to enumerate the planned cells without spending any API calls.

Usage:
    python src/ablation.py \\
        --split data/whatsup_strat200.json \\
        --image_root data/images \\
        --output_dir results/ablation/ \\
        --modes full,executor_only \\
        --ks 1,2,3
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

from env_loader import get_openai_api_key, load_project_env

logging.basicConfig(level=logging.INFO, format="%(levelname)s ablation | %(message)s")
logger = logging.getLogger("ablation")


def _run_full(args, k: int, out_path: Path) -> dict:
    """Delegates to evaluate.run_evaluation with --max_iterations k."""
    from evaluate import run_evaluation

    eval_args = SimpleNamespace(
        dataset=args.dataset,
        split=args.split,
        image_root=args.image_root,
        backend=args.backend,
        openai_key=args.openai_key,
        planner_model=args.planner_model,
        vision_model=args.vision_model,
        max_iterations=k,
        output=str(out_path),
        dump_evidence="",
    )
    return run_evaluation(eval_args)


def _run_executor_only(args, out_path: Path) -> dict:
    """Planner + Executor, no Critic. Tests the bare-VLM baseline."""
    from langchain_openai import ChatOpenAI
    import executor
    import planner
    from state import AgentState
    from evaluate import load_dataset

    api_key = get_openai_api_key(args.openai_key)
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is required for executor_only mode")
    llm = ChatOpenAI(
        model=args.planner_model,
        api_key=api_key,
        temperature=0,
        max_retries=0,
        timeout=60,
    )
    exec_cfg = {
        "backend": args.backend,
        "openai_key": api_key,
        "model": args.vision_model,
        "strict": True,
    }

    items = load_dataset(args.split, args.image_root)
    logger.info(f"[executor_only] {len(items)} items")

    results = []
    correct = 0
    abstained = 0
    for idx, item in enumerate(items):
        state = AgentState(image_path=item["image_path"], question=item["question"])
        try:
            planner.run_planner(state, llm, strict=True)
            executor.run_executor(state, exec_cfg)
        except Exception as exc:
            logger.error(f"[{idx}] error: {exc}")
            results.append({"idx": idx, "error": str(exc)})
            continue

        gt = str(item["answer"]).strip().lower()
        if state.executor_answer is None:
            pred = "abstain"
            abstained += 1
        else:
            pred = "yes" if state.executor_answer else "no"
        is_correct = pred == gt
        if is_correct:
            correct += 1
        results.append(
            {
                "idx": idx,
                "question": item["question"],
                "gt": gt,
                "pred": pred,
                "correct": is_correct,
                "abstain": pred == "abstain",
                "relation": state.relation,
            }
        )

    n = len(results)
    answered = n - abstained
    summary = {
        "mode": "executor_only",
        "n": n,
        "accuracy": round(correct / max(n, 1), 4),
        "selective_accuracy": round(correct / max(answered, 1), 4),
        "coverage": round(answered / max(n, 1), 4),
        "abstain_rate": round(abstained / max(n, 1), 4),
    }
    logger.info(f"[executor_only] {summary}")

    output = {"summary": summary, "results": results}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, indent=2))
    return output


def _per_relation_breakdown(results: list[dict]) -> dict[str, dict[str, float]]:
    bucket: dict[str, dict[str, int]] = defaultdict(lambda: {"correct": 0, "n": 0, "abstain": 0})
    for r in results:
        if "error" in r:
            continue
        # The `relation` key only exists on executor-only runs; for full
        # pipeline runs we infer relation from the question string is not
        # robust, so we skip per-relation breakdown when missing.
        rel = r.get("relation")
        if rel is None:
            continue
        b = bucket[rel]
        b["n"] += 1
        if r.get("correct"):
            b["correct"] += 1
        if r.get("abstain"):
            b["abstain"] += 1
    return {
        rel: {
            "n": b["n"],
            "accuracy": round(b["correct"] / max(b["n"], 1), 4),
            "abstain_rate": round(b["abstain"] / max(b["n"], 1), 4),
        }
        for rel, b in bucket.items()
    }


def _enumerate_cells(modes: list[str], ks: list[int]) -> list[tuple[str, int | None]]:
    """Cartesian product, but `executor_only` ignores k (no iterations)."""
    cells: list[tuple[str, int | None]] = []
    for mode in modes:
        if mode == "executor_only":
            cells.append((mode, None))
        else:
            for k in ks:
                cells.append((mode, k))
    return cells


def main():
    parser = argparse.ArgumentParser(description="Ablation harness")
    parser.add_argument("--dataset", default="whatsup", choices=["whatsup", "gqa", "3dsrbench"])
    parser.add_argument("--split", required=True)
    parser.add_argument("--image_root", required=True)
    parser.add_argument("--output_dir", required=True, help="Where to write per-cell JSON + summary.csv")
    parser.add_argument("--modes", default="full,executor_only", help="Comma-separated subset of {full, executor_only}")
    parser.add_argument("--ks", default="1,2,3", help="Comma-separated k values for full mode")
    parser.add_argument("--backend", default="openai", choices=["local", "openai"])
    parser.add_argument("--openai_key", default="")
    parser.add_argument("--planner_model", default="gpt-4o-mini")
    parser.add_argument("--vision_model", default="gpt-4o")
    parser.add_argument("--dry-run", action="store_true", help="List planned cells and exit")
    args = parser.parse_args()
    load_project_env()

    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    ks = [int(k.strip()) for k in args.ks.split(",") if k.strip()]
    cells = _enumerate_cells(modes, ks)

    logger.info(f"Planned cells ({len(cells)}):")
    for mode, k in cells:
        suffix = f"k={k}" if k is not None else "(no iterations)"
        logger.info(f"  - {mode} {suffix}")
    if args.dry_run:
        return

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_rows: list[dict] = []

    for mode, k in cells:
        cell_name = f"{mode}" if k is None else f"{mode}_k{k}"
        cell_path = out_dir / f"{cell_name}.json"
        logger.info(f"=== {cell_name} -> {cell_path} ===")
        if mode == "full":
            out = _run_full(args, k=k, out_path=cell_path)
        elif mode == "executor_only":
            out = _run_executor_only(args, out_path=cell_path)
        else:
            raise SystemExit(f"unknown mode: {mode}")
        summary = out["summary"]
        row = {
            "cell": cell_name,
            "mode": mode,
            "k": k if k is not None else "",
            **{key: summary.get(key, "") for key in (
                "n", "accuracy", "selective_accuracy", "coverage",
                "abstain_rate", "verification_rate", "avg_iterations",
            )},
        }
        # Add per-relation breakdown columns when available (executor_only only).
        per_rel = _per_relation_breakdown(out.get("results", []))
        for rel, stats in per_rel.items():
            row[f"acc_{rel}"] = stats["accuracy"]
        summary_rows.append(row)

    if summary_rows:
        all_keys = sorted({key for row in summary_rows for key in row.keys()})
        ordered = ["cell", "mode", "k", "n", "accuracy", "selective_accuracy",
                   "coverage", "abstain_rate", "verification_rate", "avg_iterations"]
        rest = [k for k in all_keys if k not in ordered]
        fieldnames = ordered + rest
        summary_path = out_dir / "summary.csv"
        with summary_path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for row in summary_rows:
                writer.writerow(row)
        logger.info(f"Summary written -> {summary_path}")


if __name__ == "__main__":
    main()
