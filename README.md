# Spatial Evidence Agent (SEA)

A **training-free** three-agent pipeline for verified spatial visual question
answering. Given an RGB image and a binary spatial question, SEA returns
yes/no/abstain **plus a Spatial Evidence Graph (SEG)** — the bounding boxes,
depths, centroid deltas, and rule evaluations that justify the answer.

The core finding: deterministic geometry is a poor *answerer* but an excellent
*router and confidence estimator*. Rather than overruling a strong VLM, SEA
uses geometry to decide **when to trust the VLM, when to commit geometry, and
when to abstain**.

## Demo

The active-perception loop in action — when the Executor and the geometric
check disagree, the Critic crops the disputed region, the VLM re-examines the
zoom, and the answer is committed only once they agree:

![Active-perception loop](results/demo/active_perception.gif)

The full pipeline walkthrough (Planner → Executor → Critic → verified answer
over several relations) is in [`results/demo/sea_demo.mp4`](results/demo/sea_demo.mp4),
regenerated with `python scripts/make_demo_video.py`.

## Results (VSR-200)

Evaluated on a stratified 200-sample split of the published
[VSR](https://github.com/cambridgeltl/visual-spatial-reasoning) benchmark
(all 8 relations), against a GPT-4o baseline on identical images:

| System | Accuracy | Selective acc. | Coverage |
| --- | --- | --- | --- |
| GPT-4o baseline | 0.720 | 0.720 | 1.00 |
| **SEA (full coverage)** | **0.790** | 0.790 | 1.00 |
| SEA (selective) | 0.715 | **0.803** | 0.89 |
| SEA (high precision, τ=0.40) | — | **0.847** | 0.30 |

A single confidence threshold dials SEA along the precision/coverage frontier;
every committed answer carries auditable geometric evidence.

## Architecture

```text
START -> Planner -> Executor -> Critic -> Output
                              -> Correction -> Executor   (zoom + re-examine)
                              -> Arbitrate                (budget exhausted)
                              -> Abstain
```

| Agent | Role |
| --- | --- |
| Planner | LangChain chat model with structured Pydantic output parses the question into `{obj1, obj2, relation}`. |
| Executor | LLaVA (local) or an OpenAI-compatible vision API; returns an initial yes/no plus object claims and a confidence. |
| Critic | Grounding-DINO localizes both objects, Depth Anything V2 estimates relative depth, then deterministic geometry rules produce a verdict and a *geometric confidence*. |

When the Executor and Critic disagree, a crop-based correction loop re-examines
the disputed region for up to `k` iterations. If they still disagree, an
**arbitration** step decides by geometric confidence: high confidence commits
geometry, low confidence defers to the VLM, no evidence abstains. Monocular
depth and the `contains` relation are confidence-capped so they always defer to
the VLM rather than overruling it.

Supported relations: `left_of`, `right_of`, `above`, `below`, `behind`,
`in_front`, `on`, `contains`.

## Install

A project-local venv is recommended:

```bash
python -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/pip install langgraph langchain-openai langchain datasets matplotlib
```

Create a local `.env` (project root or `src/`) for the OpenAI key:

```text
OPENAI_API_KEY=sk-your-key-here
```

`--openai_key` on the CLI takes precedence over `.env` and the shell env.

For real Grounding-DINO + Depth Anything runs the Critic needs:

```text
src/checkpoints/GroundingDINO_SwinT_OGC.py     # config, ~1 KB
src/checkpoints/groundingdino_swint_ogc.pth    # weights, ~660 MB
```

Pin `transformers>=4.40,<5.0` (groundingdino-py uses `BertModel.get_head_mask`,
removed in transformers 5.x). Depth Anything V2 Small weights download on first
use from HuggingFace.

> All commands run from `src/` — modules use flat imports (`import critic`).

## Single-image inference

```bash
cd src && python main.py \
  --image path/to/image.jpg \
  --question "Is the red cup to the left of the blue plate?" \
  --backend openai --k 3 \
  --save_annotation annotated.jpg \
  --save_graph evidence.json --verbose
```

`main.py`, `evaluate.py`, and `ablation.py` run in **strict mode**: Planner or
Executor model failures abort instead of falling back to regex parsing or
geometry-only behavior.

## Evaluation

```bash
cd src && python evaluate.py --dataset whatsup \
  --split ../data/vsr_strat200.json --image_root ../data/vsr_images \
  --backend openai --max_iterations 2 \
  --output ../results/run.json --dump_evidence ../results/run_evidence.json
```

Operating modes (abstention policy knobs):

- default — **selective**: abstain on detector misses; every committed answer is verified.
- `--vlm_fallback_on_miss` — **full coverage**: detector misses defer to the VLM (no abstentions).
- `--abstain_below_confidence 0.40` — **high precision**: abstain on committed answers with geometric confidence below the threshold.

`--dump_evidence` persists per-item `{relation, b1, b2, d1, d2, gt}` so
`scripts/calibrate_thresholds.py` can replay verification offline for threshold
sweeps **without re-spending API calls**.

Dataset split format:

```json
[
  {"image_path": "rel/to/image_root.jpg",
   "question": "Is the cup to the left of the plate?",
   "answer": "yes"}
]
```

Build the VSR-200 split (downloads COCO images):

```bash
python scripts/prep_vsr_split.py --n 200 \
  --out data/vsr_strat200.json --image_dir data/vsr_images
```

## Ablation, demo, and tests

```bash
# Cartesian sweep over modes x k (use --dry-run to enumerate cells, $0)
cd src && python ablation.py --split ../data/vsr_strat200.json \
  --image_root ../data/vsr_images --output_dir ../results/ablation \
  --modes full,executor_only --ks 1,2,3 --backend openai

# Streamlit demo
cd src && streamlit run app.py

# Geometry unit tests (pure CPU; mock-pipeline test skips without deps)
pytest tests/        # or: python tests/test_critic.py
```

## Configuration

`src/config.py` centralizes the Critic's thresholds in a `CriticConfig`
dataclass (slack `margin`, `on`/`contains` cutoffs, the arbitration threshold
`τ`, depth/contains confidence caps, crop padding). The ablation script and the
Streamlit sidebar vary these at runtime without touching agent code.

## Project structure

```text
src/
  state.py        Pydantic AgentState + SpatialEvidenceGraph
  config.py       CriticConfig (centralized geometry thresholds)
  planner.py      LangChain structured Planner
  executor.py     LLaVA / OpenAI Executor
  critic.py       geometric verification engine
  detector.py     Grounding-DINO wrapper
  depth.py        Depth Anything V2 wrapper (original-frame, memoized)
  pipeline.py     LangGraph orchestration (+ arbitration node)
  parsing.py      robust JSON-object extraction fallback
  env_loader.py   .env loader
  visualize.py    SEG printing, annotation, JSON export
  main.py         single-image CLI
  evaluate.py     dataset evaluation harness
  ablation.py     modes x k ablation harness
  app.py          Streamlit demo
tests/            geometry unit tests + mock-pipeline smoke test
scripts/          VSR split prep, offline threshold calibration, figure generation
docs/             proposal, design doc, mid-demo, final report, architecture.png
data/             VSR-200 split + COCO images (generated)
results/          eval outputs, ablation, figures, qualitative gallery
```
