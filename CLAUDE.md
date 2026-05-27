# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Spatial Evidence Agent (SEA): a training-free three-agent pipeline for verified
spatial visual question answering. Input is an RGB image + a binary spatial
question; output is yes/no/abstain plus a Spatial Evidence Graph (SEG) of the
geometric checks that justify the answer.

Supported relations: `left_of`, `right_of`, `above`, `below`, `behind`,
`in_front`, `on`, `contains`.

## Setup

The recommended setup uses a project-local venv at `.venv/`:

```
python -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/pip install langgraph langchain-openai langchain datasets matplotlib
```

`OPENAI_API_KEY` is read from `./.env`, `src/.env`, or the shell environment
(see `env_loader.py`); the CLI `--openai_key` overrides both.

For real Grounding-DINO + Depth Anything runs, the Critic needs:

- `src/checkpoints/GroundingDINO_SwinT_OGC.py` (config, ~1 KB)
- `src/checkpoints/groundingdino_swint_ogc.pth` (weights, ~660 MB)
- `transformers >= 4.40, < 5.0` (groundingdino-py uses `BertModel.get_head_mask`
  which was removed in transformers 5.x — pin to 4.x).
- Depth Anything V2 weights are pulled on first use from HuggingFace
  (`depth-anything/Depth-Anything-V2-Small-hf`).

## Commands

All commands must be invoked from `src/` because modules use flat imports.

Single-image inference:
```
cd src && python main.py --image PATH --question "..." --backend openai --k 3 \
  --save_annotation out.jpg --save_graph evidence.json --verbose
```

Dataset evaluation:
```
cd src && python evaluate.py --dataset whatsup --split path/to/split.json \
  --image_root path/to/images --backend openai --max_iterations 3 \
  --output results/run.json --dump_evidence results/run_evidence.json
```

The `--dump_evidence` flag persists per-item `{relation, b1, b2, d1, d2, gt}`
so `scripts/calibrate_thresholds.py` can replay verification offline for
threshold sweeps without re-spending API calls.

Ablation (cartesian product over modes × k):
```
cd src && python ablation.py --split data/vsr_strat200.json \
  --image_root data/vsr_images --output_dir results/ablation \
  --modes full,executor_only --ks 1,2,3 --backend openai
```

Use `--dry-run` to enumerate planned cells without spending any API calls.

Streamlit demo:
```
cd src && streamlit run app.py
```

Tests (pure-CPU geometry tests; mock-pipeline test auto-skips without deps):
```
pytest tests/    # or: python tests/test_critic.py
```

VSR-200 evaluation split (downloads COCO images to `data/vsr_images/`):
```
python scripts/prep_vsr_split.py --n 200 --out data/vsr_strat200.json \
  --image_dir data/vsr_images
```

Figures for the writeup (reads `results/ablation/*.json`):
```
python scripts/make_figures.py --ablation_dir results/ablation \
  --split data/vsr_strat200.json --out_dir results/figures
```

## Architecture

LangGraph orchestrates three agents through a bounded active-perception loop.
The whole pipeline mutates a single `AgentState` (Pydantic) defined in
`state.py`; every node reads/writes a `dict` form of it.

```
START → planner → executor → critic → output
                           ↘ correction → executor (with new crop)
                           ↘ abstain
```

- **Planner** (`planner.py`) — LangChain `ChatOpenAI` with Pydantic structured
  output. Parses the question into `{obj1, obj2, relation}`. In strict mode
  (used by `main.py`, `evaluate.py`, `ablation.py`) parse failure aborts the
  run; otherwise falls back to `parsing.extract_json_object` regex extraction.
- **Executor** (`executor.py`) — vision model that returns an initial yes/no
  plus two object claims (subject, reference). Three backends: `local`
  (LLaVA via transformers), `openai` (OpenAI vision API or any OpenAI-compatible
  endpoint via `openai_base`), and `mock` (deterministic JSON for smoke tests).
- **Critic** (`critic.py`) — the source of truth. Runs Grounding-DINO
  (`detector.py`) to localize both objects, Depth Anything V2 (`depth.py`) for
  relative monocular depth, then applies deterministic geometry rules
  (`_verify_relation`) using normalized bbox centers, IoU, and median depth.
  Thresholds live in `config.CriticConfig` and are passed in via `critic_config`.
- **Pipeline** (`pipeline.py`) — builds the LangGraph, defines routing.
  `output` is reached when geometric evidence exists AND the executor agrees
  with it (or `executor_answer is None`). On disagreement, `correction`
  increments `iteration` and re-runs the executor on the critic's
  pre-computed `current_crop`. If `iteration` hits `max_iterations - 1` while
  still disagreeing, control goes to the **`arbitrate`** node (not straight to
  abstain).

### Arbitration (replaces abstain-on-disagreement)

When the executor and geometry persistently disagree, `arbitrate_node`
decides by **geometric confidence** rather than always abstaining:

- `geo_confidence = min(detector confidences) × margin_clearance`, where
  margin_clearance = `clip(|signal| / scale, 0, 1)` and `signal` is `dx`/`dy`
  for position relations, `dz` for depth relations, coverage for `contains`.
  Depth relations use a smaller `scale` and have smaller, noisier signals, so
  they earn lower confidence — which is desirable, since the VLM out-performs
  geometry on depth.
- `geo_confidence >= cfg.geo_confidence_arbitration` (default 0.40) →
  **commit the geometry answer** (override the VLM), `answer_source =
  "geometry_override"`, `verified = True`.
- below threshold → **defer to the VLM**, `answer_source = "vlm_deferred"`,
  `verified = False`.
- no usable geometric evidence (detector miss) and no VLM answer → `abstain`.

Set `geo_confidence_arbitration = 1.0` to recover the old
abstain-on-every-disagreement behavior. Abstentions are still classified as
`planner_parse_error`, `detector_miss`, or `depth_noise`. `SpatialEvidenceGraph.answer_source`
records which path produced the answer (`agreement` / `geometry_override` /
`vlm_deferred` / `geometry_only`).

### Coordinate convention (post-Day-1 fix)

All bounding boxes in `state.BoundingBox` are normalized to `[0, 1]` in the
**original image** frame. When a crop is active, the Critic detects in crop
space and maps boxes back via `_map_bbox_to_original` before applying rules.

**Depth is now computed on the original full image** (`depth.estimate_depth_for_path`)
and sampled with the original-frame bbox coords, so depth values are
comparable across active-perception iterations. The depth map is memoized
in a single-entry cache so iterations 2/3 reuse it; `pipeline.run_pipeline`
calls `depth.clear_depth_cache()` in a `finally` block so memory stays flat
across an evaluation loop.

Pre-fix this was a real bug: depth was estimated on the cropped image and
sampled with crop-local detector bboxes, making d1/d2 incomparable across
crops and producing the bus/car oscillation observed in the original
`results/quantitative_sample15/eval.json`.

### Active-perception loop

The Critic, not the Correction node, decides the next crop:
- detector miss → schedule a full-image re-examination (`x1=0,y1=0,x2=1,y2=1`).
- relation evaluated but executor disagrees → tight crop around both bboxes
  with `cfg.crop_padding`.

The Correction node only increments the iteration counter; the next Executor
call reads `state.current_crop` to decide what region to send to the VLM.

### Strict vs. permissive mode

`main.py`, `evaluate.py`, and `ablation.py` set `strict_models=True` —
Planner/Executor model failures abort instead of falling back to regex
parsing or geometry-only behavior. Tests and `app.py` may permit non-strict
behavior. Detector/Depth fallbacks to mock outputs require an explicit
`allow_mock_models=True` in the critic config.

### CriticConfig (centralized thresholds)

`src/config.py` defines `CriticConfig` with the geometry-rule thresholds:

```
margin                       0.02  # position/depth tolerance
on_iou_threshold             0.05  # IoU floor for "on" relation
contains_coverage_threshold  0.70  # fraction of obj2 inside obj1
area_ratio_threshold         0.70  # obj2 must be < 70% of obj1's area
crop_padding                 0.05  # active-perception crop padding
allow_mock_models            False
```

`run_critic` accepts a `CriticConfig`, a dict, or `None` — `from_mapping`
ignores unknown keys and applies known ones. This is how the ablation script
and the Streamlit sidebar can vary thresholds at runtime.

## Conventions

- Source uses flat imports (`import critic`, `from state import ...`); commands
  must be run from `src/` as the working directory.
- All inter-agent state lives on `AgentState`; nodes serialize via
  `model_dump()` between LangGraph steps. Non-Pydantic-friendly fields
  (numpy arrays, etc.) must live outside `AgentState` — see how the depth
  map is held in a module-level cache in `depth.py` rather than on state.
- `docs/` (course deliverables) and `results/qualitative/` are reference
  artifacts kept under version control; `results/ablation/`, `results/figures/`,
  and `data/` are generated by the harness.
