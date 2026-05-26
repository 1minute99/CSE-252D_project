# Spatial Evidence Agent

A training-free three-agent pipeline for verified spatial visual question
answering. Given an RGB image and a yes/no spatial question, the system returns
a final answer plus a Spatial Evidence Graph (SEG) containing the geometric
checks used to support that answer.

## Architecture

```text
Planner -> Executor -> Critic -> Output
                       -> Correction -> Executor
                       -> Abstain
```

| Agent | Role |
| --- | --- |
| Planner | Uses a LangChain chat model with structured Pydantic output to parse `{obj1, obj2, relation}`. |
| Executor | Uses LLaVA or an OpenAI-compatible vision API to produce an initial yes/no answer and object claims. |
| Critic | Uses Grounding-DINO plus Depth Anything V2 to verify geometry. |

Supported relations: `left_of`, `right_of`, `above`, `below`, `behind`,
`in_front`, `on`, and `contains`.

## Install

```bash
pip install -r requirements.txt
```

Create a local `.env` file in the project root for the OpenAI key:

```text
OPENAI_API_KEY=sk-your-key-here
```

The code checks `.env` (project root or `src/`), then your shell environment.
Command-line `--openai_key` still works and takes precedence.

For real Grounding-DINO runs, place these files in `src/checkpoints/`:

```text
src/checkpoints/GroundingDINO_SwinT_OGC.py
src/checkpoints/groundingdino_swint_ogc.pth
```

## CLI

```bash
python src/main.py ^
  --image path/to/image.jpg ^
  --question "Is the red cup to the left of the blue plate?" ^
  --backend openai ^
  --k 3 ^
  --save_annotation annotated.jpg ^
  --save_graph evidence.json ^
  --verbose
```

`main.py` and `evaluate.py` are strict production paths. Planner and Executor
model failures stop the run instead of falling back to regex parsing or
geometry-only answers. 

## Evaluation

```bash
python src/evaluate.py ^
  --dataset whatsup ^
  --split data/whatsup_val.json ^
  --image_root data/images ^
  --backend openai ^
  --max_iterations 3 ^
  --output results/whatsup_k3.json
```

Dataset format:

```json
[
  {
    "image_path": "relative/to/image_root.jpg",
    "question": "Is the cup to the left of the plate?",
    "answer": "yes"
  }
]
```

## Project Structure

```text
src/                source code
  state.py          Pydantic state and SpatialEvidenceGraph models
  parsing.py        robust JSON-object extraction fallback
  planner.py        LangChain structured Planner
  executor.py       LLaVA/OpenAI Executor
  critic.py         geometric verification engine
  detector.py       Grounding-DINO wrapper
  depth.py          Depth Anything V2 wrapper
  pipeline.py       LangGraph orchestration
  visualize.py      SEG printing, image annotation, JSON export
  main.py           CLI entry point
  evaluate.py       dataset evaluation harness
  internet_inputs/  sample images and reference graphs
docs/               proposal, design doc, mid-demo deck, architecture.png
results/
  qualitative/      per-relation sample runs (and test.json eval split)
  quantitative_sample15/  aggregate eval.json on 15 samples
requirements.txt
```
