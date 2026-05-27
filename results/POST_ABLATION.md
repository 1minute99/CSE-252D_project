# What to run when the ablation finishes

The ablation harness writes one JSON per cell into `results/ablation/`:

```
results/ablation/
  full_k1.json
  full_k2.json
  full_k3.json
  executor_only.json
  summary.csv      # tidy headline metrics across cells
```

## 1. Generate figures (one command)

```
.venv/bin/python scripts/make_figures.py \
  --ablation_dir results/ablation \
  --split data/vsr_strat200.json \
  --out_dir results/figures
```

Produces:
- `fig1_depth_oscillation_prefix.png` (already exists)
- `fig2_k_ablation.png`
- `fig3_per_relation_accuracy.png`
- `fig4_failure_modes.png`
- `fig5_critic_vs_no_critic.png`

## 2. Auto-fill the writeup

```
.venv/bin/python scripts/fill_writeup.py \
  --ablation_dir results/ablation \
  --split data/vsr_strat200.json \
  --writeup results/writeup_results.md \
  --out results/writeup_results_filled.md
```

This replaces every `[FILL]` placeholder in `writeup_results.md` with the
real numbers from the ablation summaries and writes
`writeup_results_filled.md`. The headline numbers are also echoed to stdout.

## 3. (Optional) Threshold calibration on VSR-200

`ablation.py` doesn't dump per-item geometric evidence. To enable an offline
threshold sweep over the full VSR-200, do one more eval pass with the
`--dump_evidence` flag:

```
cd src && ../.venv/bin/python evaluate.py \
  --dataset whatsup \
  --split ../data/vsr_strat200.json \
  --image_root ../data/vsr_images \
  --backend openai \
  --max_iterations 1 \
  --output ../results/vsr200_for_calibration.json \
  --dump_evidence ../results/vsr200_evidence.json
```

Then:

```
.venv/bin/python scripts/calibrate_thresholds.py \
  --evidence results/vsr200_evidence.json \
  --report results/vsr200_calibration.json \
  --top_k 10
```

Reports the top-10 `CriticConfig` settings by macro-F1, plus per-relation
F1/TP/FP/FN/TN counts. If the best config beats defaults on macro-F1, the
new thresholds can be plugged into `src/config.py:CriticConfig` and the
full pipeline re-run.

## 4. (Optional) Live demo

```
cd src && ../.venv/bin/streamlit run app.py
```

Opens at `http://localhost:8501`. The sidebar's "Advanced critic
thresholds" expander lets you live-edit the same `CriticConfig` knobs the
calibration script sweeps — useful for showing how a single threshold flips
a verdict during the demo.

## Files produced by this work

- `src/config.py` — centralized `CriticConfig` dataclass.
- `src/ablation.py` — ablation harness.
- `tests/test_critic.py` + `tests/test_pipeline_mock.py` — geometry +
  pipeline-wiring tests.
- `scripts/prep_vsr_split.py` — builds `data/vsr_strat200.json`.
- `scripts/calibrate_thresholds.py` — offline threshold replay.
- `scripts/make_figures.py` — writeup figures.
- `scripts/fill_writeup.py` — substitutes numbers into the writeup.
- `results/writeup_results.md` — results-section draft.
- `results/postfix_12sample.json` + `results/postfix_12sample_evidence.json`
  — re-run of the original 12-sample eval after the depth-frame fix.
- `data/vsr_strat200.json` + `data/vsr_images/` — VSR eval split.
- `src/critic.py`, `src/depth.py`, `src/pipeline.py` — depth-frame fix.
- `src/main.py`, `src/evaluate.py` — Windows-path cleanup, `--dump_evidence`.
- `src/app.py` — Streamlit demo rewrite.
- `CLAUDE.md` — updated repo guide.
