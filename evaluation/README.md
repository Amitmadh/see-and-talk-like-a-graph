# Evaluation

Run from the repo root. Both evaluators score model JSONL outputs, write aggregate reports, and (unless you pass `--skip-vis`) save plots under `evaluation/vis/`.

## Experiment 1 (visual augmentation)

Scores clean baseline runs (text-only, image-only, and image+text) and reports accuracy by setting and task.

```bash
python -m evaluation.evaluation
```

Defaults to `results/baseline`. Override with `--results-root` if needed.

This writes JSON/CSV accuracy summaries and a grouped bar chart comparing modalities per task.

```bash
python -m evaluation.evaluation --skip-vis   # reports only
```

## Experiment 2 (mixed signals)

Scores mixed-signal runs (original vs corrupted graph encoding), attaches graph metadata from the GraphQA datasets, then reports which modality “wins” (image, text, or neither), broken down by task, generator algorithm, and graph size.

```bash
python -m evaluation.mixed_evaluation
```

Defaults to `results/mixed_baseline`. Override with `--results-root` if needed.

This writes scored and enriched JSON, CSV breakdowns, and per-task plots (by algorithm and by edge-count bin), plus an overall task comparison.

Skip steps if you already have intermediate reports:

```bash
python -m evaluation.mixed_evaluation --skip-score    # reuse scored JSON
python -m evaluation.mixed_evaluation --skip-enrich   # reuse enriched JSON
python -m evaluation.mixed_evaluation --skip-vis      # reports only
```
