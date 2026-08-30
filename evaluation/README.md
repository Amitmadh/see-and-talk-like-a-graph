# Evaluation

Run from the repo root. Both evaluators score model JSONL outputs, write aggregate reports, and (unless you pass `--skip-vis`) save plots under `evaluation/vis/`.

## Experiment 1 (visual augmentation)

Scores clean baseline runs (text-only, image-only, and image+text) and reports accuracy by setting and task.

```bash
python -m evaluation.evaluation
```

Defaults to `results/baseline`. Override with `--results-root` if needed.

This writes JSON/CSV accuracy summaries: a per-file aggregate, one setting×task CSV per model, and a model×modality comparison CSV. Unless `--skip-vis` is set, it also saves a grouped bar chart per model plus a three-panel plot comparing models across Text / Image / Image+Text.

```bash
python -m evaluation.evaluation --skip-vis   # reports only
```

## Experiment 2 (mixed signals)

Scores mixed-signal runs (original vs corrupted graph encoding), attaches graph metadata from the GraphQA datasets, then reports which modality “wins” (image, text, or neither), broken down by task, generator algorithm, and graph size.

```bash
python -m evaluation.mixed_evaluation
```

Defaults to `results/mixed_baseline`. Override with `--results-root` if needed.

This writes scored and enriched JSON, CSV breakdowns, and per-task plots (by algorithm and by edge-count bin), plus an overall task comparison. It also writes all-model comparisons under `evaluation/vis/all_models/`: `by_task.png` (modality wins by task), `by_edges.png` (same stacks over 10-edge bins), `by_edges_20.png` (same stacks over 20-edge bins), `by_algorithm.png` (small multiples over SFN / Complete / Star / Path), and `by_algorithm_model.png` (one row per model, focused on edge existence, shortest path, edge count, node degree, and connected nodes).

If the scored CSV already exists, regenerate only those comparisons:

```bash
python -m evaluation.plot_mixed_all_models
```

The algorithm and edge-count figures need sample-level graph metadata, so they read `evaluation/enrich_mixed_signals_results.json` by default. To rebuild only those figures:

```bash
python -m evaluation.plot_mixed_all_models --skip-task
```

To rebuild only the edge-count comparison:

```bash
python -m evaluation.plot_mixed_all_models --skip-task --skip-algorithm --skip-algorithm-model
```

Skip steps if you already have intermediate reports:

```bash
python -m evaluation.mixed_evaluation --skip-score    # reuse scored JSON
python -m evaluation.mixed_evaluation --skip-enrich   # reuse enriched JSON
python -m evaluation.mixed_evaluation --skip-vis      # reports only
```
