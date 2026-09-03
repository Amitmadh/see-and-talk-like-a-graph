"""
Error-overlap and complementary-recovery statistics for the visual-augmentation
experiment (Experiment 1).

`venn_baseline.py` draws the three-way error Venn diagrams. This module reports
the same partition as numbers, so the claims made about it in the paper can be
regenerated and checked without reading a figure.

Two reports are written:

  evaluation/error_overlap_by_cell.csv
      One row per (model, task). The seven Venn regions as counts and as a
      percentage of the samples scored in all three settings, plus which region
      is the largest. "ABC" is the shared error core: samples that text-only,
      image-only and text+image all get wrong.

  evaluation/complementary_recovery.csv
      One row per (model, task) plus an ALL row per model. Restricted to the
      complementary samples -- those solved by exactly one of the two unimodal
      settings:

          R_T = correct from text alone, wrong from the image alone
          R_I = correct from the image alone, wrong from text alone

      On these the answer is demonstrably present in one of the two inputs, so
      a model that integrated the modalities perfectly would answer all of them
      correctly in the text+image setting. The recovery rate is how many it
      actually gets, and the gap between the R_T and R_I rates is a measure of
      which modality the model falls back on when only one supports the answer.

Scoring is not reimplemented here: error sets come from
`venn_baseline.collect_error_sets`, which in turn uses `evaluation.compare`.
Only samples scored in all three settings are counted, so the three settings
are always compared on the same instances.

Run from the project root:

    python -m evaluation.error_overlap
    python -m evaluation.error_overlap --summary     # also print paper claims
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.evaluation import (  # noqa: E402
    TASK_TYPES,
    compare,
    get_file_metadata,
    load_results_file,
)

SETTINGS = ("text_only", "image_only", "text_and_image")


def collect_error_sets(results_root, image_type="spring"):
    """
    Load every baseline JSONL and group scored/error sample ids by (model, task).

    Mirrors `venn_baseline.collect_error_sets`, but without importing the
    plotting stack, so the reports can be regenerated on a machine that has no
    matplotlib-venn installed. Scoring itself is delegated to
    `evaluation.compare`, so the two agree by construction.

    Returns:
        (model, task) -> {"scored": {setting: set}, "errors": {setting: set}}
    """
    root = Path(results_root)
    grouped = defaultdict(
        lambda: {
            "scored": {setting: set() for setting in SETTINGS},
            "errors": {setting: set() for setting in SETTINGS},
        }
    )

    for path in sorted(root.rglob("*.jsonl")):
        if path.name.startswith("."):
            continue

        metadata = get_file_metadata(path, root)
        setting, task = metadata["setting"], metadata["task"]
        if setting not in SETTINGS or task not in TASK_TYPES:
            continue

        for row in load_results_file(path):
            row_image = row.get("image_type", "unknown")
            if image_type not in {None, "all"} and row_image != image_type:
                continue
            if row.get("model_answer") is None:
                continue
            sample_id = row.get("sample_id")
            model = row.get("model")
            if not sample_id or not model:
                continue

            is_correct, status, _ = compare(
                task,
                row.get("expected_answer"),
                row.get("model_answer"),
            )
            if status == "bad_expected":
                continue

            bucket = grouped[(model, task)]
            bucket["scored"][setting].add(sample_id)
            if not is_correct:
                bucket["errors"][setting].add(sample_id)

    return grouped


def region_counts(errors):
    """The seven disjoint regions of the three-way error Venn."""
    text = errors["text_only"]
    image = errors["image_only"]
    both = errors["text_and_image"]
    return {
        "A": len(text - image - both),
        "B": len(image - text - both),
        "C": len(both - text - image),
        "AB": len((text & image) - both),
        "AC": len((text & both) - image),
        "BC": len((image & both) - text),
        "ABC": len(text & image & both),
    }


# Region key -> what it means, for the CSV header and the summary.
REGION_LABELS = {
    "A": "text_only_errors",
    "B": "image_only_errors",
    "C": "both_only_errors",
    "AB": "text_and_image_only_errors",
    "AC": "text_and_both_errors",
    "BC": "image_and_both_errors",
    "ABC": "all_three_errors",
}

# Canonical task order, matching the paper's Table 1.
TASK_ORDER = [
    "node_count",
    "edge_count",
    "edge_existence",
    "node_degree",
    "connected_nodes",
    "disconnected_nodes",
    "triangle_counting",
    "shortest_path",
]

MODEL_ORDER = [
    "Qwen2.5-VL-3B-Instruct",
    "Qwen2.5-VL-7B-Instruct",
    "Phi-4-multimodal-instruct",
    "gpt-4.1-mini",
]


def _sort_key(model, task):
    m = MODEL_ORDER.index(model) if model in MODEL_ORDER else len(MODEL_ORDER)
    t = TASK_ORDER.index(task) if task in TASK_ORDER else len(TASK_ORDER)
    return (m, model, t, task)


def _common_samples(bucket):
    """Sample ids scored in all three settings."""
    return set.intersection(*(bucket["scored"][s] for s in SETTINGS))


def _restrict(bucket, common):
    """Error sets restricted to the commonly scored samples."""
    return {s: bucket["errors"][s] & common for s in SETTINGS}


def build_overlap_rows(grouped, tasks=None, models=None):
    """One row per (model, task): the seven Venn regions, counts and percent."""
    rows = []
    for (model, task) in sorted(grouped, key=lambda k: _sort_key(*k)):
        if tasks and task not in tasks:
            continue
        if models and model not in models:
            continue
        bucket = grouped[(model, task)]
        common = _common_samples(bucket)
        n = len(common)
        if not n:
            continue

        errors = _restrict(bucket, common)
        counts = region_counts(errors)
        largest = max(counts, key=counts.get)

        row = {
            "model": model,
            "task": task,
            "n_compared": n,
            "err_text_only_pct": round(100 * len(errors["text_only"]) / n, 1),
            "err_image_only_pct": round(100 * len(errors["image_only"]) / n, 1),
            "err_text_and_image_pct": round(
                100 * len(errors["text_and_image"]) / n, 1
            ),
            "largest_region": REGION_LABELS[largest],
        }
        for key, label in REGION_LABELS.items():
            row[f"{label}_n"] = counts[key]
            row[f"{label}_pct"] = round(100 * counts[key] / n, 1)
        rows.append(row)
    return rows


def build_recovery_rows(grouped, tasks=None, models=None):
    """
    One row per (model, task) plus an ALL row per model, over the samples
    solved by exactly one unimodal setting.
    """
    rows = []
    totals = {}

    for (model, task) in sorted(grouped, key=lambda k: _sort_key(*k)):
        if tasks and task not in tasks:
            continue
        if models and model not in models:
            continue
        bucket = grouped[(model, task)]
        common = _common_samples(bucket)
        if not common:
            continue

        errors = _restrict(bucket, common)
        text_err, image_err = errors["text_only"], errors["image_only"]
        both_err = errors["text_and_image"]

        # Correct from one modality alone, wrong from the other.
        r_t = (common - text_err) & image_err
        r_i = (common - image_err) & text_err

        rec_t = len(r_t - both_err)
        rec_i = len(r_i - both_err)

        agg = totals.setdefault(model, [0, 0, 0, 0])
        agg[0] += len(r_t)
        agg[1] += rec_t
        agg[2] += len(r_i)
        agg[3] += rec_i

        rows.append(
            {
                "model": model,
                "task": task,
                "n_RT": len(r_t),
                "recovered_RT": rec_t,
                "rate_RT_pct": round(100 * rec_t / len(r_t), 1) if r_t else "",
                "n_RI": len(r_i),
                "recovered_RI": rec_i,
                "rate_RI_pct": round(100 * rec_i / len(r_i), 1) if r_i else "",
                "rate_overall_pct": (
                    round(100 * (rec_t + rec_i) / (len(r_t) + len(r_i)), 1)
                    if (r_t or r_i)
                    else ""
                ),
            }
        )

    for model in sorted(totals, key=lambda m: _sort_key(m, "")):
        n_t, rec_t, n_i, rec_i = totals[model]
        rows.append(
            {
                "model": model,
                "task": "ALL",
                "n_RT": n_t,
                "recovered_RT": rec_t,
                "rate_RT_pct": round(100 * rec_t / n_t, 1) if n_t else "",
                "n_RI": n_i,
                "recovered_RI": rec_i,
                "rate_RI_pct": round(100 * rec_i / n_i, 1) if n_i else "",
                "rate_overall_pct": (
                    round(100 * (rec_t + rec_i) / (n_t + n_i), 1)
                    if (n_t + n_i)
                    else ""
                ),
            }
        )
    return rows


SHORT_NAMES = {
    "Qwen2.5-VL-3B-Instruct": "Qwen-3B",
    "Qwen2.5-VL-7B-Instruct": "Qwen-7B",
    "Phi-4-multimodal-instruct": "Phi-4",
    "gpt-4.1-mini": "GPT-4.1-mini",
}


def plot_recovery(recovery_rows, path):
    """
    Draw the paper's complementary-recovery figure from the same rows the CSV
    is written from, so the figure and the reported numbers cannot drift apart.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    totals = [r for r in recovery_rows if r["task"] == "ALL"]
    if not totals:
        raise SystemExit("No aggregate rows to plot.")

    labels = [SHORT_NAMES.get(r["model"], r["model"]) for r in totals]
    rec_t = [r["rate_RT_pct"] for r in totals]
    rec_i = [r["rate_RI_pct"] for r in totals]

    x = np.arange(len(labels))
    width = 0.36
    figure, axes = plt.subplots(figsize=(6.4, 3.0))

    bars_t = axes.bar(
        x - width / 2, rec_t, width,
        label="Text-solvable ($R_T$)",
        color="#4C9F8B", edgecolor="#2f6b5d", linewidth=0.7,
    )
    bars_i = axes.bar(
        x + width / 2, rec_i, width,
        label="Image-solvable ($R_I$)",
        color="#E08A6E", edgecolor="#a85c42", linewidth=0.7,
    )

    axes.axhline(100, ls=":", lw=0.9, c="#888")
    axes.text(
        len(labels) - 0.42, 101.5, "perfect integration",
        fontsize=7, color="#666", ha="right",
    )
    for bar in list(bars_t) + list(bars_i):
        axes.annotate(
            f"{bar.get_height():.0f}",
            (bar.get_x() + bar.get_width() / 2, bar.get_height()),
            textcoords="offset points", xytext=(0, 2),
            ha="center", fontsize=7.5,
        )

    axes.set_ylabel("Recovered by text+image (%)", fontsize=9)
    axes.set_ylim(0, 116)
    axes.set_xticks(x)
    axes.set_xticklabels(labels, fontsize=9)
    axes.tick_params(axis="y", labelsize=8)
    axes.legend(fontsize=8, frameon=False, ncol=2, loc="upper left")
    for spine in ("top", "right"):
        axes.spines[spine].set_visible(False)
    axes.set_axisbelow(True)
    axes.grid(axis="y", lw=0.4, c="#ddd")
    figure.tight_layout(pad=0.4)

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=300)
    plt.close(figure)
    print(f"Wrote {path}")


def write_csv(rows, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {path} ({len(rows)} rows)")


def print_summary(overlap_rows, recovery_rows):
    """Print the aggregate figures the paper quotes, so they can be checked."""
    n_cells = len(overlap_rows)
    core_largest = [r for r in overlap_rows if r["largest_region"] == "all_three_errors"]
    max_both_only = max(r["both_only_errors_pct"] for r in overlap_rows)

    models = sorted({r["model"] for r in overlap_rows})
    print(f"\nScope: {len(models)} model(s), {n_cells} model/task cells")
    print("\n--- Finding 2 ---")
    print(
        f"shared error core is the largest region in "
        f"{len(core_largest)}/{n_cells} cells"
    )
    print("cells where it is not:")
    for r in overlap_rows:
        if r["largest_region"] != "all_three_errors":
            print(
                f"  {r['model']:26} {r['task']:20} largest={r['largest_region']}"
                f" ({r[r['largest_region'] + '_pct']}%)"
                f" vs core {r['all_three_errors_pct']}%"
            )
    print(f"max errors unique to text+image = {max_both_only}%")

    print("\n--- Finding 3 ---")
    for r in recovery_rows:
        if r["task"] == "ALL":
            print(
                f"  {r['model']:26} R_T {r['rate_RT_pct']}% (n={r['n_RT']})   "
                f"R_I {r['rate_RI_pct']}% (n={r['n_RI']})   "
                f"overall {r['rate_overall_pct']}%"
            )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-root",
        default="results/baseline",
        help="Root of the baseline result tree.",
    )
    parser.add_argument(
        "--output-dir",
        default="evaluation",
        help="Directory the CSV reports are written to.",
    )
    parser.add_argument(
        "--image-type",
        default="spring",
        help="Keep only this image_type. Use 'all' to include every layout.",
    )
    parser.add_argument(
        "--task",
        action="append",
        dest="tasks",
        help="Restrict to one task. Repeat to pass several. "
        "Defaults to the eight tasks in the paper's Table 1.",
    )
    parser.add_argument(
        "--model",
        action="append",
        dest="models",
        help="Restrict to one model. Repeat to pass several. Defaults to the "
        "four models in the paper's Table 1.",
    )
    parser.add_argument(
        "--all-models",
        action="store_true",
        help="Include every model found, not just the Table 1 four.",
    )
    parser.add_argument(
        "--plot",
        nargs="?",
        const="See_And_Talk_Like_A_Graph paper/figures/visual_augmentation/"
        "complementary_recovery.png",
        default=None,
        help="Also regenerate the complementary-recovery figure. "
        "Optionally give an output path.",
    )
    parser.add_argument(
        "--summary",
        action="store_true",
        help="Also print the aggregate figures quoted in the paper.",
    )
    args = parser.parse_args()

    results_root = Path(args.results_root)
    if not results_root.exists():
        raise FileNotFoundError(f"Results root does not exist: {results_root}")

    tasks = set(args.tasks) if args.tasks else set(TASK_ORDER)
    if args.models:
        models = set(args.models)
    elif args.all_models:
        models = None
    else:
        models = set(MODEL_ORDER)

    grouped = collect_error_sets(results_root, image_type=args.image_type)
    if not grouped:
        raise SystemExit(f"No scorable results found under {results_root}")

    overlap_rows = build_overlap_rows(grouped, tasks=tasks, models=models)
    recovery_rows = build_recovery_rows(grouped, tasks=tasks, models=models)

    if not overlap_rows:
        raise SystemExit("No model/task pair had all three settings scored.")

    out = Path(args.output_dir)
    write_csv(overlap_rows, out / "error_overlap_by_cell.csv")
    write_csv(recovery_rows, out / "complementary_recovery.csv")

    if args.plot:
        plot_recovery(recovery_rows, args.plot)

    if args.summary:
        print_summary(overlap_rows, recovery_rows)


if __name__ == "__main__":
    main()
