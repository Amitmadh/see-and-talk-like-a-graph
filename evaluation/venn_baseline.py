"""
Error-overlap Venn diagrams for the visual-augmentation (baseline) experiment.

For every model and graph-reasoning task, compare the same samples across
the three modality settings and draw a 3-circle Venn of their errors:

    evaluation/vis/{model}/venn_baseline/{task}.png

Each circle is one setting (text-only, image-only, text-and-image).
Region labels are the percentage of compared samples that were
mispredicted in that overlap. Circle areas are proportional to each
setting's error rate.

A sample counts as an error when the existing evaluator marks it incorrect
or wrong-format. Only samples scored in all three settings are included.

Run from the project root:

    python -m evaluation.venn_baseline
"""

from __future__ import annotations

import argparse
import sys
import warnings
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib_venn import venn3


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.evaluation import (
    TASK_TYPES,
    compare,
    get_file_metadata,
    load_results_file,
)


BACKGROUND = "#F7F4EF"
INK = "#1B1B1B"
MUTED = "#5C6570"

SETTINGS = (
    "text_only",
    "image_only",
    "text_and_image",
)

SETTING_LABELS = {
    "text_only": "Text",
    "image_only": "Image",
    "text_and_image": "Image+Text",
}

SETTING_COLORS = {
    "text_only": "#2A9D8F",
    "image_only": "#E76F51",
    "text_and_image": "#1D3557",
}

# matplotlib-venn subset order: (Abc, aBc, ABc, abC, AbC, aBC, ABC)
# A = text-only, B = image-only, C = text-and-image
SUBSET_KEYS = ("A", "B", "AB", "C", "AC", "BC", "ABC")


def _pretty_task(task):
    return task.replace("_", " ")


def _as_percent(count, total):
    if not total:
        return 0.0
    return 100.0 * count / total


def _format_percent(count, total):
    return f"{_as_percent(count, total):.1f}"


def _score_rows(task, rows, image_type):
    """
    Map sample_id -> error flag for rows that the evaluator can score.

    None answers and unparseable ground truth are skipped so they do not
    appear as exclusive errors in only one setting.
    """

    scored = {}
    model = None
    used_image_type = None

    for row in rows:
        row_image = row.get("image_type", "unknown")
        if image_type not in {None, "all"} and row_image != image_type:
            continue

        if row.get("model_answer") is None:
            continue

        sample_id = row.get("sample_id")
        if not sample_id:
            continue

        is_correct, status, _ = compare(
            task,
            row.get("expected_answer"),
            row.get("model_answer"),
        )

        if status == "bad_expected":
            continue

        scored[sample_id] = not is_correct
        model = row.get("model", model)
        used_image_type = row_image

    return scored, model, used_image_type


def collect_error_sets(results_root, image_type="spring"):
    """
    Load every baseline JSONL and group error sets by (model, task).

    Returns a dict:
        (model, task) -> {
            "scored": {setting: set(sample_id)},
            "errors": {setting: set(sample_id)},
        }
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
        setting = metadata["setting"]
        task = metadata["task"]

        if setting not in SETTINGS or task not in TASK_TYPES:
            continue

        scored, model, _ = _score_rows(
            task,
            load_results_file(path),
            image_type,
        )

        if not scored or not model:
            continue

        bucket = grouped[(model, task)]
        error_ids = {sample_id for sample_id, is_error in scored.items() if is_error}
        bucket["scored"][setting].update(scored)
        bucket["errors"][setting].update(error_ids)

    return grouped


def region_counts(error_sets):
    text = error_sets["text_only"]
    image = error_sets["image_only"]
    both = error_sets["text_and_image"]

    return {
        "A": len(text - image - both),
        "B": len(image - text - both),
        "C": len(both - text - image),
        "AB": len(text & image - both),
        "AC": len(text & both - image),
        "BC": len(image & both - text),
        "ABC": len(text & image & both),
    }


def _style_venn(venn):
    if venn is None:
        return

    for label in venn.set_labels or []:
        if label is not None:
            label.set_visible(False)

    for label in venn.subset_labels or []:
        if label is None:
            continue
        label.set_color(INK)
        label.set_fontsize(11)
        label.set_fontweight("semibold")

    for patch, setting in zip(venn.patches or [], SETTINGS):
        if patch is None:
            continue
        color = SETTING_COLORS[setting]
        patch.set_edgecolor(color)
        patch.set_linewidth(1.3)
        patch.set_alpha(0.5)


def draw_error_venn(error_sets, output_path, task, model, n_compared, n_all_correct):
    counts = region_counts(error_sets)
    subsets = tuple(counts[key] for key in SUBSET_KEYS)

    fig, ax = plt.subplots(figsize=(7.1, 6.8))
    fig.patch.set_facecolor(BACKGROUND)
    ax.set_facecolor(BACKGROUND)
    ax.set_aspect("equal")
    ax.axis("off")

    ax.set_title(
        f"{_pretty_task(task)}  ·  {model}",
        color=INK,
        fontsize=13,
        pad=10,
        loc="center",
    )

    if sum(subsets) == 0:
        ax.text(
            0.5,
            0.52,
            "No errors in any setting",
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=12,
            color=MUTED,
        )
    else:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="Circle .* has zero area",
            )
            venn = venn3(
                subsets=subsets,
                set_labels=("", "", ""),
                set_colors=tuple(
                    SETTING_COLORS[setting] for setting in SETTINGS
                ),
                alpha=0.5,
                ax=ax,
                subset_label_formatter=lambda n: _format_percent(
                    n,
                    n_compared,
                ),
            )
        _style_venn(venn)

    handles = [
        Patch(
            facecolor=SETTING_COLORS[setting],
            edgecolor=SETTING_COLORS[setting],
            alpha=0.55,
            label=(
                f"{SETTING_LABELS[setting]}  "
                f"({_format_percent(len(error_sets[setting]), n_compared)}%)"
            ),
        )
        for setting in SETTINGS
    ]
    legend = ax.legend(
        handles=handles,
        loc="lower center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, -0.02),
        handlelength=1.1,
        handleheight=1.1,
        borderaxespad=0.2,
        fontsize=10,
        labelcolor=INK,
    )
    legend.set_title("Error rate by setting", prop={"size": 10})
    legend.get_title().set_color(MUTED)

    ax.text(
        0.5,
        -0.08,
        (
            f"Values are % of {n_compared} samples"
            f"   ·   {_format_percent(n_all_correct, n_compared)}% "
            f"correct in every setting"
        ),
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=9,
        color=MUTED,
    )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        output_path,
        dpi=170,
        bbox_inches="tight",
        facecolor=BACKGROUND,
        pad_inches=0.22,
    )
    plt.close(fig)
    return output_path


def write_venn_diagrams(
    grouped,
    output_root="evaluation/vis",
    models=None,
    tasks=None,
):
    """
    Write one PNG per (model, task) under
    {output_root}/{model}/venn_baseline/{task}.png
    """

    output_root = Path(output_root)
    written = []
    skipped = []

    items = sorted(grouped.items(), key=lambda item: (item[0][0], item[0][1]))

    for (model, task), bucket in items:
        if models and model not in models:
            continue
        if tasks and task not in tasks:
            continue

        scored = bucket["scored"]
        missing = [setting for setting in SETTINGS if not scored[setting]]
        if missing:
            skipped.append((model, task, missing))
            continue

        common = set.intersection(*(scored[setting] for setting in SETTINGS))
        if not common:
            skipped.append((model, task, ["no overlapping sample_ids"]))
            continue

        error_sets = {
            setting: bucket["errors"][setting] & common
            for setting in SETTINGS
        }
        all_errors = set.union(*error_sets.values())
        n_all_correct = len(common) - len(all_errors)

        output_path = (
            output_root / model / "venn_baseline" / f"{task}.png"
        )
        draw_error_venn(
            error_sets,
            output_path,
            task=task,
            model=model,
            n_compared=len(common),
            n_all_correct=n_all_correct,
        )
        written.append(output_path)
        print(f"Saved {output_path}")

    return written, skipped


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Draw per-task, per-model error Venn diagrams "
            "for the three baseline modality settings."
        )
    )
    parser.add_argument(
        "--results-root",
        default="results/baseline",
        help="Root of the baseline result tree.",
    )
    parser.add_argument(
        "--output-dir",
        default="evaluation/vis",
        help="Visualization root. Files go to {output-dir}/{model}/venn_baseline/.",
    )
    parser.add_argument(
        "--image-type",
        default="spring",
        help="Keep only this image_type. Use 'all' to include every layout.",
    )
    parser.add_argument(
        "--model",
        action="append",
        dest="models",
        help="Restrict to one model. Repeat to pass several.",
    )
    parser.add_argument(
        "--task",
        action="append",
        dest="tasks",
        help="Restrict to one task. Repeat to pass several.",
    )
    args = parser.parse_args()

    root = Path(args.results_root)
    if not root.exists():
        raise FileNotFoundError(f"Results root does not exist: {root}")

    grouped = collect_error_sets(root, image_type=args.image_type)
    written, skipped = write_venn_diagrams(
        grouped,
        output_root=args.output_dir,
        models=args.models,
        tasks=args.tasks,
    )

    print(f"\nWrote {len(written)} Venn diagram(s).")
    if skipped:
        print(f"Skipped {len(skipped)} model/task pair(s):")
        for model, task, reason in skipped:
            print(f"  {model} / {task}: missing {', '.join(reason)}")


if __name__ == "__main__":
    main()
