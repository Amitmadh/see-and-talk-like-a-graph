"""
Grouped 100% stacked bars: which modality wins per task and model.

Reads existing mixed-signals result rows (CSV or scored/enriched JSON)
and writes one comparison figure under evaluation/vis/all_models/.
"""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Rectangle

from evaluation.evaluation import TASK_TYPES


TEXT_WIN_COLOR = "#B5D8A8"
IMAGE_WIN_COLOR = "#A4C8E8"
INCORRECT_COLOR = "#E8A8A0"

MODEL_ORDER = [
    "Qwen2.5-VL-3B-Instruct",
    "Qwen2.5-VL-7B-Instruct",
    "Phi-4-multimodal-instruct",
    "llava-v1.6-mistral-7b-hf",
    "gpt-4.1-mini",
]

MODEL_LABELS = {
    "Qwen2.5-VL-3B-Instruct": "Qwen2.5 3B",
    "Qwen2.5-VL-7B-Instruct": "Qwen2.5 7B",
    "Phi-4-multimodal-instruct": "Phi-4",
    "llava-v1.6-mistral-7b-hf": "LLaVA-1.6",
    "gpt-4.1-mini": "GPT-4.1 mini",
}

MODEL_HATCHES = {
    "Qwen2.5-VL-3B-Instruct": "///",
    "Qwen2.5-VL-7B-Instruct": r"\\",
    "Phi-4-multimodal-instruct": "xx",
    "llava-v1.6-mistral-7b-hf": "..",
    "gpt-4.1-mini": "oo",
}

FALLBACK_HATCHES = ["///", r"\\", "xx", "oo", "||", "..", "--"]


def _pretty_task(task):
    return str(task).replace("_", " ").title()


def _pretty_model(model):
    return MODEL_LABELS.get(model, model)


def _ordered_tasks(tasks):
    known = [task for task in TASK_TYPES if task in tasks]
    extra = sorted(task for task in tasks if task not in TASK_TYPES)
    return known + extra


def _ordered_models(models):
    known = [model for model in MODEL_ORDER if model in models]
    extra = sorted(model for model in models if model not in known)
    return known + extra


def _model_hatch(model, index):
    if model in MODEL_HATCHES:
        return MODEL_HATCHES[model]

    return FALLBACK_HATCHES[index % len(FALLBACK_HATCHES)]


def _as_int(value):
    if value in (None, ""):
        return 0

    return int(float(value))


def load_mixed_result_rows(path):
    """
    Load mixed-signals file-level rows from CSV or JSON.

    JSON may be a list of rows or an object with a ``files`` list.
    Sample-level payloads are ignored; only the per-file counts are used.
    """

    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Mixed-signals results not found: {path}")

    if path.suffix.lower() == ".csv":
        with open(path, newline="", encoding="utf-8") as handle:
            return list(csv.DictReader(handle))

    with open(path, encoding="utf-8") as handle:
        payload = json.load(handle)

    if isinstance(payload, list):
        return payload

    rows = payload.get("files")

    if not rows:
        raise ValueError(f"No 'files' found in {path}")

    return rows


def aggregate_task_model_rows(rows, image_type="spring"):
    """
    Pool image / text / neither counts for each (task, model) pair.

    Percentages later use only the three-way modality classifications,
    matching the rest of the mixed-signals analysis.
    """

    buckets = defaultdict(
        lambda: {
            "text_wins": 0,
            "image_wins": 0,
            "neither_wins": 0,
        }
    )

    for row in rows:
        if image_type not in {None, "", "all"}:
            row_image_type = row.get("image_type")

            if row_image_type not in {None, "", image_type}:
                continue

        task = row.get("task")
        model = row.get("model")

        if not task or not model:
            continue

        entry = buckets[(task, model)]
        entry["text_wins"] += _as_int(row.get("text_wins"))
        entry["image_wins"] += _as_int(row.get("image_wins"))
        entry["neither_wins"] += _as_int(row.get("neither_wins"))

    table = {}

    for (task, model), counts in buckets.items():
        total = (
            counts["text_wins"]
            + counts["image_wins"]
            + counts["neither_wins"]
        )

        if total <= 0:
            continue

        table[(task, model)] = {
            "total": total,
            "text": counts["text_wins"] / total,
            "image": counts["image_wins"] / total,
            "incorrect": counts["neither_wins"] / total,
        }

    return table


def write_mixed_all_models_plot(
    rows,
    output_path="evaluation/vis/all_models/by_task.png",
    image_type="spring",
):
    """
    Draw one grouped 100% stacked bar chart of modality wins.

    Each x-axis cluster is a graph task. Bars inside a cluster are one
    model each, stacked as Text-win / Image-win / Incorrect.
    """

    table = aggregate_task_model_rows(rows, image_type=image_type)

    if not table:
        return None

    tasks = _ordered_tasks({task for task, _ in table})
    models = _ordered_models({model for _, model in table})

    if not tasks or not models:
        return None

    n_tasks = len(tasks)
    n_models = len(models)

    group_width = 0.84
    bar_width = group_width / n_models
    offsets = [
        -group_width / 2 + bar_width * (index + 0.5)
        for index in range(n_models)
    ]

    fig_width = max(11.0, 1.35 * n_tasks + 2.4)
    fig, ax = plt.subplots(figsize=(fig_width, 5.6))

    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    plt.rcParams["hatch.linewidth"] = 0.55
    plt.rcParams["hatch.color"] = "0.25"

    x = list(range(n_tasks))
    stack = [
        ("text", TEXT_WIN_COLOR, "Text-win"),
        ("image", IMAGE_WIN_COLOR, "Image-win"),
        ("incorrect", INCORRECT_COLOR, "Incorrect"),
    ]

    for offset, model, index in zip(offsets, models, range(n_models)):
        hatch = _model_hatch(model, index)
        bottoms = [0.0] * n_tasks
        xs = [tick + offset for tick in x]

        for key, color, _label in stack:
            heights = [
                table.get((task, model), {}).get(key, 0.0)
                for task in tasks
            ]

            ax.bar(
                xs,
                heights,
                width=bar_width,
                bottom=bottoms,
                color=color,
                hatch=hatch,
                edgecolor="black",
                linewidth=0.45,
                label="_nolegend_",
            )

            bottoms = [
                bottom + height
                for bottom, height in zip(bottoms, heights)
            ]

    ax.set_ylabel("Percentage", fontsize=11)
    ax.set_ylim(0.0, 1.0)
    ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_xlim(-0.55, n_tasks - 0.45)
    ax.set_xticks(x)
    ax.set_xticklabels(
        [_pretty_task(task) for task in tasks],
        rotation=45,
        ha="right",
    )

    ax.yaxis.grid(True, linestyle=":", linewidth=0.6, color="0.75")
    ax.set_axisbelow(True)

    for spine in ax.spines.values():
        spine.set_color("black")
        spine.set_linewidth(0.8)

    outcome_handles = [
        Patch(facecolor=color, edgecolor="black", linewidth=0.5, label=label)
        for _key, color, label in stack
    ]
    model_handles = [
        Rectangle(
            (0, 0),
            1,
            1,
            facecolor="white",
            edgecolor="black",
            linewidth=0.8,
            hatch=_model_hatch(model, index),
            label=_pretty_model(model),
        )
        for index, model in enumerate(models)
    ]

    outcome_legend = ax.legend(
        handles=outcome_handles,
        loc="upper center",
        ncol=len(stack),
        bbox_to_anchor=(0.42, 1.14),
        frameon=True,
        fancybox=False,
        edgecolor="black",
        fontsize=9,
        borderpad=0.4,
    )
    ax.add_artist(outcome_legend)

    ax.legend(
        handles=model_handles,
        loc="upper right",
        bbox_to_anchor=(1.0, 1.14),
        frameon=True,
        fancybox=False,
        edgecolor="black",
        fontsize=8.5,
        borderpad=0.4,
        handleheight=1.8,
        handlelength=1.6,
    )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig.savefig(
        output_path,
        dpi=180,
        bbox_inches="tight",
        facecolor="white",
        pad_inches=0.18,
    )
    plt.close(fig)

    return output_path


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Plot mixed-signals modality wins for every "
            "task and model from existing evaluation results."
        )
    )
    parser.add_argument(
        "--input",
        default="evaluation/mixed_signals_results.csv",
        help=(
            "Mixed-signals CSV or JSON with per-file "
            "image/text/neither counts."
        ),
    )
    parser.add_argument(
        "--output",
        default="evaluation/vis/all_models/by_task.png",
        help="Where to write the comparison PNG.",
    )
    parser.add_argument(
        "--image-type",
        default="spring",
        help="Image type to include. Use 'all' for every type.",
    )
    args = parser.parse_args()

    rows = load_mixed_result_rows(args.input)
    path = write_mixed_all_models_plot(
        rows,
        output_path=args.output,
        image_type=args.image_type,
    )

    if path is None:
        raise SystemExit("No mixed-signals modality counts found to plot.")

    print(f"Saved all-models mixed-signals plot to {path}")


if __name__ == "__main__":
    main()
