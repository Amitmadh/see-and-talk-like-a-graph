"""
All-model mixed-signals comparison figures.

``by_task`` — grouped 100% stacked bars of modality wins per task.

``by_edges`` — the same stacks, one cluster per 10-edge bin.

``by_edges_20`` — the same stacks, one cluster per 20-edge bin.

``by_algorithm`` — small multiples of the same stacks, one panel per
task, with a bar group for each focus graph shape (SFN, Complete,
Star, Path). Models sit left-to-right inside each group so the
figure stays readable without hatches.

``by_algorithm_model`` — the same stacks, one row per model and one
panel per focus task.

Reads existing mixed-signals result rows (CSV or scored/enriched JSON)
and writes figures under evaluation/vis/all_models/.
The algorithm figure needs sample-level rows (enriched JSON).
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

FOCUS_ALGORITHMS = ["sfn", "complete", "star", "path"]

ALL_ALGORITHMS = [
    "er",
    "ba",
    "sbm",
    "sfn",
    "complete",
    "star",
    "path",
]

ALGORITHM_LABELS = {
    "er": "ER",
    "ba": "BA",
    "sbm": "SBM",
    "sfn": "SFN",
    "complete": "Complete",
    "star": "Star",
    "path": "Path",
}

DEFAULT_EXCLUDED_MODELS = ("Qwen2.5-VL-3B-Instruct",)

FOCUS_TASKS_BY_MODEL = [
    "edge_existence",
    "shortest_path",
    "edge_count",
    "node_degree",
    "connected_nodes",
]

STACK_SPEC = [
    ("text", TEXT_WIN_COLOR, "Text-win"),
    ("image", IMAGE_WIN_COLOR, "Image-win"),
    ("incorrect", INCORRECT_COLOR, "Incorrect"),
]


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


def _ordered_edge_bins(bins):
    return sorted(bins, key=lambda value: int(str(value).split("-")[0]))


def _ordered_focus_tasks(tasks, focus=None):
    if not focus:
        return _ordered_tasks(tasks)

    known = [task for task in focus if task in tasks]
    extra = sorted(task for task in tasks if task not in focus)
    return known + extra


def _draw_algorithm_stacks(ax, table, task, model, algorithms):
    """Draw 100% stacked modality bars for one (task, model)."""

    n_algorithms = len(algorithms)
    x = list(range(n_algorithms))

    for boundary in range(1, n_algorithms):
        ax.axvline(
            boundary - 0.5,
            color="0.88",
            linewidth=0.6,
            zorder=0,
        )

    bottoms = [0.0] * n_algorithms

    for key, color, _label in STACK_SPEC:
        heights = []

        for algorithm in algorithms:
            entry = table.get((task, model, algorithm), {})
            heights.append(entry.get(key, 0.0) if entry else 0.0)

        ax.bar(
            x,
            heights,
            width=0.72 if n_algorithms <= 4 else 0.58,
            bottom=bottoms,
            color=color,
            edgecolor="black",
            linewidth=0.35,
            label="_nolegend_",
        )
        bottoms = [
            bottom + height
            for bottom, height in zip(bottoms, heights)
        ]

    return x


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


def _sample_algorithm(sample):
    algorithm = sample.get("algorithm")

    if algorithm:
        return str(algorithm)

    graph = sample.get("graph")

    if isinstance(graph, dict):
        algorithm = graph.get("algorithm")

        if algorithm:
            return str(algorithm)

    return None


def aggregate_task_model_algorithm_rows(
    rows,
    image_type="spring",
    algorithms=None,
    exclude_models=None,
):
    """
    Pool image / text / neither counts for each
    (task, model, algorithm) triple from sample-level rows.
    """

    focus = list(algorithms or FOCUS_ALGORITHMS)
    focus_set = set(focus)
    excluded = set(exclude_models or ())

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

        if not task or not model or model in excluded:
            continue

        samples = row.get("samples") or []

        if not samples:
            continue

        for sample in samples:
            algorithm = _sample_algorithm(sample)

            if algorithm not in focus_set:
                continue

            winner = sample.get("modality_winner")

            if winner == "text":
                buckets[(task, model, algorithm)]["text_wins"] += 1
            elif winner == "image":
                buckets[(task, model, algorithm)]["image_wins"] += 1
            elif winner == "neither":
                buckets[(task, model, algorithm)]["neither_wins"] += 1

    table = {}

    for key, counts in buckets.items():
        total = (
            counts["text_wins"]
            + counts["image_wins"]
            + counts["neither_wins"]
        )

        if total <= 0:
            continue

        table[key] = {
            "total": total,
            "text": counts["text_wins"] / total,
            "image": counts["image_wins"] / total,
            "incorrect": counts["neither_wins"] / total,
        }

    return table


def aggregate_edge_model_rows(rows, image_type="spring", bin_size=10):
    """
    Pool image / text / neither counts for each (edge bin, model) pair.

    Samples are binned with the same 10-edge intervals used by the
    mixed-signals evaluator: 0-10, 11-20, 21-30, ...
    """

    from evaluation.mixed_evaluation import get_edge_bin, get_edge_count

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

        model = row.get("model")

        if not model:
            continue

        for sample in row.get("samples") or []:
            winner = sample.get("modality_winner")

            if winner not in {"text", "image", "neither"}:
                continue

            n_edges = get_edge_count(sample)

            if n_edges is None:
                continue

            edge_bin = get_edge_bin(n_edges, bin_size=bin_size)

            if winner == "text":
                buckets[(edge_bin, model)]["text_wins"] += 1
            elif winner == "image":
                buckets[(edge_bin, model)]["image_wins"] += 1
            elif winner == "neither":
                buckets[(edge_bin, model)]["neither_wins"] += 1

    table = {}

    for key, counts in buckets.items():
        total = (
            counts["text_wins"]
            + counts["image_wins"]
            + counts["neither_wins"]
        )

        if total <= 0:
            continue

        table[key] = {
            "total": total,
            "text": counts["text_wins"] / total,
            "image": counts["image_wins"] / total,
            "incorrect": counts["neither_wins"] / total,
        }

    return table


def _write_grouped_stacked_modality_plot(
    table,
    groups,
    models,
    output_path,
    xticklabels,
    xlabel=None,
    fig_width=None,
):
    """Draw one grouped 100% stacked bar chart of modality wins."""

    n_groups = len(groups)
    n_models = len(models)

    group_width = 0.84
    bar_width = group_width / n_models
    offsets = [
        -group_width / 2 + bar_width * (index + 0.5)
        for index in range(n_models)
    ]

    if fig_width is None:
        fig_width = max(11.0, 1.35 * n_groups + 2.4)

    fig, ax = plt.subplots(figsize=(fig_width, 5.6))

    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    plt.rcParams["hatch.linewidth"] = 0.55
    plt.rcParams["hatch.color"] = "0.25"

    x = list(range(n_groups))

    for offset, model, index in zip(offsets, models, range(n_models)):
        hatch = _model_hatch(model, index)
        bottoms = [0.0] * n_groups
        xs = [tick + offset for tick in x]

        for key, color, _label in STACK_SPEC:
            heights = [
                table.get((group, model), {}).get(key, 0.0)
                for group in groups
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
    ax.set_xlim(-0.55, n_groups - 0.45)
    ax.set_xticks(x)
    ax.set_xticklabels(xticklabels, rotation=45, ha="right")

    if xlabel:
        ax.set_xlabel(xlabel, fontsize=11)

    ax.yaxis.grid(True, linestyle=":", linewidth=0.6, color="0.75")
    ax.set_axisbelow(True)

    for spine in ax.spines.values():
        spine.set_color("black")
        spine.set_linewidth(0.8)

    outcome_handles = [
        Patch(facecolor=color, edgecolor="black", linewidth=0.5, label=label)
        for _key, color, label in STACK_SPEC
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
        ncol=len(STACK_SPEC),
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

    return _write_grouped_stacked_modality_plot(
        table,
        tasks,
        models,
        output_path,
        xticklabels=[_pretty_task(task) for task in tasks],
    )


def write_mixed_all_models_edges_plot(
    rows,
    output_path="evaluation/vis/all_models/by_edges.png",
    image_type="spring",
    bin_size=10,
):
    """
    Draw one grouped 100% stacked bar chart of modality wins by graph size.

    Each x-axis cluster is an edge-count bin of ``bin_size``
    (10 → 0-10, 11-20, …; 20 → 0-20, 21-40, …). Bars inside a
    cluster are one model each, stacked as Text-win / Image-win /
    Incorrect. Counts are pooled across tasks so the figure shows
    how reliance shifts with the number of edges.
    """

    table = aggregate_edge_model_rows(
        rows,
        image_type=image_type,
        bin_size=bin_size,
    )

    if not table:
        return None

    bins = _ordered_edge_bins({edge_bin for edge_bin, _ in table})
    models = _ordered_models({model for _, model in table})

    if not bins or not models:
        return None

    return _write_grouped_stacked_modality_plot(
        table,
        bins,
        models,
        output_path,
        xticklabels=bins,
        xlabel="Number of Edges",
        fig_width=max(12.0, 1.05 * len(bins) + 2.6),
    )


def write_mixed_algorithm_models_plot(
    rows,
    output_path="evaluation/vis/all_models/by_algorithm.png",
    image_type="spring",
    algorithms=None,
    exclude_models=None,
):
    """
    Draw one small-multiples figure of modality wins by graph shape.

    Each panel is a task. Inside a panel, each x-tick is one
    algorithm (SFN / Complete / Star / Path) and holds one 100%
    stacked bar per model, left to right.
    """

    algorithms = list(algorithms or FOCUS_ALGORITHMS)
    exclude_models = tuple(
        exclude_models
        if exclude_models is not None
        else DEFAULT_EXCLUDED_MODELS
    )

    table = aggregate_task_model_algorithm_rows(
        rows,
        image_type=image_type,
        algorithms=algorithms,
        exclude_models=exclude_models,
    )

    if not table:
        return None

    tasks = _ordered_tasks({task for task, _, _ in table})
    models = _ordered_models({model for _, model, _ in table})

    if not tasks or not models:
        return None

    n_tasks = len(tasks)
    n_models = len(models)
    n_algorithms = len(algorithms)
    n_cols = 2 if n_tasks > 1 else 1
    n_rows = (n_tasks + n_cols - 1) // n_cols

    fig_width = 6.4 * n_cols + 0.6
    fig_height = 2.35 * n_rows + 0.95
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(fig_width, fig_height),
        sharey=True,
        squeeze=False,
    )

    fig.patch.set_facecolor("white")

    group_width = 0.68
    bar_width = group_width / n_models
    offsets = [
        -group_width / 2 + bar_width * (index + 0.5)
        for index in range(n_models)
    ]

    stack = [
        ("text", TEXT_WIN_COLOR, "Text-win"),
        ("image", IMAGE_WIN_COLOR, "Image-win"),
        ("incorrect", INCORRECT_COLOR, "Incorrect"),
    ]

    x = list(range(n_algorithms))

    for index, task in enumerate(tasks):
        row = index // n_cols
        col = index % n_cols
        ax = axes[row][col]
        ax.set_facecolor("white")

        for boundary in range(1, n_algorithms):
            ax.axvline(
                boundary - 0.5,
                color="0.88",
                linewidth=0.6,
                zorder=0,
            )

        for offset, model in zip(offsets, models):
            bottoms = [0.0] * n_algorithms
            xs = [tick + offset for tick in x]

            for key, color, _label in stack:
                heights = [
                    table.get((task, model, algorithm), {}).get(key, 0.0)
                    for algorithm in algorithms
                ]
                present = [
                    (task, model, algorithm) in table
                    for algorithm in algorithms
                ]

                ax.bar(
                    xs,
                    [
                        height if seen else 0.0
                        for height, seen in zip(heights, present)
                    ],
                    width=bar_width * 0.88,
                    bottom=bottoms,
                    color=color,
                    edgecolor="black",
                    linewidth=0.35,
                    label="_nolegend_",
                )

                bottoms = [
                    bottom + (height if seen else 0.0)
                    for bottom, height, seen in zip(
                        bottoms,
                        heights,
                        present,
                    )
                ]

        ax.set_title(
            _pretty_task(task),
            fontsize=10,
            pad=4,
        )
        ax.set_ylim(0.0, 1.0)
        ax.set_yticks([0.0, 0.5, 1.0])
        ax.set_xlim(-0.55, n_algorithms - 0.45)
        ax.set_xticks(x)
        ax.set_xticklabels(
            [
                ALGORITHM_LABELS.get(algorithm, algorithm)
                for algorithm in algorithms
            ],
            fontsize=9,
        )
        ax.yaxis.grid(True, linestyle=":", linewidth=0.55, color="0.78")
        ax.set_axisbelow(True)

        if col == 0:
            ax.set_ylabel("Percentage", fontsize=9)
            ax.set_yticklabels(["0", "50", "100"], fontsize=8)
        else:
            ax.tick_params(axis="y", length=0)

        for spine in ax.spines.values():
            spine.set_color("black")
            spine.set_linewidth(0.7)

    for index in range(n_tasks, n_rows * n_cols):
        axes[index // n_cols][index % n_cols].set_visible(False)

    outcome_handles = [
        Patch(facecolor=color, edgecolor="black", linewidth=0.5, label=label)
        for _key, color, label in stack
    ]
    model_order = "  ·  ".join(_pretty_model(model) for model in models)

    fig.legend(
        handles=outcome_handles,
        loc="upper center",
        ncol=len(stack),
        bbox_to_anchor=(0.5, 1.015),
        frameon=False,
        fontsize=10,
        borderpad=0.2,
    )
    fig.text(
        0.5,
        0.975,
        f"Within each algorithm, bars are  {model_order}  (left → right)",
        ha="center",
        va="center",
        fontsize=9,
        color="0.25",
    )

    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.94), w_pad=0.9, h_pad=0.95)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig.savefig(
        output_path,
        dpi=180,
        bbox_inches="tight",
        facecolor="white",
        pad_inches=0.2,
    )
    plt.close(fig)

    return output_path


def write_mixed_algorithm_by_model_plot(
    rows,
    output_path="evaluation/vis/all_models/by_algorithm_model.png",
    image_type="spring",
    algorithms=None,
    exclude_models=None,
    tasks=None,
):
    """
    Draw one figure with a row per model and a panel per focus task.

    Each panel is a 100% stacked bar chart over every graph
    generator algorithm present in GraphQA.
    """

    algorithms = list(algorithms or ALL_ALGORITHMS)
    exclude_models = tuple(
        exclude_models
        if exclude_models is not None
        else DEFAULT_EXCLUDED_MODELS
    )
    focus_tasks = list(tasks or FOCUS_TASKS_BY_MODEL)

    table = aggregate_task_model_algorithm_rows(
        rows,
        image_type=image_type,
        algorithms=algorithms,
        exclude_models=exclude_models,
    )

    if not table:
        return None

    models = _ordered_models({model for _, model, _ in table})
    plot_tasks = _ordered_focus_tasks(
        {task for task, _, _ in table},
        focus=focus_tasks,
    )
    plot_tasks = [task for task in plot_tasks if task in focus_tasks]

    if not models or not plot_tasks:
        return None

    n_models = len(models)
    n_tasks = len(plot_tasks)
    n_algorithms = len(algorithms)

    fig_width = 3.2 * n_tasks + 1.4
    fig_height = 2.25 * n_models + 1.05
    fig, axes = plt.subplots(
        n_models,
        n_tasks,
        figsize=(fig_width, fig_height),
        sharey=True,
        squeeze=False,
    )

    fig.patch.set_facecolor("white")

    for row, model in enumerate(models):
        for col, task in enumerate(plot_tasks):
            ax = axes[row][col]
            ax.set_facecolor("white")

            x = _draw_algorithm_stacks(
                ax,
                table,
                task,
                model,
                algorithms,
            )

            ax.set_ylim(0.0, 1.0)
            ax.set_yticks([0.0, 0.5, 1.0])
            ax.set_xlim(-0.55, n_algorithms - 0.45)
            ax.set_xticks(x)
            ax.yaxis.grid(True, linestyle=":", linewidth=0.55, color="0.78")
            ax.set_axisbelow(True)

            for spine in ax.spines.values():
                spine.set_color("black")
                spine.set_linewidth(0.7)

            if row == 0:
                ax.set_title(_pretty_task(task), fontsize=10, pad=5)

            if row == n_models - 1:
                ax.set_xticklabels(
                    [
                        ALGORITHM_LABELS.get(algorithm, algorithm)
                        for algorithm in algorithms
                    ],
                    fontsize=7.5,
                    rotation=45,
                    ha="right",
                )
            else:
                ax.set_xticklabels([])

            if col == 0:
                ax.set_ylabel(
                    _pretty_model(model),
                    fontsize=10,
                )
                ax.set_yticklabels(["0", "50", "100"], fontsize=8)
            else:
                ax.tick_params(axis="y", length=0)

    outcome_handles = [
        Patch(facecolor=color, edgecolor="black", linewidth=0.5, label=label)
        for _key, color, label in STACK_SPEC
    ]

    fig.legend(
        handles=outcome_handles,
        loc="upper center",
        ncol=len(STACK_SPEC),
        bbox_to_anchor=(0.5, 1.02),
        frameon=False,
        fontsize=10,
        borderpad=0.2,
    )

    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.94), w_pad=0.45, h_pad=0.55)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig.savefig(
        output_path,
        dpi=180,
        bbox_inches="tight",
        facecolor="white",
        pad_inches=0.2,
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
        "--enriched-input",
        default="evaluation/enrich_mixed_signals_results.json",
        help=(
            "Enriched JSON with sample-level algorithm "
            "labels, used for the by-algorithm figure."
        ),
    )
    parser.add_argument(
        "--output",
        default="evaluation/vis/all_models/by_task.png",
        help="Where to write the by-task comparison PNG.",
    )
    parser.add_argument(
        "--output-algorithm",
        default="evaluation/vis/all_models/by_algorithm.png",
        help="Where to write the by-algorithm small-multiples PNG.",
    )
    parser.add_argument(
        "--output-algorithm-model",
        default="evaluation/vis/all_models/by_algorithm_model.png",
        help="Where to write the per-model algorithm small-multiples PNG.",
    )
    parser.add_argument(
        "--output-edges",
        default="evaluation/vis/all_models/by_edges.png",
        help="Where to write the 10-edge-bin comparison PNG.",
    )
    parser.add_argument(
        "--output-edges-20",
        default="evaluation/vis/all_models/by_edges_20.png",
        help="Where to write the 20-edge-bin comparison PNG.",
    )
    parser.add_argument(
        "--image-type",
        default="spring",
        help="Image type to include. Use 'all' for every type.",
    )
    parser.add_argument(
        "--skip-task",
        action="store_true",
        help="Skip the by-task comparison figure.",
    )
    parser.add_argument(
        "--skip-algorithm",
        action="store_true",
        help="Skip the by-algorithm small-multiples figure.",
    )
    parser.add_argument(
        "--skip-algorithm-model",
        action="store_true",
        help="Skip the per-model algorithm small-multiples figure.",
    )
    parser.add_argument(
        "--skip-edges",
        action="store_true",
        help="Skip the by-edge-count comparison figures.",
    )
    args = parser.parse_args()

    written = []

    if not args.skip_task:
        rows = load_mixed_result_rows(args.input)
        path = write_mixed_all_models_plot(
            rows,
            output_path=args.output,
            image_type=args.image_type,
        )

        if path is None:
            raise SystemExit(
                "No mixed-signals modality counts found to plot."
            )

        written.append(path)
        print(f"Saved all-models mixed-signals plot to {path}")

    need_enriched = (
        not args.skip_algorithm
        or not args.skip_algorithm_model
        or not args.skip_edges
    )
    algorithm_rows = None

    if need_enriched:
        enriched_path = Path(args.enriched_input)

        if not enriched_path.exists() and args.input != args.enriched_input:
            enriched_path = Path(args.input)

        algorithm_rows = load_mixed_result_rows(enriched_path)

    if not args.skip_algorithm:
        algorithm_path = write_mixed_algorithm_models_plot(
            algorithm_rows,
            output_path=args.output_algorithm,
            image_type=args.image_type,
        )

        if algorithm_path is None:
            raise SystemExit(
                "No sample-level algorithm counts found to plot. "
                "Pass enriched JSON via --enriched-input."
            )

        written.append(algorithm_path)
        print(
            "Saved all-models algorithm mixed-signals "
            f"plot to {algorithm_path}"
        )

    if not args.skip_algorithm_model:
        model_path = write_mixed_algorithm_by_model_plot(
            algorithm_rows,
            output_path=args.output_algorithm_model,
            image_type=args.image_type,
        )

        if model_path is None:
            raise SystemExit(
                "No sample-level algorithm counts found "
                "for the per-model figure. "
                "Pass enriched JSON via --enriched-input."
            )

        written.append(model_path)
        print(
            "Saved per-model algorithm mixed-signals "
            f"plot to {model_path}"
        )

    if not args.skip_edges:
        edge_outputs = (
            (args.output_edges, 10),
            (args.output_edges_20, 20),
        )

        for output_path, bin_size in edge_outputs:
            edges_path = write_mixed_all_models_edges_plot(
                algorithm_rows,
                output_path=output_path,
                image_type=args.image_type,
                bin_size=bin_size,
            )

            if edges_path is None:
                raise SystemExit(
                    "No sample-level edge-count counts found to plot. "
                    "Pass enriched JSON via --enriched-input."
                )

            written.append(edges_path)
            print(
                "Saved all-models "
                f"{bin_size}-edge-bin mixed-signals "
                f"plot to {edges_path}"
            )

    if not written:
        raise SystemExit("No mixed-signals figures were requested.")


if __name__ == "__main__":
    main()
