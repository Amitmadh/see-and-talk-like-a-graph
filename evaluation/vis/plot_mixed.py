"""
Per-task mixed-signals visualizations.

Each task gets two figures:

    evaluation/vis/algorithm/<task>.png
    evaluation/vis/edges/<task>.png

Plus one comparison across tasks:

    evaluation/vis/by_task.png

When generating visualizations for multiple models, each model should
be passed through `model_name` and written to its own output directory.
The model name is displayed at the top of every plot.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.ticker import PercentFormatter

from evaluation.evaluation import TASK_TYPES


IMAGE_COLOR = "#E76F51"
TEXT_COLOR = "#2A9D8F"
NEITHER_COLOR = "#9AA5B1"
BACKGROUND = "#F7F4EF"
PANEL = "#FFFcf7"
GRID = "#E6E0D6"
INK = "#1B1B1B"
MUTED = "#5C6570"

ALGORITHM_LABELS = {
    "er": "ER",
    "ba": "BA",
    "sbm": "SBM",
    "sfn": "SFN",
    "complete": "Complete",
    "star": "Star",
    "path": "Path",
}


def _pretty_task(task):
    if task in {"ALL", "OVERALL"}:
        return "All tasks"

    return task.replace("_", " ")


def _pretty_group(name, axis):
    if axis == "algorithm":
        return ALGORITHM_LABELS.get(name, str(name).upper())

    if axis == "task":
        return _pretty_task(name)

    return str(name)


def _ordered_tasks(tasks):
    known = [
        task
        for task in TASK_TYPES
        if task in tasks
    ]

    extra = sorted(
        task
        for task in tasks
        if task not in TASK_TYPES
    )

    return known + extra


def _task_entries(algorithm_distribution):
    by_task = algorithm_distribution.get("by_task", {})
    entries = []

    for task in _ordered_tasks(by_task):
        image = 0
        text = 0
        neither = 0

        for algo in by_task[task].get("algorithms", []):
            image += algo.get("image", 0)
            text += algo.get("text", 0)
            neither += algo.get("neither", 0)

        total = image + text + neither

        entries.append({
            "task": task,
            "total": total,
            "image": image,
            "text": text,
            "neither": neither,
            "image_pct": (
                round(100.0 * image / total, 2)
                if total
                else 0.0
            ),
            "text_pct": (
                round(100.0 * text / total, 2)
                if total
                else 0.0
            ),
            "neither_pct": (
                round(100.0 * neither / total, 2)
                if total
                else 0.0
            ),
        })

    return entries


def _style_axes(ax):
    ax.set_facecolor(PANEL)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.spines["left"].set_color(GRID)
    ax.spines["bottom"].set_color(GRID)

    ax.tick_params(
        colors=MUTED,
        length=0,
    )

    ax.yaxis.grid(
        True,
        color=GRID,
        linewidth=0.8,
    )

    ax.set_axisbelow(True)


def _annotate_stack(
    ax,
    x,
    image_pct,
    text_pct,
    neither_pct,
):
    bottoms = [
        0.0,
        image_pct,
        image_pct + text_pct,
    ]

    values = [
        image_pct,
        text_pct,
        neither_pct,
    ]

    colors = [
        "white",
        "white",
        INK,
    ]

    for bottom, value, color in zip(
        bottoms,
        values,
        colors,
    ):
        if value < 11:
            continue

        ax.text(
            x,
            bottom + value / 2,
            f"{value:.0f}%",
            ha="center",
            va="center",
            fontsize=8.5,
            fontweight="bold",
            color=color,
        )


def _draw_task_figure(
    task,
    entries,
    group_key,
    axis,
    output_path,
    rotate_xticks=False,
    model_name=None,
):
    """
    Draw a single stacked-bar visualization.

    Parameters
    ----------
    task:
        Task represented by the figure.

    entries:
        Data entries used to construct the stacked bars.

    group_key:
        Field used to group the bars.

    axis:
        Axis/group type, e.g. "algorithm", "edges", or "task".

    output_path:
        Path where the PNG will be written.

    rotate_xticks:
        Whether x-axis labels should be rotated.

    model_name:
        Name of the model. If provided, it is displayed at the
        beginning of the plot title.
    """

    if not entries:
        return None

    labels = [
        _pretty_group(
            entry[group_key],
            axis,
        )
        for entry in entries
    ]

    n_values = [
        int(
            entry.get("total", 0)
            or 0
        )
        for entry in entries
    ]

    image_pcts = [
        float(
            entry.get("image_pct", 0)
            or 0
        )
        for entry in entries
    ]

    text_pcts = [
        float(
            entry.get("text_pct", 0)
            or 0
        )
        for entry in entries
    ]

    neither_pcts = [
        float(
            entry.get("neither_pct", 0)
            or 0
        )
        for entry in entries
    ]

    n_groups = len(entries)

    width = max(
        10.0,
        0.55 * n_groups + 4.5,
    )

    fig, ax = plt.subplots(
        figsize=(width, 5.4)
    )

    fig.patch.set_facecolor(
        BACKGROUND
    )

    # ---------------------------------------------------------------
    # Plot title
    #
    # Every plot now includes the model name.
    #
    # Examples:
    #
    #   Qwen2.5-VL-3B-Instruct — Connected Nodes
    #   Qwen2.5-VL-3B-Instruct — All tasks
    #   Qwen2.5-VL-7B-Instruct — Edge Count
    #
    # For the by-task comparison figure, `task` is empty, so we use
    # "Comparison Across Tasks".
    # ---------------------------------------------------------------

    if task:
        title = _pretty_task(task)
    else:
        title = "Comparison Across Tasks"

    if model_name:
        title = f"{model_name} — {title}"

    fig.suptitle(
        title,
        fontsize=18,
        fontweight="bold",
        color=INK,
        x=0.02,
        ha="left",
        y=0.98,
    )

    show_stack_labels = (
        n_groups <= 14
    )

    show_n_labels = (
        n_groups <= 18
    )

    x = list(
        range(n_groups)
    )

    bar_width = (
        0.72
        if n_groups <= 12
        else 0.82
    )

    _style_axes(ax)

    # ---------------------------------------------------------------
    # Image
    # ---------------------------------------------------------------

    ax.bar(
        x,
        image_pcts,
        width=bar_width,
        color=IMAGE_COLOR,
        edgecolor=PANEL,
        linewidth=0.6,
        label="Image",
    )

    # ---------------------------------------------------------------
    # Text
    # ---------------------------------------------------------------

    ax.bar(
        x,
        text_pcts,
        width=bar_width,
        bottom=image_pcts,
        color=TEXT_COLOR,
        edgecolor=PANEL,
        linewidth=0.6,
        label="Text",
    )

    # ---------------------------------------------------------------
    # Neither
    # ---------------------------------------------------------------

    ax.bar(
        x,
        neither_pcts,
        width=bar_width,
        bottom=[
            i + t
            for i, t in zip(
                image_pcts,
                text_pcts,
            )
        ],
        color=NEITHER_COLOR,
        edgecolor=PANEL,
        linewidth=0.6,
        label="Neither",
    )

    # ---------------------------------------------------------------
    # Labels
    # ---------------------------------------------------------------

    for i, (
        image_pct,
        text_pct,
        neither_pct,
        n,
    ) in enumerate(
        zip(
            image_pcts,
            text_pcts,
            neither_pcts,
            n_values,
        )
    ):
        if show_stack_labels:
            _annotate_stack(
                ax,
                i,
                image_pct,
                text_pct,
                neither_pct,
            )

        if show_n_labels:
            ax.text(
                i,
                103.5,
                f"n={n}",
                ha="center",
                va="bottom",
                fontsize=8,
                color=MUTED,
            )

    # ---------------------------------------------------------------
    # Axes
    # ---------------------------------------------------------------

    ax.set_ylim(
        0,
        118,
    )

    ax.yaxis.set_major_formatter(
        PercentFormatter(100)
    )

    ax.set_yticks(
        [
            0,
            25,
            50,
            75,
            100,
        ]
    )

    ax.set_xticks(x)
    ax.set_xticklabels(labels)

    if rotate_xticks:
        for label in ax.get_xticklabels():
            label.set_rotation(45)
            label.set_ha("right")

    # ---------------------------------------------------------------
    # Legend
    # ---------------------------------------------------------------

    ax.legend(
        loc="upper right",
        frameon=False,
        ncol=3,
        bbox_to_anchor=(
            1.0,
            1.16,
        ),
        fontsize=10,
        handles=[
            Patch(
                facecolor=IMAGE_COLOR,
                label="Image",
            ),
            Patch(
                facecolor=TEXT_COLOR,
                label="Text",
            ),
            Patch(
                facecolor=NEITHER_COLOR,
                label="Neither",
            ),
        ],
    )

    # ---------------------------------------------------------------
    # Save
    # ---------------------------------------------------------------

    output_path = Path(
        output_path
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fig.savefig(
        output_path,
        dpi=170,
        bbox_inches="tight",
        facecolor=BACKGROUND,
        pad_inches=0.22,
    )

    plt.close(fig)

    return output_path


def write_mixed_visualizations(
    algorithm_distribution,
    edge_distribution_by_task,
    edge_overall,
    output_dir="evaluation/vis",
    model_name=None,
):
    """
    Write one algorithm figure and one edge-count figure per task,
    plus a comparison across tasks.

    Parameters
    ----------
    algorithm_distribution:
        Algorithm-level modality distribution.

    edge_distribution_by_task:
        Edge-count modality distribution for each task.

    edge_overall:
        Pooled edge-count distribution across all tasks.

    output_dir:
        Directory in which the visualization files are written.

    model_name:
        Model name displayed at the top of every generated plot.
    """

    output_dir = Path(
        output_dir
    )

    algorithm_dir = (
        output_dir / "algorithm"
    )

    edges_dir = (
        output_dir / "edges"
    )

    written = []

    # ===================================================================
    # Algorithm visualizations
    # ===================================================================

    sections = list(
        algorithm_distribution.get(
            "by_task",
            {},
        ).items()
    )

    sections.append(
        (
            "ALL",
            algorithm_distribution.get(
                "overall",
                {},
            ),
        )
    )

    for task, payload in sections:

        entries = payload.get(
            "algorithms",
            [],
        )

        path = _draw_task_figure(
            task=task,
            entries=entries,
            group_key="algorithm",
            axis="algorithm",
            output_path=(
                algorithm_dir
                / f"{task}.png"
            ),
            rotate_xticks=False,
            model_name=model_name,
        )

        if path is not None:
            written.append(path)

    # ===================================================================
    # Edge-count visualizations
    # ===================================================================

    for task, entries in (
        edge_distribution_by_task.items()
    ):

        path = _draw_task_figure(
            task=task,
            entries=entries,
            group_key="edge_range",
            axis="edges",
            output_path=(
                edges_dir
                / f"{task}.png"
            ),
            rotate_xticks=True,
            model_name=model_name,
        )

        if path is not None:
            written.append(path)

    # ===================================================================
    # Overall edge-count visualization
    # ===================================================================

    if edge_overall:

        path = _draw_task_figure(
            task="ALL",
            entries=edge_overall,
            group_key="edge_range",
            axis="edges",
            output_path=(
                edges_dir / "ALL.png"
            ),
            rotate_xticks=True,
            model_name=model_name,
        )

        if path is not None:
            written.append(path)

    # ===================================================================
    # Comparison across tasks
    # ===================================================================

    task_entries = _task_entries(
        algorithm_distribution
    )

    if task_entries:

        path = _draw_task_figure(
            task="",
            entries=task_entries,
            group_key="task",
            axis="task",
            output_path=(
                output_dir / "by_task.png"
            ),
            rotate_xticks=True,
            model_name=model_name,
        )

        if path is not None:
            written.append(path)

    return written