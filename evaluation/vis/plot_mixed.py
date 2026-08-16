"""
Per-task mixed-signals visualizations.

Each task gets two figures:

    evaluation/vis/algorithm/<task>.png
    evaluation/vis/edges/<task>.png

The top panel is a 100% stacked modality mix (image / text / neither).
The bottom panel is accuracy, with sample size shown as a ghost bar.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.ticker import PercentFormatter


IMAGE_COLOR = "#E76F51"
TEXT_COLOR = "#2A9D8F"
NEITHER_COLOR = "#9AA5B1"
ACCURACY_COLOR = "#1D3557"
SAMPLE_COLOR = "#D8DEE6"
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

    return str(name)


def _task_accuracy(entries, n_key, correct_key="correct"):
    n = sum(entry.get(n_key, 0) or 0 for entry in entries)
    correct = sum(entry.get(correct_key, 0) or 0 for entry in entries)

    if not n:
        return 0.0

    return correct / n


def _style_axes(ax):
    ax.set_facecolor(PANEL)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(GRID)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=MUTED, length=0)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def _annotate_stack(ax, x, image_pct, text_pct, neither_pct):
    bottoms = [0.0, image_pct, image_pct + text_pct]
    values = [image_pct, text_pct, neither_pct]
    colors = ["white", "white", INK]

    for bottom, value, color in zip(bottoms, values, colors):
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
    xlabel,
    output_path,
    rotate_xticks=False,
):
    if not entries:
        return None

    labels = [
        _pretty_group(entry[group_key], axis)
        for entry in entries
    ]

    n_values = [
        int(entry.get("n_evaluated", entry.get("total", 0)) or 0)
        for entry in entries
    ]

    image_pcts = [float(entry.get("image_pct", 0) or 0) for entry in entries]
    text_pcts = [float(entry.get("text_pct", 0) or 0) for entry in entries]
    neither_pcts = [
        float(entry.get("neither_pct", 0) or 0)
        for entry in entries
    ]
    accuracies = [
        100.0 * float(entry.get("accuracy", 0) or 0)
        for entry in entries
    ]

    n_groups = len(entries)
    width = max(11.0, 0.55 * n_groups + 5.5)
    fig, (ax_mix, ax_acc) = plt.subplots(
        2,
        1,
        figsize=(width, 8.6),
        sharex=True,
        gridspec_kw={"height_ratios": [1.35, 1.0], "hspace": 0.08},
    )

    fig.patch.set_facecolor(BACKGROUND)
    fig.suptitle(
        f"{_pretty_task(task)}",
        fontsize=20,
        fontweight="bold",
        color=INK,
        x=0.02,
        ha="left",
        y=0.98,
    )

    axis_title = (
        "Which modality does the model follow?"
        if axis == "algorithm"
        else "Does graph size change the modality mix?"
    )

    fig.text(
        0.02,
        0.935,
        f"{axis_title}   ·   {xlabel.lower()}",
        fontsize=11,
        color=MUTED,
        ha="left",
    )

    show_stack_labels = n_groups <= 14
    show_n_labels = n_groups <= 18
    show_acc_labels = n_groups <= 16

    x = list(range(n_groups))
    bar_width = 0.72 if n_groups <= 12 else 0.82

    _style_axes(ax_mix)
    ax_mix.bar(
        x,
        image_pcts,
        width=bar_width,
        color=IMAGE_COLOR,
        edgecolor=PANEL,
        linewidth=0.6,
        label="Image",
    )
    ax_mix.bar(
        x,
        text_pcts,
        width=bar_width,
        bottom=image_pcts,
        color=TEXT_COLOR,
        edgecolor=PANEL,
        linewidth=0.6,
        label="Text",
    )
    ax_mix.bar(
        x,
        neither_pcts,
        width=bar_width,
        bottom=[i + t for i, t in zip(image_pcts, text_pcts)],
        color=NEITHER_COLOR,
        edgecolor=PANEL,
        linewidth=0.6,
        label="Neither",
    )

    for i, (image_pct, text_pct, neither_pct, n) in enumerate(
        zip(image_pcts, text_pcts, neither_pcts, n_values)
    ):
        if show_stack_labels:
            _annotate_stack(ax_mix, i, image_pct, text_pct, neither_pct)

        if show_n_labels:
            ax_mix.text(
                i,
                103.5,
                f"n={n}",
                ha="center",
                va="bottom",
                fontsize=8,
                color=MUTED,
            )

    ax_mix.set_ylim(0, 118)
    ax_mix.set_ylabel("Modality mix", color=MUTED, fontsize=10)
    ax_mix.yaxis.set_major_formatter(PercentFormatter(100))
    ax_mix.set_yticks([0, 25, 50, 75, 100])
    ax_mix.legend(
        loc="upper right",
        frameon=False,
        ncol=3,
        bbox_to_anchor=(1.0, 1.18),
        fontsize=10,
        handles=[
            Patch(facecolor=IMAGE_COLOR, label="Image"),
            Patch(facecolor=TEXT_COLOR, label="Text"),
            Patch(facecolor=NEITHER_COLOR, label="Neither"),
        ],
    )

    _style_axes(ax_acc)

    max_n = max(n_values) if n_values else 1
    sample_heights = [
        (n / max_n) * 100 if max_n else 0
        for n in n_values
    ]

    ax_acc.bar(
        x,
        sample_heights,
        width=bar_width,
        color=SAMPLE_COLOR,
        edgecolor=PANEL,
        linewidth=0.6,
        zorder=1,
        label="Sample size",
    )
    ax_acc.bar(
        x,
        accuracies,
        width=bar_width * 0.55,
        color=ACCURACY_COLOR,
        edgecolor=PANEL,
        linewidth=0.4,
        zorder=2,
        label="Accuracy",
    )

    overall = 100.0 * _task_accuracy(
        entries,
        "n_evaluated" if "n_evaluated" in entries[0] else "total",
    )

    ax_acc.axhline(
        overall,
        color=IMAGE_COLOR,
        linestyle=(0, (4, 3)),
        linewidth=1.4,
        zorder=3,
        label=f"Task mean  {overall:.0f}%",
    )

    for i, acc in enumerate(accuracies):
        if not show_acc_labels:
            break

        ax_acc.text(
            i,
            acc + 2.4,
            f"{acc:.0f}%",
            ha="center",
            va="bottom",
            fontsize=8.5,
            fontweight="bold",
            color=ACCURACY_COLOR,
        )

    ax_acc.set_ylim(0, 118)
    ax_acc.set_ylabel("Accuracy", color=MUTED, fontsize=10)
    ax_acc.yaxis.set_major_formatter(PercentFormatter(100))
    ax_acc.set_yticks([0, 25, 50, 75, 100])
    ax_acc.set_xticks(x)
    ax_acc.set_xticklabels(labels)
    ax_acc.set_xlabel(xlabel, color=MUTED, fontsize=10, labelpad=8)

    if rotate_xticks:
        for label in ax_acc.get_xticklabels():
            label.set_rotation(45)
            label.set_ha("right")

    ax_acc.legend(
        loc="upper right",
        frameon=False,
        ncol=3,
        fontsize=9,
    )

    fig.text(
        0.02,
        0.015,
        "Top: share of image / text / neither answers.  "
        "Bottom: accuracy on the corrupted-text target; gray bars scale with n.",
        fontsize=8.5,
        color=MUTED,
    )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig.savefig(
        output_path,
        dpi=170,
        bbox_inches="tight",
        facecolor=BACKGROUND,
        pad_inches=0.28,
    )
    plt.close(fig)

    return output_path


def write_mixed_visualizations(
    algorithm_distribution,
    edge_distribution_by_task,
    edge_overall,
    output_dir="evaluation/vis",
):
    """
    Write one algorithm figure and one edge-count figure per task.
    """

    output_dir = Path(output_dir)
    algorithm_dir = output_dir / "algorithm"
    edges_dir = output_dir / "edges"

    written = []

    sections = list(
        algorithm_distribution.get("by_task", {}).items()
    )
    sections.append(
        ("ALL", algorithm_distribution.get("overall", {}))
    )

    for task, payload in sections:
        entries = payload.get("algorithms", [])
        path = _draw_task_figure(
            task=task,
            entries=entries,
            group_key="algorithm",
            axis="algorithm",
            xlabel="Graph generator algorithm",
            output_path=algorithm_dir / f"{task}.png",
            rotate_xticks=False,
        )

        if path is not None:
            written.append(path)

    for task, entries in edge_distribution_by_task.items():
        path = _draw_task_figure(
            task=task,
            entries=entries,
            group_key="edge_range",
            axis="edges",
            xlabel="Number of edges",
            output_path=edges_dir / f"{task}.png",
            rotate_xticks=True,
        )

        if path is not None:
            written.append(path)

    if edge_overall:
        path = _draw_task_figure(
            task="ALL",
            entries=edge_overall,
            group_key="edge_range",
            axis="edges",
            xlabel="Number of edges",
            output_path=edges_dir / "ALL.png",
            rotate_xticks=True,
        )

        if path is not None:
            written.append(path)

    return written
