"""
Experiment 1: accuracy by setting and task.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

from evaluation.evaluation import (
    build_model_setting_table,
    build_setting_table,
    unique_models,
)


BACKGROUND = "#F7F4EF"
PANEL = "#FFFcf7"
GRID = "#E6E0D6"
INK = "#1B1B1B"
MUTED = "#5C6570"

SETTING_ORDER = [
    "text_only",
    "image_only",
    "text_and_image",
]

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

MODEL_ORDER = [
    "Qwen2.5-VL-3B-Instruct",
    "Qwen2.5-VL-7B-Instruct",
    "llava-v1.6-mistral-7b-hf",
]

MODEL_LABELS = {
    "Qwen2.5-VL-3B-Instruct": "Qwen-3B",
    "Qwen2.5-VL-7B-Instruct": "Qwen-7B",
    "llava-v1.6-mistral-7b-hf": "LLaVA-1.6",
}

MODEL_COLORS = {
    "Qwen2.5-VL-3B-Instruct": "#457B9D",
    "Qwen2.5-VL-7B-Instruct": "#E9C46A",
    "llava-v1.6-mistral-7b-hf": "#BC6C25",
}

FALLBACK_MODEL_COLORS = [
    "#457B9D",
    "#E9C46A",
    "#6D597A",
    "#BC6C25",
]


def _style_axes(ax):
    ax.set_facecolor(PANEL)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(GRID)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=MUTED, length=0)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def _cell_percent(cell):
    if cell == "" or cell is None:
        return 0.0

    return 100.0 * float(cell)


def _ordered_models(rows):
    present = unique_models(rows)
    ordered = [
        model
        for model in MODEL_ORDER
        if model in present
    ]
    extra = [
        model
        for model in present
        if model not in ordered
    ]
    return ordered + extra


def _model_color(model, index):
    if model in MODEL_COLORS:
        return MODEL_COLORS[model]

    return FALLBACK_MODEL_COLORS[index % len(FALLBACK_MODEL_COLORS)]


def write_baseline_plot(
    rows,
    output_path="evaluation/vis/baseline.png",
    image_type="spring",
    title=None,
):
    table, tasks = build_setting_table(
        rows,
        image_type=(
            None
            if image_type in {None, "all"}
            else image_type
        ),
    )

    tasks = [
        task
        for task in tasks
        if any(row.get(task) for row in table)
    ]

    settings = [
        setting
        for setting in SETTING_ORDER
        if any(
            row.get("setting") == setting
            for row in table
        )
    ]

    extra = sorted(
        {
            row["setting"]
            for row in table
            if row["setting"] not in settings
        }
    )
    settings.extend(extra)

    if not tasks or not settings:
        return None

    by_setting = {
        row["setting"]: row
        for row in table
    }

    n_tasks = len(tasks)
    n_settings = len(settings)
    width = max(8.5, 0.85 * n_tasks + 2.5)
    fig, ax = plt.subplots(figsize=(width, 4.6))

    fig.patch.set_facecolor(BACKGROUND)
    _style_axes(ax)

    x = list(range(n_tasks))
    group_width = 0.78
    bar_width = group_width / n_settings
    offsets = [
        -group_width / 2 + bar_width * (i + 0.5)
        for i in range(n_settings)
    ]

    for offset, setting in zip(offsets, settings):
        values = []

        for task in tasks:
            values.append(
                _cell_percent(
                    by_setting.get(setting, {}).get(task, "")
                )
            )

        ax.bar(
            [i + offset for i in x],
            values,
            width=bar_width * 0.92,
            color=SETTING_COLORS.get(setting, "#6C757D"),
            edgecolor=PANEL,
            linewidth=0.4,
            label=SETTING_LABELS.get(setting, setting),
        )

    ax.set_ylim(0, 100)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.yaxis.set_major_formatter(PercentFormatter(100))
    ax.set_xticks(x)
    ax.set_xticklabels(
        [task.replace("_", " ") for task in tasks]
    )

    for label in ax.get_xticklabels():
        label.set_rotation(35)
        label.set_ha("right")

    ax.legend(
        loc="upper right",
        frameon=False,
        ncol=n_settings,
        fontsize=9,
    )

    if title:
        ax.set_title(
            title,
            color=INK,
            fontsize=11,
            pad=10,
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


def write_model_comparison_plot(
    rows,
    output_path="evaluation/vis/baseline_by_model_and_setting.png",
    image_type="spring",
):
    table, tasks = build_model_setting_table(
        rows,
        image_type=(
            None
            if image_type in {None, "all"}
            else image_type
        ),
    )

    tasks = [
        task
        for task in tasks
        if any(row.get(task) for row in table)
    ]

    settings = [
        setting
        for setting in SETTING_ORDER
        if any(
            row.get("setting") == setting
            for row in table
        )
    ]

    extra = sorted(
        {
            row["setting"]
            for row in table
            if row["setting"] not in settings
        }
    )
    settings.extend(extra)

    models = _ordered_models(rows)

    if not tasks or not settings or not models:
        return None

    n_tasks = len(tasks)
    n_models = len(models)
    width = max(10.5, 0.95 * n_tasks * len(settings))
    fig, axes = plt.subplots(
        1,
        len(settings),
        figsize=(width, 4.8),
        sharey=True,
    )

    if len(settings) == 1:
        axes = [axes]

    fig.patch.set_facecolor(BACKGROUND)

    x = list(range(n_tasks))
    group_width = 0.78
    bar_width = group_width / n_models
    offsets = [
        -group_width / 2 + bar_width * (i + 0.5)
        for i in range(n_models)
    ]

    by_model_setting = {
        (row["model"], row["setting"]): row
        for row in table
    }

    for ax, setting in zip(axes, settings):
        _style_axes(ax)

        for offset, model, index in zip(
            offsets,
            models,
            range(n_models),
        ):
            values = [
                _cell_percent(
                    by_model_setting.get(
                        (model, setting),
                        {},
                    ).get(task, "")
                )
                for task in tasks
            ]

            ax.bar(
                [i + offset for i in x],
                values,
                width=bar_width * 0.92,
                color=_model_color(model, index),
                edgecolor=PANEL,
                linewidth=0.4,
                label=MODEL_LABELS.get(model, model),
            )

        ax.set_ylim(0, 100)
        ax.set_yticks([0, 25, 50, 75, 100])
        ax.yaxis.set_major_formatter(PercentFormatter(100))
        ax.set_xticks(x)
        ax.set_xticklabels(
            [task.replace("_", " ") for task in tasks]
        )

        for label in ax.get_xticklabels():
            label.set_rotation(35)
            label.set_ha("right")

        ax.set_title(
            SETTING_LABELS.get(setting, setting),
            color=INK,
            fontsize=11,
            pad=8,
        )

    axes[0].legend(
        loc="upper left",
        frameon=False,
        ncol=n_models,
        fontsize=9,
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
