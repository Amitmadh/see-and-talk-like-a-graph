import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path

# ---------------------------------------------------------------------------
# IMPORTANT:
#
# Reuse the exact evaluation logic from evaluation.py.
# Do NOT duplicate parsing/comparison logic here.
# ---------------------------------------------------------------------------

from evaluation.evaluation import (
    TASK_TYPES,
    evaluate_mixed_file,
    build_mixed_summary,
)
from evaluation.enrich_mixed_results import (
    DATA_DIR,
    TEXT_ENCODING,
    enrich_results,
    save_enriched_results,
    get_or_load_dataset,
    lookup_algorithm,
)


# Generator algorithms used by GraphQA, in canonical order.
ALGORITHM_ORDER = [
    "er",
    "ba",
    "sbm",
    "sfn",
    "complete",
    "star",
    "path",
]

MODALITY_WINNERS = {
    "image",
    "text",
    "neither",
}


# ===========================================================================
# Loading
# ===========================================================================

def load_enriched_results(path):
    """
    Load the already-enriched mixed-signals JSON report.

    The enrichment step has already added graph metadata to every sample,
    including:

        sample["graph"]["nodes"]
        sample["graph"]["edges"]

    Therefore this evaluator does NOT need to access the original dataset.
    """

    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"Enriched JSON does not exist: {path}"
        )

    with open(
        path,
        "r",
        encoding="utf-8",
    ) as f:
        report = json.load(f)

    rows = report.get("files", [])

    if not rows:
        raise ValueError(
            f"No 'files' found in enriched JSON: {path}"
        )

    return report, rows


# ===========================================================================
# Graph-size / edge-bin analysis
# ===========================================================================

def get_edge_count(sample):
    """
    Get the number of edges from the graph metadata already embedded
    in the enriched sample.

    Returns None if graph metadata is unavailable.
    """

    graph = sample.get("graph")

    if not isinstance(graph, dict):
        return None

    edges = graph.get("edges")

    if not isinstance(edges, list):
        return None

    return len(edges)


def get_edge_bin(n_edges, bin_size=10):
    """
    Assign an edge count to intervals of `bin_size`:

        0-10
        11-20
        21-30
        ...

    Intervals are inclusive.
    """

    if n_edges is None:
        return None

    if n_edges <= bin_size:
        return f"0-{bin_size}"

    lower = (
        ((n_edges - 1) // bin_size) * bin_size
    ) + 1

    upper = lower + bin_size - 1

    return f"{lower}-{upper}"


def build_edge_distribution(samples):
    """
    Compute modality preference as a function of graph edge count.

    Only valid three-way modality classifications are included:

        image
        text
        neither

    'both' and invalid classifications are excluded from these
    percentages because the analysis is specifically about the
    relative preference between image/text/neither.
    """

    bins = {}

    missing_graph = 0

    for sample in samples:

        winner = sample.get(
            "modality_winner"
        )

        if winner not in {
            "image",
            "text",
            "neither",
        }:
            continue

        n_edges = get_edge_count(
            sample
        )

        if n_edges is None:
            missing_graph += 1
            continue

        edge_bin = get_edge_bin(
            n_edges
        )

        if edge_bin not in bins:

            bins[edge_bin] = {
                "edge_range": edge_bin,

                "total": 0,

                "image": 0,
                "text": 0,
                "neither": 0,

                "image_pct": 0.0,
                "text_pct": 0.0,
                "neither_pct": 0.0,
            }

        entry = bins[edge_bin]

        entry["total"] += 1
        entry[winner] += 1

    # ---------------------------------------------------------------
    # Calculate percentages within each edge bin.
    # ---------------------------------------------------------------

    for entry in bins.values():

        n = entry["total"]

        if n == 0:
            continue

        entry["image_pct"] = round(
            100.0 * entry["image"] / n,
            2,
        )

        entry["text_pct"] = round(
            100.0 * entry["text"] / n,
            2,
        )

        entry["neither_pct"] = round(
            100.0 * entry["neither"] / n,
            2,
        )

    # ---------------------------------------------------------------
    # Sort bins numerically.
    # ---------------------------------------------------------------

    sorted_bins = sorted(
        bins.values(),
        key=lambda x: int(
            x["edge_range"].split("-")[0]
        ),
    )

    return {
        "bins": sorted_bins,
        "missing_graph": missing_graph,
    }


def empty_edge_bin(edge_bin):
    return {
        "edge_range": edge_bin,

        "total": 0,

        "image": 0,
        "text": 0,
        "neither": 0,

        "image_pct": 0.0,
        "text_pct": 0.0,
        "neither_pct": 0.0,

        "n_evaluated": 0,
        "correct": 0,
        "incorrect": 0,
        "wrong_format": 0,
        "accuracy": 0.0,
    }


def finalize_edge_bin(entry):
    n = entry["total"]

    if n:
        entry["image_pct"] = round(
            100.0 * entry["image"] / n,
            2,
        )

        entry["text_pct"] = round(
            100.0 * entry["text"] / n,
            2,
        )

        entry["neither_pct"] = round(
            100.0 * entry["neither"] / n,
            2,
        )

    evaluated = entry["n_evaluated"]

    entry["accuracy"] = round(
        entry["correct"] / evaluated
        if evaluated
        else 0.0,
        6,
    )

    return entry


def aggregate_edge_distributions(rows):
    """
    Pool samples across result files belonging to the same task.

    Returns:

        {
            task: [
                {
                    edge_range,
                    total,
                    image,
                    text,
                    neither,
                    image_pct,
                    text_pct,
                    neither_pct,
                    n_evaluated,
                    correct,
                    incorrect,
                    wrong_format,
                    accuracy
                },
                ...
            ]
        }

    `total` / modality percentages stay restricted to the
    image/text/neither comparison, matching the previous report.

    `n_evaluated` / `accuracy` cover every sample in the bin
    that has graph metadata.
    """

    task_bins = {}

    for row in rows:

        task = row.get(
            "task"
        )

        if task is None:
            continue

        task_bins.setdefault(
            task,
            {}
        )

        for sample in row.get(
            "samples",
            []
        ):

            n_edges = get_edge_count(
                sample
            )

            if n_edges is None:
                continue

            edge_bin = get_edge_bin(
                n_edges
            )

            if edge_bin not in task_bins[task]:
                task_bins[task][edge_bin] = empty_edge_bin(
                    edge_bin
                )

            entry = task_bins[
                task
            ][edge_bin]

            entry["n_evaluated"] += 1

            if sample.get("correct"):
                entry["correct"] += 1

            else:
                status = sample.get(
                    "correctness_status"
                )

                if status == "wrong_format":
                    entry["wrong_format"] += 1
                else:
                    entry["incorrect"] += 1

            winner = sample.get(
                "modality_winner"
            )

            # Only the three-way comparison contributes to
            # the original modality-preference fields.
            if winner not in MODALITY_WINNERS:
                continue

            entry["total"] += 1
            entry[winner] += 1

    # ---------------------------------------------------------------
    # Calculate percentages and accuracy.
    # ---------------------------------------------------------------

    for bins in task_bins.values():

        for entry in bins.values():
            finalize_edge_bin(entry)

    # ---------------------------------------------------------------
    # Sort each task's bins numerically.
    # Keep bins that have either modality or accuracy counts.
    # ---------------------------------------------------------------

    result = {}

    for task, bins in task_bins.items():

        result[task] = sorted(
            [
                entry
                for entry in bins.values()
                if entry["total"] or entry["n_evaluated"]
            ],
            key=lambda x: int(
                x["edge_range"].split("-")[0]
            ),
        )

    return result


# ===========================================================================
# Algorithm analysis
# ===========================================================================

def algorithm_sort_key(name):
    if name in ALGORITHM_ORDER:
        return (
            0,
            ALGORITHM_ORDER.index(name),
        )

    return (1, name)


def ordered_tasks(tasks):
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


def load_task_datasets(rows):
    """
    Index original GraphQA JSONL files by sample_id so algorithm
    can be recovered even if a result row was not enriched.
    """

    dataset_cache = {}

    for row in rows:

        task = row.get("task")

        if task is None:
            continue

        get_or_load_dataset(
            task,
            dataset_cache,
            data_dir=DATA_DIR,
            text_encoding=TEXT_ENCODING,
            verbose=False,
        )

    return dataset_cache


def empty_algorithm_entry(algorithm):
    return {
        "algorithm": algorithm,

        "total": 0,
        "correct": 0,
        "incorrect": 0,
        "wrong_format": 0,
        "bad_expected": 0,
        "accuracy": 0.0,

        "image": 0,
        "text": 0,
        "neither": 0,
        "both": 0,

        "valid_modality_total": 0,

        "image_pct": 0.0,
        "text_pct": 0.0,
        "neither_pct": 0.0,

        "image_win_rate": 0.0,
        "text_win_rate": 0.0,
        "neither_rate": 0.0,
        "both_rate": 0.0,
    }


def record_sample_performance(entry, sample):
    entry["total"] += 1

    if sample.get("correct"):
        entry["correct"] += 1

    else:
        status = sample.get(
            "correctness_status"
        )

        if status == "wrong_format":
            entry["wrong_format"] += 1

        elif status == "bad_expected":
            entry["bad_expected"] += 1

        else:
            entry["incorrect"] += 1

    winner = sample.get(
        "modality_winner"
    )

    if winner in MODALITY_WINNERS:
        entry[winner] += 1

    elif winner == "both":
        entry["both"] += 1


def finalize_algorithm_entry(entry):
    n = entry["total"]

    entry["accuracy"] = round(
        entry["correct"] / n
        if n
        else 0.0,
        6,
    )

    valid = (
        entry["image"]
        + entry["text"]
        + entry["neither"]
    )

    entry["valid_modality_total"] = valid

    if valid:
        entry["image_pct"] = round(
            100.0 * entry["image"] / valid,
            2,
        )

        entry["text_pct"] = round(
            100.0 * entry["text"] / valid,
            2,
        )

        entry["neither_pct"] = round(
            100.0 * entry["neither"] / valid,
            2,
        )

    denom = valid + entry["both"]

    if denom:
        entry["image_win_rate"] = round(
            entry["image"] / denom,
            6,
        )

        entry["text_win_rate"] = round(
            entry["text"] / denom,
            6,
        )

        entry["neither_rate"] = round(
            entry["neither"] / denom,
            6,
        )

        entry["both_rate"] = round(
            entry["both"] / denom,
            6,
        )

    return entry


def pack_algorithm_buckets(buckets, missing_algorithm):
    algorithms = sorted(
        buckets.values(),
        key=lambda x: algorithm_sort_key(
            x["algorithm"]
        ),
    )

    for entry in algorithms:
        finalize_algorithm_entry(entry)

    n_samples = sum(
        entry["total"]
        for entry in algorithms
    ) + missing_algorithm

    return {
        "n_samples": n_samples,
        "missing_algorithm": missing_algorithm,
        "algorithms": algorithms,
    }


def aggregate_algorithm_distributions(rows, dataset_cache):
    """
    For every task, break mixed-signals performance down by the
    graph generator algorithm that produced the sample.

    Algorithm is resolved through sample_id against the original
    GraphQA JSONL files.
    """

    by_task = {}
    missing_by_task = {}
    overall = {}
    missing_overall = 0

    for row in rows:

        task = row.get("task")

        if task is None:
            continue

        dataset_by_id = dataset_cache.get(task)

        by_task.setdefault(task, {})
        missing_by_task.setdefault(task, 0)

        for sample in row.get("samples", []):

            algorithm = lookup_algorithm(
                sample,
                dataset_by_id,
            )

            if not algorithm:
                missing_by_task[task] += 1
                missing_overall += 1
                continue

            if algorithm not in by_task[task]:
                by_task[task][algorithm] = empty_algorithm_entry(
                    algorithm
                )

            if algorithm not in overall:
                overall[algorithm] = empty_algorithm_entry(
                    algorithm
                )

            record_sample_performance(
                by_task[task][algorithm],
                sample,
            )

            record_sample_performance(
                overall[algorithm],
                sample,
            )

    result_by_task = {}

    for task in ordered_tasks(by_task):

        result_by_task[task] = pack_algorithm_buckets(
            by_task[task],
            missing_by_task.get(task, 0),
        )

    return {
        "by_task": result_by_task,
        "overall": pack_algorithm_buckets(
            overall,
            missing_overall,
        ),
    }


def pool_edge_bins(edge_distribution_by_task):
    """
    Pool edge bins across tasks for the compact ALL-task CSV row.
    """

    pooled = {}

    for bins in edge_distribution_by_task.values():

        for entry in bins:

            key = entry["edge_range"]

            if key not in pooled:
                pooled[key] = empty_edge_bin(key)

            dest = pooled[key]

            for field in [
                "total",
                "image",
                "text",
                "neither",
                "n_evaluated",
                "correct",
                "incorrect",
                "wrong_format",
            ]:
                dest[field] += entry.get(field, 0)

    return sorted(
        [
            finalize_edge_bin(entry)
            for entry in pooled.values()
        ],
        key=lambda x: int(
            x["edge_range"].split("-")[0]
        ),
    )


# ===========================================================================
# Aggregate existing result statistics
# ===========================================================================

def aggregate_counts(rows):
    """
    Aggregate counts across result files.

    Counts are pooled rather than averaging percentages.
    """

    fields = [
        "total",
        "correct",
        "incorrect",
        "wrong_format",
        "bad_expected",

        "image_wins",
        "text_wins",
        "neither_wins",
        "both_wins",

        "invalid_original",
        "invalid_corrupted",

        "valid_modality_total",
    ]

    result = {
        field: sum(
            row.get(field, 0)
            for row in rows
        )
        for field in fields
    }

    total = result["total"]
    valid = result["valid_modality_total"]

    result["accuracy"] = (
        result["correct"] / total
        if total
        else 0.0
    )

    result["image_win_rate"] = (
        result["image_wins"] / valid
        if valid
        else 0.0
    )

    result["text_win_rate"] = (
        result["text_wins"] / valid
        if valid
        else 0.0
    )

    result["neither_rate"] = (
        result["neither_wins"] / valid
        if valid
        else 0.0
    )

    result["both_rate"] = (
        result["both_wins"] / valid
        if valid
        else 0.0
    )

    return result


def build_summary(rows):
    """
    Build summaries by task, image type, and model.

    Since the enriched JSON already contains all evaluation results,
    this function simply aggregates those results.
    """

    summary = {
        "overall": aggregate_counts(rows),
        "by_task": {},
        "by_image_type": {},
        "by_model": {},
    }

    for row in rows:

        for group_name, key in [
            (
                "by_task",
                row.get("task"),
            ),
            (
                "by_image_type",
                row.get("image_type"),
            ),
            (
                "by_model",
                row.get("model"),
            ),
        ]:

            if key is None:
                continue

            summary[group_name].setdefault(
                key,
                [],
            ).append(row)

    for group_name in [
        "by_task",
        "by_image_type",
        "by_model",
    ]:

        for key, group_rows in summary[
            group_name
        ].items():

            summary[group_name][key] = (
                aggregate_counts(
                    group_rows
                )
            )

    return summary


# ===========================================================================
# Console reporting
# ===========================================================================

def print_summary(summary):
    """
    Print the overall mixed-signals modality-preference report.

    Percentages here are relative to all evaluated samples, matching
    the existing report.
    """

    task_summary = summary[
        "by_task"
    ]

    print()
    print("=" * 105)
    print(
        "MIXED-SIGNALS MODALITY PREFERENCE"
    )
    print("=" * 105)
    print()

    print(
        f"{'Task':<25}"
        f"{'N':>7}"
        f"{'Image':>20}"
        f"{'Text':>20}"
        f"{'Neither':>20}"
        f"{'Both':>16}"
    )

    print("-" * 105)

    for task in TASK_TYPES:

        if task not in task_summary:
            continue

        row = task_summary[task]

        n = row["total"]

        def fmt_count_pct(count):
            percentage = (
                100.0 * count / n
                if n
                else 0.0
            )

            return (
                f"{count} "
                f"({percentage:.1f}%)"
            )

        print(
            f"{task:<25}"
            f"{n:>7}"
            f"{fmt_count_pct(row['image_wins']):>20}"
            f"{fmt_count_pct(row['text_wins']):>20}"
            f"{fmt_count_pct(row['neither_wins']):>20}"
            f"{fmt_count_pct(row['both_wins']):>16}"
        )

    print()

    overall = summary["overall"]
    n = overall["total"]

    print("-" * 105)

    def fmt_overall(count):
        percentage = (
            100.0 * count / n
            if n
            else 0.0
        )

        return (
            f"{count} "
            f"({percentage:.1f}%)"
        )

    print(
        f"{'OVERALL':<25}"
        f"{n:>7}"
        f"{fmt_overall(overall['image_wins']):>20}"
        f"{fmt_overall(overall['text_wins']):>20}"
        f"{fmt_overall(overall['neither_wins']):>20}"
        f"{fmt_overall(overall['both_wins']):>16}"
    )

    print()


def print_edge_distributions(
    edge_distribution_by_task
):
    """
    Print modality preference for each task stratified by edge count.

    Percentages are calculated within each edge-count bin among
    image/text/neither classifications.
    """

    print()
    print("=" * 95)
    print(
        "MODALITY PREFERENCE BY GRAPH SIZE"
    )
    print("=" * 95)

    print()

    print(
        "Percentages below are among valid "
        "image/text/neither comparisons in each bin."
    )

    for task in TASK_TYPES:

        if task not in edge_distribution_by_task:
            continue

        print()
        print(
            f"## TASK: {task}"
        )

        print()

        print(
            f"{'Edges':<12}"
            f"{'N':>8}"
            f"{'Image':>20}"
            f"{'Text':>20}"
            f"{'Neither':>20}"
        )

        print("-" * 80)

        for entry in (
            edge_distribution_by_task[
                task
            ]
        ):

            print(
                f"{entry['edge_range']:<12}"
                f"{entry['total']:>8}"
                f"{entry['image']:>7}"
                f" ({entry['image_pct']:>5.1f}%)"
                f"{entry['text']:>7}"
                f" ({entry['text_pct']:>5.1f}%)"
                f"{entry['neither']:>7}"
                f" ({entry['neither_pct']:>5.1f}%)"
            )

    print()


def print_algorithm_distributions(algorithm_distribution):
    """
    Print modality preference for each task, stratified by
    graph generator algorithm.
    """

    print()
    print("=" * 95)
    print(
        "MODALITY PREFERENCE BY GRAPH GENERATOR ALGORITHM"
    )
    print("=" * 95)

    print()

    print(
        "Percentages below are among valid "
        "image/text/neither comparisons."
    )

    sections = [
        (task, payload)
        for task, payload in algorithm_distribution[
            "by_task"
        ].items()
    ]

    sections.append(
        (
            "OVERALL",
            algorithm_distribution["overall"],
        )
    )

    for task, payload in sections:

        print()
        print(
            f"## TASK: {task}"
        )

        missing = payload[
            "missing_algorithm"
        ]

        if missing:
            print(
                f"   (missing algorithm: {missing})"
            )

        print()

        print(
            f"{'Algorithm':<14}"
            f"{'N':>8}"
            f"{'Image':>20}"
            f"{'Text':>20}"
            f"{'Neither':>20}"
        )

        print("-" * 82)

        for entry in payload["algorithms"]:

            print(
                f"{entry['algorithm']:<14}"
                f"{entry['total']:>8}"
                f"{entry['image']:>7}"
                f" ({entry['image_pct']:>5.1f}%)"
                f"{entry['text']:>7}"
                f" ({entry['text_pct']:>5.1f}%)"
                f"{entry['neither']:>7}"
                f" ({entry['neither_pct']:>5.1f}%)"
            )

    print()


# ===========================================================================
# CSV
# ===========================================================================

def write_csv(rows, output_path):
    """
    Write one row per result file.

    The edge-bin analysis is stored in the JSON report rather than
    flattened into this file.
    """

    output_path = Path(
        output_path
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fields = [
        "setting",
        "task",
        "image_type",
        "model",
        "file_name",
        "path",

        "total",
        "correct",
        "incorrect",
        "wrong_format",
        "bad_expected",
        "accuracy",

        "image_wins",
        "text_wins",
        "neither_wins",
        "both_wins",

        "image_win_rate",
        "text_win_rate",
        "neither_rate",
        "both_rate",

        "valid_modality_total",

        "invalid_original",
        "invalid_corrupted",
    ]

    with open(
        output_path,
        "w",
        encoding="utf-8",
        newline="",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fields,
        )

        writer.writeheader()

        for row in rows:

            writer.writerow({
                field: row.get(
                    field,
                    "",
                )
                for field in fields
            })

    print(
        f"Saved CSV report to {output_path}"
    )


AXIS_CSV_FIELDS = [
    "task",
    "group",
    "n_evaluated",
    "correct",
    "incorrect",
    "wrong_format",
    "accuracy",
    "n_modality",
    "image",
    "text",
    "neither",
    "image_pct",
    "text_pct",
    "neither_pct",
]


def write_compact_csv(csv_rows, output_path):
    output_path = Path(output_path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        output_path,
        "w",
        encoding="utf-8",
        newline="",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=AXIS_CSV_FIELDS,
        )

        writer.writeheader()
        writer.writerows(csv_rows)

    print(
        f"Saved compact CSV report to {output_path}"
    )


def algorithm_entry_to_csv_row(task, entry):
    return {
        "task": task,
        "group": entry["algorithm"],
        "n_evaluated": entry["total"],
        "correct": entry["correct"],
        "incorrect": entry["incorrect"],
        "wrong_format": entry["wrong_format"],
        "accuracy": entry["accuracy"],
        "n_modality": entry["valid_modality_total"],
        "image": entry["image"],
        "text": entry["text"],
        "neither": entry["neither"],
        "image_pct": entry["image_pct"],
        "text_pct": entry["text_pct"],
        "neither_pct": entry["neither_pct"],
    }


def edge_entry_to_csv_row(task, entry):
    return {
        "task": task,
        "group": entry["edge_range"],
        "n_evaluated": entry.get(
            "n_evaluated",
            entry["total"],
        ),
        "correct": entry.get("correct", ""),
        "incorrect": entry.get("incorrect", ""),
        "wrong_format": entry.get("wrong_format", ""),
        "accuracy": entry.get("accuracy", ""),
        "n_modality": entry["total"],
        "image": entry["image"],
        "text": entry["text"],
        "neither": entry["neither"],
        "image_pct": entry["image_pct"],
        "text_pct": entry["text_pct"],
        "neither_pct": entry["neither_pct"],
    }


def write_algorithm_csv(algorithm_distribution, output_path):
    """
    One compact row per (task, algorithm), plus ALL-task totals.
    """

    csv_rows = []

    for task in ordered_tasks(
        algorithm_distribution["by_task"]
    ):

        payload = algorithm_distribution[
            "by_task"
        ][task]

        for entry in payload["algorithms"]:
            csv_rows.append(
                algorithm_entry_to_csv_row(
                    task,
                    entry,
                )
            )

    for entry in algorithm_distribution[
        "overall"
    ]["algorithms"]:

        csv_rows.append(
            algorithm_entry_to_csv_row(
                "ALL",
                entry,
            )
        )

    write_compact_csv(csv_rows, output_path)


def write_edges_csv(edge_distribution_by_task, output_path):
    """
    One compact row per (task, edge-count bin), plus ALL-task totals.
    """

    csv_rows = []

    for task in ordered_tasks(
        edge_distribution_by_task
    ):

        for entry in edge_distribution_by_task[task]:
            csv_rows.append(
                edge_entry_to_csv_row(
                    task,
                    entry,
                )
            )

    for entry in pool_edge_bins(
        edge_distribution_by_task
    ):

        csv_rows.append(
            edge_entry_to_csv_row(
                "ALL",
                entry,
            )
        )

    write_compact_csv(csv_rows, output_path)


def score_mixed_results(results_root):
    """
    Score every mixed-signals JSONL under results_root.
    """

    root = Path(results_root)

    if not root.exists():
        raise FileNotFoundError(
            f"Results root does not exist: {root}"
        )

    rows = []

    for path in sorted(root.rglob("*.jsonl")):

        if path.name.startswith("."):
            continue

        print(f"Scoring {path}")

        rows.append(
            evaluate_mixed_file(
                path,
                root,
            )
        )

    if not rows:
        raise ValueError(
            f"No JSONL result files found under {root}"
        )

    return {
        "generated_at": datetime.now(
            timezone.utc
        ).isoformat(),
        "results_root": str(root),
        "files": rows,
        "summary": build_mixed_summary(rows),
    }


def save_scored_results(report, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(
            report,
            f,
            indent=2,
            ensure_ascii=False,
        )

    print(f"Saved scored mixed-signals JSON to {path}")


# ===========================================================================
# Main
# ===========================================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "End-to-end mixed-signals evaluation: "
            "score result JSONL files, enrich them with "
            "GraphQA metadata, then write CSVs, JSON, "
            "and visualizations."
        )
    )

    parser.add_argument(
        "--results-root",
        default="results/mixed_baseline",
        help=(
            "Root directory of mixed-signals JSONL "
            "result files."
        ),
    )

    parser.add_argument(
        "--scored-json",
        default=(
            "evaluation/"
            "mixed_signals_results.json"
        ),
        help=(
            "Where to write the scored mixed-signals "
            "JSON report."
        ),
    )

    parser.add_argument(
        "--enriched-json",
        default=(
            "evaluation/"
            "enrich_mixed_signals_results.json"
        ),
        help=(
            "Where to write the enriched JSON report."
        ),
    )

    parser.add_argument(
        "--output-json",
        default=(
            "evaluation/"
            "mixed_signals_analysis.json"
        ),
        help=(
            "Output JSON report."
        ),
    )

    parser.add_argument(
        "--output-csv",
        default=(
            "evaluation/"
            "mixed_signals_results.csv"
        ),
        help=(
            "Output CSV report (one row per result file)."
        ),
    )

    parser.add_argument(
        "--output-algorithm-csv",
        default=(
            "evaluation/"
            "mixed_signals_by_algorithm.csv"
        ),
        help=(
            "Compact CSV: performance by graph "
            "generator algorithm."
        ),
    )

    parser.add_argument(
        "--output-edges-csv",
        default=(
            "evaluation/"
            "mixed_signals_by_edges.csv"
        ),
        help=(
            "Compact CSV: performance by graph "
            "edge-count bin."
        ),
    )

    parser.add_argument(
        "--image-type",
        default="spring",
        help=(
            "Image type to report. "
            "Use 'all' for all image types."
        ),
    )

    parser.add_argument(
        "--vis-dir",
        default="evaluation/vis",
        help=(
            "Directory for per-task visualization PNGs."
        ),
    )

    parser.add_argument(
        "--skip-vis",
        action="store_true",
        help="Skip writing visualization PNGs.",
    )

    parser.add_argument(
        "--skip-score",
        action="store_true",
        help=(
            "Skip scoring JSONL files and load "
            "--scored-json instead."
        ),
    )

    parser.add_argument(
        "--skip-enrich",
        action="store_true",
        help=(
            "Skip enrichment and load "
            "--enriched-json instead."
        ),
    )

    args = parser.parse_args()

    # ---------------------------------------------------------------
    # Score mixed-signals JSONL files
    # ---------------------------------------------------------------

    if args.skip_score and args.skip_enrich:

        enriched_report, rows = (
            load_enriched_results(
                args.enriched_json
            )
        )

    else:

        if args.skip_score:

            print(
                f"Loading scored results from "
                f"{args.scored_json}"
            )

            with open(
                args.scored_json,
                "r",
                encoding="utf-8",
            ) as f:
                scored_report = json.load(f)

        else:

            print(
                f"Scoring mixed-signals results in "
                f"{args.results_root}"
            )

            scored_report = score_mixed_results(
                args.results_root
            )

            save_scored_results(
                scored_report,
                args.scored_json,
            )

        # -----------------------------------------------------------
        # Enrich with GraphQA metadata via sample_id
        # -----------------------------------------------------------

        if args.skip_enrich:

            enriched_report, rows = (
                load_enriched_results(
                    args.enriched_json
                )
            )

        else:

            print()
            print("Enriching scored results...")

            stats = enrich_results(
                scored_report,
                data_dir=DATA_DIR,
                text_encoding=TEXT_ENCODING,
            )

            print()
            print("=" * 70)
            print("ENRICHMENT COMPLETE")
            print("=" * 70)
            print(
                f"Total corruption samples : "
                f"{stats['total_samples']}"
            )
            print(
                f"Matched                  : "
                f"{stats['matched_samples']}"
            )
            print(
                f"Missing                  : "
                f"{stats['missing_samples']}"
            )
            print(
                f"Fields added             : "
                f"{stats['fields_added']}"
            )

            save_enriched_results(
                scored_report,
                args.enriched_json,
            )

            enriched_report = scored_report
            rows = scored_report.get("files", [])

    # ---------------------------------------------------------------
    # Filter by image type
    # ---------------------------------------------------------------

    if args.image_type != "all":

        rows = [
            row
            for row in rows
            if row.get("image_type")
            == args.image_type
        ]

    if not rows:

        print(
            "No matching result files found."
        )

        return

    # ---------------------------------------------------------------
    # Calculate edge distributions
    # ---------------------------------------------------------------

    edge_distribution_by_task = (
        aggregate_edge_distributions(
            rows
        )
    )

    # ---------------------------------------------------------------
    # Calculate algorithm distributions via sample_id
    # ---------------------------------------------------------------

    dataset_cache = load_task_datasets(
        rows
    )

    algorithm_distribution = (
        aggregate_algorithm_distributions(
            rows,
            dataset_cache,
        )
    )

    # ---------------------------------------------------------------
    # Aggregate existing evaluation statistics
    # ---------------------------------------------------------------

    summary = build_summary(
        rows
    )

    # ---------------------------------------------------------------
    # Build output report
    # ---------------------------------------------------------------

    report = {
        "generated_at": (
            datetime.now(
                timezone.utc
            ).isoformat()
        ),

        "source_enriched_json": str(
            args.enriched_json
        ),

        "results_root": (
            enriched_report.get(
                "results_root"
            )
        ),

        "image_type_filter": (
            args.image_type
        ),

        "files": rows,

        "summary": summary,

        # -----------------------------------------------------------
        # NEW:
        #
        # Modality preference stratified by graph size, separately
        # for every graph reasoning task.
        # -----------------------------------------------------------

        "edge_distribution_by_task":
            edge_distribution_by_task,

        # -----------------------------------------------------------
        # Performance stratified by the graph generator algorithm
        # that produced each sample (joined via sample_id).
        # -----------------------------------------------------------

        "algorithm_distribution":
            algorithm_distribution,
    }

    # ---------------------------------------------------------------
    # Save JSON
    # ---------------------------------------------------------------

    output_json = Path(
        args.output_json
    )

    output_json.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        output_json,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            report,
            f,
            indent=2,
        )

    print(
        f"Saved JSON report to {output_json}"
    )

    # ---------------------------------------------------------------
    # Save CSV
    # ---------------------------------------------------------------

    write_csv(
        rows,
        args.output_csv,
    )

    write_algorithm_csv(
        algorithm_distribution,
        args.output_algorithm_csv,
    )

    write_edges_csv(
        edge_distribution_by_task,
        args.output_edges_csv,
    )

    if not args.skip_vis:

        try:
            from evaluation.vis.plot_mixed import (
                write_mixed_visualizations,
            )
            from evaluation.plot_mixed_all_models import (
                write_mixed_all_models_plot,
                write_mixed_algorithm_models_plot,
                write_mixed_algorithm_by_model_plot,
                write_mixed_all_models_edges_plot,
            )

        except ImportError as exc:
            print(
                "Skipping visualizations "
                f"(matplotlib unavailable): {exc}"
            )

        else:
            # Group evaluation rows by model.
            rows_by_model = {}

            for row in rows:
                model = row.get("model", "unknown_model")
                rows_by_model.setdefault(model, []).append(row)

            all_vis_paths = []

            for model, model_rows in sorted(rows_by_model.items()):

                print()
                print("=" * 80)
                print(f"Generating visualizations for model: {model}")
                print("=" * 80)

                # Compute distributions ONLY from this model's rows.
                model_edge_distribution = (
                    aggregate_edge_distributions(model_rows)
                )

                model_dataset_cache = load_task_datasets(
                    model_rows
                )

                model_algorithm_distribution = (
                    aggregate_algorithm_distributions(
                        model_rows,
                        model_dataset_cache,
                    )
                )

                # Separate directory for each model.
                model_output_dir = (
                    Path(args.vis_dir) / model
                )

                model_output_dir.mkdir(
                    parents=True,
                    exist_ok=True,
                )

                # Generate plots using only this model's data.
                model_vis_paths = write_mixed_visualizations(
                    model_algorithm_distribution,
                    model_edge_distribution,
                    pool_edge_bins(model_edge_distribution),
                    output_dir=model_output_dir,
                    model_name=model,
                )

                all_vis_paths.extend(model_vis_paths)

                print(
                    f"Saved {len(model_vis_paths)} visualizations "
                    f"for {model} to {model_output_dir}"
                )

            print(
                f"Saved {len(all_vis_paths)} model-specific "
                f"visualizations to {args.vis_dir}"
            )

            all_models_path = write_mixed_all_models_plot(
                rows,
                output_path=(
                    Path(args.vis_dir)
                    / "all_models"
                    / "by_task.png"
                ),
                image_type=args.image_type,
            )

            if all_models_path is not None:
                print(
                    "Saved all-models mixed-signals "
                    f"plot to {all_models_path}"
                )

            all_models_algorithm_path = write_mixed_algorithm_models_plot(
                rows,
                output_path=(
                    Path(args.vis_dir)
                    / "all_models"
                    / "by_algorithm.png"
                ),
                image_type=args.image_type,
            )

            if all_models_algorithm_path is not None:
                print(
                    "Saved all-models algorithm "
                    f"plot to {all_models_algorithm_path}"
                )

            all_models_by_model_path = write_mixed_algorithm_by_model_plot(
                rows,
                output_path=(
                    Path(args.vis_dir)
                    / "all_models"
                    / "by_algorithm_model.png"
                ),
                image_type=args.image_type,
            )

            if all_models_by_model_path is not None:
                print(
                    "Saved per-model algorithm "
                    f"plot to {all_models_by_model_path}"
                )

            for bin_size, filename in (
                (10, "by_edges.png"),
                (20, "by_edges_20.png"),
            ):
                all_models_edges_path = write_mixed_all_models_edges_plot(
                    rows,
                    output_path=(
                        Path(args.vis_dir)
                        / "all_models"
                        / filename
                    ),
                    image_type=args.image_type,
                    bin_size=bin_size,
                )

                if all_models_edges_path is not None:
                    print(
                        "Saved all-models "
                        f"{bin_size}-edge-bin "
                        f"plot to {all_models_edges_path}"
                    )

    # ---------------------------------------------------------------
    # Console reports
    # ---------------------------------------------------------------

    print_summary(
        summary
    )

    print_edge_distributions(
        edge_distribution_by_task
    )

    print_algorithm_distributions(
        algorithm_distribution
    )


if __name__ == "__main__":
    main()