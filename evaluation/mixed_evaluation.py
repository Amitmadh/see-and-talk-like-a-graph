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

from evaluation.evaluation import TASK_TYPES


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


def get_edge_bin(n_edges, bin_size=4):
    """
    Assign an edge count to intervals:

        0-4
        5-8
        9-12
        13-16
        ...

    Intervals are inclusive.
    """

    if n_edges is None:
        return None

    if n_edges <= bin_size:
        return "0-4"

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
                    neither_pct
                },
                ...
            ]
        }

    This is the main analysis used to distinguish:

        graph-size effect

    from:

        task-specific modality preference.
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

            winner = sample.get(
                "modality_winner"
            )

            # Only the three-way comparison.
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
                continue

            edge_bin = get_edge_bin(
                n_edges
            )

            if edge_bin not in task_bins[task]:

                task_bins[task][edge_bin] = {
                    "edge_range": edge_bin,

                    "total": 0,

                    "image": 0,
                    "text": 0,
                    "neither": 0,

                    "image_pct": 0.0,
                    "text_pct": 0.0,
                    "neither_pct": 0.0,
                }

            entry = task_bins[
                task
            ][edge_bin]

            entry["total"] += 1
            entry[winner] += 1

    # ---------------------------------------------------------------
    # Calculate percentages.
    # ---------------------------------------------------------------

    for task, bins in task_bins.items():

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
    # Sort each task's bins numerically.
    # ---------------------------------------------------------------

    result = {}

    for task, bins in task_bins.items():

        result[task] = sorted(
            bins.values(),
            key=lambda x: int(
                x["edge_range"].split("-")[0]
            ),
        )

    return result


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


# ===========================================================================
# Main
# ===========================================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Analyze an already-enriched mixed-signals "
            "evaluation report, including modality preference "
            "stratified by graph edge count."
        )
    )

    parser.add_argument(
        "--enriched-json",
        default=(
            "evaluation/"
            "enrich_mixed_signals_results.json"
        ),
        help=(
            "Path to the already-enriched JSON report."
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
            "Output CSV report."
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

    args = parser.parse_args()

    # ---------------------------------------------------------------
    # Load enriched report
    # ---------------------------------------------------------------

    enriched_report, rows = (
        load_enriched_results(
            args.enriched_json
        )
    )

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

    # ---------------------------------------------------------------
    # Console reports
    # ---------------------------------------------------------------

    print_summary(
        summary
    )

    print_edge_distributions(
        edge_distribution_by_task
    )


if __name__ == "__main__":
    main()