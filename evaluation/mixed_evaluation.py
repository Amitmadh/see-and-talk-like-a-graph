import argparse
import csv
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

# ---------------------------------------------------------------------------
# IMPORTANT:
# Reuse the exact evaluation logic from evaluation.py.
# Do NOT duplicate parsing/comparison logic here.
# ---------------------------------------------------------------------------

from evaluation.evaluation import (
    TASK_TYPES,
    clean,
    parse_expected,
    parse_model_answer,
    compare,
)


# ===========================================================================
# Loading
# ===========================================================================

def load_results_file(path):
    """
    Load either a JSON array or JSONL result file.
    """
    text = path.read_text(encoding="utf-8").strip()

    if not text:
        return []

    if text.startswith("["):
        return json.loads(text)

    return [
        json.loads(line)
        for line in text.splitlines()
        if line.strip()
    ]


def get_file_metadata(path, root):
    """
    Expected structure:

        results_root/
            setting/
                task/
                    result.jsonl
    """
    parts = path.relative_to(root).parts

    if len(parts) < 3:
        raise ValueError(
            f"Unexpected results path: {path}"
        )

    return {
        "setting": parts[0],
        "task": parts[1],
        "file_name": path.name,
    }


# ===========================================================================
# Mixed-signals comparison
# ===========================================================================

def parse_reference_answer(task, answer):
    """
    Parse an original/corrupted answer using the same parsing conventions
    as evaluation.py.

    We use parse_expected because these answers originate from the
    dataset rather than directly from the model.
    """
    return parse_expected(task, answer)


def classify_modality_winner(task, model_answer, original_answer, corrupted_answer):
    """
    Determine which source the model followed.

    IMPORTANT:
    All answer interpretation uses the parsing functions from evaluation.py.

    Returns one of:

        image
        text
        neither
        both
        invalid_model_answer
        invalid_original_answer
        invalid_corrupted_answer

    'both' is possible when original_answer == corrupted_answer.
    In a genuine corruption experiment this should normally be rare/impossible,
    but we keep it explicit rather than silently assigning a winner.
    """

    model = parse_model_answer(task, model_answer)

    if not model.valid:
        return {
            "winner": "invalid_model_answer",
            "model_valid": False,
            "original_valid": None,
            "corrupted_valid": None,
            "matches_original": False,
            "matches_corrupted": False,
        }

    original = parse_reference_answer(
        task,
        original_answer,
    )

    if not original.valid:
        return {
            "winner": "invalid_original_answer",
            "model_valid": True,
            "original_valid": False,
            "corrupted_valid": None,
            "matches_original": False,
            "matches_corrupted": False,
        }

    corrupted = parse_reference_answer(
        task,
        corrupted_answer,
    )

    if not corrupted.valid:
        return {
            "winner": "invalid_corrupted_answer",
            "model_valid": True,
            "original_valid": True,
            "corrupted_valid": False,
            "matches_original": False,
            "matches_corrupted": False,
        }

    # ---------------------------------------------------------------
    # Use the SAME semantic comparison used by evaluation.py.
    #
    # We compare parsed values rather than raw strings so that:
    #
    #   connected_nodes:
    #       "1, 2, 3" == "3, 1, 2"
    #
    # and booleans/numbers are interpreted consistently.
    # ---------------------------------------------------------------

    if task == "connected_nodes":
        matches_original = (
            set(model.value)
            == set(original.value)
        )

        matches_corrupted = (
            set(model.value)
            == set(corrupted.value)
        )

    else:
        matches_original = (
            model.value
            == original.value
        )

        matches_corrupted = (
            model.value
            == corrupted.value
        )

    # ---------------------------------------------------------------
    # Classify
    # ---------------------------------------------------------------

    if matches_original and matches_corrupted:
        winner = "both"

    elif matches_original:
        winner = "image"

    elif matches_corrupted:
        winner = "text"

    else:
        winner = "neither"

    return {
        "winner": winner,
        "model_valid": True,
        "original_valid": True,
        "corrupted_valid": True,
        "matches_original": matches_original,
        "matches_corrupted": matches_corrupted,
    }


# ===========================================================================
# Evaluate one result file
# ===========================================================================

def evaluate_file(path, root):
    metadata = get_file_metadata(
        path,
        root,
    )

    task = metadata["task"]

    if task not in TASK_TYPES:
        raise ValueError(
            f"Unknown task '{task}' in {path}"
        )

    data = load_results_file(path)

    total = 0

    correct = 0
    incorrect = 0
    wrong_format = 0
    bad_expected = 0

    image_wins = 0
    text_wins = 0
    neither_wins = 0
    both_wins = 0

    invalid_original = 0
    invalid_corrupted = 0

    # Detailed per-sample records are extremely useful for later analysis.
    sample_analysis = []

    image_type = "unknown"
    model = "unknown"

    for row in data:

        image_type = row.get(
            "image_type",
            image_type,
        )

        model = row.get(
            "model",
            model,
        )

        model_answer = row.get(
            "model_answer"
        )

        # Same convention as evaluation.py:
        # samples without a model answer are not evaluated.
        if model_answer is None:
            continue

        total += 1

        original_answer = row.get(
            "original_answer"
        )

        corrupted_answer = row.get(
            "corrupted_answer"
        )

        # ---------------------------------------------------------------
        # Ground-truth correctness
        #
        # STRICTLY delegated to evaluation.py's compare().
        # ---------------------------------------------------------------

        is_correct, correctness_status, _ = compare(
            task,
            row.get("expected_answer"),
            model_answer,
        )

        if is_correct:
            correct += 1

        elif correctness_status == "wrong_format":
            wrong_format += 1

        elif correctness_status == "bad_expected":
            bad_expected += 1

        else:
            incorrect += 1

        # ---------------------------------------------------------------
        # Mixed-signals modality attribution
        # ---------------------------------------------------------------

        modality = classify_modality_winner(
            task=task,
            model_answer=model_answer,
            original_answer=original_answer,
            corrupted_answer=corrupted_answer,
        )

        winner = modality["winner"]

        if winner == "image":
            image_wins += 1

        elif winner == "text":
            text_wins += 1

        elif winner == "neither":
            neither_wins += 1

        elif winner == "both":
            both_wins += 1

        elif winner == "invalid_original_answer":
            invalid_original += 1

        elif winner == "invalid_corrupted_answer":
            invalid_corrupted += 1

        # ---------------------------------------------------------------
        # Keep complete per-sample information.
        # ---------------------------------------------------------------

        sample_analysis.append({
            "sample_id": row.get("sample_id"),

            "expected_answer": row.get(
                "expected_answer"
            ),

            "original_answer": original_answer,

            "corrupted_answer": corrupted_answer,

            "model_answer": model_answer,

            "corruption": row.get(
                "corruption"
            ),

            "correct": is_correct,

            "correctness_status": correctness_status,

            "modality_winner": winner,

            "matches_original": modality[
                "matches_original"
            ],

            "matches_corrupted": modality[
                "matches_corrupted"
            ],
        })

    # ---------------------------------------------------------------
    # Sanity checks
    # ---------------------------------------------------------------

    classified = (
        image_wins
        + text_wins
        + neither_wins
        + both_wins
        + invalid_original
        + invalid_corrupted
    )

    if classified != total:
        raise RuntimeError(
            f"Classification accounting error in {path}: "
            f"{classified} classified != {total} evaluated"
        )

    accuracy = (
        correct / total
        if total
        else 0.0
    )

    # Usually we care about the three-way distribution among valid
    # mixed-signal comparisons.
    valid_modality_total = (
        image_wins
        + text_wins
        + neither_wins
        + both_wins
    )

    return {
        **metadata,

        "image_type": image_type,
        "model": model,

        # Basic evaluation
        "total": total,
        "correct": correct,
        "incorrect": incorrect,
        "wrong_format": wrong_format,
        "bad_expected": bad_expected,
        "accuracy": round(
            accuracy,
            6,
        ),

        # Mixed-signals analysis
        "image_wins": image_wins,
        "text_wins": text_wins,
        "neither_wins": neither_wins,
        "both_wins": both_wins,

        # Invalid reference answers
        "invalid_original": invalid_original,
        "invalid_corrupted": invalid_corrupted,

        # Useful percentages
        "image_win_rate": round(
            image_wins / valid_modality_total,
            6,
        ) if valid_modality_total else 0.0,

        "text_win_rate": round(
            text_wins / valid_modality_total,
            6,
        ) if valid_modality_total else 0.0,

        "neither_rate": round(
            neither_wins / valid_modality_total,
            6,
        ) if valid_modality_total else 0.0,

        "both_rate": round(
            both_wins / valid_modality_total,
            6,
        ) if valid_modality_total else 0.0,

        "valid_modality_total": valid_modality_total,

        "path": str(path),

        # Full sample-level diagnostic data.
        "samples": sample_analysis,
    }


# ===========================================================================
# Summary
# ===========================================================================

def aggregate_counts(rows):
    """
    Aggregate counts across files.

    Counts are pooled rather than averaging percentages, which is generally
    the right thing when all files contain the same number of samples.
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
            row[field]
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
    """

    summary = {
        "overall": aggregate_counts(rows),
        "by_task": {},
        "by_image_type": {},
        "by_model": {},
    }

    for row in rows:

        for group_name, key in [
            ("by_task", row["task"]),
            ("by_image_type", row["image_type"]),
            ("by_model", row["model"]),
        ]:
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
                aggregate_counts(group_rows)
            )

    return summary


# ===========================================================================
# Console reporting
# ===========================================================================

def print_summary(summary):
    """
    Print the mixed-signals modality-preference report.

    Percentages are calculated relative to the number of evaluated
    samples for each task.

    The accuracy column is intentionally omitted because this report
    is about modality preference, not correctness.
    """

    task_summary = summary["by_task"]

    print()
    print("=" * 105)
    print("MIXED-SIGNALS MODALITY PREFERENCE")
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

            return f"{count} ({percentage:.1f}%)"

        print(
            f"{task:<25}"
            f"{n:>7}"
            f"{fmt_count_pct(row['image_wins']):>20}"
            f"{fmt_count_pct(row['text_wins']):>20}"
            f"{fmt_count_pct(row['neither_wins']):>20}"
            f"{fmt_count_pct(row['both_wins']):>16}"
        )

    print()

    # ---------------------------------------------------------------
    # Overall
    # ---------------------------------------------------------------

    overall = summary["overall"]
    n = overall["total"]

    print("-" * 105)

    def fmt_overall(count):
        percentage = (
            100.0 * count / n
            if n
            else 0.0
        )

        return f"{count} ({percentage:.1f}%)"

    print(
        f"{'OVERALL':<25}"
        f"{n:>7}"
        f"{fmt_overall(overall['image_wins']):>20}"
        f"{fmt_overall(overall['text_wins']):>20}"
        f"{fmt_overall(overall['neither_wins']):>20}"
        f"{fmt_overall(overall['both_wins']):>16}"
    )

    print()

def write_csv(rows, output_path):
    """
    Write one row per result file.
    """

    output_path = Path(output_path)

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
                field: row[field]
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
            "Evaluate mixed-signals experiments. "
            "Correctness and answer parsing are "
            "delegated to evaluation.py."
        )
    )

    parser.add_argument(
        "--results-root",
        default="results/mixed_baseline",
    )

    parser.add_argument(
        "--output-json",
        default=(
            "evaluation/"
            "mixed_signals_results.json"
        ),
    )

    parser.add_argument(
        "--output-csv",
        default=(
            "evaluation/"
            "mixed_signals_results.csv"
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

    root = Path(
        args.results_root
    )

    if not root.exists():
        raise FileNotFoundError(
            f"Results root does not exist: {root}"
        )

    # ---------------------------------------------------------------
    # Evaluate every JSONL file.
    # ---------------------------------------------------------------

    rows = []

    for path in sorted(
        root.rglob("*.jsonl")
    ):

        if path.name.startswith("."):
            continue

        metadata = get_file_metadata(
            path,
            root,
        )

        # Filter by image type if requested.
        # We do this after loading because image_type lives in the
        # result records rather than necessarily in the directory.
        result = evaluate_file(
            path,
            root,
        )

        if (
            args.image_type != "all"
            and result["image_type"]
            != args.image_type
        ):
            continue

        rows.append(result)

    if not rows:
        print(
            "No matching result files found."
        )
        return

    # ---------------------------------------------------------------
    # Aggregate.
    # ---------------------------------------------------------------

    summary = build_summary(
        rows
    )

    # ---------------------------------------------------------------
    # JSON report.
    #
    # This contains:
    #   - file-level statistics
    #   - task-level statistics
    #   - overall statistics
    #   - every individual sample classification
    # ---------------------------------------------------------------

    report = {
        "generated_at": (
            datetime.now(
                timezone.utc
            ).isoformat()
        ),

        "results_root": str(root),

        "image_type_filter": (
            args.image_type
        ),

        "files": rows,

        "summary": summary,
    }

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
    # CSV.
    # ---------------------------------------------------------------

    write_csv(
        rows,
        args.output_csv,
    )

    # ---------------------------------------------------------------
    # Console.
    # ---------------------------------------------------------------

    print_summary(
        summary
    )


if __name__ == "__main__":
    main()