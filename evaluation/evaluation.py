import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, pstdev


TASK_TYPES = {
    "connected_nodes": "list",
    "cycle_check": "boolean",
    "edge_existence": "boolean",
    "node_count": "number",
    "node_degree": "number",
    "shortest_path": "number",
    "disconnected_nodes": "list",
    "edge_count": "number",
    "triangle_counting": "number",
}


@dataclass
class ParsedAnswer:
    value: object | None
    valid: bool
    raw: str | None


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def clean(answer):
    if answer is None:
        return None

    answer = str(answer).strip()
    return answer or None


def parse_expected(task, answer):
    """
    Ground-truth answers may be natural language.
    We therefore allow some flexibility here.
    """

    answer = clean(answer)

    task_type = TASK_TYPES[task]

    # An empty answer is a valid empty list for list-valued tasks.
    if answer is None:
        if task_type == "list":
            return ParsedAnswer([], True, "")
        
        return ParsedAnswer(None, False, None)

    task_type = TASK_TYPES[task]
    text = answer.lower()

    if task_type == "boolean":
        if re.search(r"\b(true|yes)\b", text):
            return ParsedAnswer(True, True, answer)

        if re.search(r"\b(false|no)\b", text):
            return ParsedAnswer(False, True, answer)

    elif task_type == "number":
        match = re.search(r"-?\d+", text)

        if match:
            return ParsedAnswer(
                int(match.group()),
                True,
                answer,
            )

    elif task_type == "list":
        # Extract all integer values from the expected answer.
        numbers = re.findall(r"-?\d+", text)

        if numbers:
            return ParsedAnswer(
                [int(x) for x in numbers],
                True,
                answer,
            )

        # Allow an explicit empty answer.
        if re.search(
            r"\b(none|no nodes|no other nodes)\b",
            text,
        ):
            return ParsedAnswer([], True, answer)

    return ParsedAnswer(None, False, answer)


def parse_model_answer(task, answer):
    """
    STRICT parser for model outputs.

    Accepted:

      boolean -> exactly True / False
      number  -> exactly one integer
      list    -> comma-separated integers, optionally in []

    Everything else is considered wrong format.
    """

    answer = clean(answer)

    if answer is None:
        return ParsedAnswer(None, False, None)

    task_type = TASK_TYPES[task]

    # -----------------------------------------------------------------------
    # Boolean
    # -----------------------------------------------------------------------

    if task_type == "boolean":
        if answer.lower() == "true":
            return ParsedAnswer(True, True, answer)

        if answer.lower() == "false":
            return ParsedAnswer(False, True, answer)

        return ParsedAnswer(None, False, answer)

    # -----------------------------------------------------------------------
    # Number
    # -----------------------------------------------------------------------

    if task_type == "number":
        if re.fullmatch(r"-?\d+", answer):
            return ParsedAnswer(
                int(answer),
                True,
                answer,
            )

        return ParsedAnswer(None, False, answer)

    # -----------------------------------------------------------------------
    # List
    # -----------------------------------------------------------------------

    if task_type == "list":
        original_answer = answer

        # Accept:
        #
        #   [1, 2, 3]
        #   1, 2, 3
        #
        # but NOT:
        #
        #   The nodes are 1, 2, 3
        #
        if answer.startswith("[") and answer.endswith("]"):
            answer = answer[1:-1].strip()

        if not answer:
            return ParsedAnswer(
                [],
                True,
                original_answer,
            )

        parts = [
            x.strip()
            for x in answer.split(",")
        ]

        if all(
            re.fullmatch(r"-?\d+", x)
            for x in parts
        ):
            return ParsedAnswer(
                [int(x) for x in parts],
                True,
                original_answer,
            )

        return ParsedAnswer(
            None,
            False,
            original_answer,
        )

    return ParsedAnswer(None, False, answer)


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------

def compare(task, expected_raw, predicted_raw):
    expected = parse_expected(
        task,
        expected_raw,
    )

    predicted = parse_model_answer(
        task,
        predicted_raw,
    )

    if not predicted.valid:
        return (
            False,
            "wrong_format",
            predicted.raw,
        )

    if not expected.valid:
        return (
            False,
            "bad_expected",
            expected.raw,
        )

    if TASK_TYPES[task] == "list":
        # Node lists form a set.
        # Order should therefore not matter.
        correct = (
            set(expected.value)
            == set(predicted.value)
        )
    else:
        correct = (
            expected.value
            == predicted.value
        )

    return (
        correct,
        "correct" if correct else "incorrect",
        predicted.raw,
    )


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_results_file(path):
    text = path.read_text(
        encoding="utf-8"
    ).strip()

    if not text:
        return []

    # JSON array
    if text.startswith("["):
        return json.loads(text)

    # JSONL
    return [
        json.loads(line)
        for line in text.splitlines()
        if line.strip()
    ]


def get_file_metadata(path, root):
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


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_file(path, root):
    metadata = get_file_metadata(
        path,
        root,
    )

    data = load_results_file(path)

    correct = 0
    total = 0
    wrong_format = 0
    bad_expected = 0

    wrong_format_answers = []

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

        if row.get("model_answer") is None:
            continue

        total += 1

        is_correct, status, raw = compare(
            metadata["task"],
            row.get("expected_answer"),
            row.get("model_answer"),
        )

        if is_correct:
            correct += 1

        elif status == "wrong_format":
            wrong_format += 1

            wrong_format_answers.append({
                "sample_id": row.get("sample_id"),
                "answer": raw,
            })

            print(
                f"[WRONG FORMAT] "
                f"{metadata['task']} "
                f"{row.get('sample_id')}: "
                f"{raw!r}"
            )

        elif status == "bad_expected":
            bad_expected += 1

    accuracy = (
        correct / total
        if total
        else 0.0
    )

    return {
        **metadata,
        "image_type": image_type,
        "model": model,
        "total": total,
        "correct": correct,
        "incorrect": (
            total
            - correct
            - wrong_format
        ),
        "wrong_format": wrong_format,
        "bad_expected": bad_expected,
        "accuracy": round(
            accuracy,
            6,
        ),
        "path": str(path),
        "wrong_format_answers": (
            wrong_format_answers
        ),
    }


def classify_modality(matches_original, matches_corrupted):
    if matches_original and matches_corrupted:
        return "both"

    if matches_original:
        return "image"

    if matches_corrupted:
        return "text"

    return "neither"


def evaluate_mixed_file(path, root):
    """
    Score a mixed-signals JSONL file.

    The model is compared to the original (image) answer and the
    corrupted (text) answer to decide which modality it followed.
    """

    metadata = get_file_metadata(
        path,
        root,
    )

    data = load_results_file(path)
    task = metadata["task"]

    if task not in TASK_TYPES:
        raise ValueError(
            f"Unknown task in mixed results: {task}"
        )

    correct = 0
    total = 0
    wrong_format = 0
    bad_expected = 0

    image_wins = 0
    text_wins = 0
    neither_wins = 0
    both_wins = 0
    invalid_original = 0
    invalid_corrupted = 0

    image_type = "unknown"
    model = "unknown"
    samples = []

    for row in data:
        image_type = row.get(
            "image_type",
            image_type,
        )

        model = row.get(
            "model",
            model,
        )

        if row.get("model_answer") is None:
            continue

        total += 1

        is_correct, status, raw = compare(
            task,
            row.get("expected_answer"),
            row.get("model_answer"),
        )

        if is_correct:
            correct += 1
        elif status == "wrong_format":
            wrong_format += 1
        elif status == "bad_expected":
            bad_expected += 1

        original_answer = row.get("original_answer")
        corrupted_answer = row.get(
            "corrupted_answer",
            row.get("expected_answer"),
        )

        original_expected = parse_expected(
            task,
            original_answer,
        )
        corrupted_expected = parse_expected(
            task,
            corrupted_answer,
        )

        if not original_expected.valid:
            invalid_original += 1

        if not corrupted_expected.valid:
            invalid_corrupted += 1

        matches_original, _, _ = compare(
            task,
            original_answer,
            row.get("model_answer"),
        )

        matches_corrupted, _, _ = compare(
            task,
            corrupted_answer,
            row.get("model_answer"),
        )

        winner = classify_modality(
            matches_original,
            matches_corrupted,
        )

        if winner == "image":
            image_wins += 1
        elif winner == "text":
            text_wins += 1
        elif winner == "neither":
            neither_wins += 1
        else:
            both_wins += 1

        samples.append({
            "sample_id": row.get("sample_id"),
            "expected_answer": row.get("expected_answer"),
            "original_answer": original_answer,
            "corrupted_answer": corrupted_answer,
            "model_answer": row.get("model_answer"),
            "corruption": row.get("corruption"),
            "correct": is_correct,
            "correctness_status": status,
            "modality_winner": winner,
            "matches_original": matches_original,
            "matches_corrupted": matches_corrupted,
        })

    valid_modality_total = (
        image_wins
        + text_wins
        + neither_wins
        + both_wins
    )

    accuracy = (
        correct / total
        if total
        else 0.0
    )

    return {
        **metadata,
        "image_type": image_type,
        "model": model,
        "total": total,
        "correct": correct,
        "incorrect": (
            total
            - correct
            - wrong_format
        ),
        "wrong_format": wrong_format,
        "bad_expected": bad_expected,
        "accuracy": round(
            accuracy,
            6,
        ),
        "image_wins": image_wins,
        "text_wins": text_wins,
        "neither_wins": neither_wins,
        "both_wins": both_wins,
        "invalid_original": invalid_original,
        "invalid_corrupted": invalid_corrupted,
        "image_win_rate": round(
            image_wins / valid_modality_total
            if valid_modality_total
            else 0.0,
            6,
        ),
        "text_win_rate": round(
            text_wins / valid_modality_total
            if valid_modality_total
            else 0.0,
            6,
        ),
        "neither_rate": round(
            neither_wins / valid_modality_total
            if valid_modality_total
            else 0.0,
            6,
        ),
        "both_rate": round(
            both_wins / valid_modality_total
            if valid_modality_total
            else 0.0,
            6,
        ),
        "valid_modality_total": valid_modality_total,
        "path": str(path),
        "samples": samples,
        "wrong_format_answers": [],
    }


def build_mixed_summary(rows):
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

    def aggregate(group_rows):
        result = {
            field: sum(
                row.get(field, 0)
                for row in group_rows
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

    summary = {
        "overall": aggregate(rows),
        "by_task": {},
        "by_image_type": {},
        "by_model": {},
    }

    for row in rows:
        for group_name, key in [
            ("by_task", row.get("task")),
            ("by_image_type", row.get("image_type")),
            ("by_model", row.get("model")),
        ]:
            if key is None:
                continue

            summary[group_name].setdefault(key, []).append(row)

    for group_name in ["by_task", "by_image_type", "by_model"]:
        for key, group_rows in summary[group_name].items():
            summary[group_name][key] = aggregate(group_rows)

    return summary


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def summarize(values):
    if not values:
        return {
            "count": 0,
            "mean_accuracy": 0.0,
            "std_accuracy": 0.0,
        }

    return {
        "count": len(values),
        "mean_accuracy": round(
            mean(values),
            6,
        ),
        "std_accuracy": round(
            pstdev(values),
            6,
        ),
    }


def build_summary(rows):
    summary = {
        "by_setting": {},
        "by_task": {},
        "by_image_type": {},
        "by_model": {},
    }

    def add(group, key, metric, accuracy):
        group.setdefault(
            key,
            {},
        ).setdefault(
            metric,
            [],
        ).append(accuracy)

    for row in rows:
        accuracy = row["accuracy"]

        add(
            summary["by_setting"],
            row["setting"],
            row["task"],
            accuracy,
        )

        add(
            summary["by_setting"],
            row["setting"],
            "all",
            accuracy,
        )

        add(
            summary["by_task"],
            row["task"],
            row["setting"],
            accuracy,
        )

        add(
            summary["by_task"],
            row["task"],
            "all",
            accuracy,
        )

        add(
            summary["by_image_type"],
            row["image_type"],
            row["setting"],
            accuracy,
        )

        add(
            summary["by_model"],
            row["model"],
            row["setting"],
            accuracy,
        )

    for group in summary.values():
        for key, metrics in group.items():
            for metric, values in metrics.items():
                metrics[metric] = summarize(values)

    return summary


# ---------------------------------------------------------------------------
# Accuracy table
# ---------------------------------------------------------------------------

def build_setting_table(rows, image_type="spring"):
    rows = [
        r
        for r in rows
        if (
            image_type is None
            or r["image_type"] == image_type
        )
    ]

    tasks = list(TASK_TYPES)
    settings = sorted(
        {
            r["setting"]
            for r in rows
        }
    )

    table = []

    for setting in settings:
        row = {
            "setting": setting
        }

        for task in tasks:
            matches = [
                r
                for r in rows
                if (
                    r["setting"] == setting
                    and r["task"] == task
                )
            ]

            if matches:
                average_accuracy = mean(
                    r["accuracy"]
                    for r in matches
                )
                row[task] = f"{average_accuracy:.3f}"
            else:
                row[task] = ""

        table.append(row)

    return table, tasks


def write_setting_table_csv(rows, output_path, image_type="spring"):
    table, tasks = build_setting_table(
        rows,
        image_type=image_type,
    )

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
            fieldnames=["setting"] + tasks,
        )
        writer.writeheader()
        writer.writerows(table)

    print(f"Saved setting table CSV to {output_path}")


def print_table(
    rows,
    image_type="spring",
):
    table, tasks = build_setting_table(
        rows,
        image_type=image_type,
    )

    if not table:
        print(
            f"No rows found for "
            f"image_type={image_type}"
        )
        return

    headers = [
        "setting"
    ] + tasks

    widths = [
        max(
            len(h),
            *[
                len(str(r.get(h, "")))
                for r in table
            ],
        )
        for h in headers
    ]

    fmt = " | ".join(
        f"{{:{w}}}"
        for w in widths
    )

    separator = "-+-".join(
        "-" * w
        for w in widths
    )

    print(
        fmt.format(*headers)
    )

    print(separator)

    for row in table:
        print(
            fmt.format(
                *[
                    row.get(h, "")
                    for h in headers
                ]
            )
        )


# ---------------------------------------------------------------------------
# Answer distributions
# ---------------------------------------------------------------------------

def print_answer_distributions(
    rows,
    image_type="spring",
):
    """
    Print the distribution of expected and predicted
    answers for every setting/task combination.

    This is particularly useful for detecting biases such as:

        expected:
            True  = 170
            False = 180

        model:
            True  = 350
            False = 0

    which would indicate that the model simply answers
    True for every example.
    """

    print()
    print("=" * 90)
    print("ANSWER DISTRIBUTIONS")
    print("=" * 90)

    # -----------------------------------------------------------------------
    # Iterate through individual result files.
    # -----------------------------------------------------------------------

    for metric_row in rows:

        if (
            image_type is not None
            and metric_row["image_type"]
            != image_type
        ):
            continue

        path = Path(
            metric_row["path"]
        )

        try:
            data = load_results_file(path)

        except Exception as e:
            print(
                f"[ERROR] Could not load "
                f"{path}: {e}"
            )
            continue

        task = metric_row["task"]
        setting = metric_row["setting"]

        expected_counter = Counter()
        predicted_counter = Counter()

        correct = 0
        incorrect = 0
        wrong_format = 0
        bad_expected = 0

        for sample in data:

            if sample.get(
                "model_answer"
            ) is None:
                continue

            expected = parse_expected(
                task,
                sample.get(
                    "expected_answer"
                ),
            )

            predicted = parse_model_answer(
                task,
                sample.get(
                    "model_answer"
                ),
            )

            # ---------------------------------------------------------------
            # Expected distribution
            # ---------------------------------------------------------------

            if expected.valid:
                expected_key = format_distribution_value(
                    expected.value
                )
            else:
                expected_key = "<BAD_EXPECTED>"

            expected_counter[
                expected_key
            ] += 1

            # ---------------------------------------------------------------
            # Predicted distribution
            # ---------------------------------------------------------------

            if predicted.valid:
                predicted_key = format_distribution_value(
                    predicted.value
                )
            else:
                predicted_key = "<WRONG_FORMAT>"

            predicted_counter[
                predicted_key
            ] += 1

            # ---------------------------------------------------------------
            # Evaluation status
            # ---------------------------------------------------------------

            is_correct, status, _ = compare(
                task,
                sample.get(
                    "expected_answer"
                ),
                sample.get(
                    "model_answer"
                ),
            )

            if is_correct:
                correct += 1

            elif status == "wrong_format":
                wrong_format += 1

            elif status == "bad_expected":
                bad_expected += 1

            else:
                incorrect += 1

        total = (
            correct
            + incorrect
            + wrong_format
        )

        accuracy = (
            correct / total
            if total
            else 0.0
        )

        # ---------------------------------------------------------------
        # Print
        # ---------------------------------------------------------------

        print()
        print(
            f"Setting: {setting}"
        )
        print(
            f"Task:    {task}"
        )
        print(
            f"Type:    {TASK_TYPES[task]}"
        )
        print(
            f"Image:   {metric_row['image_type']}"
        )
        print(
            f"Model:   {metric_row['model']}"
        )
        print(
            f"Samples: {total}"
        )

        print()
        print("  Expected distribution:")

        for value, count in (
            expected_counter.most_common()
        ):
            percentage = (
                100 * count / total
                if total
                else 0
            )

            print(
                f"    {value}: "
                f"{count} "
                f"({percentage:.1f}%)"
            )

        print()
        print("  Model prediction distribution:")

        for value, count in (
            predicted_counter.most_common()
        ):
            percentage = (
                100 * count / total
                if total
                else 0
            )

            print(
                f"    {value}: "
                f"{count} "
                f"({percentage:.1f}%)"
            )

        print()
        print("  Evaluation:")

        print(
            f"    Correct:       {correct}"
        )

        print(
            f"    Incorrect:     {incorrect}"
        )

        print(
            f"    Wrong format:  {wrong_format}"
        )

        print(
            f"    Bad expected:  {bad_expected}"
        )

        print(
            f"    Accuracy:      {accuracy:.3f}"
        )

        # ---------------------------------------------------------------
        # Special diagnostic for binary tasks
        # ---------------------------------------------------------------

        if TASK_TYPES[task] == "boolean":

            true_predictions = predicted_counter.get(
                "True",
                0,
            )

            false_predictions = predicted_counter.get(
                "False",
                0,
            )

            if true_predictions == total:
                print()
                print(
                    "    *** WARNING: MODEL ANSWERED "
                    "TRUE FOR EVERY SAMPLE ***"
                )

            elif false_predictions == total:
                print()
                print(
                    "    *** WARNING: MODEL ANSWERED "
                    "FALSE FOR EVERY SAMPLE ***"
                )


def format_distribution_value(value):
    """
    Convert parsed values to stable human-readable
    distribution keys.
    """

    if isinstance(value, list):
        return "[" + ", ".join(
            str(x)
            for x in sorted(set(value))
        ) + "]"

    if isinstance(value, bool):
        return (
            "True"
            if value
            else "False"
        )

    return str(value)


# ---------------------------------------------------------------------------
# Main aggregation
# ---------------------------------------------------------------------------

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Evaluate model answers "
            "from JSONL result files."
        )
    )

    parser.add_argument(
        "--results-root",
        default="results/baseline",
    )

    parser.add_argument(
        "--output-json",
        default=(
            "evaluation/"
            "aggregate_results.json"
        ),
    )

    parser.add_argument(
        "--output-csv",
        default=(
            "evaluation/"
            "aggregate_results.csv"
        ),
    )

    parser.add_argument(
        "--output-setting-csv",
        default=(
            "evaluation/"
            "aggregate_results_by_setting.csv"
        ),
    )

    parser.add_argument(
        "--image-type",
        default="spring",
        help=(
            "Image type to show in "
            "the accuracy table and "
            "distributions. Use 'all' "
            "to include everything."
        ),
    )

    parser.add_argument(
        "--mixed",
        action="store_true",
        help=(
            "Score mixed-signals result files "
            "(original vs corrupted answers) "
            "instead of clean baseline accuracy."
        ),
    )

    parser.add_argument(
        "--vis-dir",
        default="evaluation/vis",
        help="Directory for visualization PNGs.",
    )

    parser.add_argument(
        "--skip-vis",
        action="store_true",
        help="Skip writing visualization PNGs.",
    )

    args = parser.parse_args()

    root = Path(
        args.results_root
    )

    if not root.exists():
        raise FileNotFoundError(
            f"Results root does not exist: "
            f"{root}"
        )

    rows = []

    for path in sorted(
        root.rglob("*.jsonl")
    ):

        if path.name.startswith("."):
            continue

        rows.append(
            evaluate_mixed_file(
                path,
                root,
            )
            if args.mixed
            else evaluate_file(
                path,
                root,
            )
        )

    summary = (
        build_mixed_summary(rows)
        if args.mixed
        else build_summary(rows)
    )

    # -----------------------------------------------------------------------
    # Collect wrong-format answers
    # -----------------------------------------------------------------------

    wrong_format_answers = [
        {
            "setting": row["setting"],
            "task": row["task"],
            "image_type": row[
                "image_type"
            ],
            "model": row["model"],
            "file_name": row[
                "file_name"
            ],
            "sample_id": item[
                "sample_id"
            ],
            "answer": item[
                "answer"
            ],
        }
        for row in rows
        for item in row[
            "wrong_format_answers"
        ]
    ]

    # -----------------------------------------------------------------------
    # JSON report
    # -----------------------------------------------------------------------

    report = {
        "generated_at": (
            datetime.now(
                timezone.utc
            ).isoformat()
        ),
        "image_type_filter": args.image_type,
        "files": rows,
        "summary": summary,
        "wrong_format_answers": (
            wrong_format_answers
        ),
        "wrong_format_count": (
            len(wrong_format_answers)
        ),
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

    # -----------------------------------------------------------------------
    # CSV report
    # -----------------------------------------------------------------------

    csv_fields = [
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
    ]

    if args.mixed:
        csv_fields.extend([
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
        ])

    output_csv = Path(
        args.output_csv
    )

    output_csv.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        output_csv,
        "w",
        encoding="utf-8",
        newline="",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=csv_fields,
        )

        writer.writeheader()

        for row in rows:
            writer.writerow({
                key: row[key]
                for key in csv_fields
            })

    # -----------------------------------------------------------------------
    # Pivoted CSV by setting/task
    # -----------------------------------------------------------------------

    output_setting_csv = Path(
        args.output_setting_csv
    )

    write_setting_table_csv(
        rows,
        output_setting_csv,
        image_type=(
            None
            if args.image_type == "all"
            else args.image_type
        ),
    )

    # -----------------------------------------------------------------------
    # Console output
    # -----------------------------------------------------------------------

    print(
        f"Saved JSON report to "
        f"{output_json}"
    )

    print(
        f"Saved CSV report to "
        f"{output_csv}"
    )

    print(
        f"Wrong-format answers: "
        f"{len(wrong_format_answers)}"
    )

    print()

    print(
        f"Accuracy table "
        f"({args.image_type}):"
    )

    print_table(
        rows,
        image_type=(
            None
            if args.image_type == "all"
            else args.image_type
        ),
    )

    if not args.mixed and not args.skip_vis:

        try:
            from evaluation.vis.plot_baseline import (
                write_baseline_plot,
            )

        except ImportError as exc:
            print(
                "Skipping visualization "
                f"(matplotlib unavailable): {exc}"
            )

        else:
            vis_path = write_baseline_plot(
                rows,
                output_path=Path(args.vis_dir) / "baseline.png",
                image_type=args.image_type,
            )

            if vis_path is not None:
                print(
                    f"Saved visualization to {vis_path}"
                )

    # -----------------------------------------------------------------------
    # NEW: answer distributions
    # -----------------------------------------------------------------------

    # print_answer_distributions(
    #     rows,
    #     image_type=(
    #         None
    #         if args.image_type == "all"
    #         else args.image_type
    #     ),
    # )


if __name__ == "__main__":
    main()
