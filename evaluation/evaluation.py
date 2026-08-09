import argparse
import csv
import json
import re
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from statistics import mean, pstdev


TASK_TYPES = {
    "connected_nodes": "list",
    "cycle_check": "boolean",
    "edge_existence": "boolean",
    "node_count": "number",
    "node_degree": "number",
    "shortest_path": "number_or_no_path",
}

BOOLEAN_TRUE_PATTERNS = [
    r"\btrue\b",
    r"\byes\b",
    r"\byep\b",
    r"\baffirmative\b",
    r"\bthere is a cycle\b",
    r"\bcycle exists\b",
    r"\bconnected\b",
]

BOOLEAN_FALSE_PATTERNS = [
    r"\bfalse\b",
    r"\bno\b",
    r"\bnope\b",
    r"\bnot\b",
    r"\bthere is no cycle\b",
    r"\bno cycle\b",
    r"\bnot connected\b",
    r"\bdisconnected\b",
    r"\bnone\b",
]

NO_PATH_PATTERNS = [
    r"\bno path\b",
    r"\bthere is no path\b",
    r"\bnot connected\b",
    r"\bnot reachable\b",
    r"\bno route\b",
]


@dataclass
class ParsedAnswer:
    raw: str | None
    type: str | None
    value: object | None


def normalize_text(answer: str | None) -> str | None:
    if answer is None:
        return None
    answer = str(answer).strip()
    if not answer:
        return None
    return answer


def normalize_string(answer: str | None) -> str | None:
    answer = normalize_text(answer)
    if answer is None:
        return None
    text = answer.lower()
    text = re.sub(r"[\r\n]+", " ", text)
    text = re.sub(r"[^0-9a-z]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def parse_boolean(answer: str | None) -> ParsedAnswer:
    answer = normalize_text(answer)
    if answer is None:
        return ParsedAnswer(raw=answer, type=None, value=None)

    text = answer.lower()
    for pattern in BOOLEAN_TRUE_PATTERNS:
        if re.search(pattern, text):
            return ParsedAnswer(raw=answer, type="boolean", value=True)
    for pattern in BOOLEAN_FALSE_PATTERNS:
        if re.search(pattern, text):
            return ParsedAnswer(raw=answer, type="boolean", value=False)

    # Fallback on exact tokens
    if text in {"true", "false", "yes", "no", "y", "n"}:
        return ParsedAnswer(raw=answer, type="boolean", value=text in {"true", "yes", "y"})

    return ParsedAnswer(raw=answer, type=None, value=None)


def parse_number(answer: str | None) -> ParsedAnswer:
    answer = normalize_text(answer)
    if answer is None:
        return ParsedAnswer(raw=answer, type=None, value=None)

    text = answer.lower()
    for pattern in NO_PATH_PATTERNS:
        if re.search(pattern, text):
            return ParsedAnswer(raw=answer, type="no_path", value=None)

    match = re.search(r"-?\d+", text)
    if match:
        return ParsedAnswer(raw=answer, type="number", value=int(match.group()))

    return ParsedAnswer(raw=answer, type=None, value=None)


def parse_list(answer: str | None) -> ParsedAnswer:
    answer = normalize_text(answer)
    if answer is None:
        return ParsedAnswer(raw=answer, type=None, value=None)

    text = answer.strip().lower()
    text = re.sub(r"\b(and|or)\b", ",", text)
    text = re.sub(r"[^0-9a-z,]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    tokens = []
    for piece in re.split(r"[,;]+", text):
        piece = piece.strip()
        if not piece:
            continue
        if piece.isdigit():
            tokens.append(int(piece))
        else:
            tokens.append(piece)

    if tokens:
        return ParsedAnswer(raw=answer, type="list", value=tokens)

    return ParsedAnswer(raw=answer, type=None, value=None)


def parse_answer(task: str, answer: str | None) -> ParsedAnswer:
    answer = normalize_text(answer)
    if answer is None:
        return ParsedAnswer(raw=None, type=None, value=None)

    task_type = TASK_TYPES.get(task)
    if task_type == "boolean":
        parsed = parse_boolean(answer)
        if parsed.type is not None:
            return parsed
    if task_type == "number_or_no_path":
        parsed = parse_number(answer)
        if parsed.type is not None:
            return parsed
    if task_type == "number":
        parsed = parse_number(answer)
        if parsed.type is not None:
            return parsed
    if task_type == "list":
        parsed = parse_list(answer)
        if parsed.type is not None:
            return parsed

    # Generic fallback pipeline.
    parsed = parse_number(answer)
    if parsed.type is not None:
        return parsed
    parsed = parse_boolean(answer)
    if parsed.type is not None:
        return parsed
    parsed = parse_list(answer)
    if parsed.type is not None:
        return parsed

    return ParsedAnswer(raw=answer, type="text", value=normalize_string(answer))


def compare_values(expected: ParsedAnswer, predicted: ParsedAnswer) -> bool:
    if expected.type == predicted.type and expected.type is not None:
        if expected.type == "list":
            return compare_lists(expected.value, predicted.value)
        return expected.value == predicted.value

    if expected.type == "list" and predicted.type is not None:
        return compare_lists(expected.value, predicted.value)
    if predicted.type == "list" and expected.type is not None:
        return compare_lists(expected.value, predicted.value)

    if expected.type in {"boolean", "number", "no_path"} and predicted.type is not None:
        return expected.value == predicted.value
    if predicted.type in {"boolean", "number", "no_path"} and expected.type is not None:
        return expected.value == predicted.value

    if expected.raw is not None and predicted.raw is not None:
        return normalize_string(expected.raw) == normalize_string(predicted.raw)

    return False


def compare_lists(expected: list, predicted: list) -> bool:
    if expected is None or predicted is None:
        return False

    expected_norm = {normalize_string(str(v)) for v in expected if normalize_string(str(v))}
    predicted_norm = {normalize_string(str(v)) for v in predicted if normalize_string(str(v))}
    return expected_norm == predicted_norm


def load_results_file(path: Path) -> list[dict]:
    raw_text = path.read_text(encoding="utf-8").strip()
    if not raw_text:
        return []
    if raw_text.startswith("["):
        return json.loads(raw_text)
    return [json.loads(line) for line in raw_text.splitlines() if line.strip()]


def get_file_metadata(result_path: Path, root_dir: Path) -> dict:
    relative = result_path.relative_to(root_dir)
    parts = list(relative.parts)
    if len(parts) < 3:
        raise ValueError(f"Unexpected results path structure: {result_path}")

    return {
        "setting": parts[0],
        "task": parts[1],
        "file_name": result_path.name,
    }


def score_result_row(row: dict) -> bool:
    expected = parse_answer(row["task"], row.get("expected_answer"))
    predicted = parse_answer(row["task"], row.get("model_answer"))
    return compare_values(expected, predicted)


def summarize_scores(scores: list[float]) -> dict:
    if not scores:
        return {"count": 0, "mean_accuracy": 0.0, "std_accuracy": 0.0}
    return {
        "count": len(scores),
        "mean_accuracy": round(mean(scores), 6),
        "std_accuracy": round(pstdev(scores), 6),
    }


def add_summary_entry(summary: dict, group_key: str, accuracy: float) -> None:
    summary.setdefault(group_key, []).append(accuracy)


def build_nested_summary(metric_rows: list[dict]) -> dict:
    summary = {
        "by_setting": {},
        "by_task": {},
        "by_image_type": {},
        "by_model": {},
    }

    def update_group(group_dict, group_key, metric_key, accuracy):
        group_dict.setdefault(group_key, {}).setdefault(metric_key, []).append(accuracy)
        group_dict.setdefault(group_key, {}).setdefault("all", []).append(accuracy)

    for row in metric_rows:
        setting = row["setting"]
        task = row["task"]
        image_type = row.get("image_type") or "unknown"
        model_name = row.get("model") or "unknown"
        accuracy = row["accuracy"]

        update_group(summary["by_setting"], setting, task, accuracy)
        update_group(summary["by_setting"], setting, image_type, accuracy)
        update_group(summary["by_setting"], setting, "all", accuracy)

        update_group(summary["by_task"], task, setting, accuracy)
        update_group(summary["by_task"], task, "all", accuracy)

        update_group(summary["by_image_type"], image_type, setting, accuracy)
        update_group(summary["by_image_type"], image_type, "all", accuracy)

        update_group(summary["by_model"], model_name, setting, accuracy)
        update_group(summary["by_model"], model_name, "all", accuracy)

    for outer in summary.values():
        for key, metrics in outer.items():
            for metric_key, values in list(metrics.items()):
                metrics[metric_key] = summarize_scores(values)

    return summary


def build_setting_task_table(metric_rows: list[dict], image_type_filter: str | None = "spring") -> tuple[list[str], list[str], list[dict]]:
    rows = [row for row in metric_rows if image_type_filter is None or row.get("image_type") == image_type_filter]
    settings = sorted({row["setting"] for row in rows})
    tasks = list(TASK_TYPES.keys())

    table = []
    for setting in settings:
        row = {"setting": setting}
        for task in tasks:
            row[task] = ""
        table.append(row)

    for row in rows:
        for table_row in table:
            if table_row["setting"] == row["setting"]:
                table_row[row["task"]] = f"{row['accuracy']:.3f}"
                break

    return settings, tasks, table


def print_setting_task_table(metric_rows: list[dict], image_type_filter: str | None = "spring") -> None:
    settings, tasks, table = build_setting_task_table(metric_rows, image_type_filter)
    if not table:
        print(f"No rows found for image_type={image_type_filter}")
        return

    headers = ["setting"] + tasks
    widths = [max(len(h), *(len(str(row.get(h, ""))) for row in table)) for h in headers]
    fmt = " | ".join(f"{{:{w}}}" for w in widths)
    separator = "-+-".join("-" * w for w in widths)

    print(fmt.format(*headers))
    print(separator)
    for row in table:
        print(fmt.format(*(row.get(h, "") for h in headers)))


def gather_metrics(root_dir: Path) -> tuple[list[dict], dict]:
    metric_rows: list[dict] = []
    for result_path in sorted(root_dir.rglob("*.jsonl")):
        if result_path.is_dir():
            continue
        if result_path.name.startswith("."):
            continue

        metadata = get_file_metadata(result_path, root_dir)
        data = load_results_file(result_path)
        total = 0
        correct = 0
        file_image_type = "unknown"
        file_model = "unknown"

        for row in data:
            if row.get("image_type"):
                file_image_type = row["image_type"]
            if row.get("model"):
                file_model = row["model"]
            if row.get("model_answer") is None:
                continue
            total += 1
            if score_result_row({**row, **metadata}):
                correct += 1

        accuracy = 0.0 if total == 0 else correct / total
        metric_rows.append({
            "setting": metadata["setting"],
            "task": metadata["task"],
            "image_type": file_image_type,
            "model": file_model,
            "file_name": metadata["file_name"],
            "path": str(result_path),
            "total": total,
            "correct": correct,
            "accuracy": round(accuracy, 6),
        })

    return metric_rows, build_nested_summary(metric_rows)


def save_json_report(report: dict, output_file: Path) -> None:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)


def save_csv_report(rows: list[dict], output_file: Path) -> None:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    keys = [
        "setting",
        "task",
        "image_type",
        "model",
        "file_name",
        "path",
        "total",
        "correct",
        "accuracy",
    ]
    with open(output_file, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Aggregate accuracy metrics from results JSON files."
    )
    parser.add_argument(
        "--results-root",
        default="results/baseline",
        help="Root path for result files to aggregate.",
    )
    parser.add_argument(
        "--output-json",
        default="evaluation/aggregate_results.json",
        help="JSON summary output path.",
    )
    parser.add_argument(
        "--output-csv",
        default="evaluation/aggregate_results.csv",
        help="CSV file with file-level accuracy rows.",
    )
    args = parser.parse_args()

    root_dir = Path(args.results_root)
    output_json_path = Path(args.output_json)
    output_csv_path = Path(args.output_csv)

    if not root_dir.exists():
        raise FileNotFoundError(f"Results root does not exist: {root_dir}")

    metric_rows, summary = gather_metrics(root_dir)
    report = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "results_root": str(root_dir),
        "files": metric_rows,
        "summary": summary,
    }

    save_json_report(report, output_json_path)
    save_csv_report(metric_rows, output_csv_path)
    print(f"Saved JSON report to {output_json_path}")
    print(f"Saved CSV report to {output_csv_path}")
    print("\nAccuracy table by setting and task (spring image type):")
    print_setting_task_table(metric_rows, image_type_filter="spring")


if __name__ == "__main__":
    main()
