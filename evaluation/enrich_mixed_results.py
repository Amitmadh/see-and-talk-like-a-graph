import json
from pathlib import Path


# ============================================================
# CONFIG
# ============================================================

INPUT_FILE = Path("evaluation/mixed_signals_results.json")
OUTPUT_FILE = Path("evaluation/enrich_mixed_signals_results.json")

DATA_DIR = Path("data")

TEXT_ENCODING = "adjacency"


# ============================================================
# HELPERS
# ============================================================

def load_jsonl_by_sample_id(path):
    """
    Load a JSONL dataset and index it by sample_id.
    """
    samples = {}

    with open(path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()

            if not line:
                continue

            obj = json.loads(line)

            if "sample_id" not in obj:
                print(
                    f"WARNING: {path} line {line_num} "
                    "has no sample_id"
                )
                continue

            samples[obj["sample_id"]] = obj

    return samples


def dataset_path_for_task(
    task,
    data_dir=DATA_DIR,
    text_encoding=TEXT_ENCODING,
):
    """
    Path of the original GraphQA JSONL used to generate a task.
    """

    return Path(data_dir) / (
        f"{task}_{text_encoding}_test.jsonl"
    )


def get_or_load_dataset(
    task,
    dataset_cache,
    data_dir=DATA_DIR,
    text_encoding=TEXT_ENCODING,
    verbose=True,
):
    """
    Load a task dataset once and cache it by sample_id.

    Returns None when the JSONL file is missing, matching the
    previous script behaviour.
    """

    dataset_path = dataset_path_for_task(
        task,
        data_dir=data_dir,
        text_encoding=text_encoding,
    )

    if verbose:
        print()
        print("=" * 70)
        print(f"TASK: {task}")
        print(f"DATASET: {dataset_path}")

    if task in dataset_cache:
        return dataset_cache[task]

    if not dataset_path.exists():
        if verbose:
            print(f"WARNING: dataset does not exist: {dataset_path}")
        dataset_cache[task] = None
        return None

    if verbose:
        print("Loading dataset...")

    dataset_cache[task] = load_jsonl_by_sample_id(
        dataset_path
    )

    if verbose:
        print(
            f"Loaded {len(dataset_cache[task])} samples"
        )

    return dataset_cache[task]


def lookup_source_sample(sample, dataset_by_id):
    """
    Join a mixed-signals result row back to its GraphQA source
    example using sample_id.
    """

    if not dataset_by_id:
        return None

    sample_id = sample.get("sample_id")

    if sample_id is None:
        return None

    return dataset_by_id.get(sample_id)


def lookup_algorithm(sample, dataset_by_id=None):
    """
    Resolve the graph-generator algorithm for a mixed-signals
    sample via sample_id.

    Prefers the original dataset record so analysis does not
    depend on whether enrichment has already copied fields.
    Falls back to an algorithm field already present on the
    result row.
    """

    source_sample = lookup_source_sample(
        sample,
        dataset_by_id,
    )

    if source_sample is not None:
        algorithm = source_sample.get("algorithm")

        if algorithm:
            return str(algorithm)

    algorithm = sample.get("algorithm")

    if algorithm:
        return str(algorithm)

    return None


def enrich_results(
    results,
    data_dir=DATA_DIR,
    text_encoding=TEXT_ENCODING,
    dataset_cache=None,
):
    """
    Copy GraphQA metadata onto each mixed-signals sample.

    Existing fields on the result row are left unchanged.
    Returns a dict of match statistics.
    """

    if dataset_cache is None:
        dataset_cache = {}

    total_samples = 0
    matched_samples = 0
    missing_samples = 0
    fields_added = 0

    for file_result in results["files"]:

        task = file_result["task"]

        dataset_by_id = get_or_load_dataset(
            task,
            dataset_cache,
            data_dir=data_dir,
            text_encoding=text_encoding,
        )

        if dataset_by_id is None:
            continue

        for corruption_sample in file_result.get("samples", []):

            total_samples += 1

            sample_id = corruption_sample.get("sample_id")

            if sample_id is None:
                print("WARNING: corruption sample has no sample_id")
                missing_samples += 1
                continue

            source_sample = dataset_by_id.get(sample_id)

            if source_sample is None:
                print(
                    f"WARNING: sample_id not found in dataset: "
                    f"{sample_id}"
                )
                missing_samples += 1
                continue

            matched_samples += 1

            # ------------------------------------------------
            # Add ONLY fields that don't already exist
            # ------------------------------------------------

            for key, value in source_sample.items():

                if key not in corruption_sample:
                    corruption_sample[key] = value
                    fields_added += 1

    return {
        "total_samples": total_samples,
        "matched_samples": matched_samples,
        "missing_samples": missing_samples,
        "fields_added": fields_added,
        "dataset_cache": dataset_cache,
    }


def load_corruption_results(path=INPUT_FILE):
    path = Path(path)

    print(f"Loading corruption results:\n  {path}")

    with open(path, "r", encoding="utf-8") as f:
        results = json.load(f)

    print(f"Found {len(results['files'])} result files")

    return results


def save_enriched_results(results, path=OUTPUT_FILE):
    path = Path(path)

    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(
            results,
            f,
            indent=2,
            ensure_ascii=False
        )

    print()
    print(f"Saved enriched results to:")
    print(f"  {path}")


# ============================================================
# MAIN
# ============================================================

def main():

    results = load_corruption_results(INPUT_FILE)

    stats = enrich_results(
        results,
        data_dir=DATA_DIR,
        text_encoding=TEXT_ENCODING,
    )

    print()
    print("=" * 70)
    print("ENRICHMENT COMPLETE")
    print("=" * 70)

    print(f"Total corruption samples : {stats['total_samples']}")
    print(f"Matched                  : {stats['matched_samples']}")
    print(f"Missing                  : {stats['missing_samples']}")
    print(f"Fields added             : {stats['fields_added']}")

    save_enriched_results(results, OUTPUT_FILE)


if __name__ == "__main__":
    main()
