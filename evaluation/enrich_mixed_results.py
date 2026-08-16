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


# ============================================================
# LOAD CORRUPTION RESULTS
# ============================================================

print(f"Loading corruption results:\n  {INPUT_FILE}")

with open(INPUT_FILE, "r", encoding="utf-8") as f:
    results = json.load(f)

print(f"Found {len(results['files'])} result files")


# ============================================================
# CACHE DATASETS
# ============================================================

dataset_cache = {}

total_samples = 0
matched_samples = 0
missing_samples = 0
fields_added = 0


# ============================================================
# ENRICH EACH RESULT FILE
# ============================================================

for file_result in results["files"]:

    task = file_result["task"]

    dataset_path = DATA_DIR / f"{task}_{TEXT_ENCODING}_test.jsonl"

    print()
    print("=" * 70)
    print(f"TASK: {task}")
    print(f"DATASET: {dataset_path}")

    # --------------------------------------------------------
    # Load dataset only once per task
    # --------------------------------------------------------

    if task not in dataset_cache:

        if not dataset_path.exists():
            print(f"WARNING: dataset does not exist: {dataset_path}")
            dataset_cache[task] = None
        else:
            print("Loading dataset...")

            dataset_cache[task] = load_jsonl_by_sample_id(
                dataset_path
            )

            print(
                f"Loaded {len(dataset_cache[task])} samples"
            )

    dataset_by_id = dataset_cache[task]

    if dataset_by_id is None:
        continue

    # --------------------------------------------------------
    # Enrich samples
    # --------------------------------------------------------

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

        # ----------------------------------------------------
        # Add ONLY fields that don't already exist
        # ----------------------------------------------------

        for key, value in source_sample.items():

            if key not in corruption_sample:
                corruption_sample[key] = value
                fields_added += 1


# ============================================================
# SAVE ENRICHED RESULTS
# ============================================================

print()
print("=" * 70)
print("ENRICHMENT COMPLETE")
print("=" * 70)

print(f"Total corruption samples : {total_samples}")
print(f"Matched                  : {matched_samples}")
print(f"Missing                  : {missing_samples}")
print(f"Fields added             : {fields_added}")


OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(
        results,
        f,
        indent=2,
        ensure_ascii=False
    )

print()
print(f"Saved enriched results to:")
print(f"  {OUTPUT_FILE}")