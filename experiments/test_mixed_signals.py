from pathlib import Path
import tempfile

from experiments.utils import (
    load_config,
    load_results,
    get_project_root,
    log,
)

from datasets.graph_qa_dataset import load_graphqa_dataset

from experiments.corruption import (
    corrupt_dataset,
    save_corrupted_dataset,
)


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

CONFIG_PATH = "experiments/configs/mixed_baseline.yaml"

TASKS = [
    "shortest_path",
    "node_degree",
    "connected_nodes",
    "node_count",
    "edge_count",
]

N_TEST_SAMPLES = 10


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def check(condition, message):
    if not condition:
        raise AssertionError(message)


def test_load_results(root):
    """
    Verify that load_results() can read an existing baseline file.
    """

    log("Testing load_results()...")

    task = "connected_nodes"
    image_type = "spring"
    model_name = "Qwen2.5-VL-3B-Instruct"

    results_path = (
        root
        / "results"
        / "baseline"
        / "text_and_image"
        / task
        / f"{image_type}_{model_name}.jsonl"
    )

    log(f"Loading existing results: {results_path}")

    results = load_results(results_path)

    check(
        isinstance(results, list),
        "load_results() did not return a list.",
    )

    check(
        len(results) > 0,
        "Existing results file is empty.",
    )

    first = results[0]

    check(
        isinstance(first, dict),
        "Individual result is not a dictionary.",
    )

    required_fields = {
        "sample_id",
        "model_answer",
        "expected_answer",
    }

    missing = required_fields - set(first.keys())

    check(
        not missing,
        f"Result is missing fields: {missing}",
    )

    log(
        f"load_results() passed: "
        f"{len(results)} results loaded."
    )

    return results

def test_task_dataset(root, config, task):
    """
    Load one original task dataset and verify the Sample structure.
    """

    text_encoding = config["text_encoding"]

    dataset_path = (
        root
        / config["data_dir"]
        / f"{task}_{text_encoding}_test.jsonl"
    )

    log(f"Loading {task}: {dataset_path}")

    check(
        dataset_path.exists(),
        f"Dataset does not exist: {dataset_path}",
    )

    dataset = load_graphqa_dataset(
        dataset_dir=dataset_path
    )

    check(
        len(dataset) > 0,
        f"Dataset for {task} is empty.",
    )

    sample = dataset[0]

    # Basic Sample checks.
    check(
        hasattr(sample, "sample_id"),
        f"{task}: Sample has no sample_id.",
    )

    check(
        hasattr(sample, "graph"),
        f"{task}: Sample has no graph.",
    )

    check(
        hasattr(sample, "question"),
        f"{task}: Sample has no question.",
    )

    check(
        hasattr(sample, "answer"),
        f"{task}: Sample has no answer.",
    )

    check(
        hasattr(sample, "text_encoding"),
        f"{task}: Sample has no text_encoding.",
    )

    check(
        hasattr(sample, "images"),
        f"{task}: Sample has no images.",
    )

    # node_ids was added to Sample specifically for this experiment.
    check(
        hasattr(sample, "node_ids"),
        f"{task}: Sample has no node_ids.",
    )

    return dataset


def test_corruption(dataset, task, text_encoding):
    """
    Corrupt a small number of samples and verify all invariants.
    """

    samples = dataset[:N_TEST_SAMPLES]

    log(
        f"Corrupting {len(samples)} samples "
        f"for task={task}..."
    )

    corrupted = corrupt_dataset(
        samples,
        task=task,
        text_encoding=text_encoding,
    )

    check(
        len(corrupted) == len(samples),
        f"{task}: corruption changed dataset size.",
    )

    for original, corrupt in zip(samples, corrupted):

        # -------------------------------------------------------------
        # Identity
        # -------------------------------------------------------------

        check(
            original.sample_id == corrupt.sample_id,
            f"{task}: sample_id changed.",
        )

        # -------------------------------------------------------------
        # Image MUST remain unchanged.
        # -------------------------------------------------------------

        check(
            original.images == corrupt.images,
            (
                f"{task}/{original.sample_id}: "
                "images changed during text corruption."
            ),
        )

        # -------------------------------------------------------------
        # Graph SHOULD change.
        # -------------------------------------------------------------

        check(
            original.graph != corrupt.graph,
            (
                f"{task}/{original.sample_id}: "
                "graph did not change."
            ),
        )

        # -------------------------------------------------------------
        # Text SHOULD change.
        # -------------------------------------------------------------

        check(
            original.text_encoding != corrupt.text_encoding,
            (
                f"{task}/{original.sample_id}: "
                "text_encoding did not change."
            ),
        )

        # -------------------------------------------------------------
        # Answer MUST change.
        # -------------------------------------------------------------

        check(
            str(original.answer) != str(corrupt.answer),
            (
                f"{task}/{original.sample_id}: "
                f"answer did not change: "
                f"{original.answer!r}"
            ),
        )

        # -------------------------------------------------------------
        # Metadata must preserve both answers.
        # -------------------------------------------------------------

        check(
            corrupt.metadata["original_answer"]
            == original.answer,
            (
                f"{task}/{original.sample_id}: "
                "original_answer not preserved."
            ),
        )

        check(
            corrupt.metadata["corrupted_answer"]
            == corrupt.answer,
            (
                f"{task}/{original.sample_id}: "
                "corrupted_answer not preserved."
            ),
        )

        # -------------------------------------------------------------
        # Corruption information must exist.
        # -------------------------------------------------------------

        check(
            "corruption" in corrupt.metadata,
            (
                f"{task}/{original.sample_id}: "
                "corruption metadata missing."
            ),
        )

    log(
        f"{task}: corruption passed "
        f"({len(corrupted)} samples)."
    )

    return samples, corrupted


def test_save_reload(root, config, task, corrupted):
    """
    Save a temporary corrupted dataset and make sure the dataset
    loader can read it back correctly.
    """

    log(f"{task}: testing save -> reload...")

    # Use a temporary directory so we never overwrite real data.
    with tempfile.TemporaryDirectory(
        prefix="mixed_signals_test_"
    ) as tmp:

        output_path = (
            Path(tmp)
            / f"{task}_corrupted_test.jsonl"
        )

        save_corrupted_dataset(
            corrupted,
            output_path,
        )

        check(
            output_path.exists(),
            f"{task}: corrupted dataset was not saved.",
        )

        reloaded = load_graphqa_dataset(
            dataset_dir=output_path
        )

        check(
            len(reloaded) == len(corrupted),
            (
                f"{task}: reload size mismatch: "
                f"{len(reloaded)} vs {len(corrupted)}"
            ),
        )

        for before, after in zip(corrupted, reloaded):

            check(
                before.sample_id == after.sample_id,
                f"{task}: sample_id changed after reload.",
            )

            check(
                before.answer == after.answer,
                f"{task}: answer changed after reload.",
            )

            check(
                before.text_encoding == after.text_encoding,
                f"{task}: text_encoding changed after reload.",
            )

            check(
                before.images == after.images,
                f"{task}: images changed after reload.",
            )

            if before.node_ids != after.node_ids:
                print(
                    f"{task}: node_ids mismatch after reload:"
                )
                print("  BEFORE:", repr(before.node_ids), type(before.node_ids))
                print("  AFTER: ", repr(after.node_ids), type(after.node_ids))

                raise AssertionError(
                    f"{task}: node_ids changed after reload."
                )

            check(
                after.answer
                == before.answer,
                (
                    f"{task}: answer lost "
                    "during reload."
                ),
            )

            check(
                after.metadata["corrupted_answer"]
                == before.metadata["corrupted_answer"],
                (
                    f"{task}: corrupted_answer lost "
                    "during reload."
                ),
            )

    log(f"{task}: save -> reload passed.")


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():

    log("=" * 70)
    log("MIXED-SIGNALS INTEGRATION TEST")
    log("NO MODEL WILL BE LOADED")
    log("=" * 70)

    root = get_project_root()

    config = load_config(
        root / CONFIG_PATH
    )

    text_encoding = config["text_encoding"]

    log(f"Text encoding: {text_encoding}")
    log(f"Testing tasks: {TASKS}")
    log(f"Samples per task: {N_TEST_SAMPLES}")

    # -------------------------------------------------------------
    # 1. Test result loading.
    # -------------------------------------------------------------

    test_load_results(
        root,
    )

    # -------------------------------------------------------------
    # 2. Test every task.
    # -------------------------------------------------------------

    for task in TASKS:

        log("")
        log("=" * 70)
        log(f"TASK: {task}")
        log("=" * 70)

        dataset = test_task_dataset(
            root,
            config,
            task,
        )

        log(
            f"{task}: loaded {len(dataset)} samples."
        )

        original, corrupted = test_corruption(
            dataset,
            task,
            text_encoding,
        )

        # ---------------------------------------------------------
        # Print a few human-readable examples.
        # ---------------------------------------------------------

        print()
        print(f"Examples for {task}:")

        for original_sample, corrupted_sample in zip(
            original[:3],
            corrupted[:3],
        ):
            print("-" * 60)
            print("sample_id:")
            print(" ", original_sample.sample_id)

            print("question:")
            print(" ", original_sample.question)

            print("original answer:")
            print(" ", repr(original_sample.answer))

            print("corrupted answer:")
            print(" ", repr(corrupted_sample.answer))

            print("corruption:")
            print(
                " ",
                corrupted_sample.metadata["corruption"],
            )

        # ---------------------------------------------------------
        # Save + reload test.
        # ---------------------------------------------------------

        test_save_reload(
            root,
            config,
            task,
            corrupted,
        )

    # -------------------------------------------------------------
    # Finished.
    # -------------------------------------------------------------

    log("")
    log("=" * 70)
    log("ALL MIXED-SIGNALS INTEGRATION TESTS PASSED")
    log("=" * 70)
    log("No model was loaded or called.")


if __name__ == "__main__":
    main()