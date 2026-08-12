from pathlib import Path
import argparse

from experiments.utils import log, load_config, save_results, load_results, get_project_root
from datasets.graph_qa_dataset import load_graphqa_dataset
from models import get_model

from experiments.experiment_01_visual_augmentation import run_experiment, sanity_check_model
from experiments.corruption import *

from evaluation.evaluation import *




def filter_clean_correct(dataset, baseline_results_path, task):
    """
    Keep only samples that the evaluation.py definition considers
    correct in the clean image+text baseline.
    """
    baseline = load_results(baseline_results_path)

    correct_ids = set()

    for r in baseline:
        model_answer = r.get("model_answer")
        expected_answer = r.get("expected_answer")

        is_correct, status, _ = compare(
            task,
            expected_answer,
            model_answer,
        )

        if is_correct:
            correct_ids.add(r["sample_id"])

    filtered = [
        s for s in dataset
        if s.sample_id in correct_ids
    ]

    log(
        f"Filtered to clean-correct samples: "
        f"{len(filtered)}/{len(dataset)} kept"
    )

    return filtered


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="experiments/configs/mixed_baseline.yaml")
    args = parser.parse_args()

    root = get_project_root()
    config = load_config(args.config)

    tasks = config.get("tasks")
    image_types = config.get("image_types")
    model_name = config.get("model")
    model_kwargs = config.get("model_kwargs", {}) or {}
    text_encoding = config.get("text_encoding")


    model_name_in_results = "Qwen2.5-VL-3B-Instruct"

    # ============================================================
    # Validate required paths BEFORE loading the model
    # ============================================================

    if not tasks:
        raise ValueError("No tasks specified in config.")

    if not image_types:
        raise ValueError("No image_types specified in config.")

    if not model_name:
        raise ValueError("No model specified in config.")

    data_dir = root / config["data_dir"]
    output_dir = root / Path("results/baseline")

    # Check original datasets
    for task in tasks:
        dataset_path = data_dir / f"{task}_{text_encoding}_test.jsonl"

        if not dataset_path.is_file():
            raise FileNotFoundError(
                f"Dataset for task '{task}' does not exist:\n"
                f"  {dataset_path}"
            )

    # Check baseline results for every task/image combination
    for task in tasks:
        for image_type in image_types:
            baseline_path = (
                output_dir
                / "text_and_image"
                / task
                / f"{image_type}_{model_name_in_results}.jsonl"
            )

            if not baseline_path.is_file():
                raise FileNotFoundError(
                    f"Baseline results not found for "
                    f"task='{task}', image_type='{image_type}':\n"
                    f"  {baseline_path}"
                )

    log("All required datasets and baseline result files exist.")
    log(f"Data directory: {data_dir}")
    log(f"Baseline directory: {output_dir / 'text_and_image'}")



    model = None

    first_it = True
# modality is simply text+image
    for task in tasks:
        for image_type in image_types:


            log(f"Running mixed-signals experiment for task: {task}, image_type: {image_type}")

            dataset_dir = root / config["data_dir"] / f"{task}_{text_encoding}_test.jsonl"
            dataset = load_graphqa_dataset(dataset_dir=dataset_dir)

            baseline_path = (
                Path("results/baseline") / "text_and_image" / task /
                (f"{image_type}_{model_name_in_results}.jsonl")
            )
            dataset = filter_clean_correct(
            dataset,
            baseline_path,
            task,
            )
            corrupted_dataset = corrupt_dataset(
            dataset,
            task=task,
            text_encoding=text_encoding,
            )

            corrupted_dataset_path = (
            root
            / config["data_dir"]
            / "mixed_signals"
            / f"{task}_{text_encoding}_{model_name_in_results}_test.jsonl"
            )

            save_corrupted_dataset(
                corrupted_dataset,
                corrupted_dataset_path,
            )

            if first_it:
                if model_name:
                    model_kwargs.setdefault("image_root", root / config["data_dir"])
                    model = get_model(model_name, **model_kwargs)
                    log(f"Loaded model: {model.name} ({model_name})")
                first_it = False



            if model is not None and len(corrupted_dataset) > 0:
                sanity_check_model(
                    model=model,
                    sample=corrupted_dataset[0],
                    mode="image_and_text",
                    image_type=image_type,
                )

            results = run_experiment(
                corrupted_dataset,
                config=config,
                model=model,
                mode="image_and_text",
                image_type=image_type,
                batch_size=config.get("batch_size", 8),
            )

            for r, s in zip(results, corrupted_dataset):
                r["original_answer"] = s.metadata["original_answer"]
                r["corrupted_answer"] = s.answer
                r["corruption"] = s.metadata["corruption"]



            output_path = Path(config["output_dir"]) / "text_and_image" / task / f"{image_type}_corrupt-text.jsonl"
            if model:
                output_path = Path(config["output_dir"]) / "text_and_image" / task / f"{image_type}_corrupt-text_{model.name}.jsonl"

            save_results(results, output_path)