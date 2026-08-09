from pathlib import Path

EXP_MODES = {
    "text_only",
    "image_only",
    "image_and_text"
}

from experiments.utils import log
from models import get_model
from models.qwen_model import *

def run_experiment(
    samples,
    config=None,
    model=None,
    mode="image_and_text",
    image_type=None,
    batch_size=1,
):

    results = []

    log(
        f"Starting experiment: "
        f"{len(samples)} samples | "
        f"mode={mode} | "
        f"image={image_type}"
    )

    for i in range(0, len(samples), batch_size):

        batch = samples[i:i + batch_size]

        log(
            f"Processing batch {i}-{i+len(batch)-1}"
        )

        inputs = []

        for sample in batch:

            input_data = {
                "text": (
                    sample.text_encoding
                    if mode != "image_only"
                    else None
                ),

                "image": (
                    sample.images.get(image_type)
                    if mode != "text_only"
                    else None
                ),

                "sample_id": sample.sample_id,

                # the model needs the actual question to answer it
                "question": sample.question,
            }

            inputs.append(input_data)

            log(
                f"Prepared sample: {sample.sample_id}"
            )


        # No model yet
        if model is None:

            outputs = [
                {
                    "answer": None,
                    "status": "skipped_model",
                }
                for _ in inputs
            ]

        else:
            outputs = model.generate_batch(inputs)


        for sample, output in zip(batch, outputs):

            results.append({
                "sample_id": sample.sample_id,

                "question": sample.question,
                "expected_answer": sample.answer,

                "model_answer": output["answer"],

                "model": (
                    model.name
                    if model is not None
                    else None
                ),

                "image_type": image_type,

                "raw_output": output,

                # useful for debugging now
                "input_preview": {
                    "has_text": inputs[batch.index(sample)]["text"] is not None,
                    "has_image": inputs[batch.index(sample)]["image"] is not None,
                }
            })


    log(
        f"Finished experiment: "
        f"{len(samples)} samples | "
        f"mode={mode} | "
        f"image={image_type}"
    )

    return results


def sanity_check_model(model, sample, mode, image_type):
    """Run a single sample through the model and log the first answer."""
    input_data = {
        "text": sample.text_encoding if mode != "image_only" else None,
        "image": sample.images.get(image_type) if mode != "text_only" else None,
        "sample_id": sample.sample_id,
        "question": sample.question,
    }

    try:
        output = model.generate_batch([input_data])[0]
    except Exception as exc:
        log(f"Sanity check failed for sample {sample.sample_id}: {exc}")
        return

    answer = output.get("answer")
    status = output.get("status", "unknown")
    log(
        f"Sanity check sample={sample.sample_id} "
        f"mode={mode} image_type={image_type} "
        f"status={status} answer={answer!r}"
    )

    if status != "ok" or not answer:
        log("Sanity check warning: model returned an invalid or empty response.")


from experiments.utils import load_config, save_results
from datasets.graph_qa_dataset import load_graphqa_dataset

from experiments.utils import get_project_root

import argparse
import os
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--config", default="experiments/configs/baseline.yaml")
    log("Loading configuration from: " + parser.parse_args().config)

    args = parser.parse_args()


    root = get_project_root()
    config = load_config(args.config)

    modalities = config.get("modalities")
    tasks = config.get("tasks")
    image_types = config.get("image_types")
    model_name = config.get("model")
    model_kwargs = config.get("model_kwargs", {}) or {}
    text_encoding = config.get("text_encoding")

    model = None
    if model_name:
        model_kwargs.setdefault("image_root", root / config["data_dir"])
        model = get_model(model_name, **model_kwargs)
        log(f"Loaded model: {model.name} ({model_name})")
    else:
        log("No model configured; running with skipped_model outputs.")

    for modality in modalities:
        for task in tasks:
            for image_type in image_types:
        
                log(f"Running experiment for task: {task}, modality: {modality}, image_type: {image_type}")

                dataset_dir = (
                    root / 
                    config["data_dir"] /
                    f"{task}_{text_encoding}_test.jsonl"
                )
                dataset = load_graphqa_dataset(
                    dataset_dir=dataset_dir
                )

                if model is not None and len(dataset) > 0:
                    sanity_check_model(
                        model=model,
                        sample=dataset[0],
                        mode=modality,
                        image_type=image_type,
                    )

                results = run_experiment(
                    dataset,
                    config=config,
                    model=model,
                    mode=modality,
                    image_type=image_type,
                    batch_size=config.get("batch_size", 8),
                )

                output_path = Path(config["output_dir"]) / modality / task / f"{image_type}.jsonl"
                if model:
                    output_path = Path(config["output_dir"]) / modality / task / f"{image_type}_{model.name}.jsonl"

                save_results(
                    results,
                    output_path
                )
