"""Laptop end-to-end smoke test (NO GPU).

Runs the REAL dataset loader + the REAL experiment loop + the stub model on a
few real samples, across all three settings. Proves your service plugs into
Itamar's pipeline before you ever touch the cluster.

Run from the repo root:
    python -m models.test_pipeline_laptop
"""

from __future__ import annotations

from datasets.graph_qa_dataset import load_graphqa_dataset
from experiments.experiment_01_visual_augmentation import run_experiment
from experiments.utils import get_project_root
from models import get_model


def main():
    root = get_project_root()
    dataset_path = root / "data" / "cycle_check_adjacency_test.jsonl"

    dataset = load_graphqa_dataset(dataset_dir=dataset_path)
    # Just the first few samples — this is a smoke test.
    samples = dataset.samples[:3]

    model = get_model("stub", canned_answer="Yes, there is a cycle.")

    for mode in ("text_only", "image_only", "image_and_text"):
        print(f"\n########## MODE: {mode} ##########")
        results = run_experiment(
            samples,
            config=None,
            model=model,
            mode=mode,
            image_type="spring",
            batch_size=2,
        )
        for r in results:
            print(
                f"  {r['sample_id']}: "
                f"answer={r['model_answer']!r} | "
                f"expected={r['expected_answer']!r} | "
                f"has_text={r['input_preview']['has_text']} "
                f"has_image={r['input_preview']['has_image']}"
            )

    print("\nOK: real dataset + real experiment loop + stub model all connect.")


if __name__ == "__main__":
    main()
