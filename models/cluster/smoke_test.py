"""Minimal GPU smoke test: run a VLM on the first N samples of one task, print
the answer vs. gold vs. confidence, for all three settings.

This is the "fail early" check before any long experiment. Run via the SLURM
script (models/cluster/smoke_test.slurm) or directly inside an interactive
srun session:

    python models/cluster/smoke_test.py --model qwen2.5-vl --task cycle_check --n 5
"""

from __future__ import annotations

import argparse
import time

from datasets.graph_qa_dataset import load_graphqa_dataset
from experiments.utils import get_project_root
from models import get_model


SETTINGS = ["text_only", "image_only", "image_and_text"]


def make_input(sample, mode, image_type):
    return {
        "sample_id": sample.sample_id,
        "question": sample.question,
        "text": sample.text_encoding if mode != "image_only" else None,
        "image": sample.images.get(image_type) if mode != "text_only" else None,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen2.5-vl")   # or qwen-3b / llava
    ap.add_argument("--task", default="cycle_check")
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--image-type", default="spring")
    args = ap.parse_args()

    root = get_project_root()
    data_dir = root / "data"
    ds = load_graphqa_dataset(data_dir / f"{args.task}_adjacency_test.jsonl")
    samples = ds.samples[: args.n]

    print(f"Loading model {args.model!r} ...")
    t0 = time.time()
    model = get_model(args.model, image_root=str(data_dir))
    print(f"Model loaded in {time.time() - t0:.1f}s  (name={model.name})")

    for mode in SETTINGS:
        print(f"\n===== {mode} =====")
        inputs = [make_input(s, mode, args.image_type) for s in samples]
        t0 = time.time()
        outs = model.generate_batch(inputs)
        dt = time.time() - t0
        for s, o in zip(samples, outs):
            ans = (o.get("answer") or "").replace("\n", " ")[:70]
            conf = o.get("confidence")
            conf_s = f"{conf:.3f}" if isinstance(conf, float) else "n/a"
            print(f"  [{s.metadata.get('nnodes')}n] {ans!r:<74} "
                  f"| gold={s.answer!r} | conf={conf_s}")
        print(f"  ({len(inputs)} samples in {dt:.1f}s)")

    print("\nSmoke test done. If answers look sane and no OOM, you're good to scale up.")


if __name__ == "__main__":
    main()
