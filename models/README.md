# Model service (Task 2 — Model)

Turns one dataset sample into a VLM answer, for three input settings
(`text_only`, `image_only`, `image_and_text`). Plugs into the experiment loop
via a single contract: a model object with `.name` and
`.generate_batch(inputs)`.

```
Amit (data)  ->  THIS SERVICE (Yovel)  ->  Itamar (experiment loop)
```

## Files

| File | Purpose |
|------|---------|
| `base.py` | `VLMModel` interface + auto-detect setting from which fields are present |
| `prompts.py` | Prompt templates for the 3 settings |
| `stub_model.py` | Fake, dependency-free model — laptop testing / plumbing checks |
| `qwen_model.py` | Real Qwen2.5-VL wrapper (official `qwen-vl-utils` convention) |
| `__init__.py` | `get_model(name)` factory — the ">=2 models" hook |
| `poc.ipynb` | Proof-of-concept notebook (3 settings, easy/hard graphs) |
| `test_pipeline_laptop.py` | End-to-end smoke test with the stub (no GPU) |
| `requirements.txt` | Deps (stub needs almost none; Qwen needs the GPU stack) |

## Input / output contract

Each **input** dict (built by the experiment loop):

```python
{
  "text":      "<adjacency text>" | None,   # None when image_only
  "image":     "images/xxx.png"   | None,   # None when text_only
  "sample_id": "cycle_check/adjacency/test/0",
  "question":  "Q: Is there a cycle in this graph?\nA: ",
}
```

Each **output** dict:

```python
{
  "answer":     "Yes, there is a cycle.",
  "confidence": 0.87,        # mean per-token prob (None for the stub)
  "status":     "ok",
  "setting":    "image_and_text",
}
```

---

## Laptop use (no GPU)

```bash
# from repo root
python -m models.test_pipeline_laptop     # real data + real loop + stub model
```

`get_model("stub")` needs no heavy deps. Use it to verify plumbing before
spending GPU quota.

---

## Cluster use (GPU) — step by step

You have a folder + GPU access under `home/yandex/MLWG2026/`. Memory and
concurrent jobs are limited, so start tiny.

### 1. Get the code onto the cluster

```bash
cd ~/MLWG2026            # or your allocated folder
git clone https://github.com/Amitmadh/see-and-talk-like-a-graph.git
cd see-and-talk-like-a-graph
git checkout yovel/model-service     # this branch, until it's merged
```

### 2. Environment + install

```bash
python -m venv .venv && source .venv/bin/activate
# Install the torch build matching the cluster CUDA (check `nvidia-smi`).
# Then the VLM stack:
pip install -r models/requirements.txt
# If you hit `KeyError: 'qwen2_5_vl'`, install transformers from source:
pip install "git+https://github.com/huggingface/transformers"
```

### 3. Smoke-test Qwen on ~5 samples FIRST (fail early)

```bash
python -c "
from datasets.graph_qa_dataset import load_graphqa_dataset
from experiments.utils import get_project_root
from models import get_model

root = get_project_root()
ds = load_graphqa_dataset(root/'data'/'cycle_check_adjacency_test.jsonl')
m = get_model('qwen2.5-vl', image_root=str(root/'data'))   # 7B; use 'qwen-3b' if tight on memory

inputs = []
for s in ds.samples[:5]:
    inputs.append({'sample_id':s.sample_id,'question':s.question,
                   'text':s.text_encoding,'image':s.images['spring']})
for o, s in zip(m.generate_batch(inputs), ds.samples[:5]):
    print(o['answer'][:60], '| gold:', s.answer, '| conf:', o['confidence'])
"
```

Watch for: model loads without OOM, images resolve, answers look sane.
If this works, everything downstream works.

### 4. Run the PoC notebook

```bash
jupyter nbconvert --to notebook --execute --inplace models/poc.ipynb
# In the notebook, set MODEL_NAME = "qwen2.5-vl"  (or "qwen-3b")
```

Look for the "money case": text-only wrong, image+text right.

### 5. Run the real experiment (Itamar's loop)

Once the PoC looks good, the experiment loop calls the same service — just
pass a model object into `run_experiment(..., model=get_model("qwen2.5-vl"))`.

---

## Memory tips (limited quota)

- Prefer **Qwen2.5-VL-3B** (`get_model("qwen-3b")`) if the 7B OOMs.
- Keep `batch_size` small (2–4) to start; raise it once stable.
- `use_flash_attention=True` helps throughput on multi-image batches
  (needs `flash-attn` installed and a compatible GPU).
- Coordinate with teammates so you're not all grabbing the GPU at once.

## Supported models (`get_model` names)

| name | what |
|------|------|
| `"stub"` | fake, no deps (testing) |
| `"qwen2.5-vl"` / `"qwen-7b"` | Qwen2.5-VL-7B-Instruct |
| `"qwen-3b"` | Qwen2.5-VL-3B-Instruct (lighter) |

> A different-family model (e.g. LLaVA / InternVL / Llama-3.2-Vision) can be
> added behind the same factory — see the note in `__init__.py`.
