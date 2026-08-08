# Running on the TAU CS SLURM cluster

The course cluster uses **SLURM** (guide: https://www.cs.tau.ac.il/system/slurm).
You do **not** run Python directly — you submit a *job* that requests a GPU, and
SLURM runs it when a slot frees up. Your folder: `home/yandex/MLWG2026/`.

Student limits (from the TAU guide): **1 GPU per job**, `studentbatch` allows
**max 6 batch jobs**, memory is limited — request realistically or jobs get
OOM-killed.

---

## 0. Log in

```bash
ssh <username>@slurm-client.cs.tau.ac.il      # VPN required if off-campus
```

## 1. Get the code

```bash
cd ~/yandex/MLWG2026        # your allocated dir
git clone https://github.com/Amitmadh/see-and-talk-like-a-graph.git
cd see-and-talk-like-a-graph
git checkout yovel/model-service
```

## 2. Conda env  (there is NO `module load` here)

The TAU guide warns: **do not install Anaconda in HOME** (quota). Put miniconda
in your project/scratch area, then:

```bash
conda create -y -n satlag python=3.10
conda activate satlag
# Install the torch build matching the cluster GPUs (check `nvidia-smi`).
pip install -r models/requirements.txt
# If you see  KeyError: 'qwen2_5_vl' :
pip install "git+https://github.com/huggingface/transformers"
```

## 3. Model cache — keep it OFF your home quota

Qwen2.5-VL-7B is ~16 GB. Point Hugging Face's cache at a roomy dir:

```bash
export HF_HOME=~/yandex/MLWG2026/hf_cache      # add to your ~/.bashrc too
```

> **Internet caveat:** if compute nodes can't reach huggingface.co, download the
> model once on the *login* node (it caches under `$HF_HOME`), then jobs read
> from cache. Test with a tiny download first.

## 4. Fail-early smoke test (do this FIRST)

**Interactive** (watch it live, best for first try / debugging):

```bash
srun -p studentrun --gpus=1 --mem=50000 --time=30 --pty bash
conda activate satlag
export HF_HOME=~/yandex/MLWG2026/hf_cache
python models/cluster/smoke_test.py --model qwen2.5-vl --task cycle_check --n 5
```

**Or batch** (submit and walk away):

```bash
sbatch models/cluster/smoke_test.slurm
squeue --me                 # watch status
# output lands in satlag-smoke-<jobid>.out
```

Check: model loads, no OOM, images resolve, answers look sane. If tight on
memory, switch `--model qwen2.5-vl` → `--model qwen-3b`.

## 5. Try the other models

```bash
python models/cluster/smoke_test.py --model qwen-3b --task cycle_check --n 5
python models/cluster/smoke_test.py --model llava    --task cycle_check --n 5
```

## 6. PoC notebook

```bash
jupyter nbconvert --to notebook --execute --inplace models/poc.ipynb
# set MODEL_NAME = "qwen2.5-vl" inside the notebook first
```

## 7. Full experiment

Hand a real model into Itamar's loop (`run_experiment(..., model=get_model("qwen2.5-vl"))`)
and submit it on `studentbatch` (3-day limit, max 6 jobs). Start small, watch
runtime, then scale.

---

## Handy SLURM commands

| command | what |
|---|---|
| `sbatch script.slurm` | submit a batch job |
| `squeue --me` | your jobs |
| `scancel <jobid>` | cancel a job |
| `sinfo` | partitions / node states |
| `sacct -l -j <jobid>` | job accounting (memory used, etc.) |

## Partitions (student-relevant)

| partition | limit | use for |
|---|---|---|
| `studentrun` | 3 hours, interactive | smoke tests, the PoC |
| `studentbatch` | 3 days, max 6 jobs | the full experiment |
