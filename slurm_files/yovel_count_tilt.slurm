#!/bin/bash

# Count-tilt mixed-signals follow-up.
# Same 350 graphs/images as the existing JSONLs, two new text encodings
# (incident neighbourhood lists + vertical node roster), two tasks
# (node_count = tilt, cycle_check = image-heavy control), Qwen 7B only.
#
# Does NOT overwrite adjacency or adjacency_matrix results.
# Exp1/Exp2 skip a task if its result JSONL already exists.
#
# Submit with:   sbatch slurm_files/yovel_count_tilt.slurm

#SBATCH --account=gpu-students
#SBATCH --job-name=yovel_count_tilt
#SBATCH --output=/home/yandex/MLWG2026/%u/slurm_logs/%x_%j.out
#SBATCH --error=/home/yandex/MLWG2026/%u/slurm_logs/%x_%j.err
#SBATCH --partition=studentkillable
#SBATCH --time=16:00:00
#SBATCH --signal=USR1@120
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=50000
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --mail-user=yovelhatan@mail.tau.ac.il
#SBATCH --mail-type=FAIL,END

set -e

cd /home/yandex/MLWG2026/$USER/see-and-talk-like-a-graph

umask 0002

source /home/yandex/MLWG2026/itamarberger/miniconda3/etc/profile.d/conda.sh
conda activate see-talk

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HOME=/home/yandex/MLWG2026/itamarberger/huggingface
export HF_HUB_CACHE=$HF_HOME/hub
export TRITON_CACHE_DIR=/home/yandex/MLWG2026/$USER/triton_cache
export XDG_CACHE_HOME=/home/yandex/MLWG2026/$USER/cache
mkdir -p "$XDG_CACHE_HOME" "$TRITON_CACHE_DIR" \
         /home/yandex/MLWG2026/$USER/slurm_logs

unset TRANSFORMERS_CACHE

echo "Running count-tilt mixed-signals pipeline as $USER"
echo "Python: $(which python) / $(python --version 2>&1)"
echo "Conda env: $CONDA_DEFAULT_ENV ($CONDA_PREFIX)"
echo "HF_HOME: $HF_HOME"

# --- Step 0: re-encode text on the existing graphs (no new images) ---
echo "=== STEP 0/5: re-encode incident + node_roster JSONLs ==="
python -m talk_like_a_graph.reencode_dataset \
    --data_dir data \
    --tasks node_count,cycle_check \
    --target_encodings incident,node_roster \
    --source_encodings adjacency,adjacency_matrix

# --- incident ---
echo "=== STEP 1/5: Experiment 01 clean baseline (incident) ==="
python -m experiments.experiment_01_visual_augmentation \
    --config experiments/configs/baseline_7b_incident.yaml

echo "=== STEP 2/5: Experiment 02 mixed signals (incident) ==="
python -m experiments.experiment_02_mixed_signals \
    --config experiments/configs/mixed_incident.yaml

# --- node_roster ---
echo "=== STEP 3/5: Experiment 01 clean baseline (node_roster) ==="
python -m experiments.experiment_01_visual_augmentation \
    --config experiments/configs/baseline_7b_node_roster.yaml

echo "=== STEP 4/5: Experiment 02 mixed signals (node_roster) ==="
python -m experiments.experiment_02_mixed_signals \
    --config experiments/configs/mixed_node_roster.yaml

echo "=== DONE ==="
echo "Score a COPY of only these files, do not mix Itamar's adjacency JSONLs:"
echo "  results/mixed_baseline/text_and_image/{node_count,cycle_check}/spring_{incident,node_roster}_corrupt-text_Qwen2.5-VL-7B-Instruct.jsonl"
