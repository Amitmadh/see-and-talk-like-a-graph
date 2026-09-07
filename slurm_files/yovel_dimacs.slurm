#!/bin/bash

# DIMACS mixed-signals follow-up.
# Same 350 graphs/images, one encoding (`p edge n m` + edge lines),
# two tasks (node_count = tilt, cycle_check = control), Qwen 7B only.
#
# Does NOT overwrite adjacency, matrix, incident, or node_roster results.
# Exp1/Exp2 skip a task if its result JSONL already exists.
#
# Submit with:   sbatch slurm_files/yovel_dimacs.slurm

#SBATCH --account=gpu-students
#SBATCH --job-name=yovel_dimacs
#SBATCH --output=/home/yandex/MLWG2026/%u/slurm_logs/%x_%j.out
#SBATCH --error=/home/yandex/MLWG2026/%u/slurm_logs/%x_%j.err
#SBATCH --partition=studentkillable
#SBATCH --time=10:00:00
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

echo "Running DIMACS mixed-signals pipeline as $USER"
echo "Python: $(which python) / $(python --version 2>&1)"
echo "Conda env: $CONDA_DEFAULT_ENV ($CONDA_PREFIX)"
echo "HF_HOME: $HF_HOME"

echo "=== STEP 0/3: re-encode dimacs JSONLs ==="
python -m talk_like_a_graph.reencode_dataset \
    --data_dir data \
    --tasks node_count,cycle_check \
    --target_encodings dimacs \
    --source_encodings adjacency,adjacency_matrix

echo "=== STEP 1/3: Experiment 01 clean baseline (dimacs) ==="
python -m experiments.experiment_01_visual_augmentation \
    --config experiments/configs/baseline_7b_dimacs.yaml

echo "=== STEP 2/3: Experiment 02 mixed signals (dimacs) ==="
python -m experiments.experiment_02_mixed_signals \
    --config experiments/configs/mixed_dimacs.yaml

echo "=== DONE ==="
echo "Score a COPY of only these files, do not mix other encodings:"
echo "  results/mixed_baseline/text_and_image/{node_count,cycle_check}/spring_dimacs_corrupt-text_Qwen2.5-VL-7B-Instruct.jsonl"
