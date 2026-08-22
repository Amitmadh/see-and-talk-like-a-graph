#!/bin/bash

# Yovel's Mixed-Signals matrix run — chains BOTH required steps in one job:
#   1) Experiment 01 clean baseline with the adjacency_matrix encoding
#      (mixed signals only looks at samples the model got right here, so
#       this MUST finish first)
#   2) Experiment 02 mixed signals with the adjacency_matrix encoding
#
# Based on yovel_exp.slurm (already pointed at your account).
#
# Submit with:   sbatch slurm_files/yovel_matrix.slurm

#SBATCH --job-name=yovel_matrix_mixed_signals
#SBATCH --output=/home/yandex/MLWG2026/%u/slurm_logs/%x_%j.out
#SBATCH --error=/home/yandex/MLWG2026/%u/slurm_logs/%x_%j.err
#SBATCH --partition=studentbatch
#SBATCH --time=24:00:00
#SBATCH --signal=USR1@120
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=50000
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --mail-user=yovelhatan@mail.tau.ac.il
#SBATCH --mail-type=FAIL,END

# Stop the whole job if step 1 fails — never run mixed signals on a
# missing/partial baseline.
set -e

cd /home/yandex/MLWG2026/$USER/see-and-talk-like-a-graph

umask 0002

# --- Conda: point at YOUR miniconda (installed under MLWG2026/$USER, NOT $HOME) ---
source /home/yandex/MLWG2026/$USER/miniconda3/etc/profile.d/conda.sh
conda activate see-talk

# --- Hugging Face cache on the large filesystem (NOT $HOME - quota) ---
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HOME=/home/yandex/MLWG2026/$USER/huggingface
export HF_HUB_CACHE=$HF_HOME/hub
export TRITON_CACHE_DIR=/home/yandex/MLWG2026/$USER/triton_cache
export XDG_CACHE_HOME=/home/yandex/MLWG2026/$USER/cache
mkdir -p "$XDG_CACHE_HOME" "$TRITON_CACHE_DIR" \
         /home/yandex/MLWG2026/$USER/slurm_logs

unset TRANSFORMERS_CACHE

echo "Running matrix mixed-signals pipeline as $USER"
echo "Python: $(which python) / $(python --version 2>&1)"
echo "Conda env: $CONDA_DEFAULT_ENV ($CONDA_PREFIX)"
echo "HF_HOME: $HF_HOME"

# --- Step 1: clean baseline (adjacency_matrix) ---
echo "=== STEP 1/2: Experiment 01 clean baseline (adjacency_matrix) ==="
python -m experiments.experiment_01_visual_augmentation \
    --config experiments/configs/baseline_7b_matrix.yaml

# --- Step 2: mixed signals (adjacency_matrix) ---
echo "=== STEP 2/2: Experiment 02 mixed signals (adjacency_matrix) ==="
python -m experiments.experiment_02_mixed_signals \
    --config experiments/configs/mixed_adjacency_matrix.yaml

echo "=== DONE. Score with: python -m evaluation.mixed_evaluation --results-root results/mixed_baseline ==="
