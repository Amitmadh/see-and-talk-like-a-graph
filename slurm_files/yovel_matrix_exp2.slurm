#!/bin/bash

# Resume Experiment 02 only (adjacency_matrix mixed signals).
# Use this after Exp1 and the finished Exp2 tasks are already on disk.
# experiment_02 skips tasks whose result JSONL already exists.
#
# Submit with:
#   sbatch --account=gpu-students --partition=studentkillable \
#     slurm_files/yovel_matrix_exp2.slurm

#SBATCH --account=gpu-students
#SBATCH --job-name=yovel_matrix_exp2
#SBATCH --output=/home/yandex/MLWG2026/%u/slurm_logs/%x_%j.out
#SBATCH --error=/home/yandex/MLWG2026/%u/slurm_logs/%x_%j.err
#SBATCH --partition=studentkillable
#SBATCH --time=08:00:00
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

echo "Running matrix Exp2 resume as $USER"
echo "Python: $(which python) / $(python --version 2>&1)"

echo "=== Experiment 02 mixed signals (adjacency_matrix), skip existing tasks ==="
python -m experiments.experiment_02_mixed_signals \
    --config experiments/configs/mixed_adjacency_matrix.yaml

echo "=== DONE. Score with: python -m evaluation.mixed_evaluation --results-root results/mixed_baseline ==="
