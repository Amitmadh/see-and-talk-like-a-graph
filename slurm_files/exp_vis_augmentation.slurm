#!/bin/bash

#SBATCH -A gpu-research
#SBATCH --job-name=llava_visual_augmentation_experiment01
#SBATCH --output=/home/yandex/MLWG2026/%u/slurm_logs/%x_%j.out
#SBATCH --error=/home/yandex/MLWG2026/%u/slurm_logs/%x_%j.err
#SBATCH --partition=killable
#SBATCH --time=24:00:00
#SBATCH --signal=USR1@120
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=50000
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1 # GPUs in total
#SBATCH --mail-user=itamarberger@mail.tau.ac.il
#SBATCH --mail-type=FAIL,END

cd /home/yandex/MLWG2026/$USER/see-and-talk-like-a-graph

umask 0002

# Initialize Conda for BASH
source /home/yandex/MLWG2026/itamarberger/miniconda3/etc/profile.d/conda.sh
conda activate see-talk

# Hugging Face cache on large filesystem
export HF_HOME=/home/yandex/MLWG2026/itamarberger/huggingface
export HF_HUB_CACHE=$HF_HOME/hub
export TRITON_CACHE_DIR=/home/yandex/MLWG2026/itamarberger/triton_cache
export HF_TOKEN="$HF_TOKEN"
export XDG_CACHE_HOME=/home/yandex/MLWG2026/$USER/cache
mkdir -p "$XDG_CACHE_HOME"

unset TRANSFORMERS_CACHE

mkdir -p "$TRITON_CACHE_DIR"

echo "Running experiment 01 of visual augmentation"

echo "Python:"
which python
python --version

echo "Conda environment:"
echo $CONDA_PREFIX

echo "Running command: python experiments/experiment_01_visual_augmentation.py"
echo "Environment: $CONDA_DEFAULT_ENV"
echo "HF_HOME: $HF_HOME"
echo "HF_HUB_CACHE: $HF_HUB_CACHE"
echo "TRITON_CACHE_DIR: $TRITON_CACHE_DIR"
echo "TRANSFORMERS_CACHE: ${TRANSFORMERS_CACHE:-unset}"
echo "HF_HUB_OFFLINE: ${HF_HUB_OFFLINE:-0}"
python -m experiments.experiment_01_visual_augmentation
