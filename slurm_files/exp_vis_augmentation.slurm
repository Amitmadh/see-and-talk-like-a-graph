#!/bin/bash

#SBATCH --job-name=visual_augmentation_experiment01
#SBATCH --output=/home/yandex/MLWG2026/%u/slurm_logs/%x_%j.out
#SBATCH --error=/home/yandex/MLWG2026/%u/slurm_logs/%x_%j.err
#SBATCH --partition=studentkillable
#SBATCH --time=24:00:00
#SBATCH --signal=USR1@120
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=50000
#SBATCH --cpus-per-task=4
#SBATCH --mail-user=itamarberger@mail.tau.ac.il
#SBATCH --mail-type=FAIL,END

# Set working directory according to the user currently running the job
cd /home/yandex/MLWG2026/$USER/see-and-talk-like-a-graph

# Make sure that files that are created by this script are accessible by everyone
umask 0002

# activate conda environment
source /home/yandex/MLWG2026/itamarberger/miniconda3/etc/profile.d/conda.csh

conda activate see-talk

echo "Running experiment 01 of visual augmentation"

SCRIPT_CMD="python experiments/experiment_01_visual_augmentation.py"

echo "Running command : $SCRIPT_CMD"

bash $SCRIPT_CMD