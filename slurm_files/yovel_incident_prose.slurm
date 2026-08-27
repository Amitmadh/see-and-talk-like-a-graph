#!/bin/bash

# Released Talk-Like-A-Graph incident encoder (English connected-to prose).
# Tagged incident_prose so it does not overwrite our neighbourhood-list
# incident results. Same 350 graphs, node_count + cycle_check, Qwen 7B.
#
# Submit with:   sbatch slurm_files/yovel_incident_prose.slurm

#SBATCH --account=gpu-students
#SBATCH --job-name=yovel_inc_prose
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

echo "Running incident_prose mixed-signals pipeline as $USER"
echo "Python: $(which python) / $(python --version 2>&1)"

echo "=== STEP 0/3: re-encode incident_prose JSONLs ==="
python -m talk_like_a_graph.reencode_dataset \
    --data_dir data \
    --tasks node_count,cycle_check \
    --target_encodings incident_prose \
    --source_encodings adjacency,adjacency_matrix

echo "=== STEP 1/3: Experiment 01 clean baseline (incident_prose) ==="
python -m experiments.experiment_01_visual_augmentation \
    --config experiments/configs/baseline_7b_incident_prose.yaml

echo "=== STEP 2/3: Experiment 02 mixed signals (incident_prose) ==="
python -m experiments.experiment_02_mixed_signals \
    --config experiments/configs/mixed_incident_prose.yaml

echo "=== DONE ==="
echo "Score a COPY of only these files:"
echo "  results/mixed_baseline/text_and_image/{node_count,cycle_check}/spring_incident_prose_corrupt-text_Qwen2.5-VL-7B-Instruct.jsonl"
