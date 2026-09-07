#!/bin/bash

# Finish Exp1+Exp2 for ONE encoding on all 7 GraphQA tasks.
# Already-finished tasks (node_count, cycle_check) are skipped.
# Does not overwrite adjacency or adjacency_matrix results.
#
# Submit one encoding at a time (about 10-16h of GPU):
#
#   sbatch --job-name=yovel_inc_all --export=ENCODING=incident \
#     slurm_files/yovel_encoding_full.slurm
#   sbatch --job-name=yovel_roster_all --export=ENCODING=node_roster \
#     slurm_files/yovel_encoding_full.slurm
#   sbatch --job-name=yovel_dimacs_all --export=ENCODING=dimacs \
#     slurm_files/yovel_encoding_full.slurm
#   sbatch --job-name=yovel_prose_all --export=ENCODING=incident_prose \
#     slurm_files/yovel_encoding_full.slurm
#
# If a job dies, resubmit the same line; Exp1/Exp2 skip existing JSONLs.

#SBATCH --account=gpu-students
#SBATCH --job-name=yovel_enc_all
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

ALL_TASKS=node_degree,node_count,edge_existence,edge_count,disconnected_nodes,triangle_counting,cycle_check

case "${ENCODING:-}" in
  incident) BASELINE_CFG=experiments/configs/baseline_7b_incident.yaml
            MIXED_CFG=experiments/configs/mixed_incident.yaml ;;
  node_roster) BASELINE_CFG=experiments/configs/baseline_7b_node_roster.yaml
               MIXED_CFG=experiments/configs/mixed_node_roster.yaml ;;
  dimacs) BASELINE_CFG=experiments/configs/baseline_7b_dimacs.yaml
          MIXED_CFG=experiments/configs/mixed_dimacs.yaml ;;
  incident_prose) BASELINE_CFG=experiments/configs/baseline_7b_incident_prose.yaml
                  MIXED_CFG=experiments/configs/mixed_incident_prose.yaml ;;
  *)
    echo "Set ENCODING to incident, node_roster, dimacs, or incident_prose"
    echo "Example: sbatch --export=ENCODING=incident slurm_files/yovel_encoding_full.slurm"
    exit 1
    ;;
esac

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

echo "Running full 7-task mixed-signals pipeline as $USER"
echo "ENCODING=$ENCODING"
echo "Python: $(which python) / $(python --version 2>&1)"

echo "=== STEP 0/3: re-encode $ENCODING JSONLs (all 7 tasks) ==="
python -m talk_like_a_graph.reencode_dataset \
    --data_dir data \
    --tasks "$ALL_TASKS" \
    --target_encodings "$ENCODING" \
    --source_encodings adjacency,adjacency_matrix

echo "=== STEP 1/3: Experiment 01 clean baseline ($ENCODING) ==="
python -m experiments.experiment_01_visual_augmentation \
    --config "$BASELINE_CFG"

echo "=== STEP 2/3: Experiment 02 mixed signals ($ENCODING) ==="
python -m experiments.experiment_02_mixed_signals \
    --config "$MIXED_CFG"

echo "=== DONE encoding=$ENCODING ==="
echo "Score a COPY of only spring_${ENCODING}_corrupt-text_ JSONLs, not mixed_baseline as a whole."
