#!/bin/bash

# All four new encodings (not matrix, not adjacency) on all 7 GraphQA tasks.
# Exp1/Exp2 skip a task if its result JSONL already exists, so node_count
# and cycle_check (and any encoding already finished) are not redone.
#
# Wall time is 24h; four encodings x five remaining tasks is likely more
# than that. If the job is killed, resubmit this same script.
#
# Do NOT submit while another encoding job is writing the same files.
#
# Submit with:   sbatch slurm_files/yovel_encodings_remaining.slurm

#SBATCH --account=gpu-students
#SBATCH --job-name=yovel_enc_rest
#SBATCH --output=/home/yandex/MLWG2026/%u/slurm_logs/%x_%j.out
#SBATCH --error=/home/yandex/MLWG2026/%u/slurm_logs/%x_%j.err
#SBATCH --partition=studentkillable
#SBATCH --time=24:00:00
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

echo "Running remaining tasks for all four encodings as $USER"
echo "Python: $(which python) / $(python --version 2>&1)"

for ENCODING in incident node_roster dimacs incident_prose; do
  case "$ENCODING" in
    incident)
      BASELINE_CFG=experiments/configs/baseline_7b_incident.yaml
      MIXED_CFG=experiments/configs/mixed_incident.yaml
      ;;
    node_roster)
      BASELINE_CFG=experiments/configs/baseline_7b_node_roster.yaml
      MIXED_CFG=experiments/configs/mixed_node_roster.yaml
      ;;
    dimacs)
      BASELINE_CFG=experiments/configs/baseline_7b_dimacs.yaml
      MIXED_CFG=experiments/configs/mixed_dimacs.yaml
      ;;
    incident_prose)
      BASELINE_CFG=experiments/configs/baseline_7b_incident_prose.yaml
      MIXED_CFG=experiments/configs/mixed_incident_prose.yaml
      ;;
  esac

  echo "========== ENCODING=$ENCODING =========="

  echo "=== re-encode $ENCODING (all 7 tasks, skip existing JSONLs) ==="
  python -m talk_like_a_graph.reencode_dataset \
      --data_dir data \
      --tasks "$ALL_TASKS" \
      --target_encodings "$ENCODING" \
      --source_encodings adjacency,adjacency_matrix

  echo "=== Exp1 $ENCODING ==="
  python -m experiments.experiment_01_visual_augmentation \
      --config "$BASELINE_CFG"

  echo "=== Exp2 $ENCODING ==="
  python -m experiments.experiment_02_mixed_signals \
      --config "$MIXED_CFG"

  echo "=== finished encoding=$ENCODING ==="
done

echo "=== DONE all four encodings ==="
echo "Score COPY-ONLY folders per encoding, not results/mixed_baseline as a whole."
