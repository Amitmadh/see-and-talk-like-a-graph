#! /bin/sh
# Smoke-test Qwen2.5-VL on ~5 samples. Submit with:  sbatch models/cluster/smoke_test.slurm
# Uses the SHORT student partition so it schedules fast. This is the "fail early" run.

#SBATCH --job-name=satlag-smoke
#SBATCH --output=%x-%j.out          # %x=jobname %j=jobid, written to CWD
#SBATCH --error=%x-%j.err
#SBATCH --partition=studentrun      # 3h, for interactive/testing; use studentbatch for long runs
#SBATCH --time=30                    # minutes
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=50000                  # MB (~50GB); 7B model + activations need headroom
#SBATCH --gpus=1                     # students: max 1 GPU per job

# --- activate your conda env (created once, see cluster/SETUP.md) ---
source ~/miniconda3/etc/profile.d/conda.sh   # adjust path to where you installed conda
conda activate satlag

# --- keep HF model cache OFF your limited home quota ---
# Point this at a roomy location you control (scratch / project dir).
export HF_HOME="${HF_HOME:-$SLURM_SUBMIT_DIR/../hf_cache}"

cd "$SLURM_SUBMIT_DIR"
echo "Running smoke test from: $(pwd)  | GPU: $CUDA_VISIBLE_DEVICES"
nvidia-smi

python models/cluster/smoke_test.py --model qwen2.5-vl --task cycle_check --n 5
