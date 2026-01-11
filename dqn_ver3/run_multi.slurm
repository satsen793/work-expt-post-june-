#!/bin/bash
#SBATCH --job-name=dqn_multi
#SBATCH --output=logs/dqn_multi_%j.out
#SBATCH --error=logs/dqn_multi_%j.err
#SBATCH --time=24:00:00
#SBATCH --partition=cpu   # change to your GPU partition if needed
## #SBATCH --gres=gpu:1   # uncomment for GPU jobs
## #SBATCH --cpus-per-task=8
## #SBATCH --mem=16G

set -euo pipefail
mkdir -p logs

# Activate conda env
source "$HOME/miniconda3/etc/profile.d/conda.sh" || true
conda activate dqn_ver3

# Optional threading pins
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-1}

cd "$SLURM_SUBMIT_DIR"

# Parameters via env vars
SEEDS=${SEEDS:-0,1,2,3,4}
EPISODES=${EPISODES:-200}
STEPS=${STEPS:-140}
START_STEPS=${START_STEPS:-5000}
BUFFER_SIZE=${BUFFER_SIZE:-200000}
BATCH_SIZE=${BATCH_SIZE:-128}
OUT=${OUT:-logs/hpc_summary.json}
OUT_CSV=${OUT_CSV:-logs/hpc_summary.csv}

python hpc_run.py \
  --seeds "$SEEDS" \
  --episodes "$EPISODES" \
  --steps "$STEPS" \
  --start-steps "$START_STEPS" \
  --buffer-size "$BUFFER_SIZE" \
  --batch-size "$BATCH_SIZE" \
  --out "$OUT" \
  --out-csv "$OUT_CSV"
