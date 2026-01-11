#!/bin/bash
#SBATCH --job-name=dqn_ver3
#SBATCH --output=logs/dqn_%j.out
#SBATCH --error=logs/dqn_%j.err
#SBATCH --time=24:00:00
#SBATCH --partition=cpu   # change to your GPU partition if needed
## #SBATCH --gres=gpu:1   # uncomment for GPU jobs
## #SBATCH --cpus-per-task=8
## #SBATCH --mem=16G

set -euo pipefail

# Load modules if your HPC uses them (example; adjust to your cluster)
# module purge
# module load miniconda/3

# Create logs dir if missing
mkdir -p logs

# Activate conda env (assumes you created it under $HOME/.conda/envs/dqn_ver3)
source "$HOME/miniconda3/etc/profile.d/conda.sh" || true
conda activate dqn_ver3

# Optionally pin thread counts for reproducibility/throughput
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-1}
export NUMEXPR_MAX_THREADS=${NUMEXPR_MAX_THREADS:-1}

# Move to project directory if not already there
cd "$SLURM_SUBMIT_DIR"

# Training command
SEED=${SEED:-0}
STEPS=${STEPS:-200}
EPISODES=${EPISODES:-500}
START_STEPS=${START_STEPS:-5000}

python train_dqn.py \
  --seed "$SEED" \
  --steps "$STEPS" \
  --episodes "$EPISODES" \
  --start-steps "$START_STEPS"
