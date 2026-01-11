#!/bin/bash
#SBATCH -J dqn_single_gpu
#SBATCH -p gpu_h100_4
#SBATCH -N 1
#SBATCH --gpus=1
#SBATCH -t 24:00:00
#SBATCH -o logs/slurm_dqn_%j.out
#SBATCH -e logs/slurm_dqn_%j.err

# If your cluster requires gres with model name, uncomment and adjust:
# SBATCH --gres=gpu:h100:1

set -euo pipefail

module purge || true
# Optional: load a CUDA-friendly Python/module stack if required by your site
# module load anaconda/2023.09 cuda/12.1

# Create/activate conda env (idempotent)
ENV_NAME=dqn_gpu
if ! conda env list | grep -q "^${ENV_NAME}\s"; then
  conda env create -f environment-gpu.yml
fi
source $(conda info --base)/etc/profile.d/conda.sh
conda activate ${ENV_NAME}

# Sanity: show versions and CUDA
python - << 'PY'
import torch, numpy as np
print("torch:", torch.__version__, "cuda:", torch.version.cuda, "is_available:", torch.cuda.is_available())
print("device_count:", torch.cuda.device_count())
PY

# Run single-seed training on GPU
mkdir -p logs
python train_dqn.py --seed ${SEED:-0} --steps ${STEPS:-140} --episodes ${EPISODES:-100} \
  --start-steps ${START_STEPS:-5000} \
  --out-csv logs/dqn_single_${SLURM_JOB_ID}.csv \
  --out-json logs/dqn_single_${SLURM_JOB_ID}.json \
  --out-steps-csv logs/dqn_steps_${SLURM_JOB_ID}.csv
