#!/bin/bash
#SBATCH -J pets_5seed
#SBATCH -p gpu_h100_4
#SBATCH -N 1
#SBATCH --gpus=1
#SBATCH -t 24:00:00
#SBATCH -o logs/slurm_pets_%j.out
#SBATCH -e logs/slurm_pets_%j.err

set -euo pipefail

# Skip module commands - use conda directly
export PATH="$HOME/.conda/bin:$HOME/miniconda3/bin:$HOME/anaconda3/bin:$PATH"

# Initialize conda
if [ -f "$HOME/.conda/etc/profile.d/conda.sh" ]; then
    source "$HOME/.conda/etc/profile.d/conda.sh"
elif [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
elif [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/anaconda3/etc/profile.d/conda.sh"
elif command -v conda &> /dev/null; then
    eval "$(conda shell.bash hook)"
fi

# Create/activate conda env
ENV_NAME=pets_gpu
if ! conda env list | grep -q "^${ENV_NAME}\s"; then
  echo "Creating conda environment ${ENV_NAME}..."
  conda create -n ${ENV_NAME} python=3.11 -y
  conda activate ${ENV_NAME}
  pip install torch numpy
else
  conda activate ${ENV_NAME}
fi

# Sanity check
python - << 'PY'
import torch, numpy as np
print("torch:", torch.__version__, "cuda:", torch.cuda.is_available())
print("device_count:", torch.cuda.device_count())
PY

# Create output directories
mkdir -p logs results/pets

# Run PETS training (5 seeds, 295 episodes each)
echo "Starting PETS 5-seed production run..."
python pets_ver3/pets_train.py --episodes 295

echo "PETS training complete!"
