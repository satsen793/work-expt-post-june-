#!/bin/bash
#SBATCH -J dqn_multiseed_gpu
#SBATCH -p gpu_h100_4
#SBATCH -N 1
#SBATCH --gpus=1
#SBATCH -t 24:00:00
#SBATCH -o logs/slurm_multiseed_%j.out
#SBATCH -e logs/slurm_multiseed_%j.err

# If your cluster requires gres with model name, uncomment and adjust:
# SBATCH --gres=gpu:h100:1

set -euo pipefail

module purge || true
# Optional: site-specific modules
# module load anaconda/2023.09 cuda/12.1

ENV_NAME=dqn_gpu
if ! conda env list | grep -q "^${ENV_NAME}\s"; then
  conda env create -f environment-gpu.yml
fi
source $(conda info --base)/etc/profile.d/conda.sh
conda activate ${ENV_NAME}

mkdir -p logs figures

# Seeds are space-separated; default 0..4
SEEDS_STR=${SEEDS:-"0 1 2 3 4"}
python scripts/run_multiseed.py \
  --seeds ${SEEDS_STR} \
  --episodes ${EPISODES:-100} \
  --steps ${STEPS:-140} \
  --start-steps ${START_STEPS:-5000} \
  --out-json logs/multiseed_summary_${SLURM_JOB_ID}.json \
  --out-csv logs/multiseed_episodes_${SLURM_JOB_ID}.csv \
  --fig-learning figures/learning_curve_moving_avg_reward_${SLURM_JOB_ID}.png \
  --fig-modality figures/post_content_gain_by_modality_${SLURM_JOB_ID}.png \
  --fig-variance figures/variance_across_seeds_${SLURM_JOB_ID}.png \
  --fig-compute figures/compute_vs_reward_${SLURM_JOB_ID}.png
