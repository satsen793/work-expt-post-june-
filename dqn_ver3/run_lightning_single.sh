#!/bin/bash
# Quick single-seed DQN run on Lightning AI
set -euo pipefail

source $(conda info --base)/etc/profile.d/conda.sh
conda activate dqn_gpu

mkdir -p logs

python train_dqn.py \
  --seed ${SEED:-0} \
  --steps 140 \
  --episodes ${EPISODES:-100} \
  --start-steps 5000 \
  --out-csv logs/dqn_single.csv \
  --out-json logs/dqn_single.json \
  --out-steps-csv logs/dqn_steps.csv

echo "Training complete! Results in logs/"
