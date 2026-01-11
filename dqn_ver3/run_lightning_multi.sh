#!/bin/bash
# Multi-seed DQN run with all figures on Lightning AI
set -euo pipefail

source $(conda info --base)/etc/profile.d/conda.sh
conda activate dqn_gpu

mkdir -p logs figures

echo "Starting multi-seed training..."
time python scripts/run_multiseed.py \
  --seeds 0 1 2 3 4 \
  --episodes 100 \
  --steps 140 \
  --start-steps 5000 \
  --out-json logs/multiseed_summary.json \
  --out-csv logs/multiseed_episodes.csv \
  --fig-learning figures/learning_curve_moving_avg_reward.png \
  --fig-modality figures/post_content_gain_by_modality.png \
  --fig-variance figures/variance_across_seeds.png \
  --fig-compute figures/compute_vs_reward.png

echo ""
echo "All runs complete!"
echo "Results:"
echo "  - Summary: logs/multiseed_summary.json"
echo "  - Episodes CSV: logs/multiseed_episodes.csv"
echo "  - Figures: figures/*.png"
