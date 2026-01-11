#!/bin/bash
# Multi-seed DQN run using venv + pip environment
set -euo pipefail

source .venv/bin/activate
mkdir -p logs figures

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

echo "Done. Outputs in logs/ and figures/"
