#!/bin/bash
#SBATCH -J dqn_prod
#SBATCH -p gpu_h100_4
#SBATCH -N 1
#SBATCH --gpus=1
#SBATCH -t 02:00:00
#SBATCH -o logs/slurm_dqn_prod_%j.out
#SBATCH -e logs/slurm_dqn_prod_%j.err

module load python-3.10.8-gcc-11.2.0-dlcmq7k

mkdir -p logs results/dqn/figures

echo "=== DQN Production: 5 seeds, 295 episodes ==="
cd dqn_ver3
python scripts/run_multiseed.py --episodes 295

echo ""
echo "=== Verifying results ==="
ls -la ../results/dqn/
wc -l ../results/dqn/episodes.csv
cat ../results/dqn/summary.json | head -30

echo "DQN production complete!"
