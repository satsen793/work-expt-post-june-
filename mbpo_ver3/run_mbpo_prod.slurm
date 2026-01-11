#!/bin/bash
#SBATCH -J mbpo_prod
#SBATCH -p gpu_h100_4
#SBATCH -N 1
#SBATCH --gpus=1
#SBATCH -t 24:00:00
#SBATCH -o logs/slurm_mbpo_prod_%j.out
#SBATCH -e logs/slurm_mbpo_prod_%j.err

module load python-3.10.8-gcc-11.2.0-dlcmq7k

mkdir -p logs results/mbpo

echo "=== MBPO Quick Test: 5 seeds, 10 episodes ==="
python mbpo_ver3/train_mbpo.py --seeds 5 --episodes 10

echo ""
echo "=== Verifying results ==="
ls -la results/mbpo/
wc -l results/mbpo/episodes.csv

echo "MBPO production complete!"
