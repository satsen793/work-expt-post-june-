#!/bin/bash
#SBATCH -J mbpo_smoke
#SBATCH -p gpu_h100_4
#SBATCH -N 1
#SBATCH --gpus=1
#SBATCH -t 01:00:00
#SBATCH -o logs/slurm_mbpo_smoke_%j.out
#SBATCH -e logs/slurm_mbpo_smoke_%j.err

module load python-3.10.8-gcc-11.2.0-dlcmq7k

mkdir -p logs results/mbpo

echo "=== MBPO Smoke Test: 1 seed, 10 episodes ==="
python mbpo_ver3/train_mbpo.py --episodes 10 --seeds 1

echo ""
echo "=== Verifying results ==="
ls -la results/mbpo/
head -5 results/mbpo/episodes.csv 2>/dev/null || echo "ERROR: episodes.csv not found!"
cat results/mbpo/summary.json 2>/dev/null || echo "ERROR: summary.json not found!"

echo ""
echo "MBPO smoke test complete!"
