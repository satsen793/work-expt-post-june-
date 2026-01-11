#!/bin/bash
#SBATCH -J pets_smoke
#SBATCH -p gpu_h100_4
#SBATCH -N 1
#SBATCH --gpus=1
#SBATCH -t 01:00:00
#SBATCH -o logs/slurm_pets_smoke_%j.out
#SBATCH -e logs/slurm_pets_smoke_%j.err

module load python-3.10.8-gcc-11.2.0-dlcmq7k

mkdir -p logs results/pets

echo "=== PETS Smoke Test: 1 seed, 10 episodes ==="
python pets_ver3/pets_train.py --episodes 10 --seed 0

echo ""
echo "=== Verifying results ==="
ls -la results/pets/
head -5 results/pets/episodes.csv 2>/dev/null || echo "ERROR: episodes.csv not found!"
cat results/pets/summary.json 2>/dev/null || echo "ERROR: summary.json not found!"

echo ""
echo "PETS smoke test complete!"
