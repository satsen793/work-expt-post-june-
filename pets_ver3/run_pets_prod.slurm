#!/bin/bash
#SBATCH -J pets_prod
#SBATCH -p gpu_h100_4
#SBATCH -N 1
#SBATCH --gpus=1
#SBATCH -t 24:00:00
#SBATCH -o logs/slurm_pets_prod_%j.out
#SBATCH -e logs/slurm_pets_prod_%j.err

module load python-3.10.8-gcc-11.2.0-dlcmq7k

mkdir -p logs results/pets

echo "=== PETS Fast Prod: 1 seed (0), 10 episodes ==="
python pets_ver3/pets_train.py --seed 0 --episodes 10

echo ""
echo "=== Verifying results ==="
ls -la results/pets/
wc -l results/pets/episodes.csv

echo "PETS production complete!"
