#!/bin/bash
#SBATCH -J pets_prod
#SBATCH -p gpu_h100_4
#SBATCH -N 1
#SBATCH --gpus=1
#SBATCH -t 24:00:00
#SBATCH -o logs/slurm_pets_prod_%j.out
#SBATCH -e logs/slurm_pets_prod_%j.err

module load python-3.10.8-gcc-11.2.0-dlcmq7k

# Configuration variables (change these for different runs)
SEEDS="0"
EPISODES="295"
OUTPUT_DIR="results/pets_seeds_${SEEDS}_ep${EPISODES}"

mkdir -p logs $OUTPUT_DIR

# Ensure we're in the project root
cd ..

echo "=== PETS Fast Prod: ${SEEDS} seed(s), ${EPISODES} episodes ==="
python pets_ver3/pets_train.py --seed $SEEDS --episodes $EPISODES

echo ""
echo "=== Verifying results ==="
ls -la $OUTPUT_DIR/
if [ -f $OUTPUT_DIR/episodes.csv ]; then
    wc -l $OUTPUT_DIR/episodes.csv
    head -5 $OUTPUT_DIR/episodes.csv
else
    echo "ERROR: episodes.csv not found!"
fi
cat $OUTPUT_DIR/summary.json 2>/dev/null || echo "ERROR: summary.json not found!"

echo "PETS production complete!"
