#!/bin/bash
#SBATCH -J pets_prod
#SBATCH -p gpu_h100_4
#SBATCH -N 1
#SBATCH --gpus=1
#SBATCH -t 24:00:00
#SBATCH -o logs/slurm_pets_prod_%j.out
#SBATCH -e logs/slurm_pets_prod_%j.err

module load python-3.10.8-gcc-11.2.0-dlcmq7k

# Debug: Check python and environment
echo "Python version:"
python --version
echo "Python path:"
which python
echo "Current directory:"
pwd
echo "Files in current directory:"
ls -la

# Configuration variables (change these for different runs)
SEEDS="1"
EPISODES="295"
OUTPUT_DIR="results/pets_seeds_${SEEDS}_ep${EPISODES}"

mkdir -p logs $OUTPUT_DIR

# Ensure we're in the project root
cd ..

echo "=== PETS Fast Prod: ${SEEDS} seed(s), ${EPISODES} episodes ==="
echo "Running: python pets_ver3/pets_train.py --seed $SEEDS --episodes $EPISODES"
python pets_ver3/pets_train.py --seed $SEEDS --episodes $EPISODES

# Check exit code
if [ $? -eq 0 ]; then
    echo "Python command completed successfully"
else
    echo "ERROR: Python command failed with exit code $?"
fi

echo ""
echo "=== Verifying results ==="
echo "Looking for output in: results/pets_seeds_${SEEDS}_ep${EPISODES}/"
ls -la results/pets_seeds_${SEEDS}_ep${EPISODES}/ 2>/dev/null || echo "Directory not found"

OUTPUT_DIR_ACTUAL="results/pets_seeds_${SEEDS}_ep${EPISODES}"
if [ -f "$OUTPUT_DIR_ACTUAL/episodes.csv" ]; then
    echo "✓ episodes.csv found"
    wc -l $OUTPUT_DIR_ACTUAL/episodes.csv
    head -5 $OUTPUT_DIR_ACTUAL/episodes.csv
else
    echo "✗ ERROR: episodes.csv not found in $OUTPUT_DIR_ACTUAL/"
    ls -la results/ 2>/dev/null || echo "No results directory found"
fi

if [ -f "$OUTPUT_DIR_ACTUAL/summary.json" ]; then
    echo "✓ summary.json found"
    head -20 $OUTPUT_DIR_ACTUAL/summary.json
else
    echo "✗ ERROR: summary.json not found in $OUTPUT_DIR_ACTUAL/"
fi

echo "PETS production complete!"
