#!/bin/bash
#SBATCH -J compare_4_algorithms
#SBATCH -p cpu
#SBATCH -N 1
#SBATCH --cpus-per-task=4
#SBATCH -t 01:00:00
#SBATCH -o logs/slurm_compare_%j.out
#SBATCH -e logs/slurm_compare_%j.err

module load python-3.10.8-gcc-11.2.0-dlcmq7k

echo "=== Running 4-Algorithm Comparison ==="
echo "Date: $(date)"
echo "Host: $(hostname)"
echo ""

# Ensure results directories exist (in case some algorithms haven't run yet)
mkdir -p results/dqn results/pets results/mbpo results/ppo comparison logs

# Run the comparison script
python compare_all_4.py \
    --dqn results/dqn/episodes.csv \
    --pets results/pets/episodes.csv \
    --mbpo results/mbpo/episodes.csv \
    --ppo results/ppo/episodes.csv \
    --output comparison/

echo ""
echo "=== Comparison Complete ==="
echo "Check comparison/ directory for outputs:"
echo "  - table_performance_comparison.tex (LaTeX table)"
echo "  - statistical_tests.txt"
echo "  - comparison.json"
echo "  - comparison_plot.png"
echo ""
echo "SLURM job completed successfully!"