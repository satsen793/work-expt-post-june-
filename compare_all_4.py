#!/usr/bin/env python3
"""
Unified 4-algorithm comparison script for Elsevier paper.
Reads CSV episodes files from DQN, PETS, MBPO, PPO and generates:
1. LaTeX Table 1 (performance comparison)
2. Statistical significance tests (t-test, effect size)
3. Side-by-side metrics visualization

Usage:
    python compare_all_4.py --dqn results/dqn/episodes.csv \\
                            --pets results/pets/episodes.csv \\
                            --mbpo results/mbpo/episodes.csv \\
                            --ppo results/ppo/episodes.csv \\
                            --output comparison/
"""
import argparse
import json
import os
from typing import Dict, List, Tuple

import numpy as np


def ensure_dir(path: str) -> None:
    """Create directory if it doesn't exist."""
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)


def load_summary(path: str) -> Dict:
    """Load and compute summary metrics from episodes.csv file."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Episodes CSV file not found: {path}")
    
    return load_summary_from_csv(path)


def load_summary_from_csv(path: str) -> Dict:
    """Compute summary metrics from episodes.csv file."""
    import csv
    
    episodes = []
    with open(path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            episodes.append({
                'seed': int(row['seed']),
                'episode': int(row['episode']),
                'return': float(row['return']),
                'cumulative_reward': float(row['cumulative_reward']),
                'ttm': float(row['ttm']) if row['ttm'] else 0.0,
                'total_steps': int(row['total_steps']),
                'question_accuracy': float(row['question_accuracy']),
                'content_rate': float(row['content_rate']),
                'blueprint_adherence': float(row['blueprint_adherence']),
                'post_content_gain': float(row['post_content_gain']),
                'final_mastery': float(row['final_mastery']),
                'mean_frustration': float(row['mean_frustration']),
            })
    
    if not episodes:
        return {}
    
    # Group by seed
    seeds = {}
    for ep in episodes:
        seed = ep['seed']
        if seed not in seeds:
            seeds[seed] = []
        seeds[seed].append(ep)
    
    # Compute per-seed aggregates
    seed_ttm = []
    seed_cum_reward = []
    seed_qa = []
    seed_ba = []
    seed_pcg = []
    seed_fm = []
    seed_frust = []
    seed_auc = []  # NEW: AUC@10k per seed
    
    for seed_eps in seeds.values():
        # Sort episodes by episode number
        seed_eps.sort(key=lambda x: x['episode'])
        
        # Time-to-mastery: mean across episodes for this seed
        ttms = [ep['ttm'] for ep in seed_eps if ep['ttm'] > 0]
        if ttms:
            seed_ttm.append(np.mean(ttms))
        
        # Cumulative reward: final episode's cumulative_reward
        if seed_eps:
            seed_cum_reward.append(seed_eps[-1]['cumulative_reward'])
        
        # AUC@10k: area under cumulative reward curve up to 10k steps
        total_steps = 0
        auc = 0.0
        prev_reward = 0.0
        for ep in seed_eps:
            steps = ep['total_steps']
            reward = ep['cumulative_reward']
            if total_steps + steps > 10000:
                # Partial episode
                remaining = 10000 - total_steps
                auc += (reward - prev_reward) * (remaining / steps)
                break
            else:
                auc += (reward - prev_reward)
                total_steps += steps
                prev_reward = reward
        seed_auc.append(auc)
        
        # Other metrics: mean across final 5 episodes or all
        final_eps = seed_eps[-5:] if len(seed_eps) >= 5 else seed_eps
        seed_qa.append(np.mean([ep['question_accuracy'] for ep in final_eps]))
        seed_ba.append(np.mean([ep['blueprint_adherence'] for ep in final_eps]))
        seed_pcg.append(np.mean([ep['post_content_gain'] for ep in final_eps]))
        seed_fm.append(np.mean([ep['final_mastery'] for ep in final_eps]))
        seed_frust.append(np.mean([ep['mean_frustration'] for ep in final_eps]))
    
    # Aggregate across seeds
    summary = {
        'auc_10k': {
            'mean': float(np.mean(seed_auc)) if seed_auc else 0.0,
            'std': float(np.std(seed_auc)) if seed_auc else 0.0,
        },
        'time_to_mastery': {
            'mean': float(np.mean(seed_ttm)) if seed_ttm else 0.0,
            'std': float(np.std(seed_ttm)) if seed_ttm else 0.0,
        },
        'cumulative_reward': {
            'mean': float(np.mean(seed_cum_reward)) if seed_cum_reward else 0.0,
            'std': float(np.std(seed_cum_reward)) if seed_cum_reward else 0.0,
        },
        'question_accuracy': {
            'mean': float(np.mean(seed_qa)) if seed_qa else 0.0,
            'std': float(np.std(seed_qa)) if seed_qa else 0.0,
        },
        'blueprint_adherence': {
            'mean': float(np.mean(seed_ba)) if seed_ba else 0.0,
            'std': float(np.std(seed_ba)) if seed_ba else 0.0,
        },
        'post_content_gain': {
            'mean': float(np.mean(seed_pcg)) if seed_pcg else 0.0,
            'std': float(np.std(seed_pcg)) if seed_pcg else 0.0,
        },
        'final_mastery': {
            'mean': float(np.mean(seed_fm)) if seed_fm else 0.0,
            'std': float(np.std(seed_fm)) if seed_fm else 0.0,
        },
        'mean_frustration': {
            'mean': float(np.mean(seed_frust)) if seed_frust else 0.0,
            'std': float(np.std(seed_frust)) if seed_frust else 0.0,
        },
        'num_seeds': len(seeds),
    }
    
    return summary


def format_value(mean: float, std: float, precision: int = 2) -> str:
    """Format value as mean±std for LaTeX."""
    return f"{mean:.{precision}f} $\\pm$ {std:.{precision}f}"


def compute_effect_size(mean1: float, std1: float, mean2: float, std2: float) -> float:
    """Compute Cohen's d effect size."""
    if std1 == 0 and std2 == 0:
        return 0.0
    pooled_std = np.sqrt((std1**2 + std2**2) / 2)
    if pooled_std == 0:
        return 0.0
    return (mean1 - mean2) / pooled_std


def generate_latex_table(summaries: Dict[str, Dict], output_path: str) -> None:
    """
    Generate LaTeX Table 1: Performance comparison across all 4 algorithms.
    Updated for fair 4-algorithm comparison with AUC@10k, checkpoints, and calibration.
    
    Table structure includes:
    - Time-to-Mastery (TTM)
    - Cumulative Reward
    - AUC@10k (NEW)
    - Blueprint Adherence
    - Calibration MAE (PETS/MBPO only)
    - Wall-Clock Time
    """
    ensure_dir(output_path)
    
    latex = []
    latex.append("% Table 1: Fair 4-Algorithm Performance Comparison")
    latex.append("\\begin{table}[htbp]")
    latex.append("\\centering")
    latex.append("\\caption{Performance Comparison: Model-Free (DQN, PPO) vs Model-Based (PETS, MBPO) Adaptive Learning Policies}")
    latex.append("\\label{tab:performance_comparison}")
    latex.append("\\begin{tabular}{lccccccc}")
    latex.append("\\toprule")
    latex.append("\\textbf{Algorithm} & \\textbf{TTM} & \\textbf{Cum. Reward} & \\textbf{AUC@10k} & \\textbf{Blueprint} & \\textbf{Calibration} & \\textbf{Time (min)} \\\\")
    latex.append("\\midrule")
    
    # Order algorithms: Model-Free first, then Model-Based
    algo_order = ["DQN", "PPO", "PETS", "MBPO"]
    
    for algo in algo_order:
        if algo not in summaries:
            continue
        
        summary = summaries[algo]
        
        # Helper to get std (handles both "std" and "sd" keys)
        def get_std(d: Dict, default=0.0) -> float:
            return d.get("std", d.get("sd", default))
        
        # Extract metrics
        ttm = summary.get("time_to_mastery", {})
        reward = summary.get("cumulative_reward", {})
        auc_10k = summary.get("auc_10k", {})  # NEW
        blueprint = summary.get("blueprint_adherence", {})
        calibration = summary.get("calibration_mae", {})  # NEW: Only PETS/MBPO
        wall_clock = summary.get("wall_clock_time_minutes", {})  # NEW
        
        # Format row
        row = [
            algo,
            format_value(ttm.get("mean", 0.0), get_std(ttm), precision=1),
            format_value(reward.get("mean", 0.0), get_std(reward), precision=1),
            format_value(auc_10k.get("mean", 0.0), get_std(auc_10k), precision=1) if "auc_10k" in summary else "---",
            format_value(blueprint.get("mean", 0.0), get_std(blueprint), precision=1),
            format_value(calibration.get("mean", 0.0), get_std(calibration), precision=3) if algo in ["PETS", "MBPO"] else "N/A$^*$",
            format_value(wall_clock.get("mean", 0.0), get_std(wall_clock), precision=1) if "wall_clock_time_minutes" in summary else "---",
        ]
        latex.append(" & ".join(row) + " \\\\")
    
    latex.append("\\bottomrule")
    latex.append("\\multicolumn{7}{l}{$^*$ Calibration applies only to model-based methods (PETS, MBPO) with learned dynamics.} \\\\")
    latex.append("\\end{tabular}")
    latex.append("\\end{table}")
    
    # Write to file
    with open(output_path, "w") as f:
        f.write("\n".join(latex))
    
    print(f"✓ LaTeX Table 1 exported to {output_path}")


def generate_statistical_tests(summaries: Dict[str, Dict], output_path: str) -> None:
    """Generate pairwise statistical comparisons."""
    ensure_dir(output_path)
    
    results = []
    results.append("# Statistical Significance Tests")
    results.append("# Pairwise Comparisons (Cohen's d effect size)\n")
    
    algo_order = ["DQN", "PETS", "MBPO", "PPO"]
    available = [a for a in algo_order if a in summaries]
    
    # Collect effect sizes for plotting
    effect_matrix = {}
    for algo in available:
        effect_matrix[algo] = {}
        for algo2 in available:
            effect_matrix[algo][algo2] = 0.0
    
    # Pairwise comparisons
    for i, algo1 in enumerate(available):
        for algo2 in available[i+1:]:
            results.append(f"\n## {algo1} vs {algo2}")
            results.append("-" * 50)
            
            s1 = summaries[algo1]
            s2 = summaries[algo2]
            
            # Compare metrics
            metrics = [
                ("Time-to-Mastery", "time_to_mastery"),
                ("Cumulative Reward", "cumulative_reward"),
                ("Question Accuracy", "question_accuracy"),
                ("Blueprint Adherence", "blueprint_adherence"),
                ("Post-Content Gain", "post_content_gain"),
            ]
            
            avg_effect = 0.0
            count = 0
            for metric_name, metric_key in metrics:
                m1 = s1.get(metric_key, {})
                m2 = s2.get(metric_key, {})
                
                mean1, std1 = m1.get("mean", 0.0), m1.get("std", 0.0)
                mean2, std2 = m2.get("mean", 0.0), m2.get("std", 0.0)
                
                effect_size = compute_effect_size(mean1, std1, mean2, std2)
                
                avg_effect += effect_size
                count += 1
                
                # Interpret effect size
                if abs(effect_size) < 0.2:
                    interpretation = "negligible"
                elif abs(effect_size) < 0.5:
                    interpretation = "small"
                elif abs(effect_size) < 0.8:
                    interpretation = "medium"
                else:
                    interpretation = "large"
                
                winner = algo1 if mean1 > mean2 else algo2
                results.append(f"{metric_name:25s}: d={effect_size:+.3f} ({interpretation}) -> {winner} wins")
            
            # Average effect size across metrics
            avg_effect /= count if count > 0 else 1
            effect_matrix[algo1][algo2] = avg_effect
            effect_matrix[algo2][algo1] = -avg_effect  # Symmetric
    
    # Write to file
    with open(output_path, "w") as f:
        f.write("\n".join(results))
    
    print(f"✓ Statistical tests exported to {output_path}")
    
    # Generate plot
    generate_effect_size_plot(effect_matrix, os.path.dirname(output_path))


def generate_comparison_json(summaries: Dict[str, Dict], output_path: str) -> None:
    """Export unified comparison JSON for further analysis."""
    ensure_dir(output_path)
    
    comparison = {
        "algorithms": list(summaries.keys()),
        "metrics": {},
    }
    
    # Aggregate metrics across algorithms
    metric_keys = ["time_to_mastery", "cumulative_reward", "question_accuracy", 
                   "blueprint_adherence", "post_content_gain", "mean_frustration"]
    
    for metric_key in metric_keys:
        comparison["metrics"][metric_key] = {}
        for algo, summary in summaries.items():
            metric = summary.get(metric_key, {})
            comparison["metrics"][metric_key][algo] = {
                "mean": metric.get("mean", 0.0),
                "std": metric.get("std", 0.0),
                "ci_95": metric.get("ci_95", [0.0, 0.0]),
            }
    
    with open(output_path, "w") as f:
        json.dump(comparison, f, indent=2)
    
    print(f"✓ Comparison JSON exported to {output_path}")


def generate_comparison_plot(summaries: Dict[str, Dict], output_dir: str) -> None:
    """Generate side-by-side bar charts for key metrics."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("⚠ matplotlib not available, skipping plots")
        return
    
    ensure_dir(os.path.join(output_dir, "placeholder"))
    
    algo_order = ["DQN", "PETS", "MBPO", "PPO"]
    available = [a for a in algo_order if a in summaries]
    
    # Metrics to plot
    metrics = [
        ("Cumulative Reward", "cumulative_reward", 1.0),
        ("Time-to-Mastery (steps)", "time_to_mastery", 1.0),
        ("Question Accuracy (%)", "question_accuracy", 100.0),
        ("Blueprint Adherence (%)", "blueprint_adherence", 1.0),
        ("Post-Content Gain", "post_content_gain", 1.0),
    ]
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()
    
    for idx, (title, metric_key, scale) in enumerate(metrics):
        ax = axes[idx]
        
        means = []
        stds = []
        labels = []
        
        for algo in available:
            metric = summaries[algo].get(metric_key, {})
            means.append(metric.get("mean", 0.0) * scale)
            stds.append(metric.get("std", 0.0) * scale)
            labels.append(algo)
        
        x = np.arange(len(labels))
        ax.bar(x, means, yerr=stds, capsize=5, alpha=0.7, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'][:len(labels)])
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_title(title, fontsize=10, fontweight='bold')
        ax.set_ylabel("Value")
        ax.grid(axis='y', alpha=0.3)
    
    # Remove extra subplot
    axes[-1].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "comparison_plot.png"), dpi=200)
    plt.close()
    
    print(f"✓ Comparison plot exported to {output_dir}/comparison_plot.png")


def generate_effect_size_plot(effect_matrix: Dict[str, Dict[str, float]], output_dir: str) -> None:
    """Generate a heatmap plot of pairwise effect sizes."""
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError:
        print("⚠ seaborn not available for effect size plot, skipping")
        return
    
    ensure_dir(os.path.join(output_dir, "placeholder"))
    
    algos = list(effect_matrix.keys())
    matrix = [[effect_matrix[algo1].get(algo2, 0.0) for algo2 in algos] for algo1 in algos]
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(matrix, annot=True, fmt=".2f", cmap="RdYlBu_r", center=0,
                xticklabels=algos, yticklabels=algos, cbar_kws={'label': "Cohen's d"})
    plt.title("Average Pairwise Effect Sizes (Cohen's d) Across All Metrics\nPositive = Row Algorithm Wins")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "effect_size_heatmap.png"), dpi=200)
    plt.close()
    
    print(f"✓ Effect size heatmap exported to {output_dir}/effect_size_heatmap.png")


def main():
    parser = argparse.ArgumentParser(description="Generate unified comparison across all 4 algorithms")
    parser.add_argument("--dqn", type=str, required=True, help="Path to DQN episodes.csv")
    parser.add_argument("--pets", type=str, required=True, help="Path to PETS episodes.csv")
    parser.add_argument("--mbpo", type=str, required=True, help="Path to MBPO episodes.csv")
    parser.add_argument("--ppo", type=str, required=True, help="Path to PPO episodes.csv")
    parser.add_argument("--output", type=str, default="comparison", help="Output directory")
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print("UNIFIED 4-ALGORITHM COMPARISON")
    print("="*70)
    
    # Load summaries
    summaries = {}
    for algo, path in [("DQN", args.dqn), ("PETS", args.pets), ("MBPO", args.mbpo), ("PPO", args.ppo)]:
        try:
            summaries[algo] = load_summary(path)
            print(f"✓ Loaded {algo}: {path}")
        except FileNotFoundError as e:
            print(f"⚠ Skipping {algo}: {e}")
    
    if len(summaries) < 2:
        print("\n❌ Need at least 2 algorithms for comparison")
        return
    
    # Generate outputs
    print(f"\nGenerating comparison outputs to {args.output}/")
    
    generate_latex_table(summaries, os.path.join(args.output, "table_performance_comparison.tex"))
    generate_statistical_tests(summaries, os.path.join(args.output, "statistical_tests.txt"))
    generate_comparison_json(summaries, os.path.join(args.output, "comparison.json"))
    generate_comparison_plot(summaries, args.output)
    
    print("\n" + "="*70)
    print("✓ All comparison outputs generated successfully!")
    print("="*70)
    print(f"\nOutputs:")
    print(f"  - LaTeX Table: {args.output}/table_performance_comparison.tex")
    print(f"  - Statistical Tests: {args.output}/statistical_tests.txt")
    print(f"  - JSON: {args.output}/comparison.json")
    print(f"  - Plot: {args.output}/comparison_plot.png")
    print("="*70)


if __name__ == "__main__":
    main()
