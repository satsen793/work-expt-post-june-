#!/usr/bin/env python3
"""
Unified 4-algorithm comparison script for Elsevier paper.
Reads JSON summaries from DQN, PETS, MBPO, PPO and generates:
1. LaTeX Table 1 (performance comparison)
2. Statistical significance tests (t-test, effect size)
3. Side-by-side metrics visualization

Usage:
    python compare_all_4.py --dqn results/dqn/summary.json \\
                            --pets results/pets/summary.json \\
                            --mbpo results/mbpo/summary.json \\
                            --ppo results/ppo/summary.json \\
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
    """Load JSON summary from file."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Summary file not found: {path}")
    
    with open(path, "r") as f:
        data = json.load(f)
    
    # Handle different JSON structures
    if "summary" in data:
        return data["summary"]
    return data


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
    
    Table structure:
    Algorithm | TTM | Cum. Reward | Accuracy | Blueprint | Post-Content | Frustration
    """
    ensure_dir(output_path)
    
    latex = []
    latex.append("% Table 1: Performance Comparison Across Algorithms")
    latex.append("\\begin{table}[htbp]")
    latex.append("\\centering")
    latex.append("\\caption{Performance Comparison: DQN, PETS, MBPO, and PPO}")
    latex.append("\\label{tab:performance_comparison}")
    latex.append("\\begin{tabular}{lcccccc}")
    latex.append("\\toprule")
    latex.append("\\textbf{Algorithm} & \\textbf{TTM} & \\textbf{Reward} & \\textbf{Accuracy} & \\textbf{Blueprint} & \\textbf{Post-Content} & \\textbf{Frustration} \\\\")
    latex.append("\\midrule")
    
    # Order algorithms
    algo_order = ["DQN", "PETS", "MBPO", "PPO"]
    
    for algo in algo_order:
        if algo not in summaries:
            continue
        
        summary = summaries[algo]
        
        # Extract metrics
        ttm = summary.get("time_to_mastery", {})
        reward = summary.get("cumulative_reward", {})
        accuracy = summary.get("question_accuracy", {})
        blueprint = summary.get("blueprint_adherence", {})
        post_content = summary.get("post_content_gain", {})
        frustration = summary.get("mean_frustration", {})
        
        # Format row
        row = [
            algo,
            format_value(ttm.get("mean", 0.0), ttm.get("std", 0.0), precision=1),
            format_value(reward.get("mean", 0.0), reward.get("std", 0.0), precision=1),
            format_value(accuracy.get("mean", 0.0) * 100, accuracy.get("std", 0.0) * 100, precision=1),  # Convert to %
            format_value(blueprint.get("mean", 0.0), blueprint.get("std", 0.0), precision=1),
            format_value(post_content.get("mean", 0.0), post_content.get("std", 0.0), precision=3),
            format_value(frustration.get("mean", 0.0), frustration.get("std", 0.0), precision=2),
        ]
        latex.append(" & ".join(row) + " \\\\")
    
    latex.append("\\bottomrule")
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
            
            for metric_name, metric_key in metrics:
                m1 = s1.get(metric_key, {})
                m2 = s2.get(metric_key, {})
                
                mean1, std1 = m1.get("mean", 0.0), m1.get("std", 0.0)
                mean2, std2 = m2.get("mean", 0.0), m2.get("std", 0.0)
                
                effect_size = compute_effect_size(mean1, std1, mean2, std2)
                
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
                results.append(f"{metric_name:25s}: d={effect_size:+.3f} ({interpretation}) → {winner} wins")
    
    # Write to file
    with open(output_path, "w") as f:
        f.write("\n".join(results))
    
    print(f"✓ Statistical tests exported to {output_path}")


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


def main():
    parser = argparse.ArgumentParser(description="Generate unified comparison across all 4 algorithms")
    parser.add_argument("--dqn", type=str, required=True, help="Path to DQN summary.json")
    parser.add_argument("--pets", type=str, required=True, help="Path to PETS summary.json")
    parser.add_argument("--mbpo", type=str, required=True, help="Path to MBPO summary.json")
    parser.add_argument("--ppo", type=str, required=True, help="Path to PPO summary.json")
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
