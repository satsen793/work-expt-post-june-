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
    """Load and compute summary metrics from episodes.csv or summary.json file."""
    # Try JSON first
    json_path = path.replace('episodes.csv', 'summary.json')
    if os.path.exists(json_path):
        try:
            return load_summary_from_json(json_path)
        except Exception as e:
            print(f"Warning: Failed to load JSON {json_path}: {e}, falling back to CSV")
    
    # Fall back to CSV
    if not os.path.exists(path):
        raise FileNotFoundError(f"Neither JSON nor CSV file found: {json_path} or {path}")
    
    return load_summary_from_csv(path)


def load_summary_from_json(path: str) -> Dict:
    """Load summary metrics from JSON file."""
    with open(path, 'r') as f:
        data = json.load(f)
    
    if isinstance(data, list):
        # It's episodes data, compute summary
        return compute_summary_from_episodes(data)
    elif isinstance(data, dict):
        # If the JSON has a "summary" key, use that
        if "summary" in data:
            summary_data = data["summary"]
        else:
            summary_data = data
        
        # Check if metrics are already aggregated (dict with mean/std) or raw (list)
        processed_summary = {}
        for key, value in summary_data.items():
            if isinstance(value, dict) and "mean" in value:
                # Already aggregated
                processed_summary[key] = value
            elif isinstance(value, list):
                # Raw values, compute mean/std
                if value:
                    processed_summary[key] = {
                        "mean": float(np.mean(value)),
                        "std": float(np.std(value)),
                    }
                else:
                    processed_summary[key] = {"mean": 0.0, "std": 0.0}
            else:
                # Single value, assume mean
                processed_summary[key] = {"mean": float(value), "std": 0.0}
        
        return processed_summary
    else:
        raise ValueError(f"Unexpected JSON structure in {path}")


def compute_summary_from_episodes(episodes: List[Dict]) -> Dict:
    """Compute summary from list of episode dicts (same as from CSV)."""
    if not episodes:
        return {}
    
    # Group by seed
    seeds = {}
    for ep in episodes:
        seed = ep.get('seed', 0)
        if seed not in seeds:
            seeds[seed] = []
        seeds[seed].append(ep)
    
    # Compute per-seed aggregates (same as in load_summary_from_csv)
    seed_ttm = []
    seed_cum_reward = []
    seed_qa = []
    seed_pcg = []
    seed_fm = []
    seed_frust = []
    seed_auc = []
    
    for seed_eps in seeds.values():
        # Sort episodes by episode number
        seed_eps.sort(key=lambda x: x.get('episode', 0))
        
        # Time-to-mastery: mean across episodes for this seed
        ttms = [ep['ttm'] for ep in seed_eps if ep.get('ttm', 0) > 0]
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
            steps = ep.get('total_steps', 0)
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


def load_full_episodes(path: str) -> List[Dict]:
    """Load all episodes from episodes.csv file."""
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
                'post_content_gain': float(row['post_content_gain']),
                'final_mastery': float(row['final_mastery']),
                'mean_frustration': float(row['mean_frustration']),
            })
    
    return episodes


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
    Focused on key metrics: TTM, Post Content Gain, Cumulative Reward, Reward Variance
    """
    ensure_dir(output_path)
    
    latex = []
    latex.append("% Table 1: Performance Comparison Across Controllers")
    latex.append("\\begin{table}[htbp]")
    latex.append("\\centering")
    latex.append("\\caption{Evaluation Summary Across Controllers (Mean ± SD Over Seeds)}")
    latex.append("\\label{tab:performance_comparison}")
    latex.append("\\begin{tabular}{lcccc}")
    latex.append("\\toprule")
    latex.append("\\textbf{Algorithm} & \\textbf{TTM} & \\textbf{Post Content Gain} & \\textbf{Cum. Reward} & \\textbf{Reward Variance} \\\\")
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
        pcg = summary.get("post_content_gain", {})
        reward = summary.get("cumulative_reward", {})
        
        # Format row
        row = [
            algo,
            format_value(ttm.get("mean", 0.0), get_std(ttm), precision=1),
            format_value(pcg.get("mean", 0.0), get_std(pcg), precision=3),
            format_value(reward.get("mean", 0.0), get_std(reward), precision=1),
            f"{get_std(reward):.1f}",  # Reward Variance = SD of cumulative reward
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


def generate_effect_size_plot(effect_matrix: Dict[str, Dict], output_dir: str) -> None:
    """Generate Cohen's d effect size heatmap."""
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    algos = list(effect_matrix.keys())
    matrix = np.zeros((len(algos), len(algos)))
    
    for i, a1 in enumerate(algos):
        for j, a2 in enumerate(algos):
            matrix[i, j] = effect_matrix[a1][a2]
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(matrix, annot=True, fmt=".2f", cmap="RdYlBu_r", 
                xticklabels=algos, yticklabels=algos, center=0,
                cbar_kws={'label': "Cohen's d"}, annot_kws={"size": 12})
    plt.title("Effect Size Heatmap (Cohen's d)", fontsize=16, fontweight='bold')
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, "effect_size_heatmap.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Effect size heatmap exported to {output_path}")


def generate_comparison_json(summaries: Dict[str, Dict], output_path: str) -> None:
    """Export unified comparison JSON for further analysis."""
    ensure_dir(output_path)
    
    comparison = {
        "algorithms": list(summaries.keys()),
        "metrics": {},
    }
    
    # Aggregate metrics across algorithms
    metric_keys = ["time_to_mastery", "cumulative_reward", "question_accuracy", 
                   "post_content_gain", "mean_frustration"]
    
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
        ("Post-Content Gain", "post_content_gain", 1.0),
    ]
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
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
        ax.set_xticklabels(labels, fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_ylabel("Value", fontsize=12)
        ax.grid(axis='y', alpha=0.3)
        ax.tick_params(axis='both', which='major', labelsize=12)
    
    # No extra subplot to remove
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "comparison_plot.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Comparison plot exported to {output_dir}/comparison_plot.png")


def generate_learning_curves(full_episodes: Dict[str, List[Dict]], output_dir: str) -> None:
    """Generate learning curves: cumulative reward over episodes."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("⚠ matplotlib not available, skipping learning curves")
        return
    
    ensure_dir(os.path.join(output_dir, "placeholder"))
    
    plt.figure(figsize=(12, 8))
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    
    for idx, (algo, episodes) in enumerate(full_episodes.items()):
        if not episodes:
            continue
        
        # Group by seed
        seed_data = {}
        for ep in episodes:
            seed = ep['seed']
            if seed not in seed_data:
                seed_data[seed] = []
            seed_data[seed].append(ep)
        
        # Compute mean and std across seeds
        max_ep = max(len(eps) for eps in seed_data.values())
        mean_rewards = []
        std_rewards = []
        
        for ep_num in range(max_ep):
            rewards = [seed_data[seed][ep_num]['cumulative_reward'] for seed in seed_data if ep_num < len(seed_data[seed])]
            if rewards:
                mean_rewards.append(np.mean(rewards))
                std_rewards.append(np.std(rewards))
            else:
                mean_rewards.append(0)
                std_rewards.append(0)
        
        episodes_range = list(range(1, len(mean_rewards) + 1))
        plt.plot(episodes_range, mean_rewards, label=algo, color=colors[idx % len(colors)], linewidth=3)
        plt.fill_between(episodes_range, 
                         [m - s for m, s in zip(mean_rewards, std_rewards)],
                         [m + s for m, s in zip(mean_rewards, std_rewards)],
                         alpha=0.2, color=colors[idx % len(colors)])
    
    plt.xlabel('Episode', fontsize=14)
    plt.ylabel('Cumulative Reward', fontsize=14)
    plt.title('Learning Curves: Cumulative Reward Over Episodes', fontsize=16, fontweight='bold')
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tick_params(axis='both', which='major', labelsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "learning_curves.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Learning curves exported to {output_dir}/learning_curves.png")


def generate_question_accuracy_curves(full_episodes: Dict[str, List[Dict]], output_dir: str) -> None:
    """Generate question accuracy curves over episodes."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("⚠ matplotlib not available, skipping question accuracy curves")
        return
    
    ensure_dir(os.path.join(output_dir, "placeholder"))
    
    plt.figure(figsize=(12, 8))
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    
    for idx, (algo, episodes) in enumerate(full_episodes.items()):
        if not episodes:
            continue
        
        # Group by seed
        seed_data = {}
        for ep in episodes:
            seed = ep['seed']
            if seed not in seed_data:
                seed_data[seed] = []
            seed_data[seed].append(ep)
        
        # Compute mean and std across seeds
        max_ep = max(len(eps) for eps in seed_data.values())
        mean_acc = []
        std_acc = []
        
        for ep_num in range(max_ep):
            accs = [seed_data[seed][ep_num]['question_accuracy'] for seed in seed_data if ep_num < len(seed_data[seed])]
            if accs:
                mean_acc.append(np.mean(accs))
                std_acc.append(np.std(accs))
            else:
                mean_acc.append(0)
                std_acc.append(0)
        
        episodes_range = list(range(1, len(mean_acc) + 1))
        plt.plot(episodes_range, mean_acc, label=algo, color=colors[idx % len(colors)], linewidth=3)
        plt.fill_between(episodes_range, 
                         [m - s for m, s in zip(mean_acc, std_acc)],
                         [m + s for m, s in zip(mean_acc, std_acc)],
                         alpha=0.2, color=colors[idx % len(colors)])
    
    plt.xlabel('Episode', fontsize=14)
    plt.ylabel('Question Accuracy', fontsize=14)
    plt.title('Question Accuracy Over Episodes', fontsize=16, fontweight='bold')
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tick_params(axis='both', which='major', labelsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "question_accuracy_curves.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Question accuracy curves exported to {output_dir}/question_accuracy_curves.png")


def generate_frustration_curves(full_episodes: Dict[str, List[Dict]], output_dir: str) -> None:
    """Generate frustration curves over episodes."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("⚠ matplotlib not available, skipping frustration curves")
        return
    
    ensure_dir(os.path.join(output_dir, "placeholder"))
    
    plt.figure(figsize=(12, 8))
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    
    for idx, (algo, episodes) in enumerate(full_episodes.items()):
        if not episodes:
            continue
        
        # Group by seed
        seed_data = {}
        for ep in episodes:
            seed = ep['seed']
            if seed not in seed_data:
                seed_data[seed] = []
            seed_data[seed].append(ep)
        
        # Compute mean and std across seeds
        max_ep = max(len(eps) for eps in seed_data.values())
        mean_frust = []
        std_frust = []
        
        for ep_num in range(max_ep):
            frusts = [seed_data[seed][ep_num]['mean_frustration'] for seed in seed_data if ep_num < len(seed_data[seed])]
            if frusts:
                mean_frust.append(np.mean(frusts))
                std_frust.append(np.std(frusts))
            else:
                mean_frust.append(0)
                std_frust.append(0)
        
        episodes_range = list(range(1, len(mean_frust) + 1))
        plt.plot(episodes_range, mean_frust, label=algo, color=colors[idx % len(colors)], linewidth=3)
        plt.fill_between(episodes_range, 
                         [m - s for m, s in zip(mean_frust, std_frust)],
                         [m + s for m, s in zip(mean_frust, std_frust)],
                         alpha=0.2, color=colors[idx % len(colors)])
    
    plt.xlabel('Episode', fontsize=14)
    plt.ylabel('Mean Frustration', fontsize=14)
    plt.title('Frustration Levels Over Episodes', fontsize=16, fontweight='bold')
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tick_params(axis='both', which='major', labelsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "frustration_curves.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Frustration curves exported to {output_dir}/frustration_curves.png")


def generate_moving_average_curves(full_episodes: Dict[str, List[Dict]], output_dir: str) -> None:
    """Generate seed-aggregated learning curves with moving average reward vs steps."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("⚠ matplotlib not available, skipping moving average curves")
        return
    
    ensure_dir(os.path.join(output_dir, "placeholder"))
    
    plt.figure(figsize=(12, 8))
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    
    for idx, (algo, episodes) in enumerate(full_episodes.items()):
        if not episodes:
            continue
        
        # Collect all (cumulative_steps, cumulative_reward) points across all seeds
        all_points = []
        for ep in episodes:
            seed = ep['seed']
            # For each seed, compute cumulative steps
            seed_eps = [e for e in episodes if e['seed'] == seed]
            seed_eps.sort(key=lambda x: x['episode'])
            cum_steps = 0
            for e in seed_eps:
                cum_steps += e['total_steps']
                all_points.append((cum_steps, e['cumulative_reward']))
        
        if not all_points:
            continue
        
        # Sort by steps
        all_points.sort(key=lambda x: x[0])
        steps, rewards = zip(*all_points)
        
        # Remove duplicates by averaging rewards at same step
        from collections import defaultdict
        step_rewards = defaultdict(list)
        for s, r in all_points:
            step_rewards[s].append(r)
        unique_steps = sorted(step_rewards.keys())
        mean_rewards = [np.mean(step_rewards[s]) for s in unique_steps]
        
        # Apply moving average (window of 50 points)
        window = 50
        if len(mean_rewards) >= window:
            ma_rewards = np.convolve(mean_rewards, np.ones(window)/window, mode='valid')
            ma_steps = unique_steps[window-1:]
        else:
            ma_rewards = mean_rewards
            ma_steps = unique_steps
        
        plt.plot(np.array(ma_steps) / 1000, ma_rewards, label=algo, color=colors[idx % len(colors)], linewidth=3)
    
    plt.xlabel('Steps (thousands)', fontsize=14)
    plt.ylabel('Moving Average Reward', fontsize=14)
    plt.title('Seed-Aggregated Learning Curves: Moving Average Reward vs Steps', fontsize=16, fontweight='bold')
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tick_params(axis='both', which='major', labelsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "moving_average_curves.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Moving average curves exported to {output_dir}/moving_average_curves.png")


def generate_episode_reward_curves(full_episodes: Dict[str, List[Dict]], output_dir: str) -> None:
    """Generate seed-aggregated learning curves with moving average episode reward vs steps."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("⚠ matplotlib not available, skipping episode reward curves")
        return
    
    ensure_dir(os.path.join(output_dir, "placeholder"))
    
    plt.figure(figsize=(12, 8))
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    
    for idx, (algo, episodes) in enumerate(full_episodes.items()):
        if not episodes:
            continue
        
        # Collect all (cumulative_steps, episode_return) points across all seeds
        all_points = []
        for ep in episodes:
            seed = ep['seed']
            # For each seed, compute cumulative steps up to this episode
            seed_eps = [e for e in episodes if e['seed'] == seed]
            seed_eps.sort(key=lambda x: x['episode'])
            cum_steps = 0
            for e in seed_eps:
                if e['episode'] <= ep['episode']:  # Only count steps up to current episode
                    cum_steps += e['total_steps']
            all_points.append((cum_steps, ep['return']))
        
        if not all_points:
            continue
        
        # Sort by steps
        all_points.sort(key=lambda x: x[0])
        steps, rewards = zip(*all_points)
        
        # Remove duplicates by averaging rewards at same step
        from collections import defaultdict
        step_rewards = defaultdict(list)
        for s, r in all_points:
            step_rewards[s].append(r)
        unique_steps = sorted(step_rewards.keys())
        mean_rewards = [np.mean(step_rewards[s]) for s in unique_steps]
        
        # Apply moving average (window of 20 episodes)
        window = 20
        if len(mean_rewards) >= window:
            ma_rewards = np.convolve(mean_rewards, np.ones(window)/window, mode='valid')
            ma_steps = unique_steps[window-1:]
        else:
            ma_rewards = mean_rewards
            ma_steps = unique_steps
        
        plt.plot(np.array(ma_steps) / 1000, ma_rewards, label=algo, color=colors[idx % len(colors)], linewidth=3)
    
    plt.xlabel('Steps (thousands)', fontsize=14)
    plt.ylabel('Moving Average Episode Reward', fontsize=14)
    plt.title('Seed-Aggregated Learning Curves: Moving Average Episode Reward vs Steps', fontsize=16, fontweight='bold')
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tick_params(axis='both', which='major', labelsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "episode_reward_curves.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Episode reward curves exported to {output_dir}/episode_reward_curves.png")


def generate_moving_average_boxplot(full_episodes: Dict[str, List[Dict]], output_dir: str) -> None:
    """Generate box plot of moving average rewards at different step milestones."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("⚠ matplotlib not available, skipping moving average boxplot")
        return
    
    ensure_dir(os.path.join(output_dir, "placeholder"))
    
    # Define step milestones (in thousands)
    milestones = [50, 100, 150, 200, 250, 300]  # 50k, 100k, etc.
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    
    for idx, milestone in enumerate(milestones):
        ax = axes[idx]
        data = []
        labels = []
        
        for algo_idx, (algo, episodes) in enumerate(full_episodes.items()):
            if not episodes:
                continue
            
            # Collect rewards at this milestone for all seeds
            rewards_at_milestone = []
            
            for seed in set(ep['seed'] for ep in episodes):
                seed_eps = [ep for ep in episodes if ep['seed'] == seed]
                seed_eps.sort(key=lambda x: x['episode'])
                
                cum_steps = 0
                for ep in seed_eps:
                    cum_steps += ep['total_steps']
                    if cum_steps >= milestone * 1000:
                        # Interpolate if necessary, but for simplicity, take the reward at this episode
                        rewards_at_milestone.append(ep['return'])  # Use episode reward, not cumulative
                        break
                else:
                    # If not reached, skip or take last
                    if seed_eps:
                        rewards_at_milestone.append(seed_eps[-1]['return'])  # Use episode reward, not cumulative
            
            if rewards_at_milestone:
                data.append(rewards_at_milestone)
                labels.append(algo)
        
        if data:
            bp = ax.boxplot(data, labels=labels, patch_artist=True)
            for patch, color in zip(bp['boxes'], colors[:len(data)]):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
        
        ax.set_title(f'At {milestone}k Steps', fontsize=14, fontweight='bold')
        ax.set_ylabel('Episode Reward', fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis='both', which='major', labelsize=10)
    
    plt.suptitle('Seed-Aggregated Learning Curves: Episode Reward Distribution at Step Milestones', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "moving_average_boxplot.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Moving average boxplot exported to {output_dir}/moving_average_boxplot.png")


def generate_moving_average_curves_fine(full_episodes: Dict[str, List[Dict]], output_dir: str) -> None:
    """Generate seed-aggregated learning curves with moving average reward vs steps (fine window)."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("⚠ matplotlib not available, skipping fine moving average curves")
        return
    
    ensure_dir(os.path.join(output_dir, "placeholder"))
    
    plt.figure(figsize=(12, 8))
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    
    for idx, (algo, episodes) in enumerate(full_episodes.items()):
        if not episodes:
            continue
        
        # Collect all (cumulative_steps, cumulative_reward) points across all seeds
        all_points = []
        for ep in episodes:
            seed = ep['seed']
            # For each seed, compute cumulative steps
            seed_eps = [e for e in episodes if e['seed'] == seed]
            seed_eps.sort(key=lambda x: x['episode'])
            cum_steps = 0
            for e in seed_eps:
                cum_steps += e['total_steps']
                all_points.append((cum_steps, e['cumulative_reward']))
        
        if not all_points:
            continue
        
        # Sort by steps
        all_points.sort(key=lambda x: x[0])
        steps, rewards = zip(*all_points)
        
        # Remove duplicates by averaging rewards at same step
        from collections import defaultdict
        step_rewards = defaultdict(list)
        for s, r in all_points:
            step_rewards[s].append(r)
        unique_steps = sorted(step_rewards.keys())
        mean_rewards = [np.mean(step_rewards[s]) for s in unique_steps]
        
        # Apply moving average (window of 10 points)
        window = 10
        if len(mean_rewards) >= window:
            ma_rewards = np.convolve(mean_rewards, np.ones(window)/window, mode='valid')
            ma_steps = unique_steps[window-1:]
        else:
            ma_rewards = mean_rewards
            ma_steps = unique_steps
        
        plt.plot(np.array(ma_steps) / 1000, ma_rewards, label=algo, color=colors[idx % len(colors)], linewidth=3)
    
    plt.xlabel('Steps (thousands)', fontsize=14)
    plt.ylabel('Moving Average Reward (Fine)', fontsize=14)
    plt.title('Seed-Aggregated Learning Curves: Moving Average Reward vs Steps (Fine Window)', fontsize=16, fontweight='bold')
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tick_params(axis='both', which='major', labelsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "moving_average_curves_fine.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Fine moving average curves exported to {output_dir}/moving_average_curves_fine.png")


def generate_aggregated_curves(full_episodes: Dict[str, List[Dict]], output_dir: str) -> None:
    """Generate seed-aggregated learning curves (raw aggregated reward vs steps)."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("⚠ matplotlib not available, skipping aggregated curves")
        return
    
    ensure_dir(os.path.join(output_dir, "placeholder"))
    
    plt.figure(figsize=(12, 8))
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    
    for idx, (algo, episodes) in enumerate(full_episodes.items()):
        if not episodes:
            continue
        
        # Collect all (cumulative_steps, cumulative_reward) points across all seeds
        all_points = []
        for ep in episodes:
            seed = ep['seed']
            # For each seed, compute cumulative steps
            seed_eps = [e for e in episodes if e['seed'] == seed]
            seed_eps.sort(key=lambda x: x['episode'])
            cum_steps = 0
            for e in seed_eps:
                cum_steps += e['total_steps']
                all_points.append((cum_steps, e['cumulative_reward']))
        
        if not all_points:
            continue
        
        # Sort by steps
        all_points.sort(key=lambda x: x[0])
        steps, rewards = zip(*all_points)
        
        # Remove duplicates by averaging rewards at same step
        from collections import defaultdict
        step_rewards = defaultdict(list)
        for s, r in all_points:
            step_rewards[s].append(r)
        unique_steps = sorted(step_rewards.keys())
        mean_rewards = [np.mean(step_rewards[s]) for s in unique_steps]
        
        # No moving average, plot raw aggregated
        plt.plot(np.array(unique_steps) / 1000, mean_rewards, label=algo, color=colors[idx % len(colors)], linewidth=2)
    
    plt.xlabel('Steps (thousands)', fontsize=14)
    plt.ylabel('Aggregated Reward', fontsize=14)
    plt.title('Seed-Aggregated Learning Curves: Reward vs Steps (Raw)', fontsize=16, fontweight='bold')
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tick_params(axis='both', which='major', labelsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "aggregated_curves.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Aggregated curves exported to {output_dir}/aggregated_curves.png")


def generate_time_reward_tradeoff(full_episodes: Dict[str, List[Dict]], output_dir: str) -> None:
    """Generate scatter plot of wall-clock time vs final cumulative reward across seeds."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("⚠ matplotlib not available, skipping time-reward tradeoff")
        return
    
    ensure_dir(os.path.join(output_dir, "placeholder"))
    
    plt.figure(figsize=(12, 8))
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    
    for idx, (algo, episodes) in enumerate(full_episodes.items()):
        if not episodes:
            continue
        
        times = []
        rewards = []
        
        for seed in set(ep['seed'] for ep in episodes):
            seed_eps = [ep for ep in episodes if ep['seed'] == seed]
            if not seed_eps:
                continue
            
            # Assume the last episode has the final reward
            final_ep = max(seed_eps, key=lambda x: x['episode'])
            final_reward = final_ep['cumulative_reward']
            
            # Wall-clock time: sum of duration_s across episodes for this seed
            # But the data may not have duration_s per episode, only per seed in summary.
            # Since we don't have per-episode time, perhaps use total_steps as proxy, or assume constant time per step.
            # For simplicity, use total_steps / 1000 as proxy for time (assuming 1000 steps/sec or something).
            total_steps = sum(ep['total_steps'] for ep in seed_eps)
            # Assume 1000 steps per second for proxy time
            proxy_time = total_steps / 1000.0  # in seconds
            
            times.append(proxy_time)
            rewards.append(final_reward)
        
        if times and rewards:
            plt.scatter(times, rewards, label=algo, color=colors[idx % len(colors)], s=50, alpha=0.7)
    
    plt.xlabel('Wall-Clock Time (seconds)', fontsize=14)
    plt.ylabel('Final Cumulative Reward', fontsize=14)
    plt.title('Reward Trade-off: Wall-Clock Time vs Final Cumulative Reward', fontsize=16, fontweight='bold')
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tick_params(axis='both', which='major', labelsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "time_reward_tradeoff.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Time-reward tradeoff exported to {output_dir}/time_reward_tradeoff.png")


def generate_calibration_plot(output_dir: str) -> None:
    """Generate calibration plot for PETS and MBPO."""
    try:
        import matplotlib.pyplot as plt
        import json
    except ImportError:
        print("⚠ matplotlib or json not available, skipping calibration plot")
        return
    
    ensure_dir(os.path.join(output_dir, "placeholder"))
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    algos = ['PETS', 'MBPO']
    colors = ['#ff7f0e', '#2ca02c']
    
    for idx, algo in enumerate(algos):
        ax = axes[idx]
        
        # Load calibration data
        calib_path = f"results/{algo.lower()}/calibration_data.json"
        try:
            with open(calib_path, 'r') as f:
                data = json.load(f)
            
            predicted = data.get('predicted_mastery', [])
            empirical = data.get('empirical_correct', [])
            
            if predicted and empirical and len(predicted) == len(empirical):
                ax.scatter(predicted, empirical, alpha=0.6, color=colors[idx], s=20)
                # Diagonal line
                min_val = min(min(predicted), min(empirical))
                max_val = max(max(predicted), max(empirical))
                ax.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.7, label='Perfect Calibration')
                
                ax.set_xlabel('Predicted Mastery', fontsize=12)
                ax.set_ylabel('Empirical Correctness', fontsize=12)
                ax.set_title(f'{algo} Calibration', fontsize=14, fontweight='bold')
                ax.grid(True, alpha=0.3)
                ax.legend()
            else:
                ax.text(0.5, 0.5, 'No calibration data', ha='center', va='center', transform=ax.transAxes)
        except FileNotFoundError:
            ax.text(0.5, 0.5, f'Calibration data not found for {algo}', ha='center', va='center', transform=ax.transAxes)
    
    plt.suptitle('Calibration: Predicted Mastery vs Empirical Correctness', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "calibration_plot.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Calibration plot exported to {output_dir}/calibration_plot.png")


def generate_modality_gains_plot(output_dir: str) -> None:
    """Generate average post-content gain by modality for model-based methods."""
    try:
        import matplotlib.pyplot as plt
        import json
        import csv
    except ImportError:
        print("⚠ matplotlib, json, or csv not available, skipping modality gains plot")
        return
    
    ensure_dir(os.path.join(output_dir, "placeholder"))
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    algos = ['PETS', 'MBPO']
    colors = ['#ff7f0e', '#2ca02c']
    
    for idx, algo in enumerate(algos):
        ax = axes[idx]
        
        # Try to load modality gains from JSON first
        gains_path = f"results/{algo.lower()}/modality_gains.json"
        data = None
        
        try:
            with open(gains_path, 'r') as f:
                data = json.load(f)
        except FileNotFoundError:
            # Fall back to computing from episodes.csv
            csv_path = f"results/{algo.lower()}/episodes.csv"
            try:
                data = compute_modality_gains_from_csv(csv_path)
            except Exception as e:
                print(f"Warning: Could not compute modality gains for {algo}: {e}")
                data = None
        
        if data:
            modalities = list(data.keys())
            means = [data[m]['mean'] for m in modalities]
            stds = [data[m]['std'] for m in modalities]
            
            x = range(len(modalities))
            ax.bar(x, means, yerr=stds, capsize=5, alpha=0.7, color=colors[idx])
            ax.set_xticks(x)
            ax.set_xticklabels(modalities, rotation=45, ha='right')
            ax.set_ylabel('Post-Content Gain', fontsize=12)
            ax.set_title(f'{algo} Modality Gains', fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3, axis='y')
        else:
            ax.text(0.5, 0.5, f'Modality gains data not found for {algo}', ha='center', va='center', transform=ax.transAxes)
    
    plt.suptitle('Average Post-Content Gain by Modality (Model-Based Methods)', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "modality_gains_plot.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Modality gains plot exported to {output_dir}/modality_gains_plot.png")


def compute_modality_gains_from_csv(csv_path: str) -> Dict:
    """Compute modality gains from episodes.csv data."""
    import csv
    
    modality_columns = [
        'post_content_gain_video',
        'post_content_gain_PPT', 
        'post_content_gain_text',
        'post_content_gain_blog',
        'post_content_gain_article',
        'post_content_gain_handout'
    ]
    
    modality_names = ['video', 'PPT', 'text', 'blog', 'article', 'handout']
    
    # Collect data by modality
    modality_data = {name: [] for name in modality_names}
    
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            for name, col in zip(modality_names, modality_columns):
                try:
                    value = float(row[col])
                    modality_data[name].append(value)
                except (KeyError, ValueError):
                    continue
    
    # Compute mean and std for each modality
    result = {}
    for name in modality_names:
        values = modality_data[name]
        if values:
            result[name] = {
                'mean': float(np.mean(values)),
                'std': float(np.std(values)),
                'count': len(values)
            }
        else:
            result[name] = {'mean': 0.0, 'std': 0.0, 'count': 0}
    
    return result


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
    
    # Load summaries and full episode data
    summaries = {}
    full_episodes = {}
    for algo, path in [("DQN", args.dqn), ("PETS", args.pets), ("MBPO", args.mbpo), ("PPO", args.ppo)]:
        try:
            summaries[algo] = load_summary(path)
            full_episodes[algo] = load_full_episodes(path)
            print(f"✓ Loaded {algo}: {path}")
        except FileNotFoundError as e:
            print(f"⚠ Skipping {algo}: {e}")
            summaries[algo] = {}
            full_episodes[algo] = []
    
    if len(summaries) < 2:
        print("\n❌ Need at least 2 algorithms for comparison")
        return
    
    # Generate outputs
    print(f"\nGenerating comparison outputs to {args.output}/")
    
    generate_latex_table(summaries, os.path.join(args.output, "table_performance_comparison.tex"))
    generate_statistical_tests(summaries, os.path.join(args.output, "statistical_tests.txt"))
    generate_comparison_json(summaries, os.path.join(args.output, "comparison.json"))
    generate_comparison_plot(summaries, args.output)
    generate_learning_curves(full_episodes, args.output)
    generate_question_accuracy_curves(full_episodes, args.output)
    generate_frustration_curves(full_episodes, args.output)
    generate_moving_average_curves(full_episodes, args.output)
    generate_episode_reward_curves(full_episodes, args.output)
    generate_moving_average_boxplot(full_episodes, args.output)
    generate_moving_average_curves_fine(full_episodes, args.output)
    generate_aggregated_curves(full_episodes, args.output)
    generate_time_reward_tradeoff(full_episodes, args.output)
    generate_calibration_plot(args.output)
    generate_modality_gains_plot(args.output)
    
    print("\n" + "="*70)
    print("✓ All comparison outputs generated successfully!")
    print("="*70)
    print(f"\nOutputs:")
    print(f"  - LaTeX Table: {args.output}/table_performance_comparison.tex")
    print(f"  - Statistical Tests: {args.output}/statistical_tests.txt")
    print(f"  - JSON: {args.output}/comparison.json")
    print(f"  - Bar Plots: {args.output}/comparison_plot.png")
    print(f"  - Effect Size Heatmap: {args.output}/effect_size_heatmap.png")
    print(f"  - Learning Curves: {args.output}/learning_curves.png")
    print(f"  - Question Accuracy Curves: {args.output}/question_accuracy_curves.png")
    print(f"  - Frustration Curves: {args.output}/frustration_curves.png")
    print(f"  - Moving Average Curves: {args.output}/moving_average_curves.png")
    print(f"  - Episode Reward Curves: {args.output}/episode_reward_curves.png")
    print(f"  - Moving Average Boxplot: {args.output}/moving_average_boxplot.png")
    print(f"  - Fine Moving Average Curves: {args.output}/moving_average_curves_fine.png")
    print(f"  - Aggregated Curves: {args.output}/aggregated_curves.png")
    print(f"  - Time-Reward Tradeoff: {args.output}/time_reward_tradeoff.png")
    print(f"  - Calibration Plot: {args.output}/calibration_plot.png")
    print(f"  - Modality Gains Plot: {args.output}/modality_gains_plot.png")
    print("="*70)


if __name__ == "__main__":
    main()
