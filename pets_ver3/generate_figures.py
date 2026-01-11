"""
Generate all figures required for the Elsevier paper from training results.
This script creates publication-quality figures matching the paper specifications.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple

# Set publication quality style
plt.style.use('seaborn-v0_8-paper')
sns.set_context("paper", font_scale=1.2)
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']


def load_results():
    """Load all exported results from JSON files"""
    results_dir = Path("results")
    
    data = {}
    files_to_load = [
        "learning_curve_data.json",
        "performance_summary.json",
        "modality_gains.json",
        "calibration_data.json",
        "variance_data.json"
    ]
    
    for filename in files_to_load:
        filepath = results_dir / filename
        if filepath.exists():
            with open(filepath, 'r') as f:
                key = filename.replace('_data', '').replace('.json', '')
                data[key] = json.load(f)
        else:
            print(f"Warning: {filename} not found")
    
    return data


def plot_learning_curve(data: Dict):
    """Figure: Learning curve (batch mean reward across seeds)"""
    fig, ax = plt.subplots(figsize=(8, 5))
    
    episodes = np.array(data['learning_curve']['episodes'])
    mean_reward = np.array(data['learning_curve']['mean_reward'])
    std_reward = np.array(data['learning_curve']['std_reward'])
    
    ax.plot(episodes, mean_reward, linewidth=2, color='#2E86AB', label='PETS')
    ax.fill_between(episodes, 
                     mean_reward - std_reward, 
                     mean_reward + std_reward,
                     alpha=0.3, color='#2E86AB')
    
    ax.set_xlabel('Episode', fontsize=12)
    ax.set_ylabel('Cumulative Reward', fontsize=12)
    ax.set_title('PPO Learning Curve (Mean ± 1 SD)', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', frameon=True, shadow=True)
    ax.grid(True, alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig('results/learning_curve.png', bbox_inches='tight')
    plt.close()
    print("✓ Generated learning_curve.png")


def plot_modality_gains(data: Dict):
    """Figure: Post-content gain by modality"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    modality_data = data['modality_gains']
    modalities = ['video', 'PPT', 'text', 'blog', 'article', 'handout']
    means = [modality_data[mod]['mean'] for mod in modalities]
    stds = [modality_data[mod]['std'] for mod in modalities]
    
    x_pos = np.arange(len(modalities))
    colors = ['#E63946', '#F1FAEE', '#A8DADC', '#457B9D', '#1D3557', '#2A9D8F']
    
    bars = ax.bar(x_pos, means, yerr=stds, capsize=5, 
                  color=colors, edgecolor='black', linewidth=1.5, alpha=0.8)
    
    ax.set_xlabel('Content Modality', fontsize=12, fontweight='bold')
    ax.set_ylabel('Post-Content Mastery Gain', fontsize=12, fontweight='bold')
    ax.set_title('Post-Content Gain by Modality (PETS)', fontsize=14, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(modalities, rotation=0)
    ax.grid(True, axis='y', alpha=0.3, linestyle='--')
    ax.set_ylim(0, max(means) * 1.3)
    
    # Add value labels on bars
    for i, (bar, mean, std) in enumerate(zip(bars, means, stds)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + std,
                f'{mean:.3f}',
                ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('results/modality_gains.png', bbox_inches='tight')
    plt.close()
    print("✓ Generated modality_gains.png")


def plot_calibration(data: Dict):
    """Figure: Calibration curve (predicted mastery vs observed correctness)"""
    if 'calibration' not in data or not data['calibration']['predicted_mastery']:
        print("⚠ Skipping calibration plot (no data)")
        return
    
    fig, ax = plt.subplots(figsize=(8, 8))
    
    predicted = np.array(data['calibration']['predicted_mastery'])
    actual = np.array(data['calibration']['empirical_correct'])
    
    # Bin into deciles
    num_bins = 10
    bins = np.linspace(0, 1, num_bins + 1)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    
    binned_accuracy = []
    binned_confidence = []
    
    for i in range(num_bins):
        mask = (predicted >= bins[i]) & (predicted < bins[i+1])
        if mask.sum() > 0:
            binned_accuracy.append(actual[mask].mean())
            binned_confidence.append(predicted[mask].mean())
        else:
            binned_accuracy.append(np.nan)
            binned_confidence.append(bin_centers[i])
    
    # Perfect calibration line
    ax.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Perfect Calibration', alpha=0.7)
    
    # Actual calibration
    valid_mask = ~np.isnan(binned_accuracy)
    ax.plot(np.array(binned_confidence)[valid_mask], 
            np.array(binned_accuracy)[valid_mask], 
            'o-', linewidth=2.5, markersize=8, color='#E63946',
            label='PETS', markeredgecolor='black', markeredgewidth=1)
    
    ax.set_xlabel('Predicted Mastery', fontsize=12, fontweight='bold')
    ax.set_ylabel('Empirical Correctness', fontsize=12, fontweight='bold')
    ax.set_title('Calibration Curve', fontsize=14, fontweight='bold')
    ax.legend(loc='upper left', frameon=True, shadow=True, fontsize=11)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal')
    
    plt.tight_layout()
    plt.savefig('results/calibration.png', bbox_inches='tight')
    plt.close()
    print("✓ Generated calibration.png")


def plot_variance_across_seeds(data: Dict):
    """Figure: Variance of cumulative reward across random seeds"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    seed_returns = data['variance']['seed_returns']
    episodes = data['variance']['episodes']
    
    # Compute variance at each episode
    returns_array = []
    for returns in seed_returns:
        returns_array.append(returns)
    
    # Pad to same length
    max_len = max(len(r) for r in returns_array)
    padded_returns = []
    for returns in returns_array:
        padded = list(returns) + [np.nan] * (max_len - len(returns))
        padded_returns.append(padded)
    
    returns_matrix = np.array(padded_returns)
    variance = np.nanvar(returns_matrix, axis=0)
    mean_returns = np.nanmean(returns_matrix, axis=0)
    
    # Plot variance
    ax.plot(episodes[:len(variance)], variance, linewidth=2, 
            color='#E63946', label='PETS Variance')
    ax.fill_between(episodes[:len(variance)], 0, variance, alpha=0.3, color='#E63946')
    
    ax.set_xlabel('Episode', fontsize=12, fontweight='bold')
    ax.set_ylabel('Variance of Cumulative Reward', fontsize=12, fontweight='bold')
    ax.set_title('Reward Variance Across Random Seeds', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', frameon=True, shadow=True)
    ax.grid(True, alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig('results/variance_bands_all.png', bbox_inches='tight')
    plt.close()
    print("✓ Generated variance_bands_all.png")


def plot_time_to_mastery(data: Dict):
    """Figure: Time-to-mastery comparison (bar chart with CI)"""
    fig, ax = plt.subplots(figsize=(6, 6))
    
    perf = data['performance_summary']
    
    # Single algorithm for now (PETS)
    algorithms = ['PETS']
    means = [perf['time_to_mastery_mean']]
    stds = [perf['time_to_mastery_std']]
    
    x_pos = np.arange(len(algorithms))
    bars = ax.bar(x_pos, means, yerr=stds, capsize=10, 
                  color='#2E86AB', edgecolor='black', linewidth=2, alpha=0.8)
    
    ax.set_ylabel('Time to Mastery (steps)', fontsize=12, fontweight='bold')
    ax.set_title('Time-to-Mastery (Mean ± 95% CI)', fontsize=14, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(algorithms)
    ax.grid(True, axis='y', alpha=0.3, linestyle='--')
    
    # Add value labels
    for bar, mean, std in zip(bars, means, stds):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + std,
                f'{mean:.1f}±{std:.1f}',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('results/time_to_mastery_all.png', bbox_inches='tight')
    plt.close()
    print("✓ Generated time_to_mastery_all.png")


def plot_compute_vs_reward(data: Dict):
    """Figure: Compute-performance trade-off"""
    fig, ax = plt.subplots(figsize=(8, 6))
    
    perf = data['performance_summary']
    
    # Single point for PETS (would have multiple algorithms in full paper)
    wall_clock = [perf['wall_clock_mean_s']]
    rewards = [perf['cumulative_reward_mean']]
    algorithms = ['PETS']
    colors = ['#E63946']
    
    ax.scatter(wall_clock, rewards, s=200, c=colors, 
              edgecolor='black', linewidth=2, alpha=0.8, zorder=3)
    
    # Add labels
    for i, alg in enumerate(algorithms):
        ax.annotate(alg, (wall_clock[i], rewards[i]), 
                   xytext=(10, 10), textcoords='offset points',
                   fontsize=11, fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7))
    
    ax.set_xlabel('Wall-Clock Time (seconds)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Final Cumulative Reward', fontsize=12, fontweight='bold')
    ax.set_title('Compute-Reward Trade-off', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig('results/compute_vs_reward.png', bbox_inches='tight')
    plt.close()
    print("✓ Generated compute_vs_reward.png")


def generate_all_figures():
    """Generate all figures required for the paper"""
    print("="*60)
    print("GENERATING PAPER FIGURES")
    print("="*60)
    
    # Load results
    data = load_results()
    
    if not data:
        print("ERROR: No results data found. Run pets_train.py first.")
        return
    
    # Create figures directory
    Path("results").mkdir(exist_ok=True)
    
    # Generate each figure
    try:
        plot_learning_curve(data)
        plot_modality_gains(data)
        plot_calibration(data)
        plot_variance_across_seeds(data)
        plot_time_to_mastery(data)
        plot_compute_vs_reward(data)
    except Exception as e:
        print(f"ERROR generating figures: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print("\n" + "="*60)
    print("FIGURE GENERATION COMPLETE")
    print("="*60)
    print("Figures saved to results/ directory:")
    print("  - learning_curve.png")
    print("  - modality_gains.png")
    print("  - calibration.png")
    print("  - variance_bands_all.png")
    print("  - time_to_mastery_all.png")
    print("  - compute_vs_reward.png")
    print("\nThese figures can be directly included in the Elsevier paper.")


if __name__ == "__main__":
    generate_all_figures()
