#!/usr/bin/env python3
"""
Regenerate figures from existing multiseed_episodes.csv without retraining.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse
from pathlib import Path


def moving_average(values, k=10):
    if len(values) < k:
        return []
    return np.convolve(values, np.ones(k) / k, mode='valid')


def plot_learning_curve(csv_path: str, fig_path: str, window: int = 10):
    """Plot learning curve (moving avg reward) per seed."""
    df = pd.read_csv(csv_path)
    Path(fig_path).parent.mkdir(parents=True, exist_ok=True)
    
    plt.figure(figsize=(7, 4))
    for seed in df['seed'].unique():
        returns = df[df['seed'] == seed]['return'].values.tolist()
        ma = moving_average(returns, k=window)
        if len(ma) > 0:
            plt.plot(ma, alpha=0.4, linewidth=1)
    
    # Plot mean moving average
    seeds_data = []
    for seed in sorted(df['seed'].unique()):
        returns = df[df['seed'] == seed]['return'].values.tolist()
        ma = moving_average(returns, k=window)
        seeds_data.append(ma)
    
    if seeds_data:
        max_len = max(len(ma) for ma in seeds_data)
        stacked = [ma[:max_len] for ma in seeds_data if len(ma) >= max_len]
        if stacked:
            mean_ma = np.mean(np.stack(stacked), axis=0)
            plt.plot(mean_ma, color="black", linewidth=2, label=f"Mean ({window}-MA)")
            plt.legend()
    
    plt.title("Learning Curve: Moving Avg Reward")
    plt.xlabel(f"Episode ({window}-MA window)")
    plt.ylabel("Reward")
    plt.tight_layout()
    plt.savefig(fig_path, dpi=200)
    plt.close()
    print(f"✓ Saved {fig_path}")


def plot_post_content_gain_by_modality(csv_path: str, fig_path: str):
    """Plot mean post-content gain by modality."""
    df = pd.read_csv(csv_path)
    Path(fig_path).parent.mkdir(parents=True, exist_ok=True)
    
    keys = ["post_content_gain_video", "post_content_gain_PPT", "post_content_gain_text",
            "post_content_gain_blog", "post_content_gain_article", "post_content_gain_handout"]
    labels = ["video", "PPT", "text", "blog", "article", "handout"]
    
    means = [df[k].mean() for k in keys]
    
    plt.figure(figsize=(7, 4))
    plt.bar(labels, means, color="#4c78a8")
    plt.ylabel("Mean Post-Content Gain")
    plt.xticks(rotation=30)
    plt.title("Post-Content Gain by Modality (DQN)")
    plt.tight_layout()
    plt.savefig(fig_path, dpi=200)
    plt.close()
    print(f"✓ Saved {fig_path}")


def plot_variance_across_seeds(csv_path: str, fig_path: str):
    """Plot reward variance across seeds."""
    df = pd.read_csv(csv_path)
    Path(fig_path).parent.mkdir(parents=True, exist_ok=True)
    
    # Sum return per seed
    rewards = df.groupby('seed')['return'].sum().values.tolist()
    
    plt.figure(figsize=(6, 4))
    plt.boxplot(rewards, vert=False)
    plt.xlabel("Cumulative Reward")
    plt.title("Reward Variance Across Seeds (DQN)")
    plt.tight_layout()
    plt.savefig(fig_path, dpi=200)
    plt.close()
    print(f"✓ Saved {fig_path}")


def plot_compute_vs_reward(json_path: str, fig_path: str):
    """Plot compute (wall-clock time) vs reward per seed.
    
    Reads per-seed elapsed time from multiseed_summary.json.
    """
    import json
    
    Path(fig_path).parent.mkdir(parents=True, exist_ok=True)
    
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    per_seed_elapsed = data.get("per_seed_elapsed_sec", [])
    
    # Sum rewards per seed from CSV
    import os
    csv_dir = os.path.dirname(json_path)
    csv_path = os.path.join(csv_dir, "multiseed_episodes.csv")
    df = pd.read_csv(csv_path)
    per_seed_reward = df.groupby('seed')['return'].sum().values.tolist()
    
    plt.figure(figsize=(6.5, 4))
    plt.scatter(per_seed_elapsed, per_seed_reward, c="#72b7b2", s=80)
    plt.xlabel("Wall-clock per seed (s)")
    plt.ylabel("Cumulative reward per seed")
    plt.title("Compute vs Reward (DQN)")
    plt.tight_layout()
    plt.savefig(fig_path, dpi=200)
    plt.close()
    print(f"✓ Saved {fig_path}")


def main():
    parser = argparse.ArgumentParser(description="Regenerate figures from multiseed_episodes.csv")
    parser.add_argument("--csv", type=str, default="logs/multiseed_episodes.csv")
    parser.add_argument("--json", type=str, default="logs/multiseed_summary.json")
    parser.add_argument("--fig-learning", type=str, default="figures/learning_curve_moving_avg_reward.png")
    parser.add_argument("--fig-modality", type=str, default="figures/post_content_gain_by_modality.png")
    parser.add_argument("--fig-variance", type=str, default="figures/variance_across_seeds.png")
    parser.add_argument("--fig-compute", type=str, default="figures/compute_vs_reward.png")
    args = parser.parse_args()
    
    plot_learning_curve(args.csv, args.fig_learning)
    plot_post_content_gain_by_modality(args.csv, args.fig_modality)
    plot_variance_across_seeds(args.csv, args.fig_variance)
    plot_compute_vs_reward(args.json, args.fig_compute)
    print("\nDone. All figures regenerated from CSV.")


if __name__ == "__main__":
    main()
