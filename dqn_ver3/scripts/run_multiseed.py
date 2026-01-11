import argparse
import os
import time
import json
import csv
from typing import List

import numpy as np
import matplotlib.pyplot as plt

# Import training functions from the workspace
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from train_dqn import run_training, summarize_across_seeds
from shared_config import UNIFIED_SEEDS, UNIFIED_EPISODES


def ensure_dir(path: str):
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)


def moving_average(x: List[float], k: int = 10) -> np.ndarray:
    if not x:
        return np.array([])
    x = np.array(x, dtype=float)
    if len(x) < k:
        k = max(1, len(x))
    return np.convolve(x, np.ones(k) / k, mode="valid")


def aggregate_returns_across_seeds(results: List[dict]) -> List[List[float]]:
    return [r.get("returns", []) for r in results]


def write_combined_episode_csv(path: str, results: List[dict]):
    ensure_dir(path)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        header = [
            "seed",
            "episode",
            "return",
            "ttm",
            "total_steps",
            "final_mastery",
            "cumulative_reward",
            "question_accuracy",
            "content_rate",
            "blueprint_adherence",
            "post_content_gain",
            "post_content_gain_video",
            "post_content_gain_PPT",
            "post_content_gain_text",
            "post_content_gain_blog",
            "post_content_gain_article",
            "post_content_gain_handout",
            "mean_frustration",
        ]
        writer.writerow(header)
        for seed_idx, r in enumerate(results):
            returns = r.get("returns", [])
            ttms = r.get("time_to_mastery", [])
            ems = r.get("episode_metrics", [])
            for ep_idx, (ret, ttm, em) in enumerate(zip(returns, ttms, ems), start=1):
                row = [
                    seed_idx,
                    ep_idx,
                    float(ret),
                    int(ttm) if ttm is not None else 0,
                    int(em.get("total_steps", 0)),
                    float(em.get("final_mastery", 0.0)),
                    float(em.get("cumulative_reward", 0.0)),
                    float(em.get("question_accuracy", 0.0)),
                    float(em.get("content_rate", 0.0)),
                    float(em.get("blueprint_adherence", 0.0)),
                    float(em.get("post_content_gain", 0.0)),
                    float(em.get("post_content_gain_by_modality", {}).get("video", 0.0)),
                    float(em.get("post_content_gain_by_modality", {}).get("PPT", 0.0)),
                    float(em.get("post_content_gain_by_modality", {}).get("text", 0.0)),
                    float(em.get("post_content_gain_by_modality", {}).get("blog", 0.0)),
                    float(em.get("post_content_gain_by_modality", {}).get("article", 0.0)),
                    float(em.get("post_content_gain_by_modality", {}).get("handout", 0.0)),
                    float(em.get("mean_frustration", 0.0)),
                ]
                writer.writerow(row)


def plot_learning_curve(fig_path: str, returns_across_seeds: List[List[float]], window: int = 10):
    ensure_dir(fig_path)
    plt.figure(figsize=(7, 4))
    # Plot per-seed moving average
    for rs in returns_across_seeds:
        ma = moving_average(rs, k=window)
        if len(ma) > 0:
            plt.plot(ma, alpha=0.4, linewidth=1)
    # Plot mean moving average
    max_len = max((len(moving_average(rs, k=window)) for rs in returns_across_seeds), default=0)
    if max_len > 0:
        stacked = []
        for rs in returns_across_seeds:
            ma = moving_average(rs, k=window)
            if len(ma) >= max_len:
                stacked.append(ma[:max_len])
        if stacked:
            mean_ma = np.mean(np.stack(stacked), axis=0)
            plt.plot(mean_ma, color="black", linewidth=2, label="Mean ({}-MA)".format(window))
            plt.legend()
    plt.title("Learning Curve: Moving Avg Reward")
    plt.xlabel("Episode ({}-MA window)".format(window))
    plt.ylabel("Reward")
    plt.tight_layout()
    plt.savefig(fig_path, dpi=200)
    plt.close()


def plot_post_content_gain_by_modality(fig_path: str, results: List[dict]):
    ensure_dir(fig_path)
    keys = ["video", "PPT", "text", "blog", "article", "handout"]
    values = {k: [] for k in keys}
    for r in results:
        for em in r.get("episode_metrics", []):
            by_mod = em.get("post_content_gain_by_modality", {})
            for k in keys:
                values[k].append(by_mod.get(k, 0.0))
    means = [np.mean(values[k]) if values[k] else 0.0 for k in keys]
    plt.figure(figsize=(7, 4))
    plt.bar(keys, means, color="#4c78a8")
    plt.ylabel("Mean Post-Content Gain")
    plt.xticks(rotation=30)
    plt.title("Post-Content Gain by Modality (DQN)")
    plt.tight_layout()
    plt.savefig(fig_path, dpi=200)
    plt.close()


def plot_variance_across_seeds(fig_path: str, results: List[dict]):
    ensure_dir(fig_path)
    # Use cumulative reward per seed
    rewards = [float(np.sum(r.get("returns", []))) for r in results]
    plt.figure(figsize=(6, 4))
    plt.boxplot(rewards, vert=False)
    plt.xlabel("Cumulative Reward")
    plt.title("Reward Variance Across Seeds (DQN)")
    plt.tight_layout()
    plt.savefig(fig_path, dpi=200)
    plt.close()


def plot_compute_vs_reward(fig_path: str, per_seed_elapsed: List[float], per_seed_reward: List[float]):
    ensure_dir(fig_path)
    import matplotlib.pyplot as plt
    plt.figure(figsize=(6.5, 4))
    plt.scatter(per_seed_elapsed, per_seed_reward, c="#72b7b2")
    plt.xlabel("Wall-clock per seed (s)")
    plt.ylabel("Cumulative reward per seed")
    plt.title("Compute vs Reward (DQN)")
    plt.tight_layout()
    plt.savefig(fig_path, dpi=200)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Run multi-seed DQN and produce summary + figures (aligned with PETS_VER3)")
    parser.add_argument("--seeds", type=int, nargs="*", default=UNIFIED_SEEDS, 
                        help=f"Random seeds (default: {UNIFIED_SEEDS})")
    parser.add_argument("--episodes", type=int, default=UNIFIED_EPISODES,
                        help=f"Number of episodes per seed (default: {UNIFIED_EPISODES} for ~30k total steps)")
    parser.add_argument("--steps", type=int, default=140,
                        help="Max steps per episode (default 140)")
    parser.add_argument("--start-steps", type=int, default=5000,
                        help="Warmup steps before learning (default 5000)")
    parser.add_argument("--out-json", type=str, default="results/dqn/summary.json")
    parser.add_argument("--out-csv", type=str, default="results/dqn/episodes.csv")
    parser.add_argument("--fig-learning", type=str, default="results/dqn/figures/learning_curve_moving_avg_reward.png")
    parser.add_argument("--fig-modality", type=str, default="results/dqn/figures/post_content_gain_by_modality.png")
    parser.add_argument("--fig-variance", type=str, default="results/dqn/figures/variance_across_seeds.png")
    parser.add_argument("--fig-compute", type=str, default="results/dqn/figures/compute_vs_reward.png")
    parser.add_argument("--total-steps", type=int, default=None, help="Per-seed total env step budget")
    args = parser.parse_args()

    results = []
    per_seed_elapsed = []
    for seed in args.seeds:
        t0 = time.time()
        r = run_training(
            num_episodes=args.episodes,
            max_steps_per_episode=args.steps,
            start_steps=args.start_steps,
            seed=seed,
            total_steps_budget=args.total_steps,
        )
        per_seed_elapsed.append(time.time() - t0)
        results.append(r)

    summary = summarize_across_seeds(results)

    # Write summary JSON
    ensure_dir(args.out_json)
    with open(args.out_json, "w") as f:
        json.dump({"summary": summary, "per_seed_elapsed_sec": per_seed_elapsed}, f, indent=2)

    # Write combined per-episode CSV
    write_combined_episode_csv(args.out_csv, results)

    # Figures
    returns_across_seeds = aggregate_returns_across_seeds(results)
    plot_learning_curve(args.fig_learning, returns_across_seeds)
    plot_post_content_gain_by_modality(args.fig_modality, results)
    plot_variance_across_seeds(args.fig_variance, results)
    # Compute vs reward figure
    per_seed_reward = [float(np.sum(r.get("returns", []))) for r in results]
    plot_compute_vs_reward(args.fig_compute, per_seed_elapsed, per_seed_reward)

    print("Done. Summary:")
    print(json.dumps(summary, indent=2))
    print("Per-seed elapsed (s):", per_seed_elapsed)


if __name__ == "__main__":
    main()
