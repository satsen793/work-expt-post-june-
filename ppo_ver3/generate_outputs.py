import argparse
import csv
import json
import math
import os
import random
from collections import defaultdict
from statistics import mean, pstdev
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch

from ppo_train import (
    AdaptiveLearningEnv,
    CONFIG,
    PPOAgent,
    compute_blueprint_adherence,
    compute_content_rate,
    compute_cumulative_reward,
    compute_final_mastery,
    compute_mean_frustration,
    compute_post_content_gain,
    compute_question_accuracy,
    compute_time_to_mastery,
)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def train_one_seed(seed: int, cfg: Dict, max_iters: int) -> Tuple[PPOAgent, List[Tuple[int, float]]]:
    CONFIG.update(cfg)
    set_seed(seed)
    env = AdaptiveLearningEnv()
    agent = PPOAgent(CONFIG)
    total_iterations = max_iters
    rewards_curve = []
    for it in range(total_iterations):
        batch = collect_rollout(env, agent, CONFIG["rollout_steps"])
        agent.update(batch)
        mean_reward = batch.rewards.mean().item()
        total_steps = (it + 1) * CONFIG["rollout_steps"]
        rewards_curve.append((total_steps, mean_reward))
    return agent, rewards_curve


def evaluate_policy(agent: PPOAgent, episodes: int, seed: int) -> Dict:
    metrics_per_ep: List[Dict] = []
    calib_points: List[Tuple[float, int]] = []
    modality_gains = defaultdict(list)
    for ep in range(episodes):
        env = AdaptiveLearningEnv()
        state = env.reset(seed + ep)
        done = False
        while not done:
            action, _, _ = agent.select_action(state)
            state, _, done, _ = env.step(action)
        ep_log = env.episode_log
        metrics_per_ep.append({
            "cumulative_reward": compute_cumulative_reward(ep_log),
            "time_to_mastery": compute_time_to_mastery(ep_log),
            "post_content_gain": compute_post_content_gain(ep_log),
            "blueprint_adherence": compute_blueprint_adherence(ep_log),
            "question_accuracy": compute_question_accuracy(ep_log),
            "content_rate": compute_content_rate(ep_log),
            "final_mastery": compute_final_mastery(ep_log),
            "mean_frustration": compute_mean_frustration(ep_log),
        })
        for step in ep_log:
            if step.get("action_type") == "question":
                lo = step.get("result", {}).get("lo")
                if lo is not None:
                    predicted = float(step["mastery_vector"][lo])
                    actual = 1 if step.get("correct") else 0
                    calib_points.append((predicted, actual))
            if step.get("action_type") == "content":
                mod = step.get("result", {}).get("modality")
                if mod is not None:
                    modality_gains[mod].append(step.get("mastery_gain", 0.0))
    return {
        "episodes": metrics_per_ep,
        "calibration": calib_points,
        "modality_gains": {k: modality_gains[k] for k in modality_gains},
    }


def collect_rollout(env: AdaptiveLearningEnv, agent: PPOAgent, steps: int):
    from ppo_train import collect_rollout as _collect
    return _collect(env, agent, steps)


def aggregate_metrics(all_metrics: List[Dict]) -> Dict:
    agg = {}
    keys = [
        "cumulative_reward",
        "time_to_mastery",
        "post_content_gain",
        "blueprint_adherence",
        "question_accuracy",
        "content_rate",
        "final_mastery",
        "mean_frustration",
    ]
    for k in keys:
        values = [m for seed in all_metrics for m in seed["episodes"] if m[k] is not None]
        vals = [m[k] for m in values]
        agg[k] = {
            "mean": float(mean(vals)) if vals else 0.0,
            "sd": float(pstdev(vals)) if len(vals) > 1 else 0.0,
        }
    reward_by_seed = [mean([ep["cumulative_reward"] for ep in seed["episodes"]]) for seed in all_metrics]
    agg["reward_variance"] = float(pstdev(reward_by_seed)) if len(reward_by_seed) > 1 else 0.0
    return agg


def calibration_bins(calib_points: List[Tuple[float, int]], bins: int = 10):
    if not calib_points:
        return []
    calib_points = sorted(calib_points, key=lambda x: x[0])
    edges = np.linspace(0, 1, bins + 1)
    bin_sums = np.zeros(bins)
    bin_counts = np.zeros(bins)
    for pred, actual in calib_points:
        idx = min(bins - 1, int(pred * bins))
        bin_sums[idx] += actual
        bin_counts[idx] += 1
    results = []
    for i in range(bins):
        left, right = edges[i], edges[i + 1]
        acc = bin_sums[i] / bin_counts[i] if bin_counts[i] > 0 else 0.0
        conf = (left + right) / 2.0
        results.append((conf, acc))
    return results


def save_learning_curve(curves: List[List[Tuple[int, float]]], out_path: str):
    # Align on the shortest curve length for per-step aggregation
    min_len = min(len(c) for c in curves)
    rows = []
    for i in range(min_len):
        steps_i = [curves[s][i][0] for s in range(len(curves))]
        rews_i = [curves[s][i][1] for s in range(len(curves))]
        step_mean = int(sum(steps_i) / len(steps_i))
        rows.append((step_mean, float(mean(rews_i)), float(pstdev(rews_i)) if len(rews_i) > 1 else 0.0))
    with open(out_path.replace(".png", ".csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["steps", "mean_batch_reward", "sd_batch_reward"])
        for step, m, s in rows:
            writer.writerow([step, m, s])
    steps_plot = [r[0] for r in rows]
    mean_plot = [r[1] for r in rows]
    sd_plot = [r[2] for r in rows]
    upper = [m + s for m, s in zip(mean_plot, sd_plot)]
    lower = [m - s for m, s in zip(mean_plot, sd_plot)]
    plt.figure(figsize=(6, 4))
    plt.plot(steps_plot, mean_plot, label="PPO (mean across seeds)")
    plt.fill_between(steps_plot, lower, upper, color="C0", alpha=0.2, label="±1 SD")
    plt.xlabel("Env steps")
    plt.ylabel("Batch mean reward")
    plt.title("Learning curve")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def save_modality_gains(modality_gains: Dict[int, List[float]], out_path: str):
    labels = {0: "video", 1: "ppt", 2: "text", 3: "blog", 4: "article", 5: "handout"}
    rows = []
    for k, v in modality_gains.items():
        if not v:
            continue
        rows.append((labels.get(k, str(k)), float(mean(v)), float(pstdev(v)) if len(v) > 1 else 0.0))
    with open(out_path.replace(".png", ".csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["modality", "mean_gain", "sd"])
        for row in rows:
            writer.writerow(row)
    if rows:
        plt.figure(figsize=(6, 4))
        plt.bar([r[0] for r in rows], [r[1] for r in rows], yerr=[r[2] for r in rows], alpha=0.8)
        plt.ylabel("Post-content gain")
        plt.title("Post-content gain by modality")
        plt.tight_layout()
        plt.savefig(out_path, dpi=200)
        plt.close()


def save_calibration(calib_points: List[Tuple[float, int]], out_path: str):
    bins = calibration_bins(calib_points)
    if not bins:
        return
    conf, acc = zip(*bins)
    plt.figure(figsize=(4, 4))
    plt.plot(conf, acc, marker="o", label="Observed")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Ideal")
    plt.xlabel("Predicted mastery")
    plt.ylabel("Observed correctness")
    plt.title("Calibration")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    with open(out_path.replace(".png", ".csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["bin_confidence", "observed_accuracy"])
        for c, a in bins:
            writer.writerow([c, a])


def save_perf_table(agg: Dict, out_path: str):
    headers = [
        ("Cumulative Reward", "cumulative_reward"),
        ("Time-to-Mastery", "time_to_mastery"),
        ("Post-Content Gain", "post_content_gain"),
        ("Blueprint Adherence", "blueprint_adherence"),
        ("Question Accuracy", "question_accuracy"),
        ("Content Rate", "content_rate"),
        ("Final Mastery", "final_mastery"),
        ("Mean Frustration", "mean_frustration"),
        ("Reward Variance", "reward_variance"),
    ]
    lines = ["\\begin{table}[t]", "\\centering", "\\caption{PPO Performance Summary (mean$\\pm$SD across seeds)}", "\\label{tab:ppo_perf}", "\\begin{tabular}{lc}", "\\toprule", "Metric & Value \\\", "\\midrule"]
    for label, key in headers:
        val = agg.get(key, {"mean": 0.0, "sd": 0.0})
        mean_v = val["mean"] if isinstance(val, dict) else val
        sd_v = val.get("sd", 0.0) if isinstance(val, dict) else 0.0
        lines.append(f"{label} & {mean_v:.3f} $\\pm$ {sd_v:.3f} \\")
    lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}"])
    with open(out_path, "w") as f:
        f.write("\n".join(lines))


def save_modality_table(modality_gains: Dict[int, List[float]], out_path: str):
    labels = {0: "video", 1: "ppt", 2: "text", 3: "blog", 4: "article", 5: "handout"}
    rows = []
    for k, v in modality_gains.items():
        if not v:
            continue
        rows.append((labels.get(k, str(k)), float(mean(v)), float(pstdev(v)) if len(v) > 1 else 0.0))
    lines = ["\\begin{table}[t]", "\\centering", "\\caption{Post-content gain by modality (PPO)}", "\\label{tab:modality_gain}", "\\begin{tabular}{lcc}", "\\toprule", "Modality & Mean Gain & SD \\\", "\\midrule"]
    for name, m, s in rows:
        lines.append(f"{name} & {m:.3f} & {s:.3f} \\")
    lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}"])
    with open(out_path, "w") as f:
        f.write("\n".join(lines))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-seeds", type=int, default=3)
    parser.add_argument("--max-iters", type=int, default=10, help="Number of PPO update iterations per seed")
    parser.add_argument("--max-episodes", type=int, default=120)
    parser.add_argument("--max-steps-per-episode", type=int, default=140)
    parser.add_argument("--rollout-steps", type=int, default=1024)
    parser.add_argument("--eval-episodes", type=int, default=20)
    parser.add_argument("--outdir", type=str, default="results")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    base_cfg = CONFIG.copy()
    base_cfg.update({
        "max_episodes": args.max_episodes,
        "max_steps_per_episode": args.max_steps_per_episode,
        "rollout_steps": args.rollout_steps,
    })

    per_seed_metrics = []
    all_curves = []
    all_calib = []
    modality_pool = defaultdict(list)

    for seed in range(args.num_seeds):
        agent, curve = train_one_seed(seed, base_cfg.copy(), args.max_iters)
        all_curves.append(curve)
        eval_result = evaluate_policy(agent, args.eval_episodes, seed * 100)
        per_seed_metrics.append(eval_result)
        all_calib.extend(eval_result["calibration"])
        for mod, gains in eval_result["modality_gains"].items():
            modality_pool[mod].extend(gains)

    agg = aggregate_metrics(per_seed_metrics)

    summary_path = os.path.join(args.outdir, "metrics_summary.json")
    with open(summary_path, "w") as f:
        json.dump(agg, f, indent=2)

    per_seed_path = os.path.join(args.outdir, "per_seed_metrics.csv")
    with open(per_seed_path, "w", newline="") as f:
        writer = csv.writer(f)
        header = [
            "seed", "episode", "cumulative_reward", "time_to_mastery", "post_content_gain",
            "blueprint_adherence", "question_accuracy", "content_rate", "final_mastery", "mean_frustration",
        ]
        writer.writerow(header)
        for seed_idx, seed in enumerate(per_seed_metrics):
            for ep_idx, ep in enumerate(seed["episodes"]):
                writer.writerow([seed_idx, ep_idx, ep["cumulative_reward"], ep["time_to_mastery"], ep["post_content_gain"], ep["blueprint_adherence"], ep["question_accuracy"], ep["content_rate"], ep["final_mastery"], ep["mean_frustration"]])

    save_learning_curve(all_curves, os.path.join(args.outdir, "learning_curve.png"))
    save_modality_gains(modality_pool, os.path.join(args.outdir, "modality_gains.png"))
    save_calibration(all_calib, os.path.join(args.outdir, "calibration.png"))
    save_perf_table(agg, os.path.join(args.outdir, "table_ppo_perf.tex"))
    save_modality_table(modality_pool, os.path.join(args.outdir, "table_modality.tex"))

    print(f"Saved summary to {summary_path}")
    print(f"Saved per-episode metrics to {per_seed_path}")


if __name__ == "__main__":
    main()
