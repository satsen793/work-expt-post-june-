import argparse
import json
import os
from typing import List

from train_dqn import run_multi_seed_with_summary


def parse_seeds(s: str) -> List[int]:
    return [int(x.strip()) for x in s.split(',') if x.strip()]


def main():
    parser = argparse.ArgumentParser(description="Multi-seed DQN runner for HPC (summary + JSON dump)")
    parser.add_argument("--seeds", type=str, default="0,1,2,3,4", help="Comma-separated seed list (e.g., 0,1,2,3,4)")
    parser.add_argument("--episodes", type=int, default=200, help="Number of training episodes per seed")
    parser.add_argument("--steps", type=int, default=140, help="Max steps per episode")
    parser.add_argument("--start-steps", type=int, default=5000, help="Warmup steps before learning starts")
    parser.add_argument("--buffer-size", type=int, default=200_000, help="Replay buffer size")
    parser.add_argument("--batch-size", type=int, default=128, help="Training batch size")
    parser.add_argument("--out", type=str, default="logs/hpc_summary.json", help="Output JSON file path")
    parser.add_argument("--out-csv", type=str, default="logs/hpc_summary.csv", help="Output CSV file path (combined per-episode metrics across seeds)")
    args = parser.parse_args()

    seeds = parse_seeds(args.seeds)
    res = run_multi_seed_with_summary(
        seeds=seeds,
        num_episodes=args.episodes,
        buffer_size=args.buffer_size,
        batch_size=args.batch_size,
        start_steps=args.start_steps,
        max_steps_per_episode=args.steps,
    )

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(res, f, indent=2)

    summary = res["summary"]
    print("Summary (across seeds):")
    for k, v in summary.items():
        print(f"- {k}: {v}")
    print(f"Saved JSON to: {args.out}")

    # Combined CSV of per-episode metrics
    try:
        import csv
        os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
        with open(args.out_csv, "w", newline="") as f:
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
            for seed_idx, r in enumerate(res.get("results", [])):
                returns = r.get("returns", [])
                ttms = r.get("time_to_mastery", [])
                ems = r.get("episode_metrics", [])
                for ep_idx, (ret, ttm, em) in enumerate(zip(returns, ttms, ems), start=1):
                    writer.writerow([
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
                    ])
        print(f"Saved combined CSV to: {args.out_csv}")
    except Exception as e:
        print(f"CSV write failed: {e}")


if __name__ == "__main__":
    main()
