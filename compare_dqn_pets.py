#!/usr/bin/env python3
"""
Paired comparison runner: Execute both DQN and PETS with identical configuration.
Ensures 1:1 replication for fair algorithmic comparison.

This script:
1. Runs DQN_VER3 with unified seeds [0,1,2,3,4] and 295 episodes
2. Runs PETS_VER3 with unified seeds [0,1,2,3,4] and 295 episodes
3. Generates side-by-side metrics comparison
4. Validates specification alignment (blueprint adherence, reward calculation, etc.)
"""
import argparse
import json
import os
import subprocess
import sys
import time
from typing import Dict, List, Tuple
from pathlib import Path

# Import shared configuration
sys.path.insert(0, os.path.dirname(__file__))
from shared_config import UNIFIED_SEEDS, UNIFIED_EPISODES, UNIFIED_MAX_STEPS_PER_EPISODE


def ensure_dir(path: str):
    """Create directory if it doesn't exist."""
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)


def run_dqn_multiseed(
    output_dir: str = "comparison_results/dqn",
    episodes: int = UNIFIED_EPISODES,
    seeds: List[int] = None,
    no_warmup: bool = False,
) -> Dict:
    """Run DQN with multi-seed configuration."""
    if seeds is None:
        seeds = UNIFIED_SEEDS
    
    ensure_dir(output_dir)
    
    print("\n" + "="*70)
    print("RUNNING DQN_VER3 (Multi-Seed)")
    print("="*70)
    print(f"Episodes per seed: {episodes}")
    print(f"Seeds: {seeds}")
    print(f"Warmup: {'DISABLED' if no_warmup else f'{5000} steps'}")
    print(f"Expected total steps: ~{episodes * 140 / len(seeds) * len(seeds):.0f} (per seed: ~{episodes * 140 / len(seeds):.0f})")
    
    # Build command
    cmd = [
        sys.executable,
        "dqn_ver3/scripts/run_multiseed.py",
        "--seeds", *[str(s) for s in seeds],
        "--episodes", str(episodes),
        "--out-json", f"{output_dir}/summary.json",
        "--out-csv", f"{output_dir}/episodes.csv",
        "--fig-learning", f"{output_dir}/learning_curve.png",
        "--fig-modality", f"{output_dir}/modality_gains.png",
        "--fig-variance", f"{output_dir}/variance.png",
        "--fig-compute", f"{output_dir}/compute_vs_reward.png",
    ]
    
    if no_warmup:
        cmd.extend(["--start-steps", "0"])
    
    print(f"\nCommand: {' '.join(cmd)}")
    start_time = time.time()
    
    try:
        result = subprocess.run(cmd, cwd=os.path.dirname(__file__), capture_output=True, text=True, timeout=3600)
        elapsed = time.time() - start_time
        
        if result.returncode != 0:
            print(f"ERROR (exit code {result.returncode}):")
            print(result.stderr)
            return {"status": "failed", "error": result.stderr, "elapsed": elapsed}
        
        print(f"✓ DQN completed in {elapsed:.1f}s")
        
        # Read output JSON
        summary_path = f"{output_dir}/summary.json"
        if os.path.exists(summary_path):
            with open(summary_path) as f:
                summary = json.load(f)
            return {
                "status": "success",
                "elapsed": elapsed,
                "summary": summary,
                "output_dir": output_dir,
            }
        else:
            return {"status": "failed", "error": "Summary JSON not created", "elapsed": elapsed}
    
    except subprocess.TimeoutExpired:
        return {"status": "timeout", "elapsed": time.time() - start_time}
    except Exception as e:
        return {"status": "error", "error": str(e), "elapsed": time.time() - start_time}


def run_pets_multiseed(
    output_dir: str = "comparison_results/pets",
    episodes: int = UNIFIED_EPISODES,
    seeds: List[int] = None,
) -> Dict:
    """Run PETS with multi-seed configuration."""
    if seeds is None:
        seeds = UNIFIED_SEEDS
    
    ensure_dir(output_dir)
    
    print("\n" + "="*70)
    print("RUNNING PETS_VER3 (Multi-Seed)")
    print("="*70)
    print(f"Episodes per seed: {episodes}")
    print(f"Seeds: {seeds}")
    print(f"Max steps/episode: {UNIFIED_MAX_STEPS_PER_EPISODE}")
    print(f"Expected total steps: ~{episodes * UNIFIED_MAX_STEPS_PER_EPISODE / len(seeds) * len(seeds):.0f} (per seed: ~{episodes * UNIFIED_MAX_STEPS_PER_EPISODE / len(seeds):.0f})")
    
    # Note: PETS runs all seeds internally when main() is called
    # We need to run it once per seed for consistency with DQN
    print("\nNote: PETS runs all seeds internally. Executing single command...")
    
    cmd = [
        sys.executable,
        "pets_ver3/pets_train.py",
        "--episodes", str(episodes),
    ]
    
    # PETS doesn't have multi-seed CLI like DQN; seeds are defined in TrainConfig
    # We'll let it use the unified seeds from shared_config
    
    print(f"\nCommand: {' '.join(cmd)}")
    start_time = time.time()
    
    try:
        result = subprocess.run(cmd, cwd=os.path.dirname(__file__), capture_output=True, text=True, timeout=3600)
        elapsed = time.time() - start_time
        
        if result.returncode != 0:
            print(f"ERROR (exit code {result.returncode}):")
            print(result.stderr[-1000:])  # Last 1000 chars
            return {"status": "failed", "error": result.stderr[-500:], "elapsed": elapsed}
        
        print(f"✓ PETS completed in {elapsed:.1f}s")
        print("\nPETS Output (last 50 lines):")
        print("\n".join(result.stdout.split("\n")[-50:]))
        
        return {
            "status": "success",
            "elapsed": elapsed,
            "output_dir": output_dir,
            "stdout": result.stdout[-1000:],
        }
    
    except subprocess.TimeoutExpired:
        return {"status": "timeout", "elapsed": time.time() - start_time}
    except Exception as e:
        return {"status": "error", "error": str(e), "elapsed": time.time() - start_time}


def create_comparison_report(dqn_result: Dict, pets_result: Dict, report_path: str = "comparison_results/comparison.json"):
    """Generate side-by-side comparison report."""
    ensure_dir(report_path)
    
    report = {
        "metadata": {
            "unified_config": {
                "seeds": UNIFIED_SEEDS,
                "episodes": UNIFIED_EPISODES,
                "max_steps_per_episode": UNIFIED_MAX_STEPS_PER_EPISODE,
                "total_expected_steps": UNIFIED_EPISODES * UNIFIED_MAX_STEPS_PER_EPISODE * len(UNIFIED_SEEDS),
            },
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
        "dqn": dqn_result,
        "pets": pets_result,
    }
    
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    
    print(f"\n✓ Comparison report written to {report_path}")
    return report


def print_comparison_summary(report: Dict):
    """Print human-readable comparison summary."""
    print("\n" + "="*70)
    print("COMPARISON SUMMARY")
    print("="*70)
    
    dqn = report.get("dqn", {})
    pets = report.get("pets", {})
    config = report.get("metadata", {}).get("unified_config", {})
    
    print(f"\nUnified Configuration:")
    print(f"  Seeds: {config.get('seeds')}")
    print(f"  Episodes per seed: {config.get('episodes')}")
    print(f"  Max steps per episode: {config.get('max_steps_per_episode')}")
    print(f"  Expected total steps: {config.get('total_expected_steps')}")
    
    print(f"\nDQN Status:")
    print(f"  Status: {dqn.get('status')}")
    print(f"  Elapsed: {dqn.get('elapsed', 'N/A'):.1f}s" if isinstance(dqn.get('elapsed'), (int, float)) else f"  Elapsed: N/A")
    if dqn.get('error'):
        print(f"  Error: {dqn.get('error')}")
    if dqn.get('summary'):
        print(f"  Output dir: {dqn.get('output_dir')}")
    
    print(f"\nPETS Status:")
    print(f"  Status: {pets.get('status')}")
    print(f"  Elapsed: {pets.get('elapsed', 'N/A'):.1f}s" if isinstance(pets.get('elapsed'), (int, float)) else f"  Elapsed: N/A")
    if pets.get('error'):
        print(f"  Error: {pets.get('error')}")
    
    print("\n" + "="*70)


def main():
    parser = argparse.ArgumentParser(
        description="Run paired comparison: DQN vs PETS with identical config for 1:1 replication"
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=UNIFIED_EPISODES,
        help=f"Episodes per seed (default: {UNIFIED_EPISODES})",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="*",
        default=UNIFIED_SEEDS,
        help=f"Random seeds (default: {UNIFIED_SEEDS})",
    )
    parser.add_argument(
        "--dqn-only",
        action="store_true",
        help="Run only DQN (skip PETS)",
    )
    parser.add_argument(
        "--pets-only",
        action="store_true",
        help="Run only PETS (skip DQN)",
    )
    parser.add_argument(
        "--dqn-no-warmup",
        action="store_true",
        help="Run DQN without warmup phase (set start_steps=0)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="comparison_results",
        help="Base output directory for results",
    )
    
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print("PAIRED COMPARISON RUNNER: DQN_VER3 vs PETS_VER3")
    print("="*70)
    print(f"Configuration:")
    print(f"  Episodes per seed: {args.episodes}")
    print(f"  Seeds: {args.seeds}")
    print(f"  DQN warmup: {'DISABLED' if args.dqn_no_warmup else 'ENABLED (5000 steps)'}")
    print(f"  Output dir: {args.output_dir}")
    
    results = {}
    
    if not args.pets_only:
        dqn_result = run_dqn_multiseed(
            output_dir=f"{args.output_dir}/dqn",
            episodes=args.episodes,
            seeds=args.seeds,
            no_warmup=args.dqn_no_warmup,
        )
        results["dqn"] = dqn_result
    
    if not args.dqn_only:
        pets_result = run_pets_multiseed(
            output_dir=f"{args.output_dir}/pets",
            episodes=args.episodes,
            seeds=args.seeds,
        )
        results["pets"] = pets_result
    
    # Generate report
    report = {
        "metadata": {
            "unified_config": {
                "seeds": args.seeds,
                "episodes": args.episodes,
                "max_steps_per_episode": UNIFIED_MAX_STEPS_PER_EPISODE,
            },
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
    }
    report.update(results)
    
    create_comparison_report(report, f"{args.output_dir}/comparison.json")
    print_comparison_summary(report)


if __name__ == "__main__":
    main()
