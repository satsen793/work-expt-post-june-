#!/usr/bin/env python3
"""
Merge PETS multi-seed results into unified format for comparison.
Combines episodes.csv from multiple seed folders into single multi-seed format.
"""

import os
import csv
import json
from pathlib import Path
from typing import Dict, List

def merge_pets_multiseed(base_dir: str = "results") -> None:
    """
    Merge PETS seed data from separate folders into unified multi-seed format.

    Expected input folders: pets_seeds_{seed}_ep{episodes}/
    Output: results/pets/ with combined episodes.csv and individual seed files
    """
    base_path = Path(base_dir)
    pets_base = base_path / "pets"
    pets_base.mkdir(exist_ok=True)

    # Find all PETS seed folders (prioritize full 295-episode runs)
    seed_folders = []
    for item in base_path.iterdir():
        if item.is_dir() and item.name.startswith("pets_seeds_"):
            parts = item.name.split("_")
            if len(parts) >= 4:  # pets_seeds_X_epY
                try:
                    seed = int(parts[2])
                    episodes_str = parts[3]  # "ep295" or "ep10"
                    if episodes_str.startswith("ep"):
                        episodes = int(episodes_str[2:])  # Remove "ep" prefix
                        seed_folders.append((seed, episodes, item))
                except (ValueError, IndexError):
                    continue

    # Sort by seed number, prioritize 295-episode runs
    seed_folders.sort(key=lambda x: (x[0], -x[1]))  # seed asc, episodes desc

    # Group by seed, take highest episode count
    seed_data = {}
    for seed, episodes, folder in seed_folders:
        if seed not in seed_data or episodes > seed_data[seed][0]:
            seed_data[seed] = (episodes, folder)

    print(f"Found {len(seed_data)} unique seeds with data:")
    for seed, (episodes, folder) in sorted(seed_data.items()):
        print(f"  Seed {seed}: {episodes} episodes from {folder.name}")

    # Merge all episodes.csv files
    all_episodes = []
    seed_summaries = {}
    modality_gains_data = {}

    for seed, (episodes, folder) in sorted(seed_data.items()):
        episodes_file = folder / "episodes.csv"
        summary_file = folder / "summary.json"

        if not episodes_file.exists():
            print(f"Warning: {episodes_file} not found, skipping seed {seed}")
            continue

        print(f"Processing seed {seed} from {folder.name}...")

        # Read episodes data
        with open(episodes_file, 'r') as f:
            reader = csv.DictReader(f)
            seed_episodes = list(reader)

        # Verify seed consistency
        for ep in seed_episodes:
            if int(ep['seed']) != seed:
                print(f"Warning: episode has seed {ep['seed']} but expected {seed}")

        all_episodes.extend(seed_episodes)

        # Copy individual seed file
        seed_output = pets_base / f"seed{seed}_episodes.csv"
        with open(seed_output, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=reader.fieldnames)
            writer.writeheader()
            writer.writerows(seed_episodes)

        # Copy summary if available
        if summary_file.exists():
            with open(summary_file, 'r') as f:
                summary_data = json.load(f)
            seed_summaries[seed] = summary_data

            # Save individual seed summary
            seed_summary_output = pets_base / f"seed{seed}_summary.json"
            with open(seed_summary_output, 'w') as f:
                json.dump(summary_data, f, indent=2)

        # Collect modality gains data for aggregation
        modality_file = folder / "modality_gains.json"
        if modality_file.exists():
            with open(modality_file, 'r') as f:
                modality_data = json.load(f)
            if seed not in modality_gains_data:
                modality_gains_data[seed] = modality_data

    # Write combined episodes.csv
    if all_episodes:
        combined_episodes = pets_base / "episodes.csv"
        fieldnames = ['seed', 'episode', 'return', 'cumulative_reward', 'ttm', 'total_steps',
                     'question_accuracy', 'content_rate', 'blueprint_adherence', 'post_content_gain',
                     'post_content_gain_video', 'post_content_gain_PPT', 'post_content_gain_text',
                     'post_content_gain_blog', 'post_content_gain_article', 'post_content_gain_handout',
                     'final_mastery', 'mean_frustration']

        with open(combined_episodes, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_episodes)

        print(f"✓ Combined {len(all_episodes)} episodes from {len(seed_data)} seeds into {combined_episodes}")

    # Create combined summary.json (aggregate across seeds)
    if seed_summaries:
        combined_summary = aggregate_summaries(seed_summaries)
        summary_output = pets_base / "summary.json"
        with open(summary_output, 'w') as f:
            json.dump(combined_summary, f, indent=2)
        print(f"✓ Created combined summary.json with {len(seed_summaries)} seeds")

    # Create combined modality_gains.json (aggregate across seeds)
    if modality_gains_data:
        combined_modality_gains = aggregate_modality_gains(modality_gains_data)
        modality_output = pets_base / "modality_gains.json"
        with open(modality_output, 'w') as f:
            json.dump(combined_modality_gains, f, indent=2)
        print(f"✓ Created combined modality_gains.json with {len(modality_gains_data)} seeds")

        # Generate LaTeX modality table
        table_output = pets_base / "table_modality.tex"
        generate_modality_table(combined_modality_gains, table_output)
        print(f"✓ Generated modality table: {table_output}")

    print(f"\nMerged PETS data saved to: {pets_base}/")
    print("Files created:")
    print("  - episodes.csv (combined)")
    print("  - summary.json (combined)")
    for seed in sorted(seed_data.keys()):
        print(f"  - seed{seed}_episodes.csv")
        print(f"  - seed{seed}_summary.json")

def aggregate_summaries(seed_summaries: Dict[int, Dict]) -> Dict:
    """Aggregate summary statistics across seeds."""
    if not seed_summaries:
        return {}

    # Get all metric keys from first summary
    first_summary = next(iter(seed_summaries.values()))
    combined = {}

    # Aggregate each metric
    for key, value in first_summary.items():
        if isinstance(value, dict) and 'mean' in value and 'std' in value:
            # Metric with mean/std structure
            means = [s[key]['mean'] for s in seed_summaries.values() if key in s and isinstance(s[key], dict)]
            if means:
                combined[key] = {
                    'mean': float(sum(means) / len(means)),
                    'std': float(sum(s[key]['std'] for s in seed_summaries.values() if key in s and isinstance(s[key], dict)) / len(means))
                }
        elif key == 'num_seeds':
            combined[key] = len(seed_summaries)
        else:
            # Copy as-is or take first value
            combined[key] = value

    return combined

def aggregate_modality_gains(modality_gains_data: Dict[int, Dict]) -> Dict:
    """Aggregate modality gains statistics across seeds."""
    if not modality_gains_data:
        return {}

    # Get all modality types from first seed
    first_seed = next(iter(modality_gains_data.values()))
    modalities = list(first_seed.keys())

    combined = {}
    for modality in modalities:
        means = []
        stds = []
        counts = []

        for seed_data in modality_gains_data.values():
            if modality in seed_data:
                mod_data = seed_data[modality]
                means.append(mod_data.get('mean', 0))
                stds.append(mod_data.get('std', 0))
                counts.append(mod_data.get('count', 0))

        if means:
            # Calculate weighted average for mean
            total_count = sum(counts)
            if total_count > 0:
                weighted_mean = sum(m * c for m, c in zip(means, counts)) / total_count
                # Calculate combined standard deviation
                variance = sum((s**2 * c + (m - weighted_mean)**2 * c) for m, s, c in zip(means, stds, counts)) / total_count
                combined_std = variance**0.5 if variance > 0 else 0
            else:
                weighted_mean = sum(means) / len(means)
                combined_std = sum(stds) / len(stds)

            combined[modality] = {
                'mean': float(weighted_mean),
                'std': float(combined_std),
                'count': total_count
            }

    return combined

def generate_modality_table(modality_gains: Dict, output_path: Path) -> None:
    """Generate LaTeX table for modality gains."""
    latex = []
    latex.append("% Auto-generated modality gains table")
    latex.append("\\begin{tabular}{lcc}")
    latex.append("\\toprule")
    latex.append("\\textbf{Modality} & \\textbf{Mean Gain} & \\textbf{Std Dev} \\\\")
    latex.append("\\midrule")

    # Sort modalities by mean gain (descending)
    sorted_modalities = sorted(modality_gains.items(), key=lambda x: x[1]['mean'], reverse=True)

    for modality, stats in sorted_modalities:
        mean = stats['mean']
        std = stats['std']
        latex.append(f"{modality} & {mean:.3f} & {std:.3f} \\\\")

    latex.append("\\bottomrule")
    latex.append("\\end{tabular}")

    with open(output_path, 'w') as f:
        f.write('\n'.join(latex))

if __name__ == "__main__":
    merge_pets_multiseed()
    print("\nPETS multi-seed merge complete! Ready for comparison with other algorithms.")