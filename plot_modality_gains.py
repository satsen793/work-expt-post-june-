#!/usr/bin/env python3
"""
Standalone modality gains plot script.
Loads PETS data from JSON and computes MBPO data from CSV, then generates combined plot.
"""

import os
import json
import csv
from typing import Dict, List
import numpy as np

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
                'blueprint_adherence': float(row.get('blueprint_adherence', 0.0)),
                'post_content_gain_video': float(row.get('post_content_gain_video', 0.0)),
                'post_content_gain_PPT': float(row.get('post_content_gain_PPT', 0.0)),
                'post_content_gain_text': float(row.get('post_content_gain_text', 0.0)),
                'post_content_gain_blog': float(row.get('post_content_gain_blog', 0.0)),
                'post_content_gain_article': float(row.get('post_content_gain_article', 0.0)),
                'post_content_gain_handout': float(row.get('post_content_gain_handout', 0.0)),
            })
    
    return episodes

def compute_modality_gains_from_csv(csv_path: str) -> Dict:
    """Compute modality gains from episodes.csv."""
    episodes = load_full_episodes(csv_path)
    
    modality_columns = {
        'video': 'post_content_gain_video',
        'PPT': 'post_content_gain_PPT',
        'text': 'post_content_gain_text',
        'blog': 'post_content_gain_blog',
        'article': 'post_content_gain_article',
        'handout': 'post_content_gain_handout'
    }
    
    modality_gains = {}
    for mod, col in modality_columns.items():
        gains = [float(ep[col]) for ep in episodes if float(ep[col]) > 0]
        if gains:
            modality_gains[mod] = {
                'mean': float(np.mean(gains)),
                'std': float(np.std(gains)),
                'count': len(gains)
            }
        else:
            modality_gains[mod] = {
                'mean': 0.0,
                'std': 0.0,
                'count': 0
            }
    
    return modality_gains

def plot_modality_gains():
    """Generate combined modality gains plot for PETS and MBPO."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available")
        return
    
    # Load PETS data from JSON
    pets_data = None
    pets_path = "results/pets/modality_gains.json"
    if os.path.exists(pets_path):
        with open(pets_path, 'r') as f:
            pets_data = json.load(f)
        print("Loaded PETS modality gains from JSON")
    else:
        print("PETS modality_gains.json not found")
    
    # Compute MBPO data from CSV
    mbpo_data = None
    mbpo_csv = "results/mbpo/episodes.csv"
    if os.path.exists(mbpo_csv):
        mbpo_data = compute_modality_gains_from_csv(mbpo_csv)
        print("Computed MBPO modality gains from CSV")
        
        # Export MBPO data to CSV
        import csv
        with open("results/mbpo/modality_gains.csv", 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['modality', 'mean', 'std', 'count'])
            for mod, stats in mbpo_data.items():
                writer.writerow([mod, stats['mean'], stats['std'], stats['count']])
        print("Exported MBPO modality gains to CSV")
    else:
        print("MBPO episodes.csv not found")
    
    # Plot
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    algos = ['PETS', 'MBPO']
    colors = ['#ff7f0e', '#2ca02c']
    
    modalities = ['video', 'PPT', 'text', 'blog', 'article', 'handout']
    x = np.arange(len(modalities))
    width = 0.35
    
    for idx, algo in enumerate(algos):
        data = pets_data if algo == 'PETS' else mbpo_data
        if data:
            means = [data.get(m, {}).get('mean', 0.0) for m in modalities]
            stds = [data.get(m, {}).get('std', 0.0) for m in modalities]
            
            ax.bar(x + idx*width, means, width, yerr=stds, capsize=5, alpha=0.7, color=colors[idx], label=algo)
    
    ax.set_xticks(x + width/2)
    ax.set_xticklabels(modalities, rotation=45, ha='right')
    ax.set_ylabel('Post-Content Gain', fontsize=12)
    ax.set_title('Average Post-Content Gain by Modality (Model-Based Methods)', fontsize=16, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig("modality_gains_combined.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✓ Combined modality gains plot saved to modality_gains_combined.png")

if __name__ == "__main__":
    plot_modality_gains()