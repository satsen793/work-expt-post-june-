#!/usr/bin/env python3
"""
Combine all seed*_episodes.csv files into one episodes.csv
"""
import csv
import os
import glob

def combine_csvs():
    output_path = "results/dqn/episodes.csv"
    input_pattern = "results/dqn/seed*_episodes.csv"
    
    # Get all matching files
    csv_files = glob.glob(input_pattern)
    if not csv_files:
        print("No seed*_episodes.csv files found")
        return
    
    print(f"Found {len(csv_files)} CSV files: {csv_files}")
    
    all_rows = []
    header_written = False
    
    for csv_file in sorted(csv_files):
        # Extract seed from filename: seed{N}_episodes.csv -> N
        seed_str = csv_file.split('seed')[1].split('_')[0]
        seed = int(seed_str)
        
        print(f"Processing {csv_file} (seed {seed})")
        
        with open(csv_file, 'r') as f:
            reader = csv.reader(f)
            header = next(reader)
            
            if not header_written:
                # Add seed to header
                full_header = ['seed'] + header
                all_rows.append(full_header)
                header_written = True
            
            for row in reader:
                # Add seed to each row
                full_row = [seed] + row
                all_rows.append(full_row)
    
    # Write combined CSV
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(all_rows)
    
    print(f"✓ Combined CSV written to {output_path}")
    print(f"Total rows: {len(all_rows)}")

if __name__ == "__main__":
    combine_csvs()