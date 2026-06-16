#!/usr/bin/env python3
"""Generate summary CSV files for each problem size."""

import json
import os
from pathlib import Path
from collections import defaultdict
import numpy as np
import pandas as pd

def extract_distribution_type(filename):
    """Extract distribution type from filename.

    E.g., 'S042_N10_C_R50.json' -> 'C'
    """
    parts = filename.replace('.json', '').split('_')
    # Format is: S###_N##_DIST_R##
    if len(parts) >= 4:
        return parts[2]
    return None

def process_size_folder(size_dir):
    """Process all JSON files in a size folder and group by distribution type."""

    results_by_dist = defaultdict(lambda: {
        'objectives': [],
        'costs': [],
        'service_rates': []
    })

    json_files = list(Path(size_dir).glob('*.json'))

    if not json_files:
        print(f"  No JSON files found in {size_dir}")
        return None

    # Load all data
    for json_file in json_files:
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)

            dist_type = extract_distribution_type(json_file.name)
            if not dist_type:
                print(f"  Warning: Could not extract distribution type from {json_file.name}")
                continue

            results_by_dist[dist_type]['objectives'].append(data.get('objective', 0))
            results_by_dist[dist_type]['costs'].append(data.get('f2_cost', 0))
            results_by_dist[dist_type]['service_rates'].append(data.get('f1_service_rate', 0))

        except Exception as e:
            print(f"  Error reading {json_file.name}: {e}")

    # Calculate statistics
    summary_data = []
    for dist_type in sorted(results_by_dist.keys()):
        metrics = results_by_dist[dist_type]

        obj_mean = np.mean(metrics['objectives'])
        obj_std = np.std(metrics['objectives'])

        cost_mean = np.mean(metrics['costs'])
        cost_std = np.std(metrics['costs'])

        sr_mean = np.mean(metrics['service_rates'])
        sr_std = np.std(metrics['service_rates'])

        summary_data.append({
            'Distribution': dist_type,
            'Objective': f"{obj_mean:.2f}",
            'Objective_Error': f"±{obj_std:.2f}",
            'Cost': f"{cost_mean:.2f}",
            'Cost_Error': f"±{cost_std:.2f}",
            'Service_Rate': f"{sr_mean:.4f}",
            'Service_Rate_Error': f"±{sr_std:.4f}",
        })

    return pd.DataFrame(summary_data)

def main():
    base_dir = Path('/home/bxs/thesis/vrpbtw/lp/result/f')

    if not base_dir.exists():
        print(f"Error: {base_dir} does not exist")
        return

    size_folders = sorted([d for d in base_dir.iterdir() if d.is_dir()])

    for size_folder in size_folders:
        size_name = size_folder.name
        print(f"\nProcessing {size_name}...")

        df = process_size_folder(size_folder)

        if df is not None and len(df) > 0:
            output_file = size_folder / 'summary.csv'
            df.to_csv(output_file, index=False)
            print(f"  ✓ Created {output_file}")
            print(f"\n{df.to_string(index=False)}\n")
        else:
            print(f"  No data to summarize for {size_name}")

if __name__ == '__main__':
    main()
