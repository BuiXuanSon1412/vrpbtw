#!/usr/bin/env python3
"""Generate summary CSV files for baseline results."""

import json
from pathlib import Path
from collections import defaultdict
import numpy as np
import pandas as pd


def extract_metadata(filename: str) -> tuple:
    """Extract size and distribution from filename.

    E.g., 'S042_N10_C_R50.json' -> (10, 'C')
    """
    parts = filename.replace('.json', '').split('_')
    # Format: S###_N##_DIST_R##
    if len(parts) >= 4:
        size = int(parts[1][1:])  # N10 -> 10
        distribution = parts[2]    # C, R, RC
        return size, distribution
    return None, None


def process_baseline(baseline_dir: Path, baseline_name: str):
    """Process all results for a baseline method."""

    print(f"\nProcessing {baseline_name.upper()}...")

    # Group results by (size, distribution)
    results_by_group = defaultdict(lambda: {
        'fitness': [],
        'cost': [],
        'service_rate': [],
        'time': []
    })

    # Group results by size
    results_by_size = defaultdict(lambda: {
        'fitness': [],
        'cost': [],
        'service_rate': [],
        'time': []
    })

    # Scan all JSON files
    json_files = list(baseline_dir.rglob('*.json'))

    for json_file in json_files:
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)

            size, dist = extract_metadata(json_file.name)
            if not size or not dist:
                print(f"  Warning: Could not extract metadata from {json_file.name}")
                continue

            group_key = f"{dist}_N{size}"

            # Extract best metrics
            fitness = data.get('best_fitness', 0)
            cost = data.get('best_cost', 0)
            service_rate = data.get('best_service_rate', 0)
            execution_time = data.get('time', 0)

            results_by_group[group_key]['fitness'].append(fitness)
            results_by_group[group_key]['cost'].append(cost)
            results_by_group[group_key]['service_rate'].append(service_rate)
            results_by_group[group_key]['time'].append(execution_time)

            results_by_size[f"N{size}"]['fitness'].append(fitness)
            results_by_size[f"N{size}"]['cost'].append(cost)
            results_by_size[f"N{size}"]['service_rate'].append(service_rate)
            results_by_size[f"N{size}"]['time'].append(execution_time)

        except Exception as e:
            print(f"  Error reading {json_file.name}: {e}")

    # Create summary by distribution and size
    summary_by_group = []
    for group_key in sorted(results_by_group.keys()):
        metrics = results_by_group[group_key]

        fitness_mean = np.mean(metrics['fitness'])
        fitness_std = np.std(metrics['fitness'])

        cost_mean = np.mean(metrics['cost'])
        cost_std = np.std(metrics['cost'])

        sr_mean = np.mean(metrics['service_rate'])
        sr_std = np.std(metrics['service_rate'])

        time_mean = np.mean(metrics['time'])

        summary_by_group.append({
            'Distribution_Size': group_key,
            'Fitness': f"{fitness_mean:.2f}",
            'Fitness_Error': f"±{fitness_std:.2f}",
            'Cost': f"{cost_mean:.2f}",
            'Cost_Error': f"±{cost_std:.2f}",
            'Service_Rate': f"{sr_mean:.4f}",
            'Service_Rate_Error': f"±{sr_std:.4f}",
            'Avg_Time_s': f"{time_mean:.2f}",
        })

    # Create summary by size
    summary_by_size = []
    for size_key in sorted(results_by_size.keys(), key=lambda x: int(x[1:])):
        metrics = results_by_size[size_key]

        fitness_mean = np.mean(metrics['fitness'])
        fitness_std = np.std(metrics['fitness'])

        cost_mean = np.mean(metrics['cost'])
        cost_std = np.std(metrics['cost'])

        sr_mean = np.mean(metrics['service_rate'])
        sr_std = np.std(metrics['service_rate'])

        time_mean = np.mean(metrics['time'])

        summary_by_size.append({
            'Size': size_key,
            'Fitness': f"{fitness_mean:.2f}",
            'Fitness_Error': f"±{fitness_std:.2f}",
            'Cost': f"{cost_mean:.2f}",
            'Cost_Error': f"±{cost_std:.2f}",
            'Service_Rate': f"{sr_mean:.4f}",
            'Service_Rate_Error': f"±{sr_std:.4f}",
            'Avg_Time_s': f"{time_mean:.2f}",
        })

    # Save summaries
    output_dir = baseline_dir.parent / 'summary'
    output_dir.mkdir(exist_ok=True)

    # By distribution and size
    df_group = pd.DataFrame(summary_by_group)
    group_csv = output_dir / f"{baseline_name}_by_distribution_size.csv"
    df_group.to_csv(group_csv, index=False)
    print(f"  ✓ Created {group_csv}")
    print(f"\n{df_group.to_string(index=False)}\n")

    # By size
    df_size = pd.DataFrame(summary_by_size)
    size_csv = output_dir / f"{baseline_name}_by_size.csv"
    df_size.to_csv(size_csv, index=False)
    print(f"  ✓ Created {size_csv}")
    print(f"\n{df_size.to_string(index=False)}\n")


def main():
    result_dir = Path('result')

    if not result_dir.exists():
        print(f"Error: {result_dir} not found")
        return

    # Process each baseline
    for baseline_dir in sorted(result_dir.iterdir()):
        if baseline_dir.is_dir():
            baseline_name = baseline_dir.name
            process_baseline(baseline_dir, baseline_name)


if __name__ == '__main__':
    main()
