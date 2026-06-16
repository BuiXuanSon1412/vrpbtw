#!/usr/bin/env python3
"""
Option 2: Iterative Removal + LKH-3
Automatically finds maximum feasible subset by removing infeasible customers
"""

import json
import numpy as np
import os
import subprocess
import re
from copy import deepcopy

DATA_ROOT = "../data/generated/data/"
LKH_ROOT = "./lkh_files"

def load_problem_data(filename, subfolder):
    """Load problem JSON"""
    if subfolder:
        data_path = os.path.join(DATA_ROOT, subfolder, filename)
    else:
        parts = filename.split("_")
        n_folder = parts[1]
        data_path = os.path.join(DATA_ROOT, n_folder, filename)

    with open(data_path) as f:
        return json.load(f)

def create_partial_vrp(data, excluded_customers, base_name, attempt_num):
    """Create .vrp file excluding specified customers"""

    config = data["Config"]
    all_nodes = data["Nodes"]

    # Filter nodes
    filtered_nodes = [n for n in all_nodes if n["id"] not in excluded_customers]

    num_customers = len(filtered_nodes)
    num_nodes = num_customers + 2  # +2 for start/end depot

    print(f"\n  Attempt {attempt_num}: {num_customers} customers (excluded {len(excluded_customers)})")

    # Build coordinate array
    depot_coord = np.array(config["Depot"]["coord"])
    coords = [depot_coord]

    for node in filtered_nodes:
        coords.append(np.array(node["coord"]))
    coords.append(depot_coord)
    coords = np.array(coords)

    # Build distance matrix (Manhattan for truck)
    d = np.zeros((num_nodes, num_nodes))
    for i in range(num_nodes):
        for j in range(num_nodes):
            if i != j:
                d[i, j] = np.linalg.norm(coords[i] - coords[j], ord=1)

    SCALE_FACTOR = 1000
    d_int = (d * SCALE_FACTOR).astype(int)

    # Build demand dict
    demands = {0: 0}
    for i, node in enumerate(filtered_nodes, 1):
        demands[i] = node["demand"]
    demands[num_nodes - 1] = 0

    # Build time windows dict
    time_windows = {0: [0, 24]}
    for i, node in enumerate(filtered_nodes, 1):
        time_windows[i] = node["tw_h"]
    time_windows[num_nodes - 1] = [0, 24]

    # Build service times dict
    service_time = config["Vehicles"]["SERVICE_TIME_MIN"] / 60.0
    service_times = {0: 0}
    for i in range(1, num_nodes - 1):
        service_times[i] = service_time
    service_times[num_nodes - 1] = 0

    # Write .vrp file
    vrp_name = f"{base_name}_ATTEMPT{attempt_num}"
    problem_file = os.path.join(LKH_ROOT, f"{vrp_name}.vrp")
    param_file = os.path.join(LKH_ROOT, f"{vrp_name}.par")
    tour_file = os.path.join(LKH_ROOT, f"{vrp_name}.tour")

    with open(problem_file, "w") as f:
        f.write(f"NAME : {vrp_name}\n")
        f.write(f"COMMENT : Iterative removal attempt\n")
        f.write(f"TYPE : CVRPTW\n")
        f.write(f"DIMENSION : {num_nodes}\n")
        f.write(f"EDGE_WEIGHT_TYPE : EXPLICIT\n")
        f.write(f"EDGE_WEIGHT_FORMAT : FULL_MATRIX\n")
        f.write(f"CAPACITY : {config['Vehicles']['CAPACITY_TRUCK']}\n")
        f.write(f"VEHICLES : {config['Vehicles']['NUM_TRUCKS']}\n")

        f.write("EDGE_WEIGHT_SECTION\n")
        for i in range(num_nodes):
            for j in range(num_nodes):
                f.write(f"{d_int[i, j]} ")
            f.write("\n")

        f.write("DEMAND_SECTION\n")
        for i in range(num_nodes):
            f.write(f"{i+1} {demands.get(i, 0)}\n")

        f.write("DEPOT_SECTION\n")
        f.write("1\n")
        f.write("-1\n")

        f.write("TIME_WINDOW_SECTION\n")
        for i in range(num_nodes):
            tw = time_windows.get(i, [0, 24])
            early = int(tw[0] * 60)
            late = int(tw[1] * 60)
            f.write(f"{i+1} {early} {late}\n")

        f.write("SERVICE_TIME_SECTION\n")
        for i in range(num_nodes):
            st = int(service_times.get(i, 0) * 60)
            f.write(f"{i+1} {st}\n")

        f.write("EOF\n")

    # Write .par file
    with open(param_file, "w") as f:
        f.write(f"PROBLEM_FILE = {problem_file}\n")
        f.write(f"OUTPUT_TOUR_FILE = {tour_file}\n")
        f.write(f"RUNS = 1\n")
        f.write(f"TIME_LIMIT = 60\n")
        f.write(f"SEED = 1234\n")
        f.write(f"TRACE_LEVEL = 1\n")
        f.write(f"MAX_TRIALS = 1000\n")
        f.write(f"POPULATION_SIZE = 50\n")
        f.write(f"MTSP_OBJECTIVE = MINSUM\n")
        f.write(f"INITIAL_PERIOD = 1000\n")
        f.write(f"MAX_SWAPS = 1000\n")

    return param_file, vrp_name, num_customers, filtered_nodes

def parse_lkh_output(output):
    """Extract penalty and cost from LKH output"""
    penalty = None
    cost = None

    for line in output.split('\n'):
        if 'Cost =' in line and 'Best CVRPTW' in line:
            match = re.search(r'Cost = (\d+)_(\d+)', line)
            if match:
                penalty = int(match.group(1))
                cost = int(match.group(2))

    return penalty, cost

def solve_with_lkh(param_file, vrp_name):
    """Run LKH-3 and return penalty"""
    print(f"    Running LKH-3...", end='', flush=True)

    result = subprocess.run(
        [f"./lkh3/LKH-3.0.13/LKH", param_file],
        capture_output=True,
        text=True,
        timeout=120
    )

    penalty, cost = parse_lkh_output(result.stdout)

    if penalty is None:
        penalty = float('inf')

    print(f" Penalty = {penalty:,}")
    return penalty

def find_worst_customer(filtered_nodes, num_customers):
    """
    Remove customer with smallest time window (least flexible)
    """
    worst_idx = None
    worst_score = float('-inf')

    for i, node in enumerate(filtered_nodes):
        tw = node["tw_h"]
        window_width = tw[1] - tw[0]
        score = -window_width
        if score > worst_score:
            worst_score = score
            worst_idx = i

    return filtered_nodes[worst_idx]["id"]

def main():
    print("=" * 80)
    print("OPTION 2: ITERATIVE REMOVAL + LKH-3")
    print("=" * 80)

    filename = "S042_N50_C_R50.json"
    subfolder = "N50"
    base_name = filename.replace(".json", "")

    data = load_problem_data(filename, subfolder)
    all_customer_ids = set(n["id"] for n in data["Nodes"])

    print(f"\nStarting with: {len(all_customer_ids)} customers")

    excluded = set()
    attempt = 0
    best_feasible = None
    best_excluded = None

    max_attempts = 20

    while attempt < max_attempts:
        attempt += 1

        param_file, vrp_name, num_customers, filtered_nodes = create_partial_vrp(
            data, excluded, base_name, attempt
        )

        if num_customers == 0:
            print("  (No customers left!)")
            break

        penalty = solve_with_lkh(param_file, vrp_name)

        if penalty == 0:
            print(f"    ✅ FEASIBLE SOLUTION FOUND!")
            best_feasible = vrp_name
            best_excluded = excluded.copy()
            print(f"\n    Service rate: {num_customers}/50 = {num_customers*100//50}%")
            break
        else:
            print(f"    Still infeasible, removing worst customer...")
            worst_id = find_worst_customer(filtered_nodes, num_customers)
            excluded.add(worst_id)
            print(f"    Removing customer {worst_id}")

    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)

    if best_feasible:
        served = len(all_customer_ids) - len(best_excluded)
        print(f"\n✅ SUCCESS: Found feasible solution!")
        print(f"  Served: {served} / 50 customers ({served*100//50}%)")
        print(f"  Excluded: {sorted(best_excluded)}")
        print(f"  Solution file: {LKH_ROOT}/{best_feasible}.tour")
        return True
    else:
        print(f"\n❌ FAILED: Could not find feasible solution in {attempt} attempts")
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
