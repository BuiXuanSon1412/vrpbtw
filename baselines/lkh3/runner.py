"""
LKH3 Runner with Bi-Objective Evaluation

Simple wrapper that:
1. Runs existing convert.py to create valid TSPLIB files
2. Executes LKH3 solver
3. Parses tour and evaluates using bi-objective formula
"""

import json
import os
import subprocess
import sys
import argparse
from pathlib import Path
from typing import Optional, Tuple, Dict, Any
import numpy as np


def run_convert(filename: str, subfolder: Optional[str] = None,
                data_root: str = "../data/generated/data/") -> Tuple[str, str, str]:
    """Run existing convert.py to generate LKH3 files."""
    # Build command
    cmd = [sys.executable, "lkh3/convert.py", "--filename", filename]
    if subfolder:
        cmd.extend(["--subfolder", subfolder])

    result = subprocess.run(cmd, capture_output=True, text=True, cwd=os.path.dirname(os.path.abspath(__file__)) + "/..")
    if result.returncode != 0:
        raise RuntimeError(f"convert.py failed:\n{result.stderr}")

    # Parse output to find file paths
    # convert.py prints: "LKH-3 problem file created: {problem_file}"
    base_name = filename.replace(".json", "")
    problem_file = f"lkh_files/{base_name}.vrp"
    param_file = f"lkh_files/{base_name}.par"
    meta_file = f"lkh_files/{base_name}.meta.json"

    return problem_file, param_file, meta_file


def run_lkh3(param_file: str, lkh_binary: str = "./lkh3/LKH-3.0.13/LKH") -> str:
    """Run LKH3 solver."""
    tour_file = param_file.replace(".par", ".tour")

    result = subprocess.run([lkh_binary, param_file], capture_output=True, text=True, timeout=3600)

    if result.returncode != 0:
        print(f"LKH3 stderr:\n{result.stderr}", file=sys.stderr)
        print(f"LKH3 stdout:\n{result.stdout}", file=sys.stderr)

    if not os.path.exists(tour_file):
        raise FileNotFoundError(f"LKH3 tour file not created: {tour_file}\nLKH3 output:\n{result.stderr}")

    return tour_file


def parse_tour(tour_file: str) -> list:
    """Parse LKH3 tour file (1-indexed nodes)."""
    with open(tour_file, "r") as f:
        lines = f.readlines()

    # Find TOUR_SECTION
    tour_start = 0
    for i, line in enumerate(lines):
        if "TOUR_SECTION" in line:
            tour_start = i + 1
            break

    tour = []
    for line in lines[tour_start:]:
        line = line.strip()
        if line == "-1":
            break
        if line and line[0].isdigit():
            tour.append(int(line) - 1)  # Convert to 0-indexed

    return tour


def compute_objective(served_count: int, cost: float, config: Dict[str, Any]) -> float:
    """
    Compute bi-objective using service-rate-first formula.

    f = max(c_t, c_d) * max_dist * N * k - cost
    """
    N = config["General"]["NUM_CUSTOMERS"]
    k = served_count
    max_coord = config["General"]["MAX_COORD_KM"]
    max_dist = 2.0 * max_coord

    c_t = config["Vehicles"]["TRUCK_COST_UNIT"]
    c_d = config["Vehicles"]["DRONE_COST_UNIT"]
    c_b = config["Vehicles"]["FLEET_BASIS_COST"]
    num_vehicles = config["Vehicles"]["NUM_TRUCKS"]

    weight = max(c_t, c_d) * max_dist * N + c_b * num_vehicles
    service_reward = weight * k
    return service_reward - cost


def compute_tour_cost(tour: list, metadata: Dict[str, Any]) -> float:
    """Compute tour cost from distance matrix."""
    dist_matrix = np.array(metadata["distance_matrix"])
    scale_factor = metadata.get("scale_factor", 1000)

    c_t = metadata.get("truck_cost_unit", 1.0)
    cost = 0.0

    for i in range(len(tour) - 1):
        u, v = tour[i], tour[i + 1]
        dist = dist_matrix[u][v] / scale_factor
        cost += dist * c_t

    # Add fixed fleet cost
    cost += metadata.get("fleet_basis_cost", 0) * metadata.get("num_vehicles", 1)

    return cost


def extract_served_customers(tour: list, metadata: Dict[str, Any]) -> list:
    """Extract unique served customers from tour (skip depot nodes)."""
    depot_idx = metadata["depot_idx"]
    end_depot_idx = metadata.get("end_depot_idx")
    n_nodes = metadata["n_nodes"]

    served = []
    for node in tour:
        if node != depot_idx and (end_depot_idx is None or node != end_depot_idx) and 1 <= node < n_nodes:
            served.append(node)

    return list(set(served))  # Unique served customers


def solve(filename: str, subfolder: Optional[str] = None, data_root: str = "../data/generated/data/",
          lkh_binary: str = "./lkh3/LKH-3.0.13/LKH", verbose: bool = True) -> Dict[str, Any]:
    """
    Full pipeline: convert → run LKH3 → evaluate.

    Returns dict with:
      - service_rate: fraction of customers served
      - served_count: number of served customers
      - total_customers: total customers in problem
      - cost: total travel cost
      - objective: bi-objective value
      - tour: node sequence
    """
    if verbose:
        print(f"[LKH3-Solver] Converting {filename}...")

    # Determine data path for config loading
    if subfolder:
        data_path = os.path.join(data_root, subfolder, filename)
    else:
        parts = filename.split("_")
        n_folder = parts[1]
        data_path = os.path.join(data_root, n_folder, filename)

    # Load config
    with open(data_path) as f:
        data = json.load(f)
    config = data["Config"]

    # Run converter
    try:
        problem_file, param_file, meta_file = run_convert(filename, subfolder, data_root)
    except Exception as e:
        raise RuntimeError(f"Conversion failed: {e}")

    if verbose:
        print(f"  Problem file: {problem_file}")

    # Run LKH3
    if verbose:
        print(f"[LKH3-Solver] Running LKH3...")
    try:
        tour_file = run_lkh3(param_file, lkh_binary)
    except Exception as e:
        raise RuntimeError(f"LKH3 solve failed: {e}")

    # Parse solution
    if verbose:
        print(f"[LKH3-Solver] Parsing solution...")

    tour = parse_tour(tour_file)

    with open(meta_file) as f:
        metadata = json.load(f)

    served_customers = extract_served_customers(tour, metadata)
    cost = compute_tour_cost(tour, metadata)

    # Compute objective
    served_count = len(served_customers)
    total_customers = config["General"]["NUM_CUSTOMERS"]
    service_rate = served_count / total_customers if total_customers > 0 else 0.0
    objective = compute_objective(served_count, cost, config)

    result = {
        "filename": filename,
        "service_rate": service_rate,
        "served_count": served_count,
        "total_customers": total_customers,
        "cost": cost,
        "objective": objective,
        "tour": tour,
        "tour_length": len(tour),
    }

    if verbose:
        print(f"\n{'='*60}")
        print(f"SOLUTION SUMMARY")
        print(f"{'='*60}")
        print(f"Service Rate:       {result['service_rate']:.2%}")
        print(f"Served Customers:   {result['served_count']}/{result['total_customers']}")
        print(f"Total Cost:         {result['cost']:,.2f}")
        print(f"Objective Value:    {result['objective']:,.2f}")
        print(f"Tour Length:        {result['tour_length']} nodes")
        print(f"{'='*60}\n")

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Run LKH3 solver with bi-objective evaluation for Mixed VRPBTW"
    )
    parser.add_argument("--filename", type=str, required=True,
                        help="JSON problem filename (e.g., S042_N50_C_R50.json)")
    parser.add_argument("--subfolder", type=str, default=None,
                        help="Subfolder within data root")
    parser.add_argument("--data-root", type=str, default="../data/generated/data/",
                        help="Root data directory")
    parser.add_argument("--lkh-binary", type=str, default="./lkh3/LKH-3.0.13/LKH",
                        help="Path to LKH3 executable")
    parser.add_argument("--quiet", action="store_true", help="Suppress output")

    args = parser.parse_args()

    try:
        result = solve(
            filename=args.filename,
            subfolder=args.subfolder,
            data_root=args.data_root,
            lkh_binary=args.lkh_binary,
            verbose=not args.quiet
        )

        # Exit with status indicating if solution is good
        if result["service_rate"] > 0.95:
            sys.exit(0)  # Success
        else:
            sys.exit(1)  # Partial coverage

    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()
