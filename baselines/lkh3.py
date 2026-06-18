"""
LKH3 Solver for VRPBTW with Bi-Objective Evaluation

Solves instances using LKH3 heuristic with service-rate-first scalarization.
Interface compatible with ga.py and lns.py baselines.

Usage:
    # Solve single instance
    python lkh3.py --filename S042_N50_C_R50.json --subfolder N50

    # Solve all instances of specific sizes
    python lkh3.py --sizes N10 N50 N100

    # Solve all instances of specific sizes with custom LKH3 binary
    python lkh3.py --sizes N50 N100 --lkh-binary ./lkh3/LKH-3.0.13/LKH
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, Any, Optional, List

import numpy as np

from problem import Problem
from utils import save_result, get_data_files


class LKH3Solver:
    """LKH3 solver with bi-objective evaluation."""

    def __init__(self, problem: Problem, data_path: str,
                 lkh_binary: str = "./lkh3/LKH-3.0.13/LKH",
                 lkh_root: str = "./lkh_files"):
        """
        Args:
            problem: Problem instance
            data_path: Path to JSON problem file
            lkh_binary: Path to LKH3 executable
            lkh_root: Directory for LKH3 files
        """
        self.problem = problem
        self.data_path = data_path
        self.lkh_binary = lkh_binary
        self.lkh_root = lkh_root

        # Load full JSON for config
        with open(data_path) as f:
            self.data = json.load(f)
        self.config = self.data["Config"]

        os.makedirs(lkh_root, exist_ok=True)

    def solve(self) -> Dict[str, Any]:
        """
        Solve using LKH3 and evaluate with bi-objective formula.

        Returns:
            dict with keys: time, service_rate, cost, objective, tour, served_count
        """
        start_time = time.time()

        try:
            # Step 1: Generate TSPLIB problem file
            problem_file, param_file, metadata_file = self._generate_lkh_files()

            # Step 2: Run LKH3
            tour_file = self._run_lkh3(param_file)

            # Step 3: Parse tour and evaluate
            result = self._evaluate_solution(tour_file, metadata_file)

            end_time = time.time()
            result["time"] = end_time - start_time

            return result

        except Exception as e:
            raise RuntimeError(f"LKH3 solve failed: {e}")

    def _generate_lkh_files(self) -> tuple:
        """Generate TSPLIB problem file. Returns (problem_file, param_file, metadata_file)."""
        # Extract coordinates and demands from Problem
        coords_list = [np.array(node.coord) for node in self.problem.nodes]
        demands = {i: node.demand for i, node in enumerate(self.problem.nodes)}
        time_windows = {i: node.time_window for i, node in enumerate(self.problem.nodes)}
        service_times = {i: self.problem.service_time for i in range(len(self.problem.nodes))}
        service_times[0] = 0  # Depot has no service time

        # Identify backhaul indices (negative demand) BEFORE adding end depot
        backhaul_indices = [i for i, node in enumerate(self.problem.nodes) if node.demand < 0]

        # Add end depot (duplicate of start depot for return journey)
        # This is REQUIRED for LKH3 CVRPTW format
        start_depot_coord = coords_list[0]
        depot_time_window = time_windows[0]

        coords_list.append(start_depot_coord)  # End depot at same location as start
        end_depot_idx = len(coords_list) - 1

        demands[end_depot_idx] = 0
        time_windows[end_depot_idx] = depot_time_window
        service_times[end_depot_idx] = 0

        coords = np.array(coords_list)
        n_nodes = len(coords)
        base_name = Path(self.data_path).stem

        # Compute distance matrices
        d = np.zeros((n_nodes, n_nodes))
        for i in range(n_nodes):
            for j in range(n_nodes):
                if i != j:
                    d[i, j] = np.linalg.norm(coords[i] - coords[j], ord=1)

        # Scale distances to integers
        SCALE_FACTOR = 1000
        d_int = (d * SCALE_FACTOR).astype(int)

        problem_file = os.path.join(self.lkh_root, f"{base_name}.vrp")
        param_file = os.path.join(self.lkh_root, f"{base_name}.par")
        tour_file = os.path.join(self.lkh_root, f"{base_name}.tour")
        meta_file = os.path.join(self.lkh_root, f"{base_name}.meta.json")

        # Write TSPLIB VRP file
        with open(problem_file, "w") as f:
            f.write(f"NAME : {base_name}\n")
            f.write(f"COMMENT : Mixed VRPBTW instance (bi-objective)\n")
            f.write(f"TYPE : CVRPTW\n")
            f.write(f"DIMENSION : {n_nodes}\n")
            f.write(f"EDGE_WEIGHT_TYPE : EXPLICIT\n")
            f.write(f"EDGE_WEIGHT_FORMAT : FULL_MATRIX\n")
            f.write(f"CAPACITY : {int(self.config['Vehicles']['CAPACITY_TRUCK'])}\n")
            f.write(f"VEHICLES : {self.config['Vehicles']['NUM_TRUCKS']}\n")

            f.write("EDGE_WEIGHT_SECTION\n")
            for i in range(n_nodes):
                for j in range(n_nodes):
                    f.write(f"{d_int[i, j]} ")
                f.write("\n")

            f.write("DEMAND_SECTION\n")
            for i in range(n_nodes):
                f.write(f"{i+1} {int(demands.get(i, 0))}\n")

            f.write("DEPOT_SECTION\n")
            f.write("1\n")  # Only list start depot (LKH3 handles return implicitly)
            f.write("-1\n")

            f.write("TIME_WINDOW_SECTION\n")
            for i in range(n_nodes):
                tw = time_windows.get(i, (0, self.config["General"]["T_MAX_SYSTEM_H"]))
                early = int(tw[0] * 60)
                late = int(tw[1] * 60)
                f.write(f"{i+1} {early} {late}\n")

            f.write("SERVICE_TIME_SECTION\n")
            for i in range(n_nodes):
                st = int(service_times.get(i, 0) * 60)
                f.write(f"{i+1} {st}\n")

            # Add backhaul section if applicable
            if backhaul_indices:
                f.write("BACKHAUL_SECTION\n")
                for node_id in backhaul_indices:
                    f.write(f"{node_id+1}\n")

            f.write("EOF\n")

        # Write parameter file
        with open(param_file, "w") as f:
            f.write(f"PROBLEM_FILE = {os.path.abspath(problem_file)}\n")
            f.write(f"OUTPUT_TOUR_FILE = {os.path.abspath(tour_file)}\n")
            f.write(f"RUNS = 10\n")
            f.write(f"TIME_LIMIT = 3600\n")
            f.write(f"SEED = 1234\n")
            f.write(f"TRACE_LEVEL = 0\n")
            f.write(f"MAX_TRIALS = 1000\n")
            f.write(f"POPULATION_SIZE = 50\n")
            f.write(f"MTSP_OBJECTIVE = MINSUM\n")
            f.write(f"INITIAL_PERIOD = 1000\n")
            f.write(f"MAX_SWAPS = 1000\n")

        # Save metadata
        metadata = {
            "n_nodes": n_nodes,
            "num_customers": self.config["General"]["NUM_CUSTOMERS"],
            "scale_factor": SCALE_FACTOR,
            "distance_matrix": d.tolist(),
            "truck_cost_unit": self.config["Vehicles"]["TRUCK_COST_UNIT"],
            "drone_cost_unit": self.config["Vehicles"]["DRONE_COST_UNIT"],
            "fleet_basis_cost": self.config["Vehicles"]["FLEET_BASIS_COST"],
            "num_vehicles": self.config["Vehicles"]["NUM_TRUCKS"],
        }

        with open(meta_file, "w") as f:
            json.dump(metadata, f, indent=2)

        return problem_file, param_file, meta_file

    def _run_lkh3(self, param_file: str) -> str:
        """Run LKH3 solver. Returns path to tour file."""
        tour_file = param_file.replace(".par", ".tour")

        result = subprocess.run(
            [self.lkh_binary, param_file],
            capture_output=True,
            text=True,
            timeout=3600
        )

        # Check for infeasibility in output
        is_infeasible = "Successes/Runs = 0/" in result.stdout

        # Print LKH3 output for debugging
        if result.stdout:
            print("\n[LKH3 Output]")
            # Show last 20 lines of output
            lines = result.stdout.strip().split('\n')
            for line in lines[-20:]:
                print(f"  {line}")

        if result.stderr:
            print("\n[LKH3 Errors]")
            print(result.stderr)

        if is_infeasible:
            raise RuntimeError(
                "LKH3 found no feasible solution (Successes/Runs = 0/10). "
                "This instance's time windows/capacity constraints are too tight. "
                "Try: python lkh3.py --subfolder N50  (larger instances are usually feasible)"
            )

        if not os.path.exists(tour_file):
            raise FileNotFoundError(f"LKH3 did not create tour file: {tour_file}")

        return tour_file

    def _evaluate_solution(self, tour_file: str, metadata_file: str) -> Dict[str, Any]:
        """Parse tour and evaluate using bi-objective formula."""
        # Load metadata
        with open(metadata_file) as f:
            meta = json.load(f)

        # Parse tour (1-indexed nodes)
        with open(tour_file) as f:
            lines = f.readlines()

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
                tour.append(int(line) - 1)

        # Compute cost
        dist_matrix = np.array(meta["distance_matrix"])
        scale_factor = meta["scale_factor"]
        c_t = meta["truck_cost_unit"]

        cost = 0.0
        for i in range(len(tour) - 1):
            u, v = tour[i], tour[i + 1]
            dist = dist_matrix[u][v] / scale_factor
            cost += dist * c_t

        # Add fleet basis cost
        cost += meta["fleet_basis_cost"] * meta["num_vehicles"]

        # Compute service metrics
        served_customers = set(tour[1:-1])  # Skip depot at start/end
        served_count = len(served_customers)
        total_customers = meta["num_customers"]
        service_rate = served_count / total_customers if total_customers > 0 else 0.0

        # Compute bi-objective
        objective = self._compute_objective(served_count, cost)

        return {
            "service_rate": service_rate,
            "served_count": served_count,
            "cost": cost,
            "objective": objective,
            "tour": tour,
        }

    def _compute_objective(self, served_count: int, cost: float) -> float:
        """
        Compute bi-objective using service-rate-first formula.

        f = max(c_t, c_d) * max_dist * N * k - cost
        """
        N = self.config["General"]["NUM_CUSTOMERS"]
        k = served_count
        max_coord = self.config["General"]["MAX_COORD_KM"]
        max_dist = 2.0 * max_coord

        c_t = self.config["Vehicles"]["TRUCK_COST_UNIT"]
        c_d = self.config["Vehicles"]["DRONE_COST_UNIT"]
        c_b = self.config["Vehicles"]["FLEET_BASIS_COST"]
        num_vehicles = self.config["Vehicles"]["NUM_TRUCKS"]

        weight = max(c_t, c_d) * max_dist * N + c_b * num_vehicles
        service_reward = weight * k
        return service_reward - cost


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="LKH3 solver for Mixed VRPBTW with bi-objective evaluation",
        usage="python lkh3.py (--filename FILE | --subfolder FOLDER) [options]"
    )

    # Mutually exclusive: single filename or subfolder
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--filename",
        type=str,
        help="Single problem filename (e.g., S042_N50_C_R50.json)"
    )
    group.add_argument(
        "--subfolder",
        type=str,
        help="Subfolder name containing multiple files (e.g., N50, N100)"
    )

    parser.add_argument(
        "--lkh-binary",
        type=str,
        default="./lkh3/LKH-3.0.13/LKH",
        help="Path to LKH3 binary"
    )
    parser.add_argument(
        "--lkh-root",
        type=str,
        default="./lkh_files",
        help="Directory for LKH3 input/output files"
    )

    args = parser.parse_args()

    base_dir = Path(__file__).parent.parent / "data" / "generated" / "data"
    result_dir = Path(__file__).parent / "result" / "lkh3"

    print(f"\n{'='*80}")
    print("LKH3 Solver - Bi-Objective VRPBTW")
    print(f"{'='*80}\n")

    # Handle single filename
    if args.filename:
        # Infer subfolder from filename if not explicit
        parts = args.filename.split("_")
        if len(parts) > 1:
            subfolder = parts[1]
        else:
            print("ERROR: Could not infer subfolder from filename")
            sys.exit(1)

        data_path = base_dir / subfolder / args.filename
        if not data_path.exists():
            print(f"ERROR: File not found: {data_path}")
            sys.exit(1)

        print(f"Solving single instance: {args.filename}")
        print(f"  Path: {data_path}\n")

        try:
            problem = Problem(str(data_path))
            solver = LKH3Solver(problem, data_path=str(data_path),
                               lkh_binary=args.lkh_binary, lkh_root=args.lkh_root)
            result = solver.solve()

            print(f"✓ Completed in {result['time']:.2f} seconds")
            print(f"  Service Rate: {result['service_rate']:.2%}")
            print(f"  Served: {result['served_count']}/{problem.num_customer}")
            print(f"  Cost: {result['cost']:.2f}")
            print(f"  Objective: {result['objective']:.2f}\n")

            # Save result
            output_path = result_dir / subfolder / args.filename
            output_path.parent.mkdir(parents=True, exist_ok=True)
            save_result(result, output_path)
            print(f"Result saved to: {output_path}\n")

        except Exception as e:
            print(f"✗ ERROR: {str(e)}\n")
            import traceback
            traceback.print_exc()
            sys.exit(1)

    # Handle subfolder (all files in folder)
    elif args.subfolder:
        folder_path = base_dir / args.subfolder
        if not folder_path.exists():
            print(f"ERROR: Folder not found: {folder_path}")
            sys.exit(1)

        # Get all JSON files in subfolder
        data_files = list(folder_path.glob("*.json"))
        if not data_files:
            print(f"ERROR: No JSON files found in {folder_path}")
            sys.exit(1)

        data_files.sort()

        print(f"Found {len(data_files)} instances in {args.subfolder}/\n")
        for f in data_files[:5]:
            print(f"  {f.name}")
        if len(data_files) > 5:
            print(f"  ... and {len(data_files) - 5} more")
        print()

        print(f"{'-'*80}")
        print(f"LKH3 on {args.subfolder}")
        print(f"{'-'*80}\n")

        completed = 0
        succeeded = 0
        failed = 0

        for data_file in data_files:
            completed += 1
            print(f"[{completed}/{len(data_files)}] Processing: {data_file.name}...", end=" ")

            try:
                problem = Problem(str(data_file))
                solver = LKH3Solver(problem, data_path=str(data_file),
                                   lkh_binary=args.lkh_binary, lkh_root=args.lkh_root)
                result = solver.solve()

                output_path = result_dir / args.subfolder / data_file.name
                output_path.parent.mkdir(parents=True, exist_ok=True)
                save_result(result, output_path)

                print(f"✓ ({result['time']:.1f}s, SR={result['service_rate']:.0%}, "
                      f"cost={result['cost']:.0f})")
                succeeded += 1

            except Exception as e:
                print(f"✗ ({str(e)[:50]}...)")
                failed += 1
                continue

        print(f"\n{'='*80}")
        print(f"LKH3 experiments completed!")
        print(f"  Total: {len(data_files)}")
        print(f"  Succeeded: {succeeded}")
        print(f"  Failed: {failed}")
        print(f"{'='*80}\n")

        if failed > 0:
            sys.exit(1)


if __name__ == "__main__":
    main()
