"""
LKH3 Bi-objective Solver for Mixed VRPBTW
==========================================

Implements scalarized bi-objective optimization: service rate (primary) + cost (secondary).
Uses the same objective as VRPBTWEnv._compute_objective:
    f = max(c_t, c_d) * max_dist * N * k - cost

Where:
  - k: number of served customers (primary objective)
  - cost: total travel cost (secondary objective)
  - N, max_dist: problem scale factors
"""

import json
import os
import subprocess
import numpy as np
from typing import List, Tuple, Dict, Any, Optional
from pathlib import Path
from dataclasses import dataclass


@dataclass
class BiObjSolution:
    """Solution with bi-objective metrics."""
    served_customers: List[int]  # customer IDs actually served
    tour: List[int]  # full LKH3 tour (for reference)
    cost: float  # total travel cost
    served_count: int  # |served_customers|
    service_rate: float  # served_count / total_customers
    objective: float  # scalarized: service_reward - cost

    def __repr__(self) -> str:
        return (f"BiObjSolution(SR={self.service_rate:.2%}, cost={self.cost:.2f}, "
                f"obj={self.objective:.2f}, served={self.served_count})")


class BiObjLKH3Solver:
    """
    Solves bi-objective VRPBTW using LKH3 with scalarized objective.

    Workflow:
    1. Convert problem to TSPLIB format with scalarized costs
    2. Run LKH3 solver
    3. Parse tour and evaluate using service-rate-first objective
    4. Optionally explore partial coverage solutions
    """

    def __init__(self, data_path: str, lkh_root: str = "./lkh_files",
                 lkh_binary: str = "./LKH-3.0.13/LKH"):
        """
        Args:
            data_path: path to JSON problem instance
            lkh_root: directory for LKH3 input/output files
            lkh_binary: path to LKH3 executable
        """
        self.data_path = data_path
        self.lkh_root = lkh_root
        self.lkh_binary = lkh_binary
        self.metadata = {}
        self.problem_config = {}
        self.load_problem()

    def load_problem(self):
        """Load problem from JSON."""
        with open(self.data_path, "r") as f:
            data = json.load(f)

        config = data["Config"]
        self.problem_config = config
        self.num_customers = config["General"]["NUM_CUSTOMERS"]
        self.num_nodes = config["General"]["NUM_NODES"]
        self.max_coord = config["General"]["MAX_COORD_KM"]
        self.T_max = config["General"]["T_MAX_SYSTEM_H"]

        self.Q_t = config["Vehicles"]["CAPACITY_TRUCK"]
        self.Q_d = config["Vehicles"]["CAPACITY_DRONE"]
        self.num_vehicles = config["Vehicles"]["NUM_TRUCKS"]
        self.num_drones = config["Vehicles"]["NUM_DRONES"]

        self.V_TRUCK = config["Vehicles"]["V_TRUCK_KM_H"]
        self.V_DRONE = config["Vehicles"]["V_DRONE_KM_H"]
        self.c_t = config["Vehicles"]["TRUCK_COST_UNIT"]  # truck cost per km
        self.c_d = config["Vehicles"]["DRONE_COST_UNIT"]  # drone cost per km
        self.c_b = config["Vehicles"]["FLEET_BASIS_COST"]  # per-vehicle fixed cost

        self.tau_l = config["Vehicles"]["DRONE_TAKEOFF_MIN"] / 60.0
        self.tau_r = config["Vehicles"]["DRONE_LANDING_MIN"] / 60.0
        self.service_time = config["Vehicles"]["SERVICE_TIME_MIN"] / 60.0
        self.T_tilde_max = config["Vehicles"]["DRONE_DURATION_H"]

        # Parse nodes
        depot_info = config["Depot"]
        self.depot_idx = depot_info["id"]
        self.depot_coord = np.array(depot_info["coord"])
        self.depot_tw = depot_info["time_window_h"]

        self.coords = [self.depot_coord]
        self.demands = {self.depot_idx: 0}
        self.time_windows = {self.depot_idx: self.depot_tw}
        self.service_times = {self.depot_idx: 0}
        self.linehaul_indices = []
        self.backhaul_indices = []

        for node in data["Nodes"]:
            node_id = node["id"]
            self.coords.append(np.array(node["coord"]))
            self.demands[node_id] = node["demand"]
            self.time_windows[node_id] = node["tw_h"]
            self.service_times[node_id] = self.service_time

            if node["demand"] > 0:
                self.linehaul_indices.append(node_id)
            else:
                self.backhaul_indices.append(node_id)

        self.coords = np.array(self.coords)

    def compute_objective(self, served_count: int, cost: float) -> float:
        """
        Compute scalarized bi-objective using service-rate-first formula.

        f = max(c_t, c_d) * max_dist * N * k - cost

        This ensures:
        - Increasing k (served customers) is always rewarded
        - Among same k, lower cost is better
        """
        N = self.num_customers
        k = served_count
        max_dist = 2.0 * self.max_coord
        weight = max(self.c_t, self.c_d) * max_dist * N + self.c_b * self.num_vehicles
        service_reward = weight * k
        return service_reward - cost

    def convert_to_lkh(self, filename: Optional[str] = None) -> Tuple[str, str, str]:
        """
        Convert problem to TSPLIB format (standard CVRPTW).

        Returns:
            (problem_file, param_file, metadata_file) paths
        """
        if filename is None:
            filename = Path(self.data_path).stem

        os.makedirs(self.lkh_root, exist_ok=True)

        # Compute distance matrices
        n_nodes = len(self.coords)
        end_depot_idx = n_nodes
        coords = np.vstack([self.coords, self.depot_coord])

        d = np.zeros((n_nodes + 1, n_nodes + 1))
        d_tilde = np.zeros((n_nodes + 1, n_nodes + 1))

        for i in range(n_nodes + 1):
            for j in range(n_nodes + 1):
                if i != j:
                    d[i, j] = np.linalg.norm(coords[i] - coords[j], ord=1)
                    d_tilde[i, j] = np.linalg.norm(coords[i] - coords[j], ord=2)

        # Scale distances to integers for LKH
        SCALE_FACTOR = 1000
        d_int = (d * SCALE_FACTOR).astype(int)

        # Update depot end index
        self.demands[end_depot_idx] = 0
        self.time_windows[end_depot_idx] = self.depot_tw
        self.service_times[end_depot_idx] = 0

        problem_file = os.path.join(self.lkh_root, f"{filename}.vrp")
        param_file = os.path.join(self.lkh_root, f"{filename}.par")
        tour_file = os.path.join(self.lkh_root, f"{filename}.tour")

        # Write TSPLIB VRP file
        with open(problem_file, "w") as f:
            f.write(f"NAME : {filename}\n")
            f.write(f"COMMENT : Mixed VRPBTW instance (bi-objective)\n")
            f.write(f"TYPE : CVRPTW\n")
            f.write(f"DIMENSION : {n_nodes + 1}\n")
            f.write(f"EDGE_WEIGHT_TYPE : EXPLICIT\n")
            f.write(f"EDGE_WEIGHT_FORMAT : FULL_MATRIX\n")
            f.write(f"CAPACITY : {int(self.Q_t)}\n")
            f.write(f"VEHICLES : {self.num_vehicles}\n")

            f.write("EDGE_WEIGHT_SECTION\n")
            for i in range(n_nodes + 1):
                for j in range(n_nodes + 1):
                    f.write(f"{d_int[i, j]} ")
                f.write("\n")

            f.write("DEMAND_SECTION\n")
            for i in range(n_nodes + 1):
                f.write(f"{i+1} {int(self.demands.get(i, 0))}\n")

            f.write("DEPOT_SECTION\n")
            f.write("1\n")
            f.write("-1\n")

            f.write("TIME_WINDOW_SECTION\n")
            for i in range(n_nodes + 1):
                tw = self.time_windows.get(i, [0, self.T_max])
                early = int(tw[0] * 60)
                late = int(tw[1] * 60)
                f.write(f"{i+1} {early} {late}\n")

            f.write("SERVICE_TIME_SECTION\n")
            for i in range(n_nodes + 1):
                st = int(self.service_times.get(i, 0) * 60)
                f.write(f"{i+1} {st}\n")

            if self.backhaul_indices:
                f.write("BACKHAUL_SECTION\n")
                for node_id in self.backhaul_indices:
                    f.write(f"{node_id + 1}\n")

            f.write("EOF\n")

        # Write LKH parameter file
        with open(param_file, "w") as f:
            f.write(f"PROBLEM_FILE = {os.path.abspath(problem_file)}\n")
            f.write(f"OUTPUT_TOUR_FILE = {os.path.abspath(tour_file)}\n")
            f.write(f"RUNS = 10\n")
            f.write(f"TIME_LIMIT = 3600\n")
            f.write(f"SEED = 1234\n")
            f.write(f"TRACE_LEVEL = 1\n")
            f.write(f"MAX_TRIALS = 1000\n")
            f.write(f"POPULATION_SIZE = 50\n")
            f.write(f"MTSP_OBJECTIVE = MINSUM\n")
            f.write(f"INITIAL_PERIOD = 1000\n")
            f.write(f"MAX_SWAPS = 1000\n")

        # Save metadata
        metadata = {
            "filename": filename,
            "n_nodes": n_nodes + 1,
            "num_customers": self.num_customers,
            "num_vehicles": self.num_vehicles,
            "depot_idx": self.depot_idx,
            "end_depot_idx": end_depot_idx,
            "linehaul_indices": self.linehaul_indices,
            "backhaul_indices": self.backhaul_indices,
            "capacity_truck": self.Q_t,
            "capacity_drone": self.Q_d,
            "scale_factor": SCALE_FACTOR,
            "coordinates": self.coords.tolist(),
            "demands": {str(k): v for k, v in self.demands.items()},
            "distance_matrix": d.tolist(),
        }

        metadata_file = os.path.join(self.lkh_root, f"{filename}.meta.json")
        with open(metadata_file, "w") as f:
            json.dump(metadata, f, indent=2)

        return problem_file, param_file, metadata_file

    def run_lkh(self, param_file: str) -> str:
        """
        Run LKH3 solver.

        Returns:
            Path to tour file
        """
        tour_file = param_file.replace(".par", ".tour")

        result = subprocess.run(
            [self.lkh_binary, param_file],
            capture_output=True,
            text=True,
            timeout=3600
        )

        if result.returncode != 0:
            raise RuntimeError(f"LKH3 failed: {result.stderr}")

        if not os.path.exists(tour_file):
            raise FileNotFoundError(f"LKH3 did not produce tour file: {tour_file}")

        return tour_file

    def parse_tour(self, tour_file: str, metadata_file: str) -> BiObjSolution:
        """
        Parse LKH3 tour and compute bi-objective metrics.

        Returns:
            BiObjSolution with served customers, cost, and objective.
        """
        with open(metadata_file, "r") as f:
            meta = json.load(f)

        # Parse tour
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
            if line == "-1":  # End marker
                break
            if line and line[0].isdigit():
                tour.append(int(line) - 1)  # Convert to 0-indexed

        # Extract served customers (non-depot nodes in tour)
        served_customers = []
        for node_idx in tour:
            if node_idx > 0 and node_idx < meta["n_nodes"] - 1:  # Skip depot (0) and end depot
                served_customers.append(node_idx)

        # Compute cost from tour
        cost = self.compute_tour_cost(tour, meta)

        # Compute objective
        served_count = len(set(served_customers))  # Unique served customers
        objective = self.compute_objective(served_count, cost)
        service_rate = served_count / self.num_customers if self.num_customers > 0 else 0.0

        return BiObjSolution(
            served_customers=served_customers,
            tour=tour,
            cost=cost,
            served_count=served_count,
            service_rate=service_rate,
            objective=objective
        )

    def compute_tour_cost(self, tour: List[int], metadata: Dict[str, Any]) -> float:
        """Compute total travel cost for a tour."""
        dist_matrix = np.array(metadata["distance_matrix"])
        scale_factor = metadata["scale_factor"]

        cost = 0.0
        for i in range(len(tour) - 1):
            u, v = tour[i], tour[i + 1]
            # Distance in original units (distance_matrix already scaled)
            dist = dist_matrix[u][v] / scale_factor
            # Truck travel cost (could be extended for drone later)
            cost += dist * self.c_t

        # Add per-vehicle fixed cost
        cost += self.c_b * self.num_vehicles

        return cost

    def solve(self, verbose: bool = True) -> BiObjSolution:
        """
        Full pipeline: convert → run LKH3 → parse → evaluate.

        Returns:
            BiObjSolution with objective metrics
        """
        if verbose:
            print(f"[LKH3-BiObj] Converting problem...")
        problem_file, param_file, metadata_file = self.convert_to_lkh()

        if verbose:
            print(f"[LKH3-BiObj] Running LKH3...")
            print(f"  Problem file: {problem_file}")
            print(f"  Param file: {param_file}")

        tour_file = self.run_lkh(param_file)

        if verbose:
            print(f"[LKH3-BiObj] Parsing solution...")
        solution = self.parse_tour(tour_file, metadata_file)

        if verbose:
            print(f"[LKH3-BiObj] Solution: {solution}")

        return solution


def main():
    """CLI interface for bi-objective LKH3 solver."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Solve Mixed VRPBTW using LKH3 with bi-objective scalarization"
    )
    parser.add_argument("--filename", type=str, required=True,
                        help="JSON problem filename (e.g., S042_N10_C_R50.json)")
    parser.add_argument("--subfolder", type=str, default=None,
                        help="Subfolder within data/generated/data/")
    parser.add_argument("--data-root", type=str, default="../data/generated/data/",
                        help="Root data directory")
    parser.add_argument("--lkh-root", type=str, default="./lkh_files",
                        help="Directory for LKH3 files")
    parser.add_argument("--lkh-binary", type=str, default="./LKH-3.0.13/LKH",
                        help="Path to LKH3 executable")

    args = parser.parse_args()

    # Determine data path
    if args.subfolder:
        data_path = os.path.join(args.data_root, args.subfolder, args.filename)
    else:
        parts = args.filename.split("_")
        n_folder = parts[1]
        data_path = os.path.join(args.data_root, n_folder, args.filename)

    # Solve
    solver = BiObjLKH3Solver(data_path, lkh_root=args.lkh_root, lkh_binary=args.lkh_binary)
    solution = solver.solve(verbose=True)

    # Print results
    print("\n" + "="*60)
    print("SOLUTION SUMMARY")
    print("="*60)
    print(f"Service Rate:       {solution.service_rate:.2%}")
    print(f"Served Customers:   {solution.served_count}/{solver.num_customers}")
    print(f"Total Cost:         {solution.cost:.2f}")
    print(f"Objective Value:    {solution.objective:.2f}")
    print(f"Tour length:        {len(solution.tour)} nodes")
    print("="*60)


if __name__ == "__main__":
    main()
