import argparse
import json
import random
import time
from copy import deepcopy
from pathlib import Path
from typing import Dict, Any, Optional, cast

from problem import Problem, Solution, Individual
from utils import decode, cal_fitness, init_population, get_data_files, save_result


class DestroyOperator:
    """Base class for destroy operators."""

    def __call__(self, indi: Individual, destroy_size: int) -> Individual:
        raise NotImplementedError


class DestroyRandom(DestroyOperator):
    """Randomly remove destroy_size nodes from solution."""

    def __call__(self, indi: Individual, destroy_size: int) -> Individual:
        destroyed = deepcopy(indi)
        perm, mask = destroyed.chromosome

        num_nodes = len(perm)
        if destroy_size > len(perm):
            destroy_size = len(perm)

        to_remove = random.sample(range(num_nodes), min(destroy_size, num_nodes))
        for idx in to_remove:
            mask[idx] = 0

        return destroyed


class DestroyWorstCost(DestroyOperator):
    """Remove nodes that contribute most to cost."""

    def __init__(self, problem: Problem):
        self.problem = problem

    def __call__(self, indi: Individual, destroy_size: int) -> Individual:
        destroyed = deepcopy(indi)
        perm, mask = destroyed.chromosome

        solution = decode(indi, self.problem)

        node_costs = {}
        for route, trips in solution.routes:
            # Truck route costs
            for node in route.nodes[1:-1]:
                if node not in node_costs:
                    node_costs[node] = 0
                node_costs[node] += 1

            # Drone trip costs
            for trip in trips:
                for node in trip.nodes[1:-1]:
                    if node not in node_costs:
                        node_costs[node] = 0
                    node_costs[node] += 2

        sorted_nodes = sorted(node_costs.items(), key=lambda x: x[1], reverse=True)
        worst_nodes = [node for node, _ in sorted_nodes[:destroy_size]]

        for i, node_id in enumerate(perm):
            if node_id in worst_nodes:
                mask[i] = 0

        return destroyed


class DestroySpatialCluster(DestroyOperator):
    """Remove spatially clustered nodes."""

    def __init__(self, problem: Problem):
        self.problem = problem

    def __call__(
        self, indi: Individual, destroy_size: int, seed_node: Optional[int] = None
    ) -> Individual:
        destroyed = deepcopy(indi)
        perm, mask = destroyed.chromosome

        if seed_node is None:
            num_nodes = len(self.problem.nodes) - 1
            seed_node = random.randint(1, num_nodes)

        distances = []
        for i, node_id in enumerate(perm):
            if node_id >= len(self.problem.nodes):
                continue

            coord_seed = self.problem.nodes[seed_node].coord
            coord_node = self.problem.nodes[node_id].coord
            dist = (
                (coord_seed[0] - coord_node[0]) ** 2
                + (coord_seed[1] - coord_node[1]) ** 2
            ) ** 0.5
            distances.append((i, node_id, dist))

        distances.sort(key=lambda x: x[2])
        for i in range(min(destroy_size, len(distances))):
            idx, _, _ = distances[i]
            mask[idx] = 0

        return destroyed


class RepairOperator:
    """Base class for repair operators."""

    def __call__(self, destroyed: Individual) -> Individual:
        raise NotImplementedError


class RepairGreedy(RepairOperator):
    """Repair by random reinsertion of removed nodes."""

    def __call__(self, destroyed: Individual) -> Individual:
        repaired = deepcopy(destroyed)
        perm, mask = repaired.chromosome

        for i in range(len(mask)):
            if perm[i] < len(perm):  # Skip fleet delimiters
                if mask[i] == 0 and random.random() > 0.5:
                    mask[i] = random.choice([0, 1, -1])

        return repaired


class RepairBestInsertion(RepairOperator):
    """Repair by best-cost insertion of removed nodes."""

    def __call__(self, destroyed: Individual) -> Individual:
        repaired = deepcopy(destroyed)
        perm, mask = repaired.chromosome

        for i in range(len(mask)):
            if perm[i] < len(perm):  # Skip fleet delimiters
                if mask[i] == 0:
                    weights = [0.7, 0.15, 0.15]  # Truck-biased
                    mask[i] = random.choices([0, 1, -1], weights=weights)[0]

        return repaired


def get_destroy_operators(problem: Problem) -> Dict[str, DestroyOperator]:
    """Get dictionary of all destroy operators."""
    return {
        "random": DestroyRandom(),
        "worst_cost": DestroyWorstCost(problem),
        "spatial_cluster": DestroySpatialCluster(problem),
    }


def get_repair_operators() -> Dict[str, RepairOperator]:
    """Get dictionary of all repair operators."""
    return {
        "greedy": RepairGreedy(),
        "best_insertion": RepairBestInsertion(),
    }


class LNSSolver:
    """Large Neighborhood Search solver for VRPBTW using GA encoding."""

    def __init__(
        self,
        problem: Problem,
        initial_solution: Optional[Individual] = None,
        seed: int = 42,
    ):
        self.problem = problem
        self.seed = seed
        random.seed(seed)

        # Initialize operators
        self.destroy_ops = get_destroy_operators(problem)
        self.repair_ops = get_repair_operators()

        # Initial solution
        if initial_solution is None:
            initial_solution = init_population(1, seed, problem)[0]

        initial_solution = cast(Individual, initial_solution)
        self.best_indi: Individual = deepcopy(initial_solution)
        self.current_indi: Individual = deepcopy(initial_solution)

        # Evaluate initial solution
        self.best_fitness: float
        self.best_sr: float
        self.best_cost: float
        self.current_fitness: float
        self.best_fitness, self.best_sr, self.best_cost = cal_fitness(
            problem, self.best_indi
        )
        self.current_fitness = self.best_fitness

    def local_search(self, indi: Individual, max_iterations: int = 50) -> Individual:
        """Apply local search: destroy and repair operations."""
        best = deepcopy(indi)
        best_fitness, _, _ = cal_fitness(self.problem, best)

        destroy_ops_list = list(self.destroy_ops.values())
        repair_ops_list = list(self.repair_ops.values())

        for iteration in range(max_iterations):
            # Randomly choose destroy and repair operators
            destroy_op = random.choice(destroy_ops_list)
            repair_op = random.choice(repair_ops_list)

            # Random destroy size
            destroy_size = random.randint(1, max(2, len(self.problem.nodes) // 3))

            # Destroy and repair
            destroyed = destroy_op(best, destroy_size)
            repaired = repair_op(destroyed)

            # Evaluate
            fitness, sr, cost = cal_fitness(self.problem, repaired)

            # Accept if better
            if fitness > best_fitness:
                best = deepcopy(repaired)
                best_fitness = fitness

        return best

    def solve(
        self,
        max_iterations: int = 100,
        lns_iterations: int = 50,
        verbose: bool = True,
    ) -> Dict[str, Any]:
        """
        Run LNS algorithm.

        Args:
            max_iterations: Max outer iterations
            lns_iterations: Max local search iterations per outer iteration
            verbose: Print progress

        Returns:
            Dictionary with best solution info
        """
        start_time = time.time()

        history = {
            0: {
                "fitness": self.best_fitness,
                "service_rate": self.best_sr,
                "cost": self.best_cost,
            }
        }

        for iteration in range(max_iterations):
            if verbose and iteration % 10 == 0:
                print(
                    f"Iteration {iteration}: "
                    f"Best fitness={self.best_fitness:.2f}, "
                    f"Cost={self.best_cost:.2f}"
                )

            # Local search
            improved = self.local_search(self.current_indi, lns_iterations)
            fitness, sr, cost = cal_fitness(self.problem, improved)

            # Update current
            self.current_indi = improved
            self.current_fitness = fitness

            # Update best
            if fitness > self.best_fitness:
                self.best_indi = deepcopy(improved)
                self.best_fitness = fitness
                self.best_sr = sr
                self.best_cost = cost

                if verbose:
                    print(f"  *** New best: fitness={fitness:.2f}, cost={cost:.2f} ***")

            history[iteration + 1] = {
                "fitness": self.best_fitness,
                "service_rate": self.best_sr,
                "cost": self.best_cost,
            }

        end_time = time.time()

        return {
            "time": end_time - start_time,
            "history": history,
            "best_fitness": self.best_fitness,
            "best_cost": self.best_cost,
            "best_service_rate": self.best_sr,
        }

    def decode_best(self) -> Solution:
        """Decode best solution found."""
        return decode(self.best_indi, self.problem)


def main():
    """Run LNS on datasets."""
    parser = argparse.ArgumentParser(description="Run LNS solver")
    parser.add_argument("--sizes", nargs="+", default=["N10", "N20", "N50", "N100", "N150"], help="Problem sizes")
    parser.add_argument("--max-iterations", type=int, default=100, help="Maximum iterations")
    parser.add_argument("--lns-iterations", type=int, default=50, help="Local search iterations")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    base_dir = Path(__file__).parent.parent / "data" / "generated" / "data"
    result_dir = Path(__file__).parent / "result" / "lns"

    data_files = get_data_files(str(base_dir))

    if not data_files:
        print("No data files found!")
        return

    data_files = {k: v for k, v in data_files.items() if k in args.sizes}

    if not data_files:
        print(f"No data files found for sizes: {args.sizes}")
        return

    print(f"\n{'=' * 80}")
    print("LNS Solver")
    print(f"{'=' * 80}\n")
    print(f"Found {len(data_files)} problem size categories")
    for size, files in data_files.items():
        print(f"  {size}: {len(files)} files")

    for size_dir, files in data_files.items():
        print(f"\n{'-' * 80}")
        print(f"LNS on {size_dir}")
        print(f"{'-' * 80}\n")

        for data_file in files:
            print(f"Processing: {data_file.name}")

            try:
                problem = Problem(str(data_file))

                solver = LNSSolver(problem, seed=args.seed)
                result = solver.solve(
                    max_iterations=args.max_iterations,
                    lns_iterations=args.lns_iterations,
                    verbose=False,
                )

                output_path = result_dir / size_dir / data_file.name
                save_result(result, output_path, exclude_keys=["best_individual"])

                print(f"  Completed in {result['time']:.2f} seconds")
                print(f"  Best fitness: {result['best_fitness']:.2f}")

            except Exception as e:
                print(f"  ERROR: {str(e)}")
                import traceback
                traceback.print_exc()
                continue

    print(f"\n{'=' * 80}")
    print("LNS experiments completed!")
    print(f"{'=' * 80}\n")


if __name__ == "__main__":
    main()
