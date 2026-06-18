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
    """Repair by greedy insertion with feasibility bias."""

    def __call__(self, destroyed: Individual) -> Individual:
        repaired = deepcopy(destroyed)
        perm, mask = repaired.chromosome

        # Reinsertion priority: truck (most feasible) > drone > skip
        for i in range(len(mask)):
            if perm[i] < len(perm):  # Skip fleet delimiters
                if mask[i] == 0:
                    # 70% truck, 15% drone, 15% skip (prioritize service)
                    mask[i] = random.choices([1, -1, 0], weights=[0.7, 0.15, 0.15])[0]

        return repaired


class RepairBestInsertion(RepairOperator):
    """Repair by best-cost insertion - iteratively reinsert to maximize fitness."""

    def __init__(self, problem: Problem):
        self.problem = problem

    def __call__(self, destroyed: Individual) -> Individual:
        repaired = deepcopy(destroyed)
        perm, mask = repaired.chromosome

        # Get current fitness
        try:
            current_fitness, _, _ = cal_fitness(self.problem, repaired)
        except:
            # If destroyed solution is infeasible, do greedy repair
            for i in range(len(mask)):
                if mask[i] == 0 and perm[i] < len(self.problem.nodes):
                    mask[i] = random.choices([1, -1, 0], weights=[0.6, 0.2, 0.2])[0]
            return repaired

        removed_positions = [i for i in range(len(mask))
                            if mask[i] == 0 and perm[i] < len(self.problem.nodes)]

        # Greedily reinsert each removed node to maximize improvement
        for pos in removed_positions:
            best_mode = 0  # Default: skip
            best_fitness = current_fitness

            # Try each mode and pick the one that maximizes fitness
            for mode in [1, -1, 0]:
                mask[pos] = mode
                try:
                    fitness, _, _ = cal_fitness(self.problem, repaired)
                    if fitness > best_fitness:
                        best_fitness = fitness
                        best_mode = mode
                except:
                    continue

            # Accept best mode
            mask[pos] = best_mode
            current_fitness = best_fitness

        return repaired


def get_destroy_operators(problem: Problem) -> Dict[str, DestroyOperator]:
    """Get dictionary of all destroy operators."""
    return {
        "random": DestroyRandom(),
        "worst_cost": DestroyWorstCost(problem),
        "spatial_cluster": DestroySpatialCluster(problem),
    }


def get_repair_operators(problem: Problem) -> Dict[str, RepairOperator]:
    """Get dictionary of all repair operators."""
    return {
        "greedy": RepairGreedy(),
        "best_insertion": RepairBestInsertion(problem),
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
        self.repair_ops = get_repair_operators(problem)

        # Initial solution: use best from larger population (better starting point)
        if initial_solution is None:
            pop = init_population(20, seed, problem)  # Generate 20 candidates
            best_indi = pop[0]
            best_fitness, _, _ = cal_fitness(problem, best_indi)
            for indi in pop[1:]:
                fitness, _, _ = cal_fitness(problem, indi)
                if fitness > best_fitness:
                    best_indi = indi
                    best_fitness = fitness
            initial_solution = best_indi
            self.initial_pop = pop  # Store for potential restart

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

        # Adaptive parameters
        self.destroy_size_min = max(1, len(problem.nodes) // 10)
        self.destroy_size_max = max(3, len(problem.nodes) // 4)
        self.destroy_size = self.destroy_size_min

    def local_search(self, indi: Individual, max_iterations: int = 50) -> Individual:
        """Apply LNS with destroy-repair and simulated annealing acceptance."""
        import math

        best = deepcopy(indi)
        best_fitness, _, _ = cal_fitness(self.problem, best)

        destroy_ops_list = list(self.destroy_ops.values())
        repair_ops_list = list(self.repair_ops.values())

        stagnation_count = 0
        temperature = 1.0

        for iteration in range(max_iterations):
            # Adaptive destroy size: increase on stagnation
            if stagnation_count > 3:
                self.destroy_size = min(self.destroy_size + 1, self.destroy_size_max)
                stagnation_count = 0
            else:
                self.destroy_size = max(self.destroy_size_min, self.destroy_size - 0.5)

            # Randomly choose destroy and repair operators
            destroy_op = random.choice(destroy_ops_list)
            repair_op = random.choice(repair_ops_list)

            # Destroy and repair
            try:
                destroyed = destroy_op(best, int(self.destroy_size))
                repaired = repair_op(destroyed)
                fitness, sr, cost = cal_fitness(self.problem, repaired)
            except:
                stagnation_count += 1
                continue

            # Simulated annealing acceptance
            delta = fitness - best_fitness
            if delta > 0:  # Better solution - always accept
                best = deepcopy(repaired)
                best_fitness = fitness
                stagnation_count = 0
            elif temperature > 0.01 and random.random() < math.exp(delta / max(temperature, 0.1)):
                # Accept worse solution with decreasing probability
                best = deepcopy(repaired)
                best_fitness = fitness
                stagnation_count += 1
            else:
                stagnation_count += 1

            # Cool down temperature
            temperature *= 0.95

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

        stagnation_counter = 0
        restart_threshold = 20  # Restart after 20 iterations without improvement

        for iteration in range(max_iterations):
            if verbose and iteration % 10 == 0:
                print(
                    f"Iteration {iteration}: "
                    f"Best SR={self.best_sr:.0%}, Cost={self.best_cost:.2f}, "
                    f"Fitness={self.best_fitness:.2f}, Stagnation={stagnation_counter}"
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
                stagnation_counter = 0

                if verbose:
                    print(f"  *** New best: SR={sr:.0%}, cost={cost:.2f}, fitness={fitness:.2f} ***")
            else:
                stagnation_counter += 1

            # Restart from best solution in initial population if stagnant
            if stagnation_counter >= restart_threshold and hasattr(self, 'initial_pop'):
                if verbose:
                    print(f"  [RESTART] Restarting from initial population...")
                # Pick next best from initial population
                for indi in self.initial_pop:
                    f, _, _ = cal_fitness(self.problem, indi)
                    if f > self.current_fitness:
                        self.current_indi = deepcopy(indi)
                        self.current_fitness = f
                        stagnation_counter = 0
                        break

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
    parser.add_argument(
        "--sizes",
        nargs="+",
        default=["N10", "N20", "N50", "N100", "N150"],
        help="Problem sizes",
    )
    parser.add_argument(
        "--max-iterations", type=int, default=100, help="Maximum iterations"
    )
    parser.add_argument(
        "--lns-iterations", type=int, default=50, help="Local search iterations"
    )
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
                save_result(result, output_path)

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
