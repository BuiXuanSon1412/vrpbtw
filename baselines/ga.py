import argparse
import json
import multiprocessing
import time
from copy import deepcopy
from pathlib import Path
from typing import Dict, Any

import numpy as np

from problem import Problem, Individual
from utils import init_population, cal_fitness, crossover, mutation, get_data_files, save_result


class Population:
    def __init__(self, pop_size):
        self.pop_size = pop_size
        self.indivs = []

    def pre_indi_gen(self, indi_list):
        if len(indi_list) != self.pop_size:
            raise ValueError(
                "The length of the list must be equal to the population size"
            )
        self.indivs = deepcopy(indi_list)

    def gen_offspring(
        self,
        problem,
        crossover_operator,
        mutation_operator,
        crossover_rate,
        mutation_rate,
    ):
        offspring = []
        for _ in range(self.pop_size):
            parent1, parent2 = np.random.choice(self.indivs, 2, replace=False)
            if np.random.rand() < crossover_rate:
                off1, off2 = crossover_operator(problem, parent1, parent2)
            else:
                off1 = Individual(deepcopy(parent1.chromosome))
                off2 = Individual(deepcopy(parent2.chromosome))
            if np.random.rand() < mutation_rate:
                off1 = mutation_operator(problem, off1)
                off2 = mutation_operator(problem, off2)
            offspring.append(off1)
            offspring.append(off2)
        return offspring

    def selection(self):
        self.indivs.sort(key=lambda ind: ind.fitness[0], reverse=True)
        self.indivs = self.indivs[: self.pop_size]


class GASolver:
    """Genetic Algorithm solver for VRPBTW."""

    def __init__(
        self,
        problem: Problem,
        pop_size: int,
        max_gen: int,
        crossover_rate: float = 0.9,
        mutation_rate: float = 0.1,
        processing_number: int = 1,
        seed: int = 42,
    ):
        self.problem = problem
        self.pop_size = pop_size
        self.max_gen = max_gen
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.processing_number = processing_number
        self.seed = seed

    def run(self) -> Dict[str, Any]:
        """
        Run GA algorithm.

        Returns:
            Dictionary with history of fitness, service_rate, and cost
        """
        history = {}
        indi_list = init_population(self.pop_size, self.seed, self.problem)
        pop = Population(self.pop_size)
        print("Population initialization")

        pop.pre_indi_gen(indi_list)

        pool = multiprocessing.Pool(self.processing_number)
        arg = []
        for individual in pop.indivs:
            arg.append((self.problem, individual))
        result = pool.starmap(cal_fitness, arg)
        for individual, fitness in zip(pop.indivs, result):
            individual.fitness = fitness

        pop.selection()

        history[0] = {
            "fitness": pop.indivs[0].fitness[0],
            "service_rate": pop.indivs[0].fitness[1],
            "cost": pop.indivs[0].fitness[2],
        }

        for gen in range(self.max_gen):
            offspring = pop.gen_offspring(
                self.problem,
                crossover,
                mutation,
                self.crossover_rate,
                self.mutation_rate,
            )
            arg = []
            for individual in offspring:
                arg.append((self.problem, individual))

            result = pool.starmap(cal_fitness, arg)
            for individual, fitness in zip(offspring, result):
                individual.fitness = fitness
            pop.indivs.extend(offspring)

            pop.selection()

            print("Generation {}: Done".format(gen + 1))
            fitness, service_rate, cost = pop.indivs[0].fitness
            history[gen + 1] = {
                "fitness": fitness,
                "service_rate": service_rate,
                "cost": cost,
            }

        pool.close()
        return history


def main():
    """Run GA on datasets."""
    parser = argparse.ArgumentParser(description="Run GA solver")
    parser.add_argument(
        "--sizes",
        nargs="+",
        default=["N10", "N20", "N50", "N100", "N150"],
        help="Problem sizes",
    )
    parser.add_argument("--pop-size", type=int, default=100, help="Population size")
    parser.add_argument("--max-gen", type=int, default=100, help="Maximum generations")
    parser.add_argument(
        "--processing-number", type=int, default=4, help="Number of processes"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    base_dir = Path(__file__).parent.parent / "data" / "generated" / "data"
    result_dir = Path(__file__).parent / "result" / "ga"

    data_files = get_data_files(str(base_dir))

    if not data_files:
        print("No data files found!")
        return

    data_files = {k: v for k, v in data_files.items() if k in args.sizes}

    if not data_files:
        print(f"No data files found for sizes: {args.sizes}")
        return

    print(f"\n{'=' * 80}")
    print("GA Solver")
    print(f"{'=' * 80}\n")
    print(f"Found {len(data_files)} problem size categories")
    for size, files in data_files.items():
        print(f"  {size}: {len(files)} files")

    for size_dir, files in data_files.items():
        print(f"\n{'-' * 80}")
        print(f"GA on {size_dir}")
        print(f"{'-' * 80}\n")

        for data_file in files:
            print(f"Processing: {data_file.name}")

            try:
                problem = Problem(str(data_file))

                start_time = time.time()
                solver = GASolver(
                    problem=problem,
                    pop_size=args.pop_size,
                    max_gen=args.max_gen,
                    processing_number=args.processing_number,
                    seed=args.seed,
                )
                history = solver.run()
                end_time = time.time()

                result = {
                    "time": end_time - start_time,
                    "history": history,
                    "best_fitness": history[max(history.keys())]["fitness"],
                    "best_cost": history[max(history.keys())]["cost"],
                    "best_service_rate": history[max(history.keys())]["service_rate"],
                }

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
    print("GA experiments completed!")
    print(f"{'=' * 80}\n")


if __name__ == "__main__":
    main()
