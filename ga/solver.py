import multiprocessing
import random
import numpy as np
from copy import deepcopy

from problem import Problem
from population import Individual, Population
from utils import init_population, cal_fitness, repair
from operators import crossover_PMX, mutation_flip

W_TARDINESS = 0.5
W_COST = 0.5


def scalarize(tardiness: float, cost: float) -> float:
    return W_TARDINESS * tardiness + W_COST * cost


def cal_scalar_fitness(problem: Problem, indi: Individual):
    chro, tardiness, cost = cal_fitness(problem, indi)
    scalar = scalarize(tardiness, cost)
    return chro, scalar, tardiness, cost


# GA population
class GAPopulation(Population):
    def __init__(self, pop_size: int):
        super().__init__(pop_size)

    # Selection
    def tournament_selection(self, k: int = 3) -> Individual:
        """k-tournament selection (lower scalar fitness wins)."""
        candidates = random.sample(self.indivs, min(k, len(self.indivs)))
        return min(candidates, key=lambda ind: ind.scalar_fitness)

    # Offspring generation 
    def gen_offspring_ga(self,problem: Problem,crossover_operator,mutation_operator,crossover_rate: float,mutation_rate: float):
        offspring = []
        while len(offspring) < self.pop_size:
            p1 = self.tournament_selection()
            p2 = self.tournament_selection()
            if np.random.rand() < crossover_rate:
                off1, off2 = crossover_operator(problem, p1, p2)
            else:
                off1 = Individual(deepcopy(p1.chromosome))
                off2 = Individual(deepcopy(p2.chromosome))
            if np.random.rand() < mutation_rate:
                off1 = mutation_operator(problem, off1)
            if np.random.rand() < mutation_rate:
                off2 = mutation_operator(problem, off2)
            offspring.append(off1)
            offspring.append(off2)
        return offspring[: self.pop_size]

    # Survivor selection 
    def natural_selection(self, combined: list):
        combined.sort(key=lambda ind: ind.scalar_fitness)
        self.indivs = combined[: self.pop_size]


def _evaluate_population(pool, problem, individuals):
    args = [(problem, ind) for ind in individuals]
    results = pool.starmap(cal_scalar_fitness, args)
    for ind, res in zip(individuals, results):
        chro, scalar, tardiness, cost = res
        ind.chromosome = chro          # repaired chromosome
        ind.scalar_fitness = scalar
        ind.objectives = [tardiness, cost]
    return individuals


def run_ga(processing_number: int,problem: Problem,indi_list: list,pop_size: int,max_gen: int,crossover_operator=crossover_PMX,mutation_operator=mutation_flip,crossover_rate: float = 0.9,mutation_rate: float = 0.1,verbose: bool = True,):
    print("GA (w_tardiness=0.5, w_cost=0.5)")

    ga_pop = GAPopulation(pop_size)
    ga_pop.pre_indi_gen(indi_list)

    history = {}
    pool = multiprocessing.Pool(processing_number)

    # Gen 0
    _evaluate_population(pool, problem, ga_pop.indivs)
    ga_pop.indivs.sort(key=lambda ind: ind.scalar_fitness)

    best = deepcopy(ga_pop.indivs[0])
    history[0] = _snapshot(ga_pop.indivs[0])

    if verbose:
        _print_gen(0, ga_pop.indivs[0])

    for gen in range(1, max_gen + 1):
        offspring = ga_pop.gen_offspring_ga(
            problem,
            crossover_operator,
            mutation_operator,
            crossover_rate,
            mutation_rate,
        )

        _evaluate_population(pool, problem, offspring)

        # (μ + λ) selection
        combined = ga_pop.indivs + offspring
        ga_pop.natural_selection(combined)

        # Track best
        if ga_pop.indivs[0].scalar_fitness < best.scalar_fitness:
            best = deepcopy(ga_pop.indivs[0])

        history[gen] = _snapshot(ga_pop.indivs[0])

        if verbose:
            _print_gen(gen, ga_pop.indivs[0])

    pool.close()
    pool.join()

    print(
        f"\nGA Done | best scalar={best.scalar_fitness:.4f} "
        f"| tardiness={best.objectives[0]:.4f} | cost={best.objectives[1]:.2f}"
    )
    return {"history": history, "best": best}


# Utilities
def _snapshot(ind: Individual) -> dict:
    return {
        "scalar": ind.scalar_fitness,
        "tardiness": ind.objectives[0],
        "cost": ind.objectives[1],
    }


def _print_gen(gen: int, best_ind: Individual):
    print(
        f"  Gen {gen:>4d} | scalar={best_ind.scalar_fitness:>12.4f} "
        f"| tardiness={best_ind.objectives[0]:>8.4f} "
        f"| cost={best_ind.objectives[1]:>10.2f}"
    )


# Entry point (example usage)
if __name__ == "__main__":
    import json
    import sys
    import os   

    DATA_PATH = sys.argv[1] if len(sys.argv) > 1 else "./data/N10/"
    POP_SIZE = 50
    MAX_GEN = 100
    N_WORKERS = max(1, multiprocessing.cpu_count() - 1)
    SEED = 42

    problem = Problem(DATA_PATH)
    indi_list = init_population(POP_SIZE, SEED, problem)

    result = run_ga(
        processing_number=N_WORKERS,
        problem=problem,
        indi_list=indi_list,
        pop_size=POP_SIZE,
        max_gen=MAX_GEN,
        crossover_operator=crossover_PMX,
        mutation_operator=mutation_flip,
        crossover_rate=0.9,
        mutation_rate=0.1,
        verbose=True,
    )

    base_filename = os.path.splitext(os.path.basename(DATA_PATH))[0]
    if not base_filename: # Trường hợp DATA_PATH là đường dẫn thư mục
        base_filename = "result"
    parent_dir = os.path.basename(os.path.dirname(os.path.normpath(DATA_PATH)))
    output_dir = os.path.join("result_ga", parent_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    # 4. Tạo tên file JSON mới: result_ga/N10/ten_file_goc.json
    out_path = os.path.join(output_dir, f"{base_filename}.json")
    
    with open(out_path, "w") as f:
        json.dump(result["history"], f, indent=2)
    
    print(f"\n[OK] History saved to: {out_path}")