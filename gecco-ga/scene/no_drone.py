import random
from population import Individual
from problem import Problem
from operators import crossover_PMX
from moo_algorithm.moead import run_moead, init_weight_vectors_2d
from moo_algorithm.nsga_ii import run_nsga_ii
from moo_algorithm.nsga_iii import run_nsga_iii
from moo_algorithm.pfg_moea import run_pfgmoea
from moo_algorithm.agea import run_agea
from moo_algorithm.iagea import run_iagea
from moo_algorithm.ciagea import run_ciagea


def mutation_flip(problem: Problem, indi: Individual, num_flips=None):
    return indi


ALGORITHMS = {
    "CIAGEA": {
        "runner": run_ciagea,
        "params": {
            "init_div": 10,
            "crossover_operator": crossover_PMX,
            "mutation_operator": mutation_flip,
            "crossover_rate": 0.9,
            "mutation_rate": 0.1,
        },
        "ref_point": [24, 100000],
    },
    "IAGEA": {
        "runner": run_iagea,
        "params": {
            "init_div": 10,
            "crossover_operator": crossover_PMX,
            "mutation_operator": mutation_flip,
            "crossover_rate": 0.9,
            "mutation_rate": 0.1,
        },
        "ref_point": [24, 100000],
    },
    "AGEA": {
        "runner": run_agea,
        "params": {
            "init_div": 10,
            "crossover_operator": crossover_PMX,
            "mutation_operator": mutation_flip,
            "crossover_rate": 0.9,
            "mutation_rate": 0.1,
        },
        "ref_point": [24, 100000],
    },
    "PFG_MOEA": {
        "runner": run_pfgmoea,
        "params": {
            "GK": 10,
            "sigma": 0.1,
            "crossover_operator": crossover_PMX,
            "mutation_operator": mutation_flip,
            "crossover_rate": 0.9,
            "mutation_rate": 0.1,
        },
        "ref_point": [24, 100000],
    },
    "MOEAD": {
        "runner": run_moead,
        "params": {
            "neighborhood_size": 20,
            "init_weight_vectors": init_weight_vectors_2d,
            "crossover_operator": crossover_PMX,
            "mutation_operator": mutation_flip,
        },
        "ref_point": [24, 100000],
    },
    "NSGA_II": {
        "runner": run_nsga_ii,
        "params": {
            "crossover_operator": crossover_PMX,
            "mutation_operator": mutation_flip,
            "crossover_rate": 0.9,
            "mutation_rate": 0.1,
        },
        "ref_point": [24, 100000],
    },
    "NSGA_III": {
        "runner": run_nsga_iii,
        "params": {
            "crossover_operator": crossover_PMX,
            "mutation_operator": mutation_flip,
            "crossover_rate": 0.9,
            "mutation_rate": 0.1,
        },
        "ref_point": [24, 100000],
    },
}


def init_population(num_indi, seed, problem):
    random.seed(seed)
    popu = []
    for _ in range(num_indi):
        chromosome = []

        num_fleet = problem.num_fleet
        num_nodes = len(problem.nodes) - 1

        chromosome = []

        population = list(range(1, num_nodes + num_fleet))
        nodes = random.sample(population, len(population))

        mask = [0 for _ in range(num_nodes + num_fleet - 1)]
        chromosome = [nodes, mask]

        indi = Individual(chromosome)
        popu.append(indi)

    return popu
