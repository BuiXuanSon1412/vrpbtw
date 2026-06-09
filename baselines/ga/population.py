import multiprocessing
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/..")

from copy import deepcopy
import numpy as np


class Individual:
    def __init__(self, chromosome):
        self.chromosome = chromosome
        self.fitness = None


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
                # off1, off2 = deepcopy(parent1), deepcopy(parent2)
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


def run_ga(
    processing_number,
    problem,
    pop_size,
    indi_list,
    max_gen,
    crossover_operator,
    mutation_operator,
    crossover_rate,
    mutation_rate,
    cal_fitness,
):
    history = {}
    pop = Population(pop_size)
    print("Population initialization")

    pop.pre_indi_gen(indi_list)

    pool = multiprocessing.Pool(processing_number)
    arg = []
    for individual in pop.indivs:
        arg.append((problem, individual))
    result = pool.starmap(cal_fitness, arg)
    for individual, fitness in zip(pop.indivs, result):
        individual.fitness = fitness

    pop.selection()

    history[0] = {
        "fitness": pop.indivs[0].fitness[0],
        "service_rate": pop.indivs[0].fitness[1],
        "cost": pop.indivs[0].fitness[2],
    }

    for gen in range(max_gen):
        offspring = pop.gen_offspring(
            problem,
            crossover_operator,
            mutation_operator,
            crossover_rate,
            mutation_rate,
        )
        arg = []
        for individual in offspring:
            arg.append((problem, individual))

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
