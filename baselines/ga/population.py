import multiprocessing
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/..")

from copy import deepcopy
import numpy as np


class Individual:
    def __init__(self, chromosome):
        self.chromosome = chromosome
        self.objectives = None  # Objectives vector

        self.domination_count = None  # be dominated
        self.dominated_solutions = None  # dominate
        self.crowding_distance = None
        self.rank = None

    # Dominate operator
    def dominates(self, other_individual):
        if not self.objectives or not other_individual.objectives:
            return

        tolerance = 0
        and_condition = True
        or_condition = False
        for first, second in zip(self.objectives, other_individual.objectives):
            and_condition = and_condition and (first <= second + tolerance)
            or_condition = or_condition or (first < second - tolerance)
        return and_condition and or_condition


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


class NSGAIIPopulation(Population):
    def __init__(self, pop_size):
        super().__init__(pop_size)
        self.ParetoFront = []

    def fast_nondominated_sort_crowding_distance(self, indi_list):
        ParetoFront = [[]]
        for individual in indi_list:
            individual.domination_count = 0
            individual.dominated_solutions = []
            for other_individual in indi_list:
                if individual.dominates(other_individual):
                    individual.dominated_solutions.append(other_individual)
                elif other_individual.dominates(individual):
                    individual.domination_count += 1
            if individual.domination_count == 0:
                individual.rank = 0
                ParetoFront[0].append(individual)
        i = 0
        while len(ParetoFront[i]) > 0:
            temp = []
            for individual in ParetoFront[i]:
                for other_individual in individual.dominated_solutions:
                    other_individual.domination_count -= 1
                    if other_individual.domination_count == 0:
                        other_individual.rank = i + 1
                        temp.append(other_individual)
            i = i + 1
            ParetoFront.append(temp)
        for front in ParetoFront:
            self.calculate_crowding_distance(front)
        return ParetoFront

    def calculate_crowding_distance(self, front):
        if len(front) > 0:
            solutions_num = len(front)
            for individual in front:
                individual.crowding_distance = 0

            for m in range(len(front[0].objectives)):
                front.sort(key=lambda individual: individual.objectives[m])
                front[0].crowding_distance = 10**9
                front[solutions_num - 1].crowding_distance = 10**9
                m_values = [individual.objectives[m] for individual in front]
                scale = max(m_values) - min(m_values)
                if scale == 0:
                    scale = 1
                for i in range(1, solutions_num - 1):
                    front[i].crowding_distance += (
                        front[i + 1].objectives[m] - front[i - 1].objectives[m]
                    ) / scale

    # Crowding Operator
    def crowding_operator(self, individual, other_individual):
        if (individual.rank < other_individual.rank) or (
            (individual.rank == other_individual.rank)
            and (individual.crowding_distance > other_individual.crowding_distance)
        ):
            return 1
        else:
            return -1

    def natural_selection(self):
        self.ParetoFront = self.fast_nondominated_sort_crowding_distance(self.indivs)
        new_indivs = []
        new_fronts = []
        front_num = 0
        while len(new_indivs) + len(self.ParetoFront[front_num]) <= self.pop_size:
            new_indivs.extend(self.ParetoFront[front_num])
            new_fronts.append(self.ParetoFront[front_num])
            if len(new_indivs) == self.pop_size:
                break
            front_num += 1
        self.calculate_crowding_distance(self.ParetoFront[front_num])
        self.ParetoFront[front_num].sort(
            key=lambda individual: individual.crowding_distance, reverse=True
        )
        number_remain = self.pop_size - len(new_indivs)
        new_indivs.extend(self.ParetoFront[front_num][0:number_remain])
        new_fronts.append(self.ParetoFront[front_num][0:number_remain])
        self.ParetoFront = new_fronts
        self.indivs = new_indivs


def run_ga(
    processing_number,
    problem,
    indi_list,
    pop_size,
    max_gen,
    crossover_operator,
    mutation_operator,
    crossover_rate,
    mutation_rate,
    cal_fitness,
):
    history = {}
    nsga_ii_pop = NSGAIIPopulation(pop_size)
    nsga_ii_pop.pre_indi_gen(indi_list)

    pool = multiprocessing.Pool(processing_number)
    arg = []
    for individual in nsga_ii_pop.indivs:
        arg.append((problem, individual))
    result = pool.starmap(cal_fitness, arg)
    for individual, fitness in zip(nsga_ii_pop.indivs, result):
        individual.chromosome = fitness[0]
        individual.objectives = fitness[1:]
    nsga_ii_pop.natural_selection()
    Pareto_store = []
    for indi in nsga_ii_pop.ParetoFront[0]:
        Pareto_store.append(list(indi.objectives))
    history[0] = Pareto_store

    for gen in range(max_gen):
        Pareto_store = []
        offspring = nsga_ii_pop.gen_offspring(
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
            individual.chromosome = fitness[0]
            individual.objectives = fitness[1:]
        nsga_ii_pop.indivs.extend(offspring)
        nsga_ii_pop.natural_selection()
        print("Generation {}: Done".format(gen + 1))
        for indi in nsga_ii_pop.ParetoFront[0]:
            Pareto_store.append(list(indi.objectives))
        history[gen + 1] = Pareto_store
    pool.close()
    return history
