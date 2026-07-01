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
