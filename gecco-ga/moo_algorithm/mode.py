import multiprocessing
import sys
import os
import numpy as np

# Add the parent directory to the module search path
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/..")
from moo_algorithm.metric import cal_hv_front
from population import Population, Individual


class MODEPopulation(Population):
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

    def gen_offspring_de(self, F, CR):
        offspring = []
        for i in range(self.pop_size):
            indi1, indi2 = np.random.choice(self.indivs, 2, replace=False)
            off = rand_1(self.indivs[i], indi1, indi2, F, CR)
            offspring.append(off)
        return offspring


import random


# def rand_1(indi: Individual, indi1: Individual, indi2: Individual, F, CR):
#    new_indi = Individual()
#    new_indi.chromosome = indi.chromosome + F * (indi1.chromosome - indi2.chromosome)
#    new_indi.repair()
#    for i in range(len(indi.chromosome)):
#        if random.random() > CR:
#            new_indi.chromosome[i] = indi.chromosome[i]
#    return new_indi


def rand_1(indi: Individual, indi1: Individual, indi2: Individual, F, CR):
    new_indi = Individual(indi.chromosome + F * (indi1.chromosome - indi2.chromosome))
    # new_indi.repair()
    for i in range(len(indi.chromosome)):
        if random.random() > CR:
            new_indi.chromosome[i] = indi.chromosome[i]
    return new_indi


def run_mode(
    processing_number, problem, indi_list, pop_size, max_gen, F, CR, cal_fitness
):
    print("MODE")
    history = {}
    mode_pop = MODEPopulation(pop_size)
    mode_pop.pre_indi_gen(indi_list)

    pool = multiprocessing.Pool(processing_number)
    arg = []
    for individual in mode_pop.indivs:
        arg.append((problem, individual))
    result = pool.starmap(cal_fitness, arg)
    for individual, fitness in zip(mode_pop.indivs, result):
        individual.objectives = fitness
    mode_pop.natural_selection()
    print("Generation 0: Done")
    Pareto_store = []
    for indi in mode_pop.ParetoFront[0]:
        Pareto_store.append(list(indi.objectives))
    history[0] = Pareto_store

    for gen in range(max_gen):
        Pareto_store = []
        offspring = mode_pop.gen_offspring_de(F, CR)
        arg = []
        for individual in offspring:
            arg.append((problem, individual))
        result = pool.starmap(cal_fitness, arg)
        for individual, fitness in zip(offspring, result):
            individual.objectives = fitness
        mode_pop.indivs.extend(offspring)
        mode_pop.natural_selection()
        print("Generation {}: Done".format(gen + 1))
        for indi in mode_pop.ParetoFront[0]:
            Pareto_store.append(list(indi.objectives))
        history[gen + 1] = Pareto_store
    pool.close()
    # print(
    #    "MODE Done: ", cal_hv_front(mode_pop.ParetoFront[0], np.array([1, 1, 10, 10]))
    # )
    print("MODE Done: ", cal_hv_front(mode_pop.ParetoFront[0], np.array([10, 100000])))
    return history

