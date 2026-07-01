# def generate_reference_points(num_objs, num_divisions_per_obj=4):
#     """Generates reference points for NSGA-III selection. This code is based on
#     jMetal NSGA-III implementation <https://github.com/jMetal/jMetal>.
#     """
#     def gen_refs_recursive(work_point, num_objs, left, total, depth):
#         if depth == num_objs - 1:
#             work_point[depth] = left / total
#             ref = ReferencePoint(copy.deepcopy(work_point))
#             return [ref]
#         else:
#             res = []
#             for i in range(left):
#                 w
#
#
# ork_point[depth] = i / total
#                 res = res + gen_refs_recursive(
#                     work_point, num_objs, left - i, total, depth + 1
#                 )
#             return res

#     return gen_refs_recursive(
#         [0] * num_objs,
#         num_objs,
#         num_objs * num_divisions_per_obj,
#         num_objs * num_divisions_per_obj,
#         0,
#     )


# def find_ideal_point(indivs):
#     m = len(indivs[0].objectives)
#     ideal_point = [np.infty]*m
#     for indi in indivs:
#         for i in range(m):
#             ideal_point[i] = min(ideal_point[i], indi.objectives[i])
#     return ideal_point


# def find_extreme_points(individuals):
#     "Finds the individuals with extreme values for each objective function."
#     return [
#         sorted(individuals, key=lambda ind: ind.objectives[o])[-1]
#         for o in range(len(individuals[0].objectives))
#     ]


# def construct_hyperplane(individuals, extreme_points):
#     "Calculates the axis intersects for a set of individuals and its extremes."

#     def has_duplicate_individuals(individuals):
#         for i in range(len(individuals)):
#             for j in range(i + 1, len(individuals)):
#                 if individuals[i].objectives == individuals[j].objectives:
#                     return True
#         return False

#     num_objs = len(individuals[0].objectives)

#     if has_duplicate_individuals(extreme_points):
#         intercepts = [extreme_points[m].objectives[m] for m in range(num_objs)]
#     else:
#         b = np.ones(num_objs)
#         A = [point.objectives for point in extreme_points]
#         x = np.linalg.solve(A, b)
#         intercepts = 1 / x
#     return intercepts


# def normalize_objective(individual, m, intercepts, ideal_point, epsilon=1e-20):
#     "Normalizes an objective."
#     if np.abs(intercepts[m] - ideal_point[m]) > epsilon:
#         return individual.objectives[m] / (intercepts[m] - ideal_point[m])
#     else:
#         return individual.objectives[m] / epsilon


# def normalize_objectives(individuals, intercepts, ideal_point):
#     """Normalizes individuals using the hyperplane defined by the intercepts as
#     reference. Corresponds to Algorithm 2 of Deb & Jain (2014)."""
#     num_objs = len(individuals[0].objectives)

#     for ind in individuals:
#         ind.normalized_values = list(
#             [
#                 normalize_objective(ind, m, intercepts, ideal_point)
#                 for m in range(num_objs)
#             ]
#         )
#     return individuals


# def perpendicular_distance(direction, point):
#     k = np.dot(direction, point) / np.sum(np.power(direction, 2))
#     d = np.sum(
#         np.power(np.subtract(np.multiply(direction, [k] * len(direction)), point), 2)
#     )
#     return np.sqrt(d)


# def associate(individuals, reference_points):
#     """Associates individuals to reference points and calculates niche number.
#     Corresponds to Algorithm 3 of Deb & Jain (2014)."""
#     # tools.sortLogNondominated(individuals, len(individuals))

#     num_objs = len(individuals[0].objectives)

#     for ind in individuals:
#         rp_dists = [
#             (rp, perpendicular_distance(ind.objectives, rp))
#             for rp in reference_points
#         ]
#         best_rp, best_dist = sorted(rp_dists, key=lambda rpd: rpd[1])[0]

#         ind.reference_point = best_rp
#         ind.ref_point_distance = best_dist

#         best_rp.associations_count += 1
#         best_rp.associations.append(ind)


# def niching_select(individuals, k):
#     """Secondary niched selection based on reference points. Corresponds to
#     steps 13-17 of Algorithm 1 and to Algorithm 4."""

#     if len(individuals) == k:
#         return individuals

#     ideal_point = find_ideal_point(individuals)
#     extremes = find_extreme_points(individuals)

#     intercepts = construct_hyperplane(individuals, extremes)
#     normalize_objectives(individuals, intercepts, ideal_point)

#     reference_points = generate_reference_points(len(individuals[0].objectives))
#     associate(individuals, reference_points)

#     res = []
#     while len(res) < k:
#         min_assoc_rp = min(reference_points, key=lambda rp: rp.associations_count)

#         min_assoc_rps = [
#             rp
#             for rp in reference_points
#             if rp.associations_count == min_assoc_rp.associations_count
#         ]

#         chosen_rp = min_assoc_rps[random.randint(0, len(min_assoc_rps) - 1)]

#         if chosen_rp.associations:
#             if chosen_rp.associations_count == 0:
#                 sel = min(
#                     chosen_rp.associations, key=lambda ind: ind.ref_point_distance
#                 )
#             else:
#                 sel = chosen_rp.associations[random.randint(0, len(chosen_rp.associations) - 1)]
#             res += [sel]
#             chosen_rp.associations.remove(sel)
#             chosen_rp.associations_count += 1
#             individuals.remove(sel)
#         else:
#             reference_points.remove(chosen_rp)

#     return res


# def sel_nsga_iii(individuals, k):
#     """Implements NSGA-III selection as described in
#     Deb, K., & Jain, H. (2014). An Evolutionary Many-Objective Optimization
#     Algorithm Using Reference-Point-Based Nondominated Sorting Approach,
#     Part I: Solving Problems With Box Constraints. IEEE Transactions on
#     Evolutionary Computation, 18(4), 577-601. doi: 10.1109/TEVC.2013.2281535.
#     """
#     assert len(individuals) >= k

#     if len(individuals) == k:
#         return individuals

#     # Algorithm 1 steps 4--8
#     fronts = fast_nondominated_sort(individuals)

#     limit = 0
#     res = []
#     for f, front in enumerate(fronts):
#         res += front
#         if len(res) > k:
#             limit = f
#             break
#     selection = []
#     if limit > 0:
#         for f in range(limit):
#             selection += fronts[f]
#     selection += niching_select(fronts[limit], k - len(selection))
#     return selection


# def fast_nondominated_sort(indi_list):
#     ParetoFront = [[]]
#     for individual in indi_list:
#         individual.domination_count = 0
#         individual.dominated_solutions = []
#         for other_individual in indi_list:
#             if individual.dominates(other_individual):
#                 individual.dominated_solutions.append(other_individual)
#             elif other_individual.dominates(individual):
#                 individual.domination_count += 1
#         if individual.domination_count == 0:
#             individual.rank = 0
#             ParetoFront[0].append(individual)
#     i = 0
#     while len(ParetoFront[i]) > 0:
#         temp = []
#         for individual in ParetoFront[i]:
#             for other_individual in individual.dominated_solutions:
#                 other_individual.domination_count -= 1
#                 if other_individual.domination_count == 0:
#                     other_individual.rank = i + 1
#                     temp.append(other_individual)
#         i = i + 1
#         ParetoFront.append(temp)
#     return ParetoFront


# class NSGAIIIPopulation(Population):
#     def __init__(self, pop_size):
#         super().__init__(pop_size)
#         self.ParetoFront = []
#     def fast_nondominated_sort(self):
#         self.ParetoFront = fast_nondominated_sort(self.indivs)

#     def natural_selection(self):
#         self.indivs = sel_nsga_iii(self.indivs, self.pop_size)


# def run_nsga_iii(processing_number, problem, indi_list, pop_size, max_gen, crossover_operator, mutation_operator,
#                 crossover_rate, mutation_rate, cal_fitness):
#     print("NSGA-III")
#     nsga_iii_pop = NSGAIIIPopulation(pop_size)
#     nsga_iii_pop.pre_indi_gen(indi_list)

#     history = {}
#     pool = multiprocessing.Pool(processing_number)
#     arg = []
#     for individual in nsga_iii_pop.indivs:
#         arg.append((problem, individual))
#     result = pool.starmap(cal_fitness, arg)
#     for individual, fitness in zip(nsga_iii_pop.indivs, result):
#         individual.objectives = fitness
#     nsga_iii_pop.fast_nondominated_sort()
#     pool.close()

#     print("Generation 0: Done")
#     Pareto_store = []
#     for indi in nsga_iii_pop.ParetoFront[0]:
#         Pareto_store.append(list(indi.objectives))
#     history[0] = Pareto_store

#     for gen in range(max_gen):
#         offspring = nsga_iii_pop.gen_offspring(problem, crossover_operator, mutation_operator, crossover_rate, mutation_rate)
#         pool = multiprocessing.Pool(processing_number)
#         arg = []
#         for individual in offspring:
#             arg.append((problem, individual))
#         result = pool.starmap(cal_fitness, arg)
#         for individual, fitness in zip(offspring, result):
#             individual.objectives = fitness
#         pool.close()
#         # for indi in offspring:
#         #     indi.objectives = cal_fitness(problem, indi)
#         nsga_iii_pop.indivs.extend(offspring)
#         nsga_iii_pop.natural_selection()
#         nsga_iii_pop.fast_nondominated_sort()
#         print(f"Generation {gen+1}: Done")
#         Pareto_store = []
#         for indi in nsga_iii_pop.ParetoFront[0]:
#             Pareto_store.append(list(indi.objectives))
#         history[gen+1] = Pareto_store

#     pool.close()

#     print("NSGA-III Done: ", cal_hv_front(nsga_iii_pop.ParetoFront[0], np.array([1,1,10,10])))
#     return history

import multiprocessing
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/..")
from moo_algorithm.metric import cal_hv_front
from population import Population, Individual
import copy
import random
import numpy as np


class ReferencePoint(list):
    def __init__(self, *args):
        list.__init__(self, *args)
        self.associations_count = 0
        self.associations = []


def generate_reference_points(num_objs, num_divisions_per_obj=4):
    def gen_refs_recursive(work_point, num_objs, left, total, depth):
        if depth == num_objs - 1:
            work_point[depth] = left / total
            ref = ReferencePoint(copy.deepcopy(work_point))
            return [ref]
        else:
            res = []
            for i in range(left):
                work_point[depth] = i / total
                res = res + gen_refs_recursive(
                    work_point, num_objs, left - i, total, depth + 1
                )
            return res

    return gen_refs_recursive(
        [0] * num_objs,
        num_objs,
        num_objs * num_divisions_per_obj,
        num_objs * num_divisions_per_obj,
        0,
    )


def find_ideal_point(indivs):
    m = len(indivs[0].objectives)
    ideal_point = [np.inf] * m
    for indi in indivs:
        for i in range(m):
            ideal_point[i] = min(ideal_point[i], indi.objectives[i])
    return ideal_point


def find_extreme_points(individuals):
    return [
        sorted(individuals, key=lambda ind: ind.objectives[o])[-1]
        for o in range(len(individuals[0].objectives))
    ]


def construct_hyperplane(individuals, extreme_points):
    def has_duplicate_individuals(individuals):
        for i in range(len(individuals)):
            for j in range(i + 1, len(individuals)):
                if individuals[i].objectives == individuals[j].objectives:
                    return True
        return False

    num_objs = len(individuals[0].objectives)

    if has_duplicate_individuals(extreme_points):
        intercepts = [extreme_points[m].objectives[m] for m in range(num_objs)]
    else:
        b = np.ones(num_objs)
        A = [point.objectives for point in extreme_points]
        x = np.linalg.solve(A, b)
        intercepts = 1 / x
    return intercepts


def normalize_objective(individual, m, intercepts, ideal_point, epsilon=1e-20):
    if np.abs(intercepts[m] - ideal_point[m]) > epsilon:
        return individual.objectives[m] / (intercepts[m] - ideal_point[m])
    else:
        return individual.objectives[m] / epsilon


def normalize_objectives(individuals, intercepts, ideal_point):
    num_objs = len(individuals[0].objectives)

    for ind in individuals:
        ind.normalized_values = list(
            [
                normalize_objective(ind, m, intercepts, ideal_point)
                for m in range(num_objs)
            ]
        )
    return individuals


def perpendicular_distance(direction, point):
    k = np.dot(direction, point) / np.sum(np.power(direction, 2))
    d = np.sum(
        np.power(np.subtract(np.multiply(direction, [k] * len(direction)), point), 2)
    )
    return np.sqrt(d)


def associate(individuals, reference_points):
    num_objs = len(individuals[0].objectives)

    for ind in individuals:
        rp_dists = [
            (rp, perpendicular_distance(ind.objectives, rp)) for rp in reference_points
        ]
        best_rp, best_dist = sorted(rp_dists, key=lambda rpd: rpd[1])[0]

        ind.reference_point = best_rp
        ind.ref_point_distance = best_dist

        best_rp.associations_count += 1
        best_rp.associations.append(ind)


def niching_select(individuals, k):
    if len(individuals) == k:
        return individuals

    ideal_point = find_ideal_point(individuals)
    extremes = find_extreme_points(individuals)

    intercepts = construct_hyperplane(individuals, extremes)
    normalize_objectives(individuals, intercepts, ideal_point)

    reference_points = generate_reference_points(len(individuals[0].objectives))
    associate(individuals, reference_points)

    res = []
    while len(res) < k:
        min_assoc_rp = min(reference_points, key=lambda rp: rp.associations_count)

        min_assoc_rps = [
            rp
            for rp in reference_points
            if rp.associations_count == min_assoc_rp.associations_count
        ]

        chosen_rp = min_assoc_rps[random.randint(0, len(min_assoc_rps) - 1)]

        if chosen_rp.associations:
            if chosen_rp.associations_count == 0:
                sel = min(
                    chosen_rp.associations, key=lambda ind: ind.ref_point_distance
                )
            else:
                sel = chosen_rp.associations[
                    random.randint(0, len(chosen_rp.associations) - 1)
                ]
            res += [sel]
            chosen_rp.associations.remove(sel)
            chosen_rp.associations_count += 1
            individuals.remove(sel)
        else:
            reference_points.remove(chosen_rp)

    return res


def sel_nsga_iii(individuals, k):
    assert len(individuals) >= k

    if len(individuals) == k:
        return individuals

    fronts = fast_nondominated_sort(individuals)

    limit = 0
    res = []
    for f, front in enumerate(fronts):
        res += front
        if len(res) > k:
            limit = f
            break
    selection = []
    if limit > 0:
        for f in range(limit):
            selection += fronts[f]
    selection += niching_select(fronts[limit], k - len(selection))
    return selection


def fast_nondominated_sort(indi_list):
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
    return ParetoFront


class NSGAIIIPopulation(Population):
    def __init__(self, pop_size):
        super().__init__(pop_size)
        self.ParetoFront = []

    def fast_nondominated_sort(self):
        self.ParetoFront = fast_nondominated_sort(self.indivs)

    def natural_selection(self):
        self.indivs = sel_nsga_iii(self.indivs, self.pop_size)


def run_nsga_iii(
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
    print("NSGA-III")
    nsga_iii_pop = NSGAIIIPopulation(pop_size)
    nsga_iii_pop.pre_indi_gen(indi_list)

    history = {}
    pool = multiprocessing.Pool(processing_number)
    arg = []

    for individual in nsga_iii_pop.indivs:
        arg.append((problem, individual))
    result = pool.starmap(cal_fitness, arg, chunksize=10)

    for individual, fitness in zip(nsga_iii_pop.indivs, result):
        individual.chromosome = fitness[0]
        individual.objectives = fitness[1:]

    # for individual in nsga_iii_pop.indivs:
    #     fitness = cal_fitness(problem, individual)
    #     individual.chromosome = fitness[0]
    #     individual.objectives = fitness[1:]

    nsga_iii_pop.fast_nondominated_sort()
    pool.close()
    pool.join()

    print("Generation 0: Done")
    Pareto_store = []
    for indi in nsga_iii_pop.ParetoFront[0]:
        Pareto_store.append(list(indi.objectives))
    history[0] = Pareto_store

    for gen in range(max_gen):
        print("Bắt đầu tạo offspring")
        offspring = nsga_iii_pop.gen_offspring(
            problem,
            crossover_operator,
            mutation_operator,
            crossover_rate,
            mutation_rate,
        )
        print("Bắt đầu tính fitness: ", len(offspring))
        pool = multiprocessing.Pool(processing_number)
        arg = []
        for individual in offspring:
            arg.append((problem, individual))

        result = pool.starmap(cal_fitness, arg, chunksize=10)
        for individual, fitness in zip(offspring, result):
            individual.chromosome = fitness[0]
            individual.objectives = fitness[1:]
        pool.close()
        pool.join()
        print("Tinh fitness xong")
        # for individual in offspring:
        #     fitness = cal_fitness(problem, individual)
        #     individual.chromosome = fitness[0]
        #     individual.objectives = fitness[1:]

        nsga_iii_pop.indivs.extend(offspring)
        nsga_iii_pop.natural_selection()
        nsga_iii_pop.fast_nondominated_sort()
        print(f"Generation {gen + 1}: Done")
        # print(len(nsga_iii_pop.ParetoFront[0]))
        Pareto_store = []
        for indi in nsga_iii_pop.ParetoFront[0]:
            Pareto_store.append(list(indi.objectives))
        history[gen + 1] = Pareto_store
    # pool.close()
    # print("NSGA-III Done: ", cal_hv_front(nsga_iii_pop.ParetoFront[0], np.array([100000,10000,100000])))
    print(
        "NSGA-III Done: ",
        cal_hv_front(nsga_iii_pop.ParetoFront[0], np.array([10, 100000])),
    )
    return history
