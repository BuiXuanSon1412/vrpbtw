from copy import deepcopy
import enum
import random
from typing import List, Optional, Tuple, Any
import numpy as np
from collections import deque

from problem import Problem, Solution, Route
from population import Individual


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

        mask = random.choices(
            [0, 1, -1], weights=[0.7, 0.15, 0.15], k=(num_nodes + num_fleet - 1)
        )
        chromosome = [nodes, mask]

        indi = Individual(chromosome)
        popu.append(indi)
    return popu


def crossover(problem: Problem, parent1: Individual, parent2: Individual):
    """
    Partially Mapped Crossover (PMX) for permutation-based chromosomes.
    Operates on two-layer chromosome: [permutation, binary_mask]
    """
    size = len(parent1.chromosome[0])
    p1_perm, p1_bin = parent1.chromosome
    p2_perm, p2_bin = parent2.chromosome

    cx_point1 = random.randint(0, size - 2)
    cx_point2 = random.randint(cx_point1 + 1, size - 1)

    off1_perm = [None] * size
    off2_perm = [None] * size

    off1_perm[cx_point1 : cx_point2 + 1] = p1_perm[cx_point1 : cx_point2 + 1]
    off2_perm[cx_point1 : cx_point2 + 1] = p2_perm[cx_point1 : cx_point2 + 1]

    for i in range(cx_point1, cx_point2 + 1):
        if p2_perm[i] not in off1_perm:
            pos = i
            while off1_perm[pos] is not None:
                pos = p2_perm.index(p1_perm[pos])
            off1_perm[pos] = p2_perm[i]

        if p1_perm[i] not in off2_perm:
            pos = i
            while off2_perm[pos] is not None:
                pos = p1_perm.index(p2_perm[pos])
            off2_perm[pos] = p1_perm[i]

    for i in range(size):
        if off1_perm[i] is None:
            off1_perm[i] = p2_perm[i]
        if off2_perm[i] is None:
            off2_perm[i] = p1_perm[i]

    idx_map1 = {val: i for i, val in enumerate(off1_perm)}
    idx_map2 = {val: i for i, val in enumerate(off2_perm)}
    off1_bin = [0] * size
    off2_bin = [0] * size

    for i, val in enumerate(p1_perm):
        off1_bin[idx_map1[val]] = p1_bin[i]
    for i, val in enumerate(p2_perm):
        off2_bin[idx_map2[val]] = p2_bin[i]

    off1 = Individual([off1_perm, off1_bin])
    off2 = Individual([off2_perm, off2_bin])
    return off1, off2


def mutation(problem: Problem, indi: Individual, num_flips=None):
    """
    Flip mutation: randomly flip binary mask genes with probability.
    Flipped 0s become random [-1, 1], flipped 1s become 0.
    """
    perm, bin_vec = indi.chromosome
    size = len(perm)

    if num_flips is None:
        num_flips = max(1, size // 10)

    new_bin = bin_vec.copy()

    flip_positions = random.sample(range(size), num_flips)
    for pos in flip_positions:
        if new_bin[pos]:
            new_bin[pos] = 0
        else:
            new_bin[pos] = random.choice([-1, 1])

    offspring = Individual([perm.copy(), new_bin])
    return offspring


def partition(chro, problem):
    seqs = []
    start = 0
    n_node = len(problem.nodes)
    for i in range(len(chro[0])):
        if chro[0][i] >= n_node:
            if i > start:
                seq = []
                for j in range(start, i):
                    seq.append((chro[0][j], chro[1][j]))
                seqs.append(seq)
            start = i + 1
        elif i == len(chro[0]) - 1:
            if i + 1 > start:
                seq = []
                for j in range(start, i + 1):
                    seq.append((chro[0][j], chro[1][j]))
                seqs.append(seq)

    return seqs


# unset mask layer if its node capacity overload the drone capacity itself
def unset(chro, problem: Problem):
    n_node = len(problem.nodes)
    for i in range(len(chro[0])):
        if (
            chro[0][i] < n_node
            and chro[1][i]
            and abs(problem.nodes[chro[0][i]].demand) > problem.drone_capacity
        ):
            chro[1][i] = 0


# check if the truck overloads
def overload(chro, problem: Problem):
    demands = [node.demand for node in problem.nodes]
    load = 0
    for i in range(len(chro[0])):
        if chro[0][i] >= len(problem.nodes) or i == len(chro[0]) - 1:
            if load > problem.truck_capacity:
                load = 0
                return True
            else:
                load = 0
        else:
            load += abs(demands[chro[0][i]])

    return False


def dronable(chro, problem: Problem):
    n_node = len(problem.nodes)
    # the sequential drone nodes must have same sign
    for i in range(1, len(chro[0])):
        if not chro[1][i] or chro[0][i] >= n_node or chro[0][i - 1] >= n_node:
            continue
        if chro[1][i - 1] + chro[1][i] == 0:
            chro[1][i] = chro[1][i - 1]


def balance(chro, problem: Problem):
    demands = [node.demand for node in problem.nodes]

    delimiter_queue = deque()
    node_queue = deque()
    n_node = len(problem.nodes)
    for i in range(len(chro[0])):
        if chro[0][i] >= n_node:
            delimiter_queue.append((chro[0][i], chro[1][i]))
        else:
            node_queue.append((chro[0][i], chro[1][i]))

    avg_count = (n_node - 1) / problem.num_fleet
    load = 0
    count = 0
    new_genes = []
    while node_queue:
        gene = node_queue.popleft()
        if load + abs(demands[gene[0]]) <= problem.truck_capacity and count < avg_count:
            load += abs(demands[gene[0]])
            count = count + 1
        else:
            new_genes.append(delimiter_queue.popleft())
            load = 0
            count = 0
        new_genes.append(gene)

    return np.array(new_genes).T.tolist()


def repair(chro, problem):
    unset(chro, problem)

    if overload(chro, problem):
        chro = balance(chro, problem)

    dronable(chro, problem)


def routing(seq, problem: Problem):
    end_depot = len(problem.nodes)
    nodes = seq + [[end_depot, 0]]
    # separate truck route and drone trips in one sequence
    route = []
    trips = []
    trip = []
    for idx, node in enumerate(nodes):
        if not node[1]:
            route.append(node[0])
        elif node[0] < len(problem.nodes):
            if trip and nodes[trip[-1]][1] != node[1]:
                trips.append(deepcopy(trip))
                trip.clear()
            trip.append(idx)
    if trip:
        trips.append(trip)

    truck_route = [0]
    truck_arrive = [0.0]
    truck_service = [0.0]
    truck_depart = [0.0]
    truck_load = [0]

    drone_trips = []

    truck_time = 0.0

    for node in route:
        tmp_truck_time = (
            truck_time
            + problem.truck_time_matrix[truck_route[-1]][node % len(problem.nodes)]
        )

        # time feasible
        if (
            tmp_truck_time
            > problem.nodes[node % len(problem.nodes)].time_window[1]
            - problem.service_time
        ):
            continue

        # system time limit
        if (
            tmp_truck_time
            + problem.service_time
            + problem.truck_time_matrix[node % len(problem.nodes)][0]
            > problem.system_duration
        ):
            continue

        # load capacity
        max_linehaul = problem.truck_capacity - max(truck_load, default=0)
        demand = problem.nodes[node % len(problem.nodes)].demand

        if (demand < 0 and truck_load[-1] - demand > problem.truck_capacity) or (
            demand > 0 and demand > max_linehaul
        ):
            continue

        truck_route.append(node)
        truck_load.append(truck_load[-1])
        truck_arrive.append(tmp_truck_time)
        truck_service.append(
            max(tmp_truck_time, problem.nodes[node % len(problem.nodes)].time_window[0])
        )
        truck_depart.append(truck_service[-1] + problem.service_time)

        if demand > 0:
            for idx, _ in enumerate(truck_load[:-1]):
                truck_load[idx] += demand
        else:
            truck_load[-1] += demand

    d_trips = []
    drone_trips = []
    drone_arrives = []
    drone_departs = []

    for trip in trips:
        drone_trip = []
        drone_load = []
        drone_arrive = []
        drone_depart = []

        for idx in trip:
            # launch indices
            if not drone_trip:
                launch_indices = []
                launch_idx = idx - 1
                while launch_idx >= 0 and (nodes[launch_idx][1] == 0):
                    if nodes[launch_idx][1] + nodes[idx][1] == 0:
                        break
                    if (
                        nodes[launch_idx][1] == 0
                        and nodes[launch_idx][0] in truck_route
                    ):
                        launch_indices.append(launch_idx)

                    launch_idx = launch_idx - 1

                # land indices
                land_indices = []
                land_idx = trip[-1] + 1
                while land_idx < len(nodes):
                    if nodes[land_idx][1] + nodes[idx][1] == 0:
                        break
                    if nodes[land_idx][1] == 0 and nodes[land_idx][0] in truck_route:
                        land_indices.append(land_idx)

                    land_idx = land_idx + 1

                if not launch_indices or not land_indices:
                    continue

                for launch_idx in launch_indices:
                    launch_node = nodes[launch_idx][0]
                    node = nodes[idx][0]

                    ub_depart = (
                        problem.nodes[node].time_window[1]
                        - problem.service_time
                        - problem.drone_time_matrix[node][launch_node]
                    )

                    # can serve on time
                    if ub_depart > truck_depart[truck_route.index(launch_node)]:
                        break

                    # load feasible
                    demand = problem.nodes[node].demand
                    max_linehaul = problem.truck_capacity - max(
                        truck_load[: truck_route.index(launch_node)], default=0
                    )

                    if demand > 0 and demand > max_linehaul:
                        continue

                    launch_depart = max(
                        ub_depart, truck_arrive[truck_route.index(launch_node)]
                    )

                    node_arrive = max(
                        launch_depart + problem.drone_time_matrix[launch_node][node],
                        problem.nodes[node].time_window[0],
                    )
                    node_depart = node_arrive + problem.service_time

                    found_pair = False
                    for land_idx in land_indices:
                        land_node = nodes[land_idx][0]
                        land_arrive = max(
                            node_depart
                            + problem.drone_time_matrix[node][
                                land_node % len(problem.nodes)
                            ],
                            problem.nodes[land_node % len(problem.nodes)].time_window[
                                0
                            ],
                        )
                        if land_arrive - launch_depart > problem.drone_trip_duration:
                            break

                        # backhaul feasible
                        if (
                            demand < 0
                            and max(truck_load[truck_route.index(land_node) :])
                            + abs(demand)
                            > problem.truck_capacity
                        ):
                            continue

                        drone_trip.extend([launch_node, node, land_node])
                        if demand > 0:
                            drone_load.extend([demand, 0, 0])
                        else:
                            drone_load.extend([0, abs(demand), 0])

                        drone_arrive.extend([0, node_arrive, land_arrive])
                        drone_depart.extend([launch_depart, node_depart, 0])
                        found_pair = True
                        break
                    if found_pair:
                        break
            else:
                node = nodes[idx][0]
                # linehaul feasible
                demand = problem.nodes[node].demand
                max_linehaul = min(
                    problem.truck_capacity
                    - max(truck_load[: truck_route.index(drone_trip[0])], default=0),
                    problem.drone_capacity - max(drone_load[:-1], default=0),
                )
                if demand > 0 and demand > max_linehaul:
                    continue

                # land indices
                land_indices = []
                land_idx = trip[-1] + 1

                while land_idx < len(nodes):
                    if nodes[land_idx][1] + nodes[idx][1] == 0:
                        break
                    if nodes[land_idx][1] == 0 and nodes[land_idx][0] in truck_route:
                        land_indices.append(land_idx)

                    land_idx = land_idx + 1

                last_depart = drone_depart[-2]
                last_node = drone_trip[-2]
                node_arrive = max(
                    last_depart + problem.drone_time_matrix[last_node][node],
                    problem.nodes[node].time_window[0],
                )
                # serve on time
                if (
                    node_arrive
                    > problem.nodes[node].time_window[1] - problem.service_time
                ):
                    continue

                node_depart = node_arrive + problem.service_time

                found_pair = False
                for land_idx in land_indices:
                    land_node = nodes[land_idx][0]
                    land_arrive = max(
                        node_depart
                        + problem.drone_time_matrix[node][
                            land_node % len(problem.nodes)
                        ],
                        problem.nodes[land_node % len(problem.nodes)].time_window[0],
                    )
                    # drone trip duration
                    if land_arrive - drone_depart[0] > problem.drone_trip_duration:
                        break
                    # backhaul feasible
                    if (
                        demand < 0
                        and max(
                            truck_load[
                                truck_route.index(land_node % len(problem.nodes)) :
                            ]
                        )
                        + abs(demand)
                        > problem.truck_capacity
                    ):
                        continue

                    drone_trip.extend([node, land_node])
                    drone_depart.extend([node_depart, 0])
                    drone_arrive.extend([node_arrive, land_arrive])
                    found_pair = True
                    break
                if found_pair:
                    break

        if drone_trip:
            drone_trips.append(drone_trip)
            drone_arrives.append(drone_arrive)
            drone_departs.append(drone_depart)

    t_route = Route(
        nodes=truck_route,
        arrival=truck_arrive,
        departure=truck_depart,
    )

    for idx in range(len(drone_trips)):
        d_trips.append(
            Route(
                nodes=drone_trips[idx],
                arrival=drone_arrives[idx],
                departure=drone_departs[idx],
            )
        )

    return (t_route, d_trips)


def decode(indi: Individual, problem: Problem) -> Solution:
    chro = indi.chromosome

    repaired = deepcopy(chro)
    repair(repaired, problem)

    seqs = partition(repaired, problem)
    routes = []

    for seq in seqs:
        route, trips = routing(seq, problem)

        routes.append((route, trips))

    return Solution(routes)


def cal_drone_route_distance(route, problem: Problem):
    n_node = len(problem.nodes)
    distance = sum(
        [
            problem.drone_distance_matrix[route[i] % n_node][route[i - 1] % n_node]
            for i in range(1, len(route))
        ]
    )
    return distance


def cal_truck_route_distance(route, problem: Problem):
    n_node = len(problem.nodes)
    distance = sum(
        [
            problem.truck_distance_matrix[route[i] % n_node][route[i - 1] % n_node]
            for i in range(1, len(route))
        ]
    )
    return distance


def cal_service_rate(solution, problem):
    routes = solution.routes
    served = 0
    for route, trips in routes:
        served += len(route.nodes) - 2
        for trip in trips:
            served += len(trip.nodes) - 2

    return served


def cal_cost(solution, problem):
    routes = solution.routes
    cost = 0.0

    for route, trips in routes:
        cost += (
            cal_truck_route_distance([node for node in route.nodes], problem)
            * problem.truck_cost
        )
        cost += (
            sum(
                [
                    cal_drone_route_distance([node for node in trip.nodes], problem)
                    for trip in trips
                ]
            )
            * problem.drone_cost
        )

    cost += len(routes) * problem.basis_cost
    return cost


def cal_objective(problem: Problem, solution: Solution):
    cost = cal_cost(solution, problem)
    service_rate = cal_service_rate(solution, problem)
    weight = (
        problem.num_customer * 2 * problem.max_coord
        + problem.num_fleet * problem.basis_cost
    )

    return service_rate * weight + cost


def cal_fitness(problem: Problem, indi: Individual):
    solution = decode(indi, problem)

    fitness = cal_objective(problem, solution)
    service_rate = cal_service_rate(solution, problem)
    cost = cal_cost(solution, problem)
    return fitness, service_rate, cost


def cal_all_metrics(problem: Problem, indi: Individual):
    """Calculate fitness, service_rate, and cost for an individual."""
    solution = decode(indi, problem)
    fitness = cal_objective(problem, solution)
    service_rate = cal_service_rate(solution, problem)
    cost = cal_cost(solution, problem)
    return fitness, service_rate, cost
