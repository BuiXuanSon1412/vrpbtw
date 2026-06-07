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


#
def dronable(chro, problem: Problem):
    n_node = len(problem.nodes)
    # the sequential drone nodes must have same sign
    for i in range(1, len(chro[0])):
        if not chro[1][i] or chro[0][i] >= n_node or chro[0][i - 1] >= n_node:
            continue
        if chro[1][i - 1] + chro[1][i] == 0:
            chro[1][i] = chro[1][i - 1]

    # re-distribute nodes into trips
    trip = []
    demands = [node.demand for node in problem.nodes]
    for i in range(len(chro[0])):
        if chro[0][i] >= n_node:
            trip.clear()

        elif chro[1][i]:
            if not trip or trip[-1][1] + chro[1][i] == 0:
                trip.clear()
                trip.append((chro[0][i], chro[1][i]))
            else:
                # verify sum of interior edges
                temp_trip = trip + [(chro[0][i], chro[1][i])]
                temp_in_distance = cal_route_distance(
                    [node[0] for node in temp_trip], problem
                )

                # verify drone load
                overloaded = False
                load = sum(
                    [demands[node[0]] for node in temp_trip if demands[node[0]] > 0]
                )
                if load > problem.drone_capacity:
                    overloaded = True
                else:
                    for node in temp_trip:
                        load -= demands[node[0]]
                        if load > problem.drone_capacity:
                            overloaded = True
                            break

                # violate traveling range and payload
                # solved by: switch sign of mask value from current violated gene
                if (
                    temp_in_distance > problem.drone_speed * problem.drone_trip_duration
                    or overloaded
                ):
                    if chro[1][i - 1] == chro[1][i] and chro[0][i - 1] < n_node:
                        chro[1][i] = 0

                    for j in range(i, len(chro[0])):
                        chro[1][j] = chro[1][j] * (-1)
                    trip.clear()
                trip.append((chro[0][i], chro[1][i]))


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


def schedule(
    route, trips, problem: Problem
) -> Tuple[Optional[Route], Optional[List[Route]], float, float]:
    n_node = len(problem.nodes)
    # Initialize Route objects
    t_route = Route(
        nodes=route,
        arrival=[0.0] * len(route),
        departure=[0.0] * len(route),
        service=[0.0] * len(route),
    )

    d_trips = []

    for trip in trips:
        d_trips.append(
            Route(
                nodes=trip,
                arrival=[0.0] * len(trip),
                departure=[0.0] * len(trip),
                service=[0.0] * len(trip),
            )
        )
    trip_idx = 0

    # --- Forward Pass ---
    for i in range(len(route)):
        curr_node = route[i]
        # 1. Truck Arrival Time
        if i > 0:
            prev_node = route[i - 1]
            dist = problem.drone_distance_matrix[prev_node][curr_node % n_node]
            travel_time = dist / problem.truck_speed
            t_route.arrival[i] = t_route.departure[i - 1] + travel_time

        # 2. Service Time (Earliest possible start)
        if i > 0 and i < len(route) - 1:
            tw_start = problem.nodes[curr_node].time_window[0]
            t_route.service[i] = max(t_route.arrival[i], tw_start)

        # 3. Complete current drone trip
        if trip_idx < len(d_trips) and curr_node == d_trips[trip_idx].nodes[0]:
            curr_trip = d_trips[trip_idx]
            dist = problem.drone_distance_matrix[curr_node][curr_trip.nodes[1]]
            travel_time = dist / problem.drone_speed
            temp_launch = (
                problem.nodes[curr_trip.nodes[1]].time_window[0]
                - problem.land_time
                - travel_time
                - problem.launch_time
            )
            curr_trip.departure[0] = max(temp_launch, t_route.arrival[i])
            t_route.departure[i] = max(curr_trip.departure[0], t_route.service[i])
            for i in range(1, len(curr_trip.nodes)):
                prev_node = curr_trip.nodes[i - 1]
                dist = problem.drone_distance_matrix[prev_node][
                    curr_trip.nodes[i] % n_node
                ]
                travel_time = dist / problem.drone_speed
                curr_trip.arrival[i] = (
                    curr_trip.departure[i - 1] + problem.launch_time + travel_time
                )

                if i < len(curr_trip.nodes) - 1:
                    tw_start = problem.nodes[curr_trip.nodes[i]].time_window[0]
                    curr_trip.service[i] = max(
                        tw_start, curr_trip.arrival[i] + problem.land_time
                    )

        else:
            if i == 0:
                t_route.departure[i] = 0
            elif i < len(t_route.nodes):
                if trip_idx < len(d_trips) and curr_node == d_trips[trip_idx].nodes[-1]:
                    t_route.departure[i] = max(
                        d_trips[trip_idx].arrival[-1] + problem.land_time,
                        t_route.service[i] + problem.service_time,
                    )
                    trip_idx = trip_idx + 1
                    if (
                        trip_idx < len(d_trips)
                        and curr_node == d_trips[trip_idx].nodes[0]
                    ):
                        curr_trip = d_trips[trip_idx]
                        dist = problem.drone_distance_matrix[curr_node][
                            curr_trip.nodes[1]
                        ]
                        travel_time = dist / problem.drone_speed
                        temp_launch = (
                            problem.nodes[curr_trip.nodes[1]].time_window[0]
                            - problem.land_time
                            - travel_time
                            - problem.launch_time
                        )
                        curr_trip.departure[0] = max(temp_launch, t_route.arrival[i])
                        t_route.departure[i] = max(
                            curr_trip.departure[0], t_route.service[i]
                        )
                        for i in range(1, len(curr_trip.nodes)):
                            prev_node = curr_trip.nodes[i - 1]
                            dist = problem.drone_distance_matrix[prev_node][
                                curr_trip.nodes[i] % n_node
                            ]
                            travel_time = dist / problem.drone_speed
                            curr_trip.arrival[i] = (
                                curr_trip.departure[i - 1]
                                + problem.launch_time
                                + travel_time
                            )

                            if i < len(curr_trip.nodes) - 1:
                                tw_start = problem.nodes[
                                    curr_trip.nodes[i]
                                ].time_window[0]
                                curr_trip.service[i] = max(
                                    tw_start, curr_trip.arrival[i] + problem.land_time
                                )
                else:
                    t_route.departure[i] = t_route.service[i] + problem.service_time

    # --- Max Tardiness ---
    max_tardiness = 0.0

    # Check Truck Customers
    for i in range(1, len(route) - 1):
        tw_end = problem.nodes[route[i]].time_window[1]
        tardiness = max(0.0, (t_route.service[i] + problem.service_time) - tw_end)
        max_tardiness = max(max_tardiness, tardiness)

    # Check Drone Customers
    for d_trip in d_trips:
        for i in range(1, len(d_trip.nodes) - 1):
            tw_end = problem.nodes[d_trip.nodes[i]].time_window[1]
            tardiness = max(0.0, d_trip.service[i] + problem.service_time - tw_end)
            max_tardiness = max(max_tardiness, tardiness)

    total_cost = cal_route_distance([node for node in route], problem)
    total_cost += sum(
        [cal_route_distance([node for node in trip], problem) for trip in trips]
    )
    return t_route, d_trips, max_tardiness, total_cost


def routing(seq, problem: Problem):
    end_depot = len(problem.nodes)
    nodes = [(0, 0)] + seq + [(end_depot, 0)]

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

    return (
        route,
        trips,
    )


def decode(indi: Individual, problem: Problem) -> Solution:
    chro = indi.chromosome

    seqs = partition(chro, problem)
    routes = []

    for seq in seqs:
        route, trips = routing(seq, problem)

        routes.append((route, trips))

    return Solution(routes)


def cal_route_distance(route, problem: Problem):
    n_node = len(problem.nodes)
    distance = sum(
        [
            problem.drone_distance_matrix[route[i] % n_node][route[i - 1] % n_node]
            for i in range(1, len(route))
        ]
    )
    return distance


def cal_service_rate(solution, problem):
    routes = problem.routes
    served = 0
    for route, trips in routes:
        served += len(route) - 2
        for trip in trips:
            served += len(trip) - 2

    return served


def cal_cost(solution, problem):
    routes = solution.routes
    cost = 0.0

    for route, trips in routes:
        cost += (
            cal_route_distance([node for node in route.nodes], problem)
            * problem.truck_cost
        )
        cost += (
            sum(
                [
                    cal_route_distance([node for node in trip.nodes], problem)
                    for trip in trips
                ]
            )
            * problem.drone_cost
        )

    cost += len(routes) * problem.basis_cost
    return cost


def cal_objective(problem: Problem, indi: Individual):
    chro = indi.chromosome
    solution = decode(indi, problem)
    cost = cal_cost(solution, problem)
    service_rate = cal_service_rate(solution, problem)
    weight = problem.num_customer * 2 * problem.max_coord


def cal_fitness(problem: Problem, indi: Individual):
    chro = indi.chromosome
    solution = decode(indi, problem)

    fitness = cal_objective(problem, indi)
    return (chro,)
