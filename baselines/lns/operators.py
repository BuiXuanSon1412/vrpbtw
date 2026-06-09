from copy import deepcopy
import math
import random
from typing import Callable, Iterable, List, Sequence, Tuple

from ga.population import Individual
from ga.problem import Problem

try:
    from .utils import balance, cal_cost, decode, dronable, overload, unset
except ImportError:
    from utils import balance, cal_cost, decode, dronable, overload, unset


Gene = Tuple[int, int]
Chromosome = List[List[int]]


def clone_chromosome(chromosome: Chromosome) -> Chromosome:
    return [chromosome[0].copy(), chromosome[1].copy()]


def service_weight(problem: Problem) -> float:
    max_coord = max(max(node.coord) for node in problem.nodes)
    num_customers = len(problem.nodes) - 1
    return (
        max(problem.truck_cost, problem.drone_cost)
        * num_customers
        * 2
        * max_coord
        + problem.basis_cost * problem.num_fleet
    )


def served_nodes(chromosome: Chromosome, problem: Problem) -> set:
    n_node = len(problem.nodes)
    return {node for node in chromosome[0] if 0 < node < n_node}


def service_rate(served_count: int, problem: Problem) -> float:
    num_customers = len(problem.nodes) - 1
    if num_customers <= 0:
        return 0.0
    return served_count / num_customers


def solution_service_stats(solution, problem: Problem) -> Tuple[int, float]:
    served = set()
    max_tardiness = [0.0]

    def visit(route, idx):
        node = route.nodes[idx]
        if node <= 0 or node >= len(problem.nodes):
            return

        tw_end = problem.nodes[node].time_window[1]
        completion = route.service[idx] + problem.service_time
        tardiness = max(0.0, completion - tw_end)
        if tardiness <= 1e-9:
            served.add(node)
        max_tardiness[0] = max(max_tardiness[0], tardiness)

    for route, trips in solution.routes:
        for idx in range(1, len(route.nodes) - 1):
            visit(route, idx)
        for trip in trips:
            for idx in range(1, len(trip.nodes) - 1):
                visit(trip, idx)

    return len(served), max_tardiness[0]


def scalarize(
    cost: float,
    served_count: int,
    tardiness: float,
    problem: Problem,
    tardiness_penalty: float = 0.0,
) -> float:
    return (
        cost
        - service_weight(problem) * served_count
        + tardiness_penalty * tardiness
    )


def repair_chromosome(chromosome: Chromosome, problem: Problem) -> Chromosome:
    repaired = clone_chromosome(chromosome)
    unset(repaired, problem)
    if overload(repaired, problem):
        repaired[:] = balance(repaired, problem)
    dronable(repaired, problem)
    return repaired


def evaluate(
    problem: Problem,
    chromosome: Chromosome,
    tardiness_penalty: float = 0.0,
) -> Tuple[float, float, float, int, float, Chromosome]:
    repaired = repair_chromosome(chromosome, problem)
    individual = Individual(repaired)
    _, cost_solution = decode(individual, problem)
    fixed_chro = individual.chromosome
    cost = cal_cost(cost_solution, problem)
    served_count, tardiness = solution_service_stats(cost_solution, problem)
    rate = service_rate(served_count, problem)
    score = scalarize(cost, served_count, tardiness, problem, tardiness_penalty)
    return score, cost, rate, served_count, tardiness, fixed_chro


def chromosome_to_genes(chromosome: Chromosome) -> List[Gene]:
    return list(zip(chromosome[0], chromosome[1]))


def genes_to_chromosome(genes: Sequence[Gene]) -> Chromosome:
    return [[node for node, _ in genes], [mask for _, mask in genes]]


def customer_positions(chromosome: Chromosome, problem: Problem) -> List[int]:
    n_node = len(problem.nodes)
    return [idx for idx, node in enumerate(chromosome[0]) if node < n_node]


def missing_customer_genes(chromosome: Chromosome, problem: Problem) -> List[Gene]:
    present = served_nodes(chromosome, problem)
    return [
        (node_id, 0)
        for node_id in range(1, len(problem.nodes))
        if node_id not in present
    ]


def _remove_positions(chromosome: Chromosome, positions: Iterable[int]):
    remove_set = set(positions)
    remaining = []
    removed = []

    for idx, gene in enumerate(chromosome_to_genes(chromosome)):
        if idx in remove_set:
            removed.append(gene)
        else:
            remaining.append(gene)

    return genes_to_chromosome(remaining), removed


def destroy_random(
    chromosome: Chromosome,
    problem: Problem,
    remove_count: int,
    rng: random.Random,
) -> Tuple[Chromosome, List[Gene]]:
    positions = customer_positions(chromosome, problem)
    remove_count = min(remove_count, len(positions))
    return _remove_positions(chromosome, rng.sample(positions, remove_count))


def destroy_route(
    chromosome: Chromosome,
    problem: Problem,
    remove_count: int,
    rng: random.Random,
) -> Tuple[Chromosome, List[Gene]]:
    n_node = len(problem.nodes)
    routes = []
    start = 0

    for idx, node in enumerate(chromosome[0]):
        if node >= n_node:
            routes.append(list(range(start, idx)))
            start = idx + 1
    if start < len(chromosome[0]):
        routes.append(list(range(start, len(chromosome[0]))))

    routes = [[idx for idx in route if chromosome[0][idx] < n_node] for route in routes]
    routes = [route for route in routes if route]
    if not routes:
        return destroy_random(chromosome, problem, remove_count, rng)

    route = rng.choice(routes)
    remove_count = min(remove_count, len(route))
    return _remove_positions(chromosome, rng.sample(route, remove_count))


def destroy_related(
    chromosome: Chromosome,
    problem: Problem,
    remove_count: int,
    rng: random.Random,
) -> Tuple[Chromosome, List[Gene]]:
    positions = customer_positions(chromosome, problem)
    if not positions:
        return clone_chromosome(chromosome), []

    seed_pos = rng.choice(positions)
    seed_node = chromosome[0][seed_pos]
    ranked = sorted(
        positions,
        key=lambda pos: problem.distance_matrix[seed_node][chromosome[0][pos]],
    )
    return _remove_positions(chromosome, ranked[: min(remove_count, len(ranked))])


def destroy_worst(
    chromosome: Chromosome,
    problem: Problem,
    remove_count: int,
    rng: random.Random,
) -> Tuple[Chromosome, List[Gene]]:
    positions = customer_positions(chromosome, problem)
    if not positions:
        return clone_chromosome(chromosome), []

    genes = chromosome_to_genes(chromosome)
    n_node = len(problem.nodes)
    end_depot = len(problem.nodes)
    scores = []

    for pos in positions:
        prev_node = 0
        next_node = end_depot

        for left in range(pos - 1, -1, -1):
            if genes[left][0] < n_node:
                prev_node = genes[left][0]
                break
            prev_node = 0

        for right in range(pos + 1, len(genes)):
            if genes[right][0] < n_node:
                next_node = genes[right][0]
                break
            next_node = end_depot

        node = genes[pos][0]
        saving = (
            problem.distance_matrix[prev_node % n_node][node]
            + problem.distance_matrix[node][next_node % n_node]
            - problem.distance_matrix[prev_node % n_node][next_node % n_node]
        )
        scores.append((saving, pos))

    scores.sort(reverse=True)
    selected = [pos for _, pos in scores[: min(remove_count, len(scores))]]
    return _remove_positions(chromosome, selected)


DESTROY_OPERATORS: List[
    Callable[[Chromosome, Problem, int, random.Random], Tuple[Chromosome, List[Gene]]]
] = [destroy_random, destroy_route, destroy_related, destroy_worst]


def greedy_repair(
    partial: Chromosome,
    removed: Sequence[Gene],
    problem: Problem,
    rng: random.Random,
    tardiness_penalty: float = 0.0,
    position_sample: int = 25,
    mask_choices: Sequence[int] = (0, 1, -1),
) -> Chromosome:
    current = clone_chromosome(partial)
    pending = []
    seen = set()
    for gene in removed:
        if gene[0] in seen:
            continue
        pending.append(gene)
        seen.add(gene[0])
    rng.shuffle(pending)

    for node, old_mask in pending:
        choices = list(dict.fromkeys([old_mask, *mask_choices]))
        candidate_positions = list(range(len(current[0]) + 1))
        if len(candidate_positions) > position_sample:
            candidate_positions = rng.sample(candidate_positions, position_sample)
            candidate_positions.extend([0, len(current[0])])
            candidate_positions = sorted(set(candidate_positions))

        score, cost, rate, served, tardiness, fixed = evaluate(
            problem, current, tardiness_penalty
        )
        best = (score, -served, cost)
        best_chro = fixed
        for pos in candidate_positions:
            for mask in choices:
                trial = clone_chromosome(current)
                trial[0].insert(pos, node)
                trial[1].insert(pos, mask)
                score, cost, rate, served, tardiness, fixed = evaluate(
                    problem, trial, tardiness_penalty
                )
                key = (score, -served, cost)
                if key < best:
                    best = key
                    best_chro = fixed

        current = best_chro

    return current


def random_repair(
    partial: Chromosome,
    removed: Sequence[Gene],
    problem: Problem,
    rng: random.Random,
) -> Chromosome:
    genes = chromosome_to_genes(partial)
    seen = {node for node, _ in genes}
    for gene in removed:
        if gene[0] in seen:
            continue
        genes.insert(rng.randint(0, len(genes)), gene)
        seen.add(gene[0])
    return repair_chromosome(genes_to_chromosome(genes), problem)


def mutate_drone_masks(
    chromosome: Chromosome,
    problem: Problem,
    rng: random.Random,
    mutation_rate: float = 0.05,
) -> Chromosome:
    mutated = clone_chromosome(chromosome)
    n_node = len(problem.nodes)
    for idx, node in enumerate(mutated[0]):
        if node >= n_node or rng.random() > mutation_rate:
            continue
        if mutated[1][idx]:
            mutated[1][idx] = 0
        else:
            mutated[1][idx] = rng.choice([-1, 1])
    return repair_chromosome(mutated, problem)


def lns_neighbor(
    chromosome: Chromosome,
    problem: Problem,
    rng: random.Random,
    destroy_rate: float = 0.2,
    tardiness_penalty: float = 0.0,
    position_sample: int = 25,
) -> Chromosome:
    positions = customer_positions(chromosome, problem)
    if not positions:
        missing = missing_customer_genes(chromosome, problem)
        if not missing:
            return clone_chromosome(chromosome)
        extra_count = max(1, math.ceil(len(missing) * destroy_rate))
        return greedy_repair(
            chromosome,
            rng.sample(missing, min(extra_count, len(missing))),
            problem,
            rng,
            tardiness_penalty=tardiness_penalty,
            position_sample=position_sample,
        )

    remove_count = max(1, math.ceil(len(positions) * destroy_rate))
    destroy = rng.choice(DESTROY_OPERATORS)
    partial, removed = destroy(chromosome, problem, remove_count, rng)
    missing = missing_customer_genes(partial, problem)
    if missing:
        extra_count = min(len(missing), remove_count)
        removed.extend(rng.sample(missing, extra_count))

    if rng.random() < 0.85:
        candidate = greedy_repair(
            partial,
            removed,
            problem,
            rng,
            tardiness_penalty=tardiness_penalty,
            position_sample=position_sample,
        )
    else:
        candidate = random_repair(partial, removed, problem, rng)

    return mutate_drone_masks(candidate, problem, rng, mutation_rate=0.03)
