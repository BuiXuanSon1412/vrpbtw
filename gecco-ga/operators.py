from copy import deepcopy
import random
from problem import Problem
from population import Individual


# crossover
def pmx_crossover(parent1, parent2):
    # Extract layers
    p1_layer1, p1_layer2 = parent1.chromosome[0], parent1.chromosome[1]
    p2_layer1, p2_layer2 = parent2.chromosome[0], parent2.chromosome[1]

    size = len(p1_layer1)
    # choose two crossover points
    cx_point1, cx_point2 = sorted(random.sample(range(size), 2))

    def pmx_one_layer(l1, l2):
        offspring = [None] * size
        # copy the segment
        offspring[cx_point1:cx_point2] = l1[cx_point1:cx_point2]
        # mapping for duplicates
        for i in range(cx_point1, cx_point2):
            if l2[i] not in offspring:
                pos = i
                val = l2[i]
                while offspring[pos] is not None:
                    pos = l2.index(l1[pos])
                offspring[pos] = val
        # fill remaining with l2
        for i in range(size):
            if offspring[i] is None:
                offspring[i] = l2[i]
        return offspring

    child1_layer1 = pmx_one_layer(p1_layer1, p2_layer1)
    child2_layer1 = pmx_one_layer(p2_layer1, p1_layer1)

    # second layer: simple one-point crossover for flags
    cut = random.randint(1, size - 1)
    child1_layer2 = p1_layer2[:cut] + p2_layer2[cut:]
    child2_layer2 = p2_layer2[:cut] + p1_layer2[cut:]

    return (
        Individual([child1_layer1, child1_layer2]),
        Individual([child2_layer1, child2_layer2]),
    )


def ox_crossover(problem, parent1, parent2):
    p1_layer1, p1_layer2 = parent1.chromosome[0], parent1.chromosome[1]
    p2_layer1, p2_layer2 = parent2.chromosome[0], parent2.chromosome[1]

    size = len(p1_layer1)
    cx_point1, cx_point2 = sorted(random.sample(range(size), 2))

    def ox_one_layer(l1, l2):
        offspring = [None] * size
        offspring[cx_point1:cx_point2] = l1[cx_point1:cx_point2]
        fill_values = [x for x in l2 if x not in offspring]
        pos = cx_point2
        for val in fill_values:
            if pos >= size:
                pos = 0
            while offspring[pos] is not None:
                pos += 1
                if pos >= size:
                    pos = 0
            offspring[pos] = val
        return offspring

    child1_layer1 = ox_one_layer(p1_layer1, p2_layer1)
    child2_layer1 = ox_one_layer(p2_layer1, p1_layer1)

    # second layer: simple one-point crossover
    cut = random.randint(1, size - 1)
    child1_layer2 = p1_layer2[:cut] + p2_layer2[cut:]
    child2_layer2 = p2_layer2[:cut] + p1_layer2[cut:]

    return (
        Individual([child1_layer1, child1_layer2]),
        Individual([child2_layer1, child2_layer2]),
    )


def cx_crossover(problem, parent1, parent2):
    p1_layer1, p1_layer2 = parent1.chromosome[0], parent1.chromosome[1]
    p2_layer1, p2_layer2 = parent2.chromosome[0], parent2.chromosome[1]

    size = len(p1_layer1)

    def cx_one_layer(l1, l2):
        offspring = [None] * size
        visited = [False] * size
        index = 0
        while not all(visited):
            if visited[index]:
                index = visited.index(False)
            start = index
            while not visited[index]:
                offspring[index] = l1[index]
                visited[index] = True
                index = l1.index(l2[index])
            index = visited.index(False) if not all(visited) else 0
            # swap cycle to second parent
            for i in range(size):
                if offspring[i] is None:
                    offspring[i] = l2[i]
        return offspring

    child1_layer1 = cx_one_layer(p1_layer1, p2_layer1)
    child2_layer1 = cx_one_layer(p2_layer1, p1_layer1)

    # second layer: simple one-point crossover
    cut = random.randint(1, size - 1)
    child1_layer2 = p1_layer2[:cut] + p2_layer2[cut:]
    child2_layer2 = p2_layer2[:cut] + p1_layer2[cut:]

    return (
        Individual([child1_layer1, child1_layer2]),
        Individual([child2_layer1, child2_layer2]),
    )


def swap_mutation(individual):
    layer1, layer2 = individual.chromosome[0], individual.chromosome[1]
    size = len(layer1)
    i, j = random.sample(range(size), 2)
    # mutate layer1
    layer1[i], layer1[j] = layer1[j], layer1[i]
    # optionally also mutate layer2 in same positions
    layer2[i], layer2[j] = layer2[j], layer2[i]
    return Individual([layer1[:], layer2[:]])


def inversion_mutation(individual):
    layer1, layer2 = individual.chromosome[0], individual.chromosome[1]
    size = len(layer1)
    i, j = sorted(random.sample(range(size), 2))
    # reverse segment
    layer1[i:j] = reversed(layer1[i:j])
    layer2[i:j] = reversed(layer2[i:j])
    return Individual([layer1[:], layer2[:]])


def insertion_mutation(individual):
    layer1, layer2 = individual.chromosome[0], individual.chromosome[1]
    size = len(layer1)
    i, j = random.sample(range(size), 2)
    val1 = layer1.pop(i)
    val2 = layer2.pop(i)
    layer1.insert(j, val1)
    layer2.insert(j, val2)
    return Individual([layer1[:], layer2[:]])


def scramble_mutation(individual):
    layer1, layer2 = individual.chromosome[0], individual.chromosome[1]
    size = len(layer1)
    i, j = sorted(random.sample(range(size), 2))
    # scramble inside segment
    segment1 = layer1[i:j]
    segment2 = layer2[i:j]
    combined = list(zip(segment1, segment2))
    random.shuffle(combined)
    layer1[i:j], layer2[i:j] = zip(*combined)
    return Individual([layer1[:], layer2[:]])


def crossover_PMX(problem: Problem, parent1: Individual, parent2: Individual):
    size = len(parent1.chromosome[0])
    p1_perm, p1_bin = parent1.chromosome
    p2_perm, p2_bin = parent2.chromosome

    # chọn 2 điểm cắt
    cx_point1 = random.randint(0, size - 2)
    cx_point2 = random.randint(cx_point1 + 1, size - 1)

    off1_perm = [None] * size
    off2_perm = [None] * size

    # copy đoạn cắt từ cha
    off1_perm[cx_point1 : cx_point2 + 1] = p1_perm[cx_point1 : cx_point2 + 1]
    off2_perm[cx_point1 : cx_point2 + 1] = p2_perm[cx_point1 : cx_point2 + 1]

    # PMX mapping
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

    # điền chỗ trống
    for i in range(size):
        if off1_perm[i] is None:
            off1_perm[i] = p2_perm[i]
        if off2_perm[i] is None:
            off2_perm[i] = p1_perm[i]

    # ánh xạ lại binary theo vị trí
    idx_map1 = {val: i for i, val in enumerate(off1_perm)}
    idx_map2 = {val: i for i, val in enumerate(off2_perm)}
    off1_bin = [0] * size
    off2_bin = [0] * size

    for i, val in enumerate(p1_perm):
        off1_bin[idx_map1[val]] = p1_bin[i]
    for i, val in enumerate(p2_perm):
        off2_bin[idx_map2[val]] = p2_bin[i]

    # trả về offspring
    off1 = Individual([off1_perm, off1_bin])
    off2 = Individual([off2_perm, off2_bin])
    return off1, off2


def mutation_flip(problem: Problem, indi: Individual, num_flips=None):
    perm, bin_vec = indi.chromosome
    size = len(perm)

    if num_flips is None:
        num_flips = max(1, size // 10)  # mặc định 10% số gene

    # copy để không sửa trực tiếp parent
    new_bin = bin_vec.copy()

    flip_positions = random.sample(range(size), num_flips)
    for pos in flip_positions:
        if new_bin[pos]:
            new_bin[pos] = 0
        else:
            new_bin[pos] = random.choice([-1, 1])

    # tạo offspring mới
    offspring = Individual([perm.copy(), new_bin])
    return offspring
