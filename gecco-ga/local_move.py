import random
from population import Individual


def local_swap(individual):
    """Swap two random positions in permutation"""
    perm, mask = individual.chromosome
    i, j = random.sample(range(len(perm)), 2)
    perm[i], perm[j] = perm[j], perm[i]
    return Individual([perm, mask])


def local_insert(individual):
    """Remove and reinsert at different position"""
    perm, mask = individual.chromosome
    i = random.randint(0, len(perm) - 1)
    j = random.randint(0, len(perm) - 1)
    val = perm.pop(i)
    perm.insert(j, val)
    return Individual([perm, mask])


def local_invert(individual):
    """Reverse a substring"""
    perm, mask = individual.chromosome
    i, j = sorted(random.sample(range(len(perm)), 2))
    perm[i:j] = perm[i:j][::-1]
    return Individual([perm, mask])


def local_flip(individual, problem):
    """Flip drone assignments strategically"""
    perm, mask = individual.chromosome
    # Flip 1-2 positions
    positions = random.sample(range(len(mask)), min(2, len(mask)))
    for pos in positions:
        if mask[pos] == 0:
            mask[pos] = random.choice([-1, 1])
        else:
            mask[pos] = 0
    return Individual([perm, mask])
