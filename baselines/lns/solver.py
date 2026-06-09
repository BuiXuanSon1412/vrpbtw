import argparse
from copy import deepcopy
import json
import math
from pathlib import Path
import random
import sys
import time
from typing import Dict, Iterable, List, Optional

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from ga.population import Individual
from ga.problem import Problem

try:
    from .operators import evaluate, lns_neighbor
    from .utils import init_population
except ImportError:
    from operators import evaluate, lns_neighbor
    from utils import init_population


def _evaluate_individual(
    problem: Problem,
    individual: Individual,
    tardiness_penalty: float,
):
    score, cost, service_rate, served_count, tardiness, fixed_chro = evaluate(
        problem, individual.chromosome, tardiness_penalty
    )
    individual.chromosome = fixed_chro
    individual.scalar_fitness = score
    individual.objectives = [cost, service_rate]
    individual.served_count = served_count
    individual.tardiness = tardiness
    return individual


def _snapshot(individual: Individual) -> Dict[str, float]:
    return {
        "scalar": individual.scalar_fitness,
        "cost": individual.objectives[0],
        "service_rate": individual.objectives[1],
        "served_count": individual.served_count,
        "tardiness_penalty_value": individual.tardiness,
    }


def _print_iter(iteration: int, current: Individual, best: Individual):
    print(
        f"  Iter {iteration:>5d} | current={current.scalar_fitness:>12.4f} "
        f"| best={best.scalar_fitness:>12.4f} "
        f"| cost={best.objectives[0]:>10.2f} "
        f"| service_rate={best.objectives[1]:>6.2%} "
        f"| served={best.served_count:>4d}"
    )


def _initial_solution(
    problem: Problem,
    indi_list: Optional[List[Individual]],
    pop_size: int,
    seed: int,
    tardiness_penalty: float,
) -> Individual:
    if not indi_list:
        indi_list = init_population(pop_size, seed, problem)

    evaluated = [
        _evaluate_individual(
            problem, Individual(deepcopy(ind.chromosome)), tardiness_penalty
        )
        for ind in indi_list
    ]
    return min(evaluated, key=lambda ind: ind.scalar_fitness)


def _accept(
    candidate_score: float,
    current_score: float,
    temperature: float,
    rng: random.Random,
) -> bool:
    if candidate_score <= current_score:
        return True
    if temperature <= 1e-12:
        return False
    return rng.random() < math.exp(-(candidate_score - current_score) / temperature)


def run_lns(
    problem: Problem,
    indi_list: Optional[List[Individual]] = None,
    pop_size: int = 50,
    max_gen: int = 200,
    seed: int = 42,
    destroy_rate: float = 0.2,
    tardiness_penalty: float = 0.0,
    initial_temperature: float = 1.0,
    cooling_rate: float = 0.995,
    position_sample: int = 25,
    verbose: bool = True,
) -> Dict[str, object]:
    rng = random.Random(seed)
    random.seed(seed)

    current = _initial_solution(problem, indi_list, pop_size, seed, tardiness_penalty)
    best = deepcopy(current)
    history = {0: _snapshot(best)}

    temperature = initial_temperature
    if verbose:
        print(
            "LNS "
            f"(destroy_rate={destroy_rate}, tardiness_penalty={tardiness_penalty}, "
            f"temp={initial_temperature}, cooling={cooling_rate})"
        )
        _print_iter(0, current, best)

    for iteration in range(1, max_gen + 1):
        candidate_chro = lns_neighbor(
            current.chromosome,
            problem,
            rng,
            destroy_rate=destroy_rate,
            tardiness_penalty=tardiness_penalty,
            position_sample=position_sample,
        )
        candidate = _evaluate_individual(
            problem, Individual(candidate_chro), tardiness_penalty
        )

        if _accept(
            candidate.scalar_fitness, current.scalar_fitness, temperature, rng
        ):
            current = candidate

        if current.scalar_fitness < best.scalar_fitness:
            best = deepcopy(current)

        history[iteration] = _snapshot(best)
        temperature *= cooling_rate

        if verbose and (iteration == 1 or iteration % 10 == 0):
            _print_iter(iteration, current, best)

    if verbose:
        print(
            f"\nLNS Done | best scalar={best.scalar_fitness:.4f} "
            f"| cost={best.objectives[0]:.2f} "
            f"| service_rate={best.objectives[1]:.2%} "
            f"| served={best.served_count}"
        )

    return {"history": history, "best": best}


def iter_data_files(data_path: Path) -> Iterable[Path]:
    if data_path.is_file():
        yield data_path
        return

    for file_path in sorted(data_path.rglob("*.json")):
        yield file_path


def _serializable_result(result: Dict[str, object]) -> Dict[str, object]:
    best = result["best"]
    return {
        "history": result["history"],
        "best": {
            "chromosome": best.chromosome,
            "scalar": best.scalar_fitness,
            "cost": best.objectives[0],
            "service_rate": best.objectives[1],
            "served_count": best.served_count,
            "tardiness_penalty_value": best.tardiness,
        },
    }


def solve_file(
    data_file: Path,
    output_root: Path,
    args,
) -> Dict[str, object]:
    start = time.time()
    problem = Problem(str(data_file))
    initial = init_population(args.pop_size, args.seed, problem)

    result = run_lns(
        problem=problem,
        indi_list=initial,
        pop_size=args.pop_size,
        max_gen=args.max_gen,
        seed=args.seed,
        destroy_rate=args.destroy_rate,
        tardiness_penalty=args.tardiness_penalty,
        initial_temperature=args.initial_temperature,
        cooling_rate=args.cooling_rate,
        position_sample=args.position_sample,
        verbose=args.verbose,
    )
    elapsed = time.time() - start

    serializable = _serializable_result(result)
    serializable["time"] = elapsed
    serializable["data"] = str(data_file)

    size_dir = data_file.parent.name
    out_path = output_root / size_dir / data_file.name
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(serializable, f, indent=2)

    print(f"[OK] {data_file.name} -> {out_path} ({elapsed:.2f}s)")
    return serializable


def parse_args():
    parser = argparse.ArgumentParser(description="Large Neighborhood Search for VRPBTW")
    parser.add_argument(
        "--data",
        type=Path,
        default=ROOT / "data" / "generated" / "data",
        help="JSON instance file or directory. Default: data/generated/data",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "baselines" / "lns" / "result",
        help="Directory for JSON results",
    )
    parser.add_argument("--pop-size", type=int, default=50)
    parser.add_argument("--max-gen", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--destroy-rate", type=float, default=0.2)
    parser.add_argument("--tardiness-penalty", type=float, default=0.0)
    parser.add_argument("--initial-temperature", type=float, default=1.0)
    parser.add_argument("--cooling-rate", type=float, default=0.995)
    parser.add_argument("--position-sample", type=int, default=25)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of files to solve when --data is a directory",
    )
    args = parser.parse_args()
    args.verbose = not args.quiet
    return args


def main():
    args = parse_args()
    files = list(iter_data_files(args.data))
    if args.limit is not None:
        files = files[: args.limit]

    if not files:
        raise FileNotFoundError(f"No JSON files found under {args.data}")

    print(f"Found {len(files)} instance(s)")
    for data_file in files:
        solve_file(data_file, args.output, args)


if __name__ == "__main__":
    main()
