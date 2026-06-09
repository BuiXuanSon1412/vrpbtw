import os
import sys
import time
import json
from pathlib import Path
from typing import Dict, Any

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from problem import Problem
from utils import init_population, cal_fitness, crossover, mutation
from population import run_ga


def run_ga_on_problem(
    processing_number: int,
    problem: Problem,
    pop_size: int,
    max_gen: int,
    crossover_rate: float = 0.9,
    mutation_rate: float = 0.1,
    seed=42,
) -> Dict[str, Any]:
    """
    Run GA algorithm on a single problem.

    Args:
        processing_number: Number of processes for multiprocessing
        problem: Problem instance
        pop_size: Population size
        max_gen: Maximum generations
        crossover_rate: Crossover rate
        mutation_rate: Mutation rate

    Returns:
        Dictionary containing 'time' and 'history' keys
    """
    start_time = time.time()

    # Initialize population
    indi_list = init_population(pop_size, seed, problem)
    # Run GA
    history = run_ga(
        processing_number=processing_number,
        problem=problem,
        pop_size=pop_size,
        indi_list=indi_list,
        max_gen=max_gen,
        crossover_operator=crossover,
        mutation_operator=mutation,
        crossover_rate=crossover_rate,
        mutation_rate=mutation_rate,
        cal_fitness=cal_fitness,
    )

    end_time = time.time()

    return {"time": end_time - start_time, "history": history}


def get_data_files(base_dir: str = "../../data/generated/data") -> Dict[str, list]:
    """
    Get all JSON data files organized by problem size.

    Args:
        base_dir: Base directory containing data files

    Returns:
        Dictionary mapping size folder (e.g., 'N10') to list of file paths
    """
    data_files = {}
    base_path = Path(base_dir).resolve()

    if not base_path.exists():
        print(f"Warning: Data directory {base_dir} does not exist!")
        return data_files

    # Iterate through size directories (N10, N20, N50, etc.)
    for size_dir in sorted(base_path.iterdir()):
        if size_dir.is_dir() and size_dir.name.startswith("N"):
            json_files = list(size_dir.glob("*.json"))
            if json_files:
                data_files[size_dir.name] = sorted(json_files)

    return data_files


def save_result(result: Dict[str, Any], output_path: Path):
    """
    Save result to JSON file.

    Args:
        result: Result dictionary to save
        output_path: Path to output file
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"Saved result to {output_path}")


def main():
    # Configuration
    POP_SIZE = 100
    MAX_GEN = 100
    PROCESSING_NUMBER = 12
    BASE_DATA_DIR = "../../data/generated/data"
    BASE_RESULT_DIR = "./result"
    SIZES = ["N10", "N20", "N50", "N100", "N150"]
    # SIZES = ["N10"]
    SEED = 42
    # Get all data files
    data_files = get_data_files(BASE_DATA_DIR)

    if not data_files:
        print("No data files found!")
        return

    # Filter to only allowed sizes
    data_files = {k: v for k, v in data_files.items() if k in SIZES}

    if not data_files:
        print(f"No data files found for sizes: {SIZES}")
        return

    print(f"Found {len(data_files)} problem size categories")
    for size, files in data_files.items():
        print(f"  {size}: {len(files)} files")

    for size_dir, files in data_files.items():
        print(f"\n{'-' * 80}")
        print(f"GA on {size_dir}")
        print(f"{'-' * 80}\n")

        for data_file in files:
            print(f"Processing: {data_file.name}")

            try:
                # Load problem
                problem = Problem(str(data_file))
                # Run GA
                result = run_ga_on_problem(
                    processing_number=PROCESSING_NUMBER,
                    problem=problem,
                    pop_size=POP_SIZE,
                    max_gen=MAX_GEN,
                    seed=SEED,
                )

                # Save result
                output_path = Path(BASE_RESULT_DIR) / size_dir / data_file.name
                save_result(result, output_path)

                print(f"  Completed in {result['time']:.2f} seconds")

            except Exception as e:
                print(f"  ERROR: {str(e)}")
                import traceback

                traceback.print_exc()
                continue

    print(f"\n{'=' * 80}")
    print("All experiments completed!")
    print(f"{'=' * 80}\n")


if __name__ == "__main__":
    main()
