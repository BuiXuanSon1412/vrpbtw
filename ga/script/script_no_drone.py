import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from problem import Problem
from scene.no_drone import init_population
from scene.no_drone import ALGORITHMS
from run import run_algorithm_on_data, get_data_files, save_result

# thay đổi thành phần trong size_dirs để xác định bộ chạy
size_dirs = ["N100", "N200", "N400"]
algorithms = ["MOEAD", "NSGA_II", "NSGA_III", "PFG_MOEA", "AGEA", "CIAGEA"]


def main():
    # Configuration
    POP_SIZE = 100
    MAX_GEN = 200
    PROCESSING_NUMBER = 12
    INITIAL_SEED = 42  # Renamed to clarify it's the starting point
    BASE_DATA_DIR = "../data/generated/data"
    BASE_RESULT_DIR = "./result/no_drone"
    NUM_RUNS = 5  # Number of times to run each instance

    # Get all data files
    data_files = get_data_files(BASE_DATA_DIR)

    if not data_files:
        print("No data files found!")
        return

    # Loop for multiple runs (1 to 5)
    for run_count in range(1, NUM_RUNS + 1):
        # Update seed for this specific run
        current_seed = INITIAL_SEED + (run_count - 1)

        print(f"\n{'#' * 80}")
        print(f"### STARTING BATCH RUN {run_count} (SEED: {current_seed}) ###")
        print(f"{'#' * 80}\n")

        # Original logic nested inside the run loop
        for size_dir, files in data_files.items():
            if size_dir not in size_dirs:
                continue

            for algorithm_name, algorithm_config in ALGORITHMS.items():
                if algorithm_name not in algorithms:
                    continue

                algorithm_runner = algorithm_config["runner"]
                algorithm_params = algorithm_config["params"]

                print(f"\n{'=' * 40}")
                print(
                    f"Run {run_count} | Algorithm: {algorithm_name} | Size: {size_dir}"
                )
                print(f"{'=' * 40}")

                for data_file in files:
                    # Update output path structure: result/{count}/{algorithm}/N{nodes}/{file_name}.json
                    output_path = (
                        Path(BASE_RESULT_DIR)
                        / str(run_count)  # Store in count folder (1, 2, 3, 4, 5)
                        / algorithm_name
                        / size_dir
                        / data_file.name
                    )

                    # Create parent directories if they don't exist
                    output_path.parent.mkdir(parents=True, exist_ok=True)

                    # Check if result already exists to avoid redundant work
                    if output_path.exists():
                        print(f"  Skipping: {data_file.name} (already exists)")
                        continue

                    print(f"  Processing: {data_file.name}")

                    try:
                        # Load problem
                        problem = Problem(str(data_file))

                        # Initialize population with the INCREMENTED seed
                        indi_list = init_population(POP_SIZE, current_seed, problem)

                        # Run algorithm
                        result = run_algorithm_on_data(
                            algorithm_runner=algorithm_runner,
                            algorithm_params=algorithm_params,
                            problem=problem,
                            indi_list=indi_list,
                            pop_size=POP_SIZE,
                            max_gen=MAX_GEN,
                            processing_number=PROCESSING_NUMBER,
                        )

                        # Save result
                        save_result(result, output_path)

                        print(f"    Completed in {result['time']:.2f} seconds")

                    except Exception as e:
                        print(f"    ERROR on {data_file.name}: {str(e)}")
                        import traceback

                        traceback.print_exc()
                        continue

    print(f"\\n{'=' * 80}")
    print("All 5 batch runs completed!")
    print(f"{'=' * 80}\\n")


if __name__ == "__main__":
    main()
