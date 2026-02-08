from pathlib import Path
from script.script_cal_igd import find_nadir_point_igd, selection_nondominate
import numpy as np
from moo_algorithm.metric import cal_igd


def cal_igd_one_dataset(result_dir, size_dir, instance_file, run_number=None):
    """
    Calculate IGD for all algorithms on one dataset/instance.

    Args:
        result_dir: Base result directory (e.g., "./result/raw/drone")
        size_dir: Size directory (e.g., "N100")
        instance_file: Instance filename (e.g., "S042_N100_RC_R50.json")
        run_number: Optional run number (1-5)

    Returns:
        Tuple of IGD values for all algorithms
    """
    if run_number is not None:
        result_dir = str(Path(result_dir) / str(run_number))

    print(result_dir)
    algorithms = ["MOEAD", "NSGA_II", "NSGA_III", "PFG_MOEA", "AGEA", "CIAGEA"]

    try:
        # 1. Load all data and find global nadir point
        nadir_point, all_data = find_nadir_point_igd(
            result_dir, size_dir, instance_file, algorithms
        )

        # 2. Collect final generation Pareto fronts for each algorithm
        all_pareto_data = {}
        for algo in algorithms:
            if algo in all_data:
                history = all_data[algo]
                final_gen_key = str(max([int(k) for k in history.keys()]))
                pareto_points = np.array(history[final_gen_key])
                all_pareto_data[algo] = pareto_points
            else:
                # If algorithm data not found, use empty array
                all_pareto_data[algo] = np.array([]).reshape(0, 2)

        # 3. Normalize all Pareto fronts by the global nadir point
        normalized_pareto_data = {}
        for algo, pareto_set in all_pareto_data.items():
            if pareto_set.shape[0] > 0:
                normalized_pareto_data[algo] = pareto_set / nadir_point
            else:
                normalized_pareto_data[algo] = pareto_set

        # 4. Combine all normalized solutions and find approximate Pareto front
        valid_solutions = [
            data for data in normalized_pareto_data.values() if data.shape[0] > 0
        ]

        if not valid_solutions:
            print(f"Warning: No valid solutions found for {instance_file}")
            return tuple([0.0] * len(algorithms))

        all_normalized_solutions = np.concatenate(valid_solutions, axis=0)
        approximate_front = selection_nondominate(all_normalized_solutions)

        # 5. Calculate IGD for each algorithm
        results_igd = {}
        for algo in algorithms:
            if normalized_pareto_data[algo].shape[0] > 0 and algo == "MOEAD":
                igd_value = cal_igd(normalized_pareto_data[algo], approximate_front)
                results_igd[algo] = igd_value
            else:
                # No solutions for this algorithm
                results_igd[algo] = float("inf")
        return tuple(results_igd[algo] for algo in algorithms)

    except Exception as e:
        print(f"Error calculating IGD for {instance_file}: {e}")
        import traceback

        traceback.print_exc()
        return tuple([float("inf")] * len(algorithms))


result_dir = "./result/raw/drone"
size_dir = "N100"
instance_files = [
    "S044_N100_R_R50.json",
    "S042_N100_C_R50.json",
    "S043_N100_C_R50.json",
    "S044_N100_C_R50.json",
    "S042_N100_RC_R50.json",
]
run_number = 4
for instance_file in instance_files:
    result = cal_igd_one_dataset(result_dir, size_dir, instance_file, run_number)
    print(instance_file)
    print(result)
