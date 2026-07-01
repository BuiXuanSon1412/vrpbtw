import json
import numpy as np
from pathlib import Path
from moo_algorithm.metric import cal_hv  # Reusing your existing pymoo-based HV function

# --- Configuration ---
RESULT_DIR = "./result"
# Since you are dividing by nadir, the nadir becomes [1, 1].
# We use 1.1 as the reference point to include the boundary solutions.NORMALIZED_REF_POINT = np.array([1.1, 1.1])
NORMALIZED_REF_POINT = np.array([1.1, 1.1])


def get_final_front(json_path):
    """Loads the last generation's objectives from the JSON file."""
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    history = data.get("history", {})
    if not history:
        return None

    # Identify the last generation (usually '99')
    last_gen = str(max([int(k) for k in history.keys()]))
    return np.array(history[last_gen])


files = ["S043_N100_RC_R50.json"]
algorithms = ["NSGA_II", "NSGA_III", "MOEAD", "PFG_MOEA", "AGEA", "IAGEA", "CIAGEA"]


def main():
    result_path = Path(RESULT_DIR)
    if not result_path.exists():
        print(f"Error: {RESULT_DIR} not found.")
        return

    # 1. Group files by instance to find the shared Nadir Point
    instance_groups = {}
    for json_file in result_path.glob("**/*.json"):
        if json_file.name not in files:
            continue

        instance_name = json_file.name
        algorithm = json_file.parts[1]
        if algorithm not in algorithms:
            continue
        if instance_name not in instance_groups:
            instance_groups[instance_name] = []
        instance_groups[instance_name].append((algorithm, json_file))

    print(f"{'Instance':<30} | {'Algorithm':<12} | {'Norm HV (v/Nadir)':<15}")
    print("-" * 70)

    for instance, entries in instance_groups.items():
        all_fronts_data = []

        # Load all algorithm results for this specific problem file
        for algo, path in entries:
            front = get_final_front(path)
            if front is not None:
                all_fronts_data.append((algo, front))

        if not all_fronts_data:
            continue

        # 2. Calculate the Global Nadir for this specific instance
        # This is the worst value seen by ANY algorithm on this problem
        combined_points = np.vstack([data[1] for data in all_fronts_data])
        global_nadir = np.max(combined_points, axis=0)

        # Avoid division by zero
        global_nadir[global_nadir == 0] = 1e-9

        # 3. Calculate HV for each algorithm using your normalization formula
        for algo, front in all_fronts_data:
            # Formula: normalized_objectives = objectives / nadir_point
            normalized_front = front / global_nadir

            # Calculate HV using normalized front and [1.1, 1.1]
            hv_val = cal_hv(normalized_front, NORMALIZED_REF_POINT)

            print(f"{instance:<30} | {algo:<12} | {hv_val:>15.6f}")

        print("\n")


if __name__ == "__main__":
    main()
