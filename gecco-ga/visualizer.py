import os
import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

from moo_algorithm.metric import cal_hv

# --- Configuration ---
RESULT_DIR = "result"
IMAGE_DIR = "img"
# Using your specific reference point for HV calculation
REF_POINT = np.array([100, 700000])


def visualize_instance(json_path):
    # 1. Load Data
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    history = data.get("history", {})
    if not history:
        print(f"No history found in {json_path}")
        return

    # 2. Setup Output Directory
    relative_path = Path(json_path).relative_to(RESULT_DIR)
    instance_name = relative_path.stem
    output_folder = Path(IMAGE_DIR) / relative_path.parent / instance_name
    output_folder.mkdir(parents=True, exist_ok=True)

    # 3. Calculate HV over all generations using YOUR cal_hv function
    gens = sorted([int(g) for g in history.keys()])
    hv_values = []

    for g in gens:
        front = np.array(history[str(g)])
        # Reuse your cal_hv function here
        hv = cal_hv(front, REF_POINT)
        hv_values.append(hv)

    # --- PLOT 1: HV Convergence ---
    # This shows how your cal_hv value grows as the algorithm runs
    plt.figure(figsize=(8, 5))
    plt.plot(gens, hv_values, color="#1f77b4", linewidth=2, label="HV Growth")
    plt.title(f"HV Convergence: {instance_name}")
    plt.xlabel("Generation")
    plt.ylabel("Hypervolume (HV)")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.savefig(output_folder / "convergence_hv.png")
    plt.close()

    # --- PLOT 2: Pareto Front Scatter ---
    plt.figure(figsize=(8, 6))

    # Final Front (Last Gen)
    # Using str(max(gens)) to ensure we get "99" or whatever the last index is
    final_gen_key = str(max(gens))
    final_front = np.array(history[final_gen_key])

    # Plotting Objective 1 (x) and Objective 2 (y)
    plt.scatter(
        final_front[:, 0],
        final_front[:, 1],
        c="#d62728",
        marker="o",
        s=30,
        edgecolors="k",
        label=f"Final Front (Gen {final_gen_key})",
    )

    plt.title(f"Final Pareto Front: {instance_name}")
    plt.xlabel("Objective 1")
    plt.ylabel("Objective 2")
    plt.legend()
    plt.grid(True, linestyle=":", alpha=0.5)
    plt.savefig(output_folder / "pareto_front_final.png")
    plt.close()

    print(f"Success: {instance_name} visualized using your cal_hv logic.")


def main():
    result_path = Path(RESULT_DIR)
    json_files = list(result_path.glob("**/*.json"))

    for json_file in json_files:
        try:
            visualize_instance(str(json_file))
        except Exception as e:
            print(f"Error processing {json_file}: {e}")


files = ["S043_N100_RC_R50.json"]
algorithms = ["NSGA_II", "NSGA_III", "MOEAD", "PFG_MOEA", "AGEA", "IAGEA", "CIAGEA"]

RESULT_DIR = "./result/"
if __name__ == "__main__":
    # main()
    for algo in algorithms:
        for file in files:
            size_dir = file.split("_")[1]
            json_file = Path(RESULT_DIR) / algo / size_dir / file
            try:
                visualize_instance(str(json_file))
            except Exception as e:
                print(f"Error processing {json_file}: {e}")
