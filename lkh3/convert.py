import numpy as np
import json
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument(
    "--filename",
    type=str,
    required=True,
    help="Relative JSON filename, e.g. S042_N5_C_R50.json",
)
parser.add_argument(
    "--subfolder",
    type=str,
    help="Subfolder name (optional)",
)

args = parser.parse_args()

DATA_ROOT = "../data/generated/data/"
LKH_ROOT = "./lkh_files"

filename = args.filename
subfolder = args.subfolder

# Determine data path
if subfolder:
    data_path = os.path.join(DATA_ROOT, subfolder, filename)
else:
    parts = filename.split("_")
    n_folder = parts[1]
    data_path = os.path.join(DATA_ROOT, n_folder, filename)

# Load JSON data
with open(data_path, "r") as f:
    data = json.load(f)

config = data["Config"]
nodes = data["Nodes"]

# Extract configuration
num_customers = config["General"]["NUM_CUSTOMERS"]
num_nodes = config["General"]["NUM_NODES"]
MAX_COORD = config["General"]["MAX_COORD_KM"]
T_max = config["General"]["T_MAX_SYSTEM_H"]

T_tilde_max = config["Vehicles"]["DRONE_DURATION_H"]
V_TRUCK = config["Vehicles"]["V_TRUCK_KM_H"]
V_DRONE = config["Vehicles"]["V_DRONE_KM_H"]
Q = config["Vehicles"]["CAPACITY_TRUCK"]
Q_tilde = config["Vehicles"]["CAPACITY_DRONE"]
num_vehicles = config["Vehicles"]["NUM_TRUCKS"]
num_drones = config["Vehicles"]["NUM_DRONES"]

tau_l = config["Vehicles"]["DRONE_TAKEOFF_MIN"] / 60.0
tau_r = config["Vehicles"]["DRONE_LANDING_MIN"] / 60.0
service_time = config["Vehicles"]["SERVICE_TIME_MIN"] / 60.0

depot_info = config["Depot"]
depot_idx = depot_info["id"]
depot_coord = np.array(depot_info["coord"])
depot_tw = depot_info["time_window_h"]

# Build node lists
coords = [depot_coord]
demands = {depot_idx: 0}
time_windows = {depot_idx: depot_tw}
service_times = {depot_idx: 0}

linehaul_indices = []
backhaul_indices = []

for node in nodes:
    node_id = node["id"]
    coords.append(np.array(node["coord"]))
    demands[node_id] = node["demand"]
    time_windows[node_id] = node["tw_h"]
    service_times[node_id] = service_time
    
    # Backhaul classification based on demand sign
    if node["demand"] > 0:  # Linehaul (pickup/delivery)
        linehaul_indices.append(node_id)
    elif node["demand"] < 0:  # Backhaul (return/collection)
        backhaul_indices.append(node_id)

coords = np.array(coords)

# Add end depot
n_nodes = len(coords)
end_depot_idx = n_nodes
coords = np.vstack([coords, depot_coord])
demands[end_depot_idx] = 0
time_windows[end_depot_idx] = depot_tw
service_times[end_depot_idx] = 0

n_nodes = len(coords)

# Calculate distance matrices
d = np.zeros((n_nodes, n_nodes))
d_tilde = np.zeros((n_nodes, n_nodes))

for i in range(n_nodes):
    for j in range(n_nodes):
        if i != j:
            d[i, j] = np.linalg.norm(coords[i] - coords[j], ord=1)  # Manhattan for truck
            d_tilde[i, j] = np.linalg.norm(coords[i] - coords[j], ord=2)  # Euclidean for drone

# Scale distances to integer (LKH works better with integers)
SCALE_FACTOR = 1000
d_int = (d * SCALE_FACTOR).astype(int)
d_tilde_int = (d_tilde * SCALE_FACTOR).astype(int)

# Time matrices
t = d / V_TRUCK
t_tilde = d_tilde / V_DRONE

# Prepare LKH problem file
os.makedirs(LKH_ROOT, exist_ok=True)

base_name = filename.replace(".json", "")
problem_file = os.path.join(LKH_ROOT, f"{base_name}.vrp")
param_file = os.path.join(LKH_ROOT, f"{base_name}.par")
tour_file = os.path.join(LKH_ROOT, f"{base_name}.tour")

# Write TSPLIB format VRP file with time windows and backhaul
with open(problem_file, "w") as f:
    f.write(f"NAME : {base_name}\n")
    f.write(f"COMMENT : CTDVRPTWB instance\n")
    f.write(f"TYPE : CVRP\n")
    f.write(f"DIMENSION : {n_nodes}\n")
    f.write(f"EDGE_WEIGHT_TYPE : EXPLICIT\n")
    f.write(f"EDGE_WEIGHT_FORMAT : FULL_MATRIX\n")
    f.write(f"CAPACITY : {Q}\n")
    f.write(f"VEHICLES : {num_vehicles}\n")
    
    # Edge weight matrix (truck distances)
    f.write("EDGE_WEIGHT_SECTION\n")
    for i in range(n_nodes):
        for j in range(n_nodes):
            f.write(f"{d_int[i, j]} ")
        f.write("\n")
    
    # Demand section
    f.write("DEMAND_SECTION\n")
    for i in range(n_nodes):
        f.write(f"{i+1} {int(demands.get(i, 0))}\n")
    
    # Depot section
    f.write("DEPOT_SECTION\n")
    f.write("1\n")
    f.write("-1\n")
    
    # Time window section (scaled to integer minutes)
    f.write("TIME_WINDOW_SECTION\n")
    for i in range(n_nodes):
        tw = time_windows.get(i, [0, T_max])
        early = int(tw[0] * 60)  # Convert hours to minutes
        late = int(tw[1] * 60)
        f.write(f"{i+1} {early} {late}\n")
    
    # Service time section (in minutes)
    f.write("SERVICE_TIME_SECTION\n")
    for i in range(n_nodes):
        st = int(service_times.get(i, 0) * 60)
        f.write(f"{i+1} {st}\n")
    
    # Backhaul section - nodes with q < 0
    if backhaul_indices:
        f.write("BACKHAUL_SECTION\n")
        for node_id in backhaul_indices:
            f.write(f"{node_id+1}\n")
    
    f.write("EOF\n")

# Write LKH parameter file
with open(param_file, "w") as f:
    f.write(f"PROBLEM_FILE = {problem_file}\n")
    f.write(f"OUTPUT_TOUR_FILE = {tour_file}\n")
    f.write(f"RUNS = 10\n")
    f.write(f"TIME_LIMIT = 3600\n")  # 1 hour time limit
    f.write(f"SEED = 1234\n")
    f.write(f"TRACE_LEVEL = 1\n")
    f.write(f"MAX_TRIALS = 1000\n")
    f.write(f"POPULATION_SIZE = 50\n")
    
    # LKH-3 specific parameters for VRPTW
    f.write(f"MTSP_OBJECTIVE = MINSUM\n")
    f.write(f"INITIAL_PERIOD = 1000\n")
    f.write(f"MAX_SWAPS = 1000\n")

# Save metadata for later processing
metadata = {
    "filename": filename,
    "n_nodes": n_nodes,
    "num_customers": num_customers,
    "num_vehicles": num_vehicles,
    "num_drones": num_drones,
    "depot_idx": depot_idx,
    "end_depot_idx": end_depot_idx,
    "linehaul_indices": linehaul_indices,
    "backhaul_indices": backhaul_indices,
    "capacity_truck": Q,
    "capacity_drone": Q_tilde,
    "drone_duration": T_tilde_max,
    "tau_l": tau_l,
    "tau_r": tau_r,
    "service_time": service_time,
    "V_TRUCK": V_TRUCK,
    "V_DRONE": V_DRONE,
    "scale_factor": SCALE_FACTOR,
    "coords": coords.tolist(),
    "demands": demands,
    "time_windows": time_windows,
    "service_times": service_times,
    "distance_matrix": d.tolist(),
    "drone_distance_matrix": d_tilde.tolist(),
}

metadata_file = os.path.join(LKH_ROOT, f"{base_name}.meta.json")
with open(metadata_file, "w") as f:
    json.dump(metadata, f, indent=2)

print(f" LKH-3 problem file created: {problem_file}")
print(f" LKH-3 parameter file created: {param_file}")
print(f" Metadata file created: {metadata_file}")
print(f"\nTo solve, run: ./LKH-3.0.13/LKH {param_file}")