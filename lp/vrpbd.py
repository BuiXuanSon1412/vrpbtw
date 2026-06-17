from ortools.linear_solver import pywraplp
import numpy as np
import json
import argparse
import sys
import os
import time

parser = argparse.ArgumentParser()
parser.add_argument(
    "--filename",
    type=str,
    help="Relative JSON filename, e.g. S042_N5_C_3G_R50.json",
)
parser.add_argument(
    "--subfolder",
    type=str,
    help="Relative JSON filename, e.g. S042_N5_C_3G_R50.json",
)

args = parser.parse_args()

DATA_ROOT = "../data/generated/data"
RESULT_ROOT = "./result/"

filename = args.filename
subfolder = args.subfolder

files = []
if filename and subfolder:
    print("Either one: filename OR subfolder")
    sys.exit(1)

if filename:
    files.append(filename)
else:
    subfolder_path = os.path.join(DATA_ROOT, subfolder)
    files = [
        file
        for file in os.listdir(subfolder_path)
        if os.path.isfile((os.path.join(subfolder_path, file)))
    ]


def run(filename):
    parts = filename.split("_")
    subfolder = parts[1]
    data_path = os.path.join(DATA_ROOT, subfolder, filename)

    n_folder = parts[1]  # "N10"

    output_dir = os.path.join(RESULT_ROOT, "f", n_folder)
    os.makedirs(output_dir, exist_ok=True)

    output_path = os.path.join(output_dir, filename)

    with open(data_path, "r") as f:
        data = json.load(f)

    config = data["Config"]
    nodes = data["Nodes"]

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

    c = config["Vehicles"]["TRUCK_COST_UNIT"]
    c_tilde = config["Vehicles"]["DRONE_COST_UNIT"]
    c_b = config["Vehicles"]["FLEET_BASIS_COST"]

    depot_info = config["Depot"]
    depot_idx = depot_info["id"]
    depot_coord = np.array(depot_info["coord"])
    depot_tw = depot_info["time_window_h"]

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

        if node["type"] == "LINEHAUL":
            linehaul_indices.append(node_id)
        else:
            backhaul_indices.append(node_id)

    coords = np.array(coords)

    n_nodes = len(coords)
    end_depot_idx = n_nodes
    coords = np.vstack([coords, depot_coord])
    demands[end_depot_idx] = 0
    time_windows[end_depot_idx] = depot_tw
    service_times[end_depot_idx] = 0

    n_nodes = len(coords)

    L = linehaul_indices
    B = backhaul_indices
    C = L + B
    N = [depot_idx] + C
    N_end = C + [end_depot_idx]
    N_all = [depot_idx] + C + [end_depot_idx]

    d = np.zeros((n_nodes, n_nodes))
    d_tilde = np.zeros((n_nodes, n_nodes))

    for i in range(n_nodes):
        for j in range(n_nodes):
            if i != j:
                d[i, j] = np.linalg.norm(coords[i] - coords[j], ord=1)
                d_tilde[i, j] = np.linalg.norm(coords[i] - coords[j], ord=2)

    t = d / V_TRUCK
    t_tilde = d_tilde / V_DRONE

    q = demands
    t_start = {i: time_windows[i][0] for i in range(n_nodes)}
    t_end = {i: time_windows[i][1] for i in range(n_nodes)}
    s = service_times

    t_start[end_depot_idx] = time_windows[end_depot_idx][0]
    t_end[end_depot_idx] = time_windows[end_depot_idx][1]
    s[end_depot_idx] = 0

    K = list(range(1, num_vehicles + 1))
    num_drone_routes = num_drones
    R = list(range(1, num_drone_routes + 1))

    M = 10000.0
    M_Q = Q + 1
    M_Q_tilde = Q_tilde + 1
    M_node = end_depot_idx + 1
    M_edge = M_node * M_node
    M_T = T_max + 1

    w1 = 1.0
    w2 = 0.0

    # Create OR-Tools solver
    solver = pywraplp.Solver.CreateSolver("SCIP")
    if not solver:
        print("Could not create solver SCIP")
        return

    # Decision variables
    x = {}
    for k in K:
        for i in N_all:
            x[k, i] = solver.BoolVar(f"x_{k}_{i}")

    y = {}
    for k in K:
        for i in N_all:
            for j in N_all:
                if i != j:
                    y[k, i, j] = solver.BoolVar(f"y_{k}_{i}_{j}")

    z = {}
    for k in K:
        for i in N_all:
            z[k, i] = solver.IntVar(0, solver.infinity(), f"z_{k}_{i}")

    p = {}
    for k in K:
        for i in N_all:
            p[k, i] = solver.NumVar(0, solver.infinity(), f"p_{k}_{i}")

    a = {}
    for k in K:
        for i in N_all:
            a[k, i] = solver.NumVar(0, solver.infinity(), f"a_{k}_{i}")

    b = {}
    for k in K:
        for i in N_all:
            b[k, i] = solver.NumVar(0, solver.infinity(), f"b_{k}_{i}")

    x_tilde = {}
    for k in K:
        for r in R:
            for i in N_all:
                x_tilde[k, r, i] = solver.BoolVar(f"x_tilde_{k}_{r}_{i}")

    y_tilde = {}
    for k in K:
        for r in R:
            for i in N_all:
                for j in N_all:
                    if i != j:
                        y_tilde[k, r, i, j] = solver.BoolVar(f"y_tilde_{k}_{r}_{i}_{j}")

    z_tilde = {}
    for k in K:
        for r in R:
            for i in N_all:
                z_tilde[k, r, i] = solver.IntVar(
                    0, solver.infinity(), f"z_tilde_{k}_{r}_{i}"
                )

    lambda_var = {}
    for k in K:
        for r in R:
            for i in N_all:
                lambda_var[k, r, i] = solver.BoolVar(f"lambda_{k}_{r}_{i}")

    varrho = {}
    for k in K:
        for r in R:
            for i in N_all:
                varrho[k, r, i] = solver.BoolVar(f"varrho_{k}_{r}_{i}")

    p_tilde = {}
    for k in K:
        for r in R:
            for i in N_all:
                p_tilde[k, r, i] = solver.NumVar(
                    0, solver.infinity(), f"p_tilde_{k}_{r}_{i}"
                )

    a_tilde = {}
    for k in K:
        for i in N_all:
            a_tilde[k, i] = solver.NumVar(0, solver.infinity(), f"a_tilde_{k}_{i}")

    b_tilde = {}
    for k in K:
        for i in N_all:
            b_tilde[k, i] = solver.NumVar(0, solver.infinity(), f"b_tilde_{k}_{i}")

    h = {}
    for k in K:
        for r in R:
            for i in N_all:
                h[k, r, i] = solver.NumVar(0, solver.infinity(), f"h_{k}_{r}_{i}")

    Z_lambda = {}
    for k in K:
        for r in R:
            for i in N_all:
                Z_lambda[k, r, i] = solver.NumVar(
                    0, solver.infinity(), f"Z_lambda_{k}_{r}_{i}"
                )

    Z_varrho = {}
    for k in K:
        for r in R:
            for i in N_end:
                Z_varrho[k, r, i] = solver.NumVar(
                    0, solver.infinity(), f"Z_varrho_{k}_{r}_{i}"
                )

    xi = {}
    for i in N_all:
        xi[i] = solver.NumVar(0, solver.infinity(), f"xi_{i}")

    # tardiness = solver.NumVar(0, solver.infinity(), 'tardiness')
    # f1
    sr_expr = solver.Sum([x[k, i] for k in K for i in C]) + solver.Sum(
        [x_tilde[k, r, i] for k in K for r in R for i in C]
    )

    # Objective function
    # f2
    cost_expr = (
        solver.Sum(
            [
                y[k, i, j] * c * d[i][j]
                for k in K
                for i in N_all
                for j in N_end
                if i != j
            ]
        )
        + solver.Sum(
            [
                y_tilde[k, r, i, j] * c_tilde * d_tilde[i][j]
                for k in K
                for r in R
                for i in N_all
                for j in N_end
                if i != j
            ]
        )
        + solver.Sum([y[k, 0, j] * c_b for k in K for j in C])
    )

    # maximize: f = w * f1 - f2
    weight = max(c_tilde, c) * len(C) * 2 * MAX_COORD + c_b * num_vehicles
    solver.Minimize(cost_expr - weight * sr_expr)

    # solver.Minimize(w1 * cost_expr + w2 * tardiness)

    # Constraints
    # Tardiness constraints
    # for i in C:
    #     solver.Add(tardiness >= xi[i] + s[i] - t_end[i])

    # Each customer served once (constraint 3)
    for i in C:
        solver.Add(
            solver.Sum([x[k, i] for k in K])
            + solver.Sum([x_tilde[k, r, i] for k in K for r in R])
            <= 1
        )

    # If no launching node, no edges traversed
    for k in K:
        for r in R:
            solver.Add(
                solver.Sum([y_tilde[k, r, i, j] for i in N for j in N_end if i != j])
                <= M * solver.Sum([lambda_var[k, r, i] for i in N])
            )

    # No launching and landing at same node
    for k in K:
        for r in R:
            for i in C:
                solver.Add(lambda_var[k, r, i] + varrho[k, r, i] <= 1)

    for k in K:
        for r in R:
            for i in N:
                for j in N_end:
                    if i != j:
                        solver.Add(
                            y_tilde[k, r, i, j]
                            <= solver.Sum([x_tilde[k, r, h] for h in C])
                        )

    # Constraints 4-5
    for k in K:
        solver.Add(solver.Sum([y[k, 0, j] for j in C]) <= 1)
        solver.Add(solver.Sum([y[k, i, end_depot_idx] for i in C]) <= 1)

    for k in K:
        for i in C:
            solver.Add(solver.Sum([y[k, j, i] for j in N if j != i]) == x[k, i])
            solver.Add(solver.Sum([y[k, i, j] for j in N_end if j != i]) == x[k, i])

    # Constraints 6-7
    for k in K:
        for r in R:
            solver.Add(solver.Sum([lambda_var[k, r, i] for i in N]) <= 1)
            solver.Add(solver.Sum([varrho[k, r, j] for j in N_end]) <= 1)
            solver.Add(
                solver.Sum([lambda_var[k, r, i] for i in N])
                == solver.Sum([varrho[k, r, j] for j in N_end])
            )

    # Constraint 8
    for k in K:
        for r in R:
            for i in C:
                solver.Add(lambda_var[k, r, i] <= x[k, i])
                solver.Add(varrho[k, r, i] <= x[k, i])

    # Constraints 9-13
    for k in K:
        for r in R:
            for i in C:
                in_flow = solver.Sum([y_tilde[k, r, j, i] for j in N if i != j])
                solver.Add(
                    in_flow - x_tilde[k, r, i]
                    <= M_node * (lambda_var[k, r, i] + varrho[k, r, i])
                )
                solver.Add(
                    in_flow - x_tilde[k, r, i]
                    >= -M_node * (lambda_var[k, r, i] + varrho[k, r, i])
                )

    for k in K:
        for r in R:
            for i in N:
                out_flow = solver.Sum([y_tilde[k, r, i, j] for j in N_end if i != j])
                solver.Add(
                    out_flow - x_tilde[k, r, i]
                    <= M_node * (lambda_var[k, r, i] + varrho[k, r, i])
                )
                solver.Add(
                    out_flow - x_tilde[k, r, i]
                    >= -M_node * (lambda_var[k, r, i] + varrho[k, r, i])
                )

            solver.Add(varrho[k, r, 0] == 0)
            solver.Add(lambda_var[k, r, end_depot_idx] == 0)

    # Constraints 14-15
    for k in K:
        for r in R:
            for i in N:
                solver.Add(
                    lambda_var[k, r, i]
                    <= solver.Sum([y_tilde[k, r, i, j] for j in N_end if j != i])
                )
                solver.Add(
                    lambda_var[k, r, i]
                    >= x[k, i]
                    + solver.Sum([y_tilde[k, r, i, j] for j in N_end if j != i])
                    - 1
                )

    # Constraints 16-17
    for k in K:
        for r in R:
            for i in N_end:
                solver.Add(
                    varrho[k, r, i]
                    <= solver.Sum([y_tilde[k, r, j, i] for j in N if j != i])
                )
                solver.Add(
                    varrho[k, r, i]
                    >= x[k, i]
                    + solver.Sum([y_tilde[k, r, j, i] for j in N if j != i])
                    - 1
                )

    # Constraints 18-21
    for k in K:
        for r in R:
            for i in N:
                for j in N_end:
                    if i != j:
                        solver.Add(z[k, i] - z[k, j] + 1 <= M * (1 - y[k, i, j]))
                        solver.Add(z[k, i] - z[k, j] + 1 >= -M * (1 - y[k, i, j]))
                        solver.Add(
                            z_tilde[k, r, i] - z_tilde[k, r, j] + 1
                            <= M * (1 - y_tilde[k, r, i, j])
                        )
                        solver.Add(
                            z_tilde[k, r, i] - z_tilde[k, r, j] + 1
                            >= -M * (1 - y_tilde[k, r, i, j])
                        )

    for k in K:
        for r in R:
            for i in N:
                for j in N_end:
                    if i != j:
                        solver.Add(
                            z[k, j]
                            >= z[k, i]
                            + 1
                            - M * (2 - lambda_var[k, r, i] - varrho[k, r, j])
                        )

    # Constraints 22-23
    for k in K:
        solver.Add(
            p[k, 0]
            == solver.Sum([q[u] * x[k, u] for u in L])
            + solver.Sum([q[u] * x_tilde[k, r, u] for u in L for r in R])
            - solver.Sum([Z_lambda[k, r, 0] for r in R])
        )
        solver.Add(
            p[k, end_depot_idx]
            == -solver.Sum([q[u] * x[k, u] for u in B])
            - solver.Sum([q[u] * x_tilde[k, r, u] for u in B for r in R])
        )

    # Constraints 24-25
    for k in K:
        for i in N:
            for j in N_end:
                if i == 0 and j == end_depot_idx:
                    continue
                if i != j:
                    load_change = (
                        -q[j]
                        - solver.Sum([Z_lambda[k, r, j] for r in R])
                        + solver.Sum([Z_varrho[k, r, j] for r in R])
                    )
                    solver.Add(
                        p[k, j] <= p[k, i] + load_change + M_Q * (1 - y[k, i, j])
                    )
                    solver.Add(
                        p[k, j] >= p[k, i] + load_change - M_Q * (1 - y[k, i, j])
                    )

    for k in K:
        for i in N_all:
            solver.Add(p[k, i] <= Q)
            solver.Add(p[k, i] >= 0)

    # Constraints 26-35
    for k in K:
        for r in R:
            for j in N:
                solver.Add(
                    Z_lambda[k, r, j]
                    <= p_tilde[k, r, j] + Q_tilde * (1 - lambda_var[k, r, j])
                )
                solver.Add(Z_lambda[k, r, j] <= Q_tilde * lambda_var[k, r, j])
                solver.Add(
                    Z_lambda[k, r, j]
                    >= p_tilde[k, r, j] - Q_tilde * (1 - lambda_var[k, r, j])
                )

    for k in K:
        for r in R:
            for j in N_end:
                solver.Add(
                    Z_varrho[k, r, j]
                    <= p_tilde[k, r, j] + Q_tilde * (1 - varrho[k, r, j])
                )
                solver.Add(Z_varrho[k, r, j] <= Q_tilde * varrho[k, r, j])
                solver.Add(
                    Z_varrho[k, r, j]
                    >= p_tilde[k, r, j] - Q_tilde * (1 - varrho[k, r, j])
                )

    # Constraints 36-37
    for k in K:
        for r in R:
            drone_pickup = solver.Sum([q[u] * x_tilde[k, r, u] for u in L])
            for i in N:
                solver.Add(
                    p_tilde[k, r, i]
                    <= drone_pickup + M_Q_tilde * (1 - lambda_var[k, r, i])
                )
                solver.Add(
                    p_tilde[k, r, i]
                    >= drone_pickup - M_Q_tilde * (1 - lambda_var[k, r, i])
                )

    # Constraints 38-39
    for k in K:
        for r in R:
            drone_delivery = solver.Sum([q[u] * x_tilde[k, r, u] for u in B])
            for j in N_end:
                solver.Add(
                    p_tilde[k, r, j]
                    >= drone_delivery - M_Q_tilde * (1 - varrho[k, r, j])
                )
                solver.Add(
                    p_tilde[k, r, j]
                    <= drone_delivery + M_Q_tilde * (1 - varrho[k, r, j])
                )

    # Constraints 40-41
    for k in K:
        for r in R:
            for i in N:
                for j in N_end:
                    if i != j:
                        solver.Add(
                            p_tilde[k, r, j]
                            <= p_tilde[k, r, i]
                            - q[j] * x_tilde[k, r, j]
                            + M_Q_tilde * (1 - y_tilde[k, r, i, j])
                        )
                        solver.Add(
                            p_tilde[k, r, j]
                            >= p_tilde[k, r, i]
                            - q[j] * x_tilde[k, r, j]
                            - M_Q_tilde * (1 - y_tilde[k, r, i, j])
                        )

    # Constraint 42
    for k in K:
        for r in R:
            for i in N + [end_depot_idx]:
                solver.Add(p_tilde[k, r, i] <= Q_tilde)
                solver.Add(p_tilde[k, r, i] >= 0)

    # Constraints 43-45
    # for i in C:
    #     solver.Add(xi[i] >= t_start[i])

    for i in C:
        sigma_i = solver.Sum([x[k, i] for k in K]) + solver.Sum(
            [x_tilde[k, r, i] for k in K for r in R]
        )
        solver.Add(xi[i] >= t_start[i] - M_T * (1 - sigma_i))
        solver.Add(xi[i] <= t_end[i] - s[i] + M_T * (1 - sigma_i))

    for k in K:
        for i in C:
            solver.Add(xi[i] >= a[k, i] - M_T * (1 - x[k, i]))
            solver.Add(xi[i] <= b[k, i] - s[i] + M_T * (1 - x[k, i]))

    for k in K:
        for r in R:
            for i in C:
                solver.Add(
                    xi[i] >= a_tilde[k, i] + tau_r - M_T * (1 - x_tilde[k, r, i])
                )
                solver.Add(xi[i] <= b_tilde[k, i] - s[i] + M_T * (1 - x_tilde[k, r, i]))

    # Constraints 46-47
    for k in K:
        for i in N:
            for j in N_end:
                if i != j:
                    solver.Add(a[k, j] >= b[k, i] + t[i][j] - M_T * (1 - y[k, i, j]))
                    solver.Add(a[k, j] <= b[k, i] + t[i][j] + M_T * (1 - y[k, i, j]))

    # Constraints 48-49
    for k in K:
        for r in R:
            for i in N:
                for j in N_end:
                    if i != j:
                        solver.Add(
                            a_tilde[k, j]
                            >= b_tilde[k, i]
                            + tau_l
                            + t_tilde[i][j]
                            - M_T * (1 - y_tilde[k, r, i, j])
                        )
                        solver.Add(
                            a_tilde[k, j]
                            <= b_tilde[k, i]
                            + tau_l
                            + t_tilde[i][j]
                            + M_T * (1 - y_tilde[k, r, i, j])
                        )

    # Constraint 51
    for k in K:
        for r in R:
            for i in N:
                solver.Add(b_tilde[k, i] <= b[k, i] + M_T * (1 - lambda_var[k, r, i]))
                solver.Add(b_tilde[k, i] >= a[k, i] - M_T * (1 - lambda_var[k, r, i]))

    # Constraints 54-55
    for k in K:
        for r in R:
            for j in N_end:
                solver.Add(
                    a_tilde[k, j] + h[k, r, j] + tau_r
                    >= a[k, j] - M_T * (1 - varrho[k, r, j])
                )
                solver.Add(
                    a_tilde[k, j] + h[k, r, j] + tau_r
                    <= b[k, j] + M_T * (1 - varrho[k, r, j])
                )

    # Constraints 56-58
    for k in K:
        for r in R:
            for i in C:
                solver.Add(
                    h[k, r, i]
                    >= xi[i] - a_tilde[k, i] - tau_r - M_T * (1 - x_tilde[k, r, i])
                )
                solver.Add(
                    h[k, r, i]
                    <= xi[i] - a_tilde[k, i] - tau_r + M_T * (1 - x_tilde[k, r, i])
                )
                solver.Add(h[k, r, i] >= 0)

    # Constraint 59
    for k in K:
        for r in R:
            flight_time = solver.Sum(
                [
                    y_tilde[k, r, i, j] * t_tilde[i][j]
                    for i in N
                    for j in N_end
                    if i != j
                ]
            )
            launch_time = solver.Sum([lambda_var[k, r, i] * tau_l for i in N])
            land_time = solver.Sum([varrho[k, r, j] * tau_r for j in N_end])
            service_time_total = solver.Sum(
                [x_tilde[k, r, i] * (tau_l + tau_r + s[i]) for i in C]
            )
            wait_time = solver.Sum([h[k, r, i] for i in N])

            total_time = (
                flight_time + launch_time + land_time + service_time_total + wait_time
            )
            solver.Add(total_time <= T_tilde_max)

    # Constraint system time limit
    for k in K:
        solver.Add(a[k, end_depot_idx] <= T_max)

    # Constraints 76-77
    for k in K:
        for r in R[:-1]:
            for i in N_end:
                for j in N:
                    solver.Add(
                        z[k, j]
                        >= z[k, i] - M * (2 - varrho[k, r, i] - lambda_var[k, r + 1, j])
                    )

            for i in N_end:
                for j in N:
                    solver.Add(
                        a_tilde[k, j]
                        >= b_tilde[k, i]
                        + tau_l
                        - M_T * (2 - varrho[k, r, i] - lambda_var[k, r + 1, j])
                    )

    # Constraint 78
    for k in K:
        for r in R[:-1]:
            solver.Add(
                solver.Sum([lambda_var[k, r + 1, i] for i in N])
                <= solver.Sum([lambda_var[k, r, i] for i in N])
            )

    # Constraint 79
    for k in K:
        for r in R:
            solver.Add(
                solver.Sum([lambda_var[k, r, i] for i in N])
                <= solver.Sum([y[k, 0, j] for j in C])
            )

    # Constraint 80
    for k in K[:-1]:
        solver.Add(
            solver.Sum([y[k, 0, j] for j in C])
            >= solver.Sum([y[k + 1, 0, j] for j in C])
        )

    print(
        f"Model created with {solver.NumVariables()} variables and {solver.NumConstraints()} constraints"
    )
    print("\nSolving the model")

    # Set solver parameters
    solver.SetTimeLimit(36000 * 1000)  # 36000 seconds in milliseconds

    start_time = time.time()
    status = solver.Solve()
    running_time = time.time() - start_time

    if status == pywraplp.Solver.OPTIMAL or status == pywraplp.Solver.FEASIBLE:
        status_str = "Optimal" if status == pywraplp.Solver.OPTIMAL else "Feasible"

        result_data = {
            # "weights": {
            #     "cost": w1,
            #     "tardiness": w2,
            # },
            "status": status_str,
            "objective": -solver.Objective().Value(),
            "f2_cost": cost_expr.solution_value(),
            "f1_num_served": int(round(sr_expr.solution_value())),
            "f1_service_rate": sr_expr.solution_value() / len(C),
            "time": running_time,
            "routes": [],
        }

        print("[TEST]: x")
        for k, v in x.items():
            print(f"\t {k}: {v.solution_value()}")
        print("[TEST]: x_tilde")
        for k, v in x_tilde.items():
            print(f"\t {k}: {v.solution_value()}")
        print("[TEST]: y")
        for k, v in y.items():
            print(f"\t {k}: {v.solution_value()}")
        print("[TEST]: y_tilde")
        for k, v in y_tilde.items():
            print(f"\t {k}: {v.solution_value()}")

        print("[TEST]: lambda")
        for k, v in lambda_var.items():
            print(f"\t {k}: {v.solution_value()}")

        print("[TEST]: varrho")
        for k, v in varrho.items():
            print(f"\t {k}: {v.solution_value()}")
        print("[TEST]: p", Q)
        for k, v in p.items():
            print(f"\t {k}: {v.solution_value()}")
        print("[TEST]: p_tilde", Q_tilde)
        for k, v in p_tilde.items():
            print(f"\t {k}: {v.solution_value()}")

        print(
            f"[TEST]: f1 (served) = {int(round(sr_expr.solution_value()))} / {len(C)}"
        )
        print(f"[TEST]: f2 (cost)   = {cost_expr.solution_value():.4f}")
        print(f"[TEST]: f  (obj)    = {solver.Objective().Value():.4f}")

        for k in K:
            route = [0]
            current = 0
            visited = {0}

            # 1. Reconstruct Truck Route
            max_iter = len(N_all) * 2
            for _ in range(max_iter):
                found = False
                for j in N_end:  # Use N_all to ensure end_depot_idx is reachable
                    if (
                        j not in visited
                        and y[k, current, j].solution_value()
                        and y[k, current, j].solution_value() > 0.5
                    ):
                        route.append(j)
                        visited.add(j)
                        current = j
                        found = True
                        break
                if not found or current == end_depot_idx:
                    break

            # 2. Process Route Data (Only if truck left the depot)
            if len(route) > 1:
                route_info = {
                    "id": int(k),
                    "route": route,
                    "arrival": [
                        a[k, i].solution_value() if idx > 0 else None
                        for idx, i in enumerate(route)
                    ],
                    "departure": [
                        b[k, i].solution_value() if idx < len(route) - 1 else None
                        for idx, i in enumerate(route)
                    ],
                    "service": [
                        xi[i].solution_value()
                        if idx < len(route) - 1 and idx > 0
                        else None
                        for idx, i in enumerate(route)
                    ],
                    "trips": [],
                }

                # 3. Process Drone Trips (Sorties)
                for r in R:
                    # Check if this specific trip r for truck k was used
                    # We identify the launch node first
                    launch_nodes = [
                        i
                        for i in N
                        if lambda_var[k, r, i].solution_value()
                        and lambda_var[k, r, i].solution_value() > 0.5
                    ]

                    if launch_nodes:
                        launch_node = launch_nodes[0]
                        drone_route = [launch_node]
                        drone_current = launch_node
                        drone_visited = {launch_node}

                        # Reconstruct the drone path for this specific sortie
                        for _ in range(len(N_all)):
                            found_next = False
                            for j in N_all:
                                if (
                                    j not in drone_visited
                                    and y_tilde[k, r, drone_current, j].solution_value()
                                    and y_tilde[k, r, drone_current, j].solution_value()
                                    > 0.5
                                ):
                                    drone_route.append(j)
                                    drone_visited.add(j)
                                    drone_current = j
                                    found_next = True
                                    break
                            if not found_next:
                                break

                        if len(drone_route) < 3:
                            break
                        # Retrieve arrival/departure for drone trip
                        trip_info = {
                            "id": int(r),
                            "route": drone_route,
                            "arrival": [
                                a_tilde[k, i].solution_value() if idx > 0 else None
                                for idx, i in enumerate(drone_route)
                            ],
                            "departure": [
                                b_tilde[k, i].solution_value()
                                if idx < len(drone_route) - 1
                                else None
                                for idx, i in enumerate(drone_route)
                            ],
                            "service": [
                                xi[i].solution_value()
                                if idx < len(drone_route) - 1 and idx > 0
                                else None
                                for idx, i in enumerate(drone_route)
                            ],
                        }
                        route_info["trips"].append(trip_info)

                result_data["routes"].append(route_info)
        # Write to file
        with open(output_path, "w") as f:
            json.dump(result_data, f, indent=2)

        print(f"✔ Results saved to {output_path}")


for file in files:
    run(file)