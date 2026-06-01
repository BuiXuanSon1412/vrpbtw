import pulp as pl
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

filename = args.filename  # e.g., "S42_N5_C_U_R50.json"
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

    # Prepare output path
    output_dir = os.path.join(RESULT_ROOT, subfolder)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
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

    c = 1.0
    c_tilde = 0.2

    M = 10000.0
    M_Q = Q + 1
    M_Q_tilde = Q_tilde + 1
    M_node = end_depot_idx + 1
    M_edge = M_node * M_node
    M_T = T_max + 1

    w1 = 0.0
    w2 = 1.0

    model = pl.LpProblem("VRPBD", pl.LpMinimize)

    x = pl.LpVariable.dicts("x", ((k, i) for k in K for i in N_all), cat="Binary")
    y = pl.LpVariable.dicts(
        "y",
        ((k, i, j) for k in K for i in N_all for j in N_all if i != j),
        cat="Binary",
    )
    z = pl.LpVariable.dicts(
        "z", ((k, i) for k in K for i in N_all), lowBound=0, cat="Integer"
    )
    p = pl.LpVariable.dicts(
        "p", ((k, i) for k in K for i in N_all), lowBound=0, cat="Continuous"
    )
    a = pl.LpVariable.dicts(
        "a", ((k, i) for k in K for i in N_all), lowBound=0, cat="Continuous"
    )
    b = pl.LpVariable.dicts(
        "b", ((k, i) for k in K for i in N_all), lowBound=0, cat="Continuous"
    )

    x_tilde = pl.LpVariable.dicts(
        "x_tilde", ((k, r, i) for k in K for r in R for i in N_all), cat="Binary"
    )
    y_tilde = pl.LpVariable.dicts(
        "y_tilde",
        ((k, r, i, j) for k in K for r in R for i in N_all for j in N_all if i != j),
        cat="Binary",
    )
    z_tilde = pl.LpVariable.dicts(
        "z_tilde",
        ((k, r, i) for k in K for r in R for i in N_all),
        lowBound=0,
        cat="Integer",
    )
    lambda_var = pl.LpVariable.dicts(
        "lambda", ((k, r, i) for k in K for r in R for i in N_all), cat="Binary"
    )
    varrho = pl.LpVariable.dicts(
        "varrho", ((k, r, i) for k in K for r in R for i in N_all), cat="Binary"
    )
    p_tilde = pl.LpVariable.dicts(
        "p_tilde",
        ((k, r, i) for k in K for r in R for i in N_all),
        lowBound=0,
        cat="Continuous",
    )
    a_tilde = pl.LpVariable.dicts(
        "a_tilde", ((k, i) for k in K for i in N_all), lowBound=0, cat="Continuous"
    )
    b_tilde = pl.LpVariable.dicts(
        "b_tilde", ((k, i) for k in K for i in N_all), lowBound=0, cat="Continuous"
    )
    h = pl.LpVariable.dicts(
        "h",
        ((k, r, i) for k in K for r in R for i in N_all),
        lowBound=0,
        cat="Continuous",
    )

    Z_lambda = pl.LpVariable.dicts(
        "Z_lambda",
        ((k, r, i) for k in K for r in R for i in N_all),
        lowBound=0,
        cat="Continuous",
    )
    Z_varrho = pl.LpVariable.dicts(
        "Z_varrho",
        ((k, r, i) for k in K for r in R for i in N_end),
        lowBound=0,
        cat="Continuous",
    )

    xi = pl.LpVariable.dicts("xi", (i for i in N_all), lowBound=0, cat="Continuous")

    tardiness = pl.LpVariable("tardiness", lowBound=0, cat="Continuous")

    cost = pl.lpSum(
        [y[k, i, j] * c * d[i][j] for k in K for i in N_all for j in N_end if i != j]
    ) + pl.lpSum(
        [
            y_tilde[k, r, i, j] * c_tilde * d_tilde[i][j]
            for k in K
            for r in R
            for i in N_all
            for j in N_end
            if i != j
        ]
    )

    model += w1 * cost + w2 * tardiness

    for i in C:
        model += tardiness >= xi[i] + s[i] - t_end[i], f"tardiness_{i}"

    # 3
    # each customer is served once
    for i in C:
        model += (
            pl.lpSum([x[k, i] for k in K])
            + pl.lpSum([x_tilde[k, r, i] for k in K for r in R])
            == 1
        )

    # 10 if no launching node, no edge are traversed
    for k in K:
        for r in R:
            model += pl.lpSum(
                [y_tilde[k, r, i, j] for i in N for j in N_end if i != j]
            ) <= M * pl.lpSum([lambda_var[k, r, i] for i in N])

    # 8 no launching and landing at the same node
    for k in K:
        for r in R:
            for i in C:
                model += lambda_var[k, r, i] + varrho[k, r, i] <= 1

    for k in K:
        for r in R:
            for i in N:
                for j in N_end:
                    if i != j:
                        model += y_tilde[k, r, i, j] <= pl.lpSum(
                            x_tilde[k, r, h] for h in C
                        )

    # 4-5
    for k in K:
        model += pl.lpSum([y[k, 0, j] for j in C]) <= 1
        model += pl.lpSum([y[k, i, end_depot_idx] for i in C]) <= 1

    for k in K:
        for i in C:
            model += pl.lpSum([y[k, j, i] for j in N if j != i]) == x[k, i]
            model += pl.lpSum([y[k, i, j] for j in N_end if j != i]) == x[k, i]

    # 6-7
    for k in K:
        for r in R:
            model += pl.lpSum([lambda_var[k, r, i] for i in N]) <= 1
            model += pl.lpSum([varrho[k, r, j] for j in N_end]) <= 1
            model += pl.lpSum([lambda_var[k, r, i] for i in N]) == pl.lpSum(
                [varrho[k, r, j] for j in N_end]
            )

    # 8
    for k in K:
        for r in R:
            for i in C:
                model += lambda_var[k, r, i] <= x[k, i]
                model += varrho[k, r, i] <= x[k, i]

    # 9-13
    for k in K:
        for r in R:
            for i in C:
                model += pl.lpSum([y_tilde[k, r, j, i] for j in N if i != j]) - x_tilde[
                    k, r, i
                ] <= M_node * (lambda_var[k, r, i] + varrho[k, r, i])
                model += pl.lpSum([y_tilde[k, r, j, i] for j in N if i != j]) - x_tilde[
                    k, r, i
                ] >= -M_node * (lambda_var[k, r, i] + varrho[k, r, i])
    for k in K:
        for r in R:
            for i in N:
                model += pl.lpSum(
                    [y_tilde[k, r, i, j] for j in N_end if i != j]
                ) - x_tilde[k, r, i] <= M_node * (lambda_var[k, r, i] + varrho[k, r, i])
                model += pl.lpSum(
                    [y_tilde[k, r, i, j] for j in N_end if i != j]
                ) - x_tilde[k, r, i] >= -M_node * (
                    lambda_var[k, r, i] + varrho[k, r, i]
                )

            model += varrho[k, r, 0] == 0
            model += lambda_var[k, r, end_depot_idx] == 0
    # 14-15
    for k in K:
        for r in R:
            for i in N:
                model += lambda_var[k, r, i] <= pl.lpSum(
                    [y_tilde[k, r, i, j] for j in N_end if j != i]
                )
                model += (
                    lambda_var[k, r, i]
                    >= x[k, i]
                    + pl.lpSum([y_tilde[k, r, i, j] for j in N_end if j != i])
                    - 1
                )

    # 16-17
    for k in K:
        for r in R:
            for i in N_end:
                model += varrho[k, r, i] <= pl.lpSum(
                    [y_tilde[k, r, j, i] for j in N if j != i]
                )
                model += (
                    varrho[k, r, i]
                    >= x[k, i]
                    + pl.lpSum([y_tilde[k, r, j, i] for j in N if j != i])
                    - 1
                )

    # 18-21
    for k in K:
        for r in R:
            for i in N:
                for j in N_end:
                    if i != j:
                        model += z[k, i] - z[k, j] + 1 <= M * (1 - y[k, i, j])
                        model += z[k, i] - z[k, j] + 1 >= -M * (1 - y[k, i, j])
                        model += z_tilde[k, r, i] - z_tilde[k, r, j] + 1 <= M * (
                            1 - y_tilde[k, r, i, j]
                        )
                        model += z_tilde[k, r, i] - z_tilde[k, r, j] + 1 >= -M * (
                            1 - y_tilde[k, r, i, j]
                        )

    for k in K:
        for r in R:
            for i in N:
                for j in N_end:
                    if i != j:
                        model += z[k, j] >= z[k, i] + 1 - M * (
                            2 - lambda_var[k, r, i] - varrho[k, r, j]
                        )

    # 22-23
    for k in K:
        model += p[k, 0] == pl.lpSum([q[u] * x[k, u] for u in L]) + pl.lpSum(
            [q[u] * x_tilde[k, r, u] for u in L for r in R]
            - pl.lpSum(Z_lambda[k, r, 0] for r in R)
        )
        model += p[k, end_depot_idx] == -pl.lpSum(
            [q[u] * x[k, u] for u in B]
        ) - pl.lpSum([q[u] * x_tilde[k, r, u] for u in B for r in R])

    # 24-25
    for k in K:
        for i in N:
            for j in N_end:
                if i == 0 and j == end_depot_idx:
                    continue
                if i != j:
                    load_change = (
                        -q[j]
                        - pl.lpSum([Z_lambda[k, r, j] for r in R])
                        + pl.lpSum([Z_varrho[k, r, j] for r in R])
                    )
                    model += p[k, j] <= p[k, i] + load_change + M_Q * (1 - y[k, i, j])
                    model += p[k, j] >= p[k, i] + load_change - M_Q * (1 - y[k, i, j])

    for k in K:
        for i in N_all:
            model += p[k, i] <= Q
            model += p[k, i] >= 0

    # 26-35
    for k in K:
        for r in R:
            for j in N:
                model += Z_lambda[k, r, j] <= p_tilde[k, r, j] + Q_tilde * (
                    1 - lambda_var[k, r, j]
                )
                model += Z_lambda[k, r, j] <= Q_tilde * lambda_var[k, r, j]
                model += Z_lambda[k, r, j] >= p_tilde[k, r, j] - Q_tilde * (
                    1 - lambda_var[k, r, j]
                )
    for k in K:
        for r in R:
            for j in N_end:
                model += Z_varrho[k, r, j] <= p_tilde[k, r, j] + Q_tilde * (
                    1 - varrho[k, r, j]
                )
                model += Z_varrho[k, r, j] <= Q_tilde * varrho[k, r, j]
                model += Z_varrho[k, r, j] >= p_tilde[k, r, j] - Q_tilde * (
                    1 - varrho[k, r, j]
                )

    # 36-37
    for k in K:
        for r in R:
            drone_pickup = pl.lpSum([q[u] * x_tilde[k, r, u] for u in L])
            for i in N:
                model += p_tilde[k, r, i] <= drone_pickup + M_Q_tilde * (
                    1 - lambda_var[k, r, i]
                )
                model += p_tilde[k, r, i] >= drone_pickup - M_Q_tilde * (
                    1 - lambda_var[k, r, i]
                )

    # 38-39
    for k in K:
        for r in R:
            drone_delivery = pl.lpSum([q[u] * x_tilde[k, r, u] for u in B])
            for j in N_end:
                model += p_tilde[k, r, j] >= drone_delivery - M_Q_tilde * (
                    1 - varrho[k, r, j]
                )
                model += p_tilde[k, r, j] <= drone_delivery + M_Q_tilde * (
                    1 - varrho[k, r, j]
                )

    # 40-41
    for k in K:
        for r in R:
            for i in N:
                for j in N_end:
                    if i != j:
                        model += p_tilde[k, r, j] <= p_tilde[k, r, i] - q[j] * x_tilde[
                            k, r, j
                        ] + M_Q_tilde * (1 - y_tilde[k, r, i, j])
                        model += p_tilde[k, r, j] >= p_tilde[k, r, i] - q[j] * x_tilde[
                            k, r, j
                        ] - M_Q_tilde * (1 - y_tilde[k, r, i, j])

    # 42
    for k in K:
        for r in R:
            for i in N + [end_depot_idx]:
                model += p_tilde[k, r, i] <= Q_tilde
                model += p_tilde[k, r, i] >= 0

    # 43-45
    for i in C:
        model += xi[i] >= t_start[i]

    for k in K:
        for i in C:
            model += xi[i] >= a[k, i] - M_T * (1 - x[k, i])
            model += xi[i] <= b[k, i] - s[i] + M_T * (1 - x[k, i])
    for k in K:
        for r in R:
            for i in C:
                model += xi[i] >= a_tilde[k, i] + tau_r - M_T * (1 - x_tilde[k, r, i])
                model += xi[i] <= b_tilde[k, i] - s[i] + M_T * (1 - x_tilde[k, r, i])
    # 46-47
    for k in K:
        for i in N:
            for j in N_end:
                if i != j:
                    model += a[k, j] >= b[k, i] + t[i][j] - M_T * (1 - y[k, i, j])
                    model += a[k, j] <= b[k, i] + t[i][j] + M_T * (1 - y[k, i, j])

    # 48-49
    for k in K:
        for r in R:
            for i in N:
                for j in N_end:
                    if i != j:
                        model += a_tilde[k, j] >= b_tilde[k, i] + tau_l + t_tilde[i][
                            j
                        ] - M_T * (1 - y_tilde[k, r, i, j])
                        model += a_tilde[k, j] <= b_tilde[k, i] + tau_l + t_tilde[i][
                            j
                        ] + M_T * (1 - y_tilde[k, r, i, j])

    # 51
    for k in K:
        for r in R:
            for i in N:
                model += b_tilde[k, i] <= b[k, i] + M_T * (1 - lambda_var[k, r, i])
                model += b_tilde[k, i] >= a[k, i] - M_T * (1 - lambda_var[k, r, i])

    # 54-55
    for k in K:
        for r in R:
            for j in N_end:
                model += a_tilde[k, j] + h[k, r, j] + tau_r >= a[k, j] - M_T * (
                    1 - varrho[k, r, j]
                )
                model += a_tilde[k, j] + h[k, r, j] + tau_r <= b[k, j] + M_T * (
                    1 - varrho[k, r, j]
                )

    # 56-58
    for k in K:
        for r in R:
            for i in C:
                model += h[k, r, i] >= xi[i] - a_tilde[k, i] - tau_r - M_T * (
                    1 - x_tilde[k, r, i]
                )
                model += h[k, r, i] <= xi[i] - a_tilde[k, i] - tau_r + M_T * (
                    1 - x_tilde[k, r, i]
                )
                model += h[k, r, i] >= 0

    # 59
    for k in K:
        for r in R:
            flight_time = pl.lpSum(
                [
                    y_tilde[k, r, i, j] * t_tilde[i][j]
                    for i in N
                    for j in N_end
                    if i != j
                ]
            )
            launch_time = pl.lpSum([lambda_var[k, r, i] * tau_l for i in N])
            land_time = pl.lpSum([varrho[k, r, j] * tau_r for j in N_end])
            service_time = pl.lpSum(
                [x_tilde[k, r, i] * (tau_l + tau_r + s[i]) for i in C]
            )
            wait_time = pl.lpSum([h[k, r, i] for i in N])

            total_time = (
                flight_time + launch_time + land_time + service_time + wait_time
            )
            model += total_time <= T_tilde_max

    # 76-77
    for k in K:
        for r in R[:-1]:
            for i in N_end:
                for j in N:
                    model += z[k, j] >= z[k, i] - M * (
                        2 - varrho[k, r, i] - lambda_var[k, r + 1, j]
                    )

            for i in N_end:
                for j in N:
                    model += a_tilde[k, j] >= b_tilde[k, i] + tau_l - M_T * (
                        2 - varrho[k, r, i] - lambda_var[k, r + 1, j]
                    )

    # 78
    for k in K:
        for r in R[:-1]:
            model += pl.lpSum(lambda_var[k, r + 1, i] for i in N) <= pl.lpSum(
                lambda_var[k, r, i] for i in N
            )

    # 79
    for k in K:
        for r in R:
            model += pl.lpSum(lambda_var[k, r, i] for i in N) <= pl.lpSum(
                y[k, 0, j] for j in C
            )

    # 80
    for k in K[:-1]:
        model += pl.lpSum(y[k, 0, j] for j in C) >= pl.lpSum(y[k + 1, 0, j] for j in C)

    print(
        f"Model created with {len(model.variables())} variables and {len(model.constraints)} constraints"
    )
    print("\nSolving the model")

    model.writeLP("model.lp")

    solver = pl.PULP_CBC_CMD(timeLimit=36000, gapRel=0.05, msg=True)

    start_time = time.time()
    model.solve(solver)
    running_time = time.time() - start_time

    if model.status == pl.LpStatusOptimal or model.status == pl.LpStatusNotSolved:
        result_data = {
            "weights": {
                "cost": w1,
                "tardiness": w2,
            },
            "status": pl.LpStatus[model.status],
            "objective": pl.value(model.objective),
            "time": running_time,
            "routes": [],
        }

        print("[TEST]: x")
        for k, v in x.items():
            print(f"\t {k, pl.value(v)}")
        print("[TEST]: x_tilde")
        for k, v in x_tilde.items():
            print(f"\t {k, pl.value(v)}")
        print("[TEST]: y")
        for k, v in y.items():
            print(f"\t {k, pl.value(v)}")
        print("[TEST]: y_tilde")
        for k, v in y_tilde.items():
            print(f"\t {k, pl.value(v)}")

        print("[TEST]: lambda")
        for k, v in lambda_var.items():
            print(f"\t {k, pl.value(v)}")

        print("[TEST]: varrho")
        for k, v in varrho.items():
            print(f"\t {k, pl.value(v)}")
        print("[TEST]: p", Q)
        for k, v in p.items():
            print(f"\t {k, pl.value(v)}")
        print("[TEST]: p_tilde", Q_tilde)
        for k, v in p_tilde.items():
            print(f"\t {k, pl.value(v)}")

        print("[TEST]: tardiness")
        print(pl.value(tardiness))

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
                        and y[k, current, j].varValue
                        and pl.value(y[k, current, j]) > 0.5
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
                        pl.value(a[k, i]) if idx > 0 else None
                        for idx, i in enumerate(route)
                    ],
                    "departure": [
                        pl.value(b[k, i]) if idx < len(route) - 1 else None
                        for idx, i in enumerate(route)
                    ],
                    "service": [
                        pl.value(xi[i]) if idx < len(route) - 1 and idx > 0 else None
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
                        if pl.value(lambda_var[k, r, i])
                        and pl.value(lambda_var[k, r, i]) > 0.5
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
                                    and y_tilde[k, r, drone_current, j].varValue
                                    and pl.value(y_tilde[k, r, drone_current, j]) > 0.5
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
                                pl.value(a_tilde[k, i]) if idx > 0 else None
                                for idx, i in enumerate(drone_route)
                            ],
                            "departure": [
                                pl.value(b_tilde[k, i])
                                if idx < len(drone_route) - 1
                                else None
                                for idx, i in enumerate(drone_route)
                            ],
                            "service": [
                                pl.value(xi[i])
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
