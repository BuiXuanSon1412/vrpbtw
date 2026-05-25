"""Debug test to understand why _compute_trip_start always returns inf."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from impl.mvrpbtw import MVRPBTWEnv
from core import SeedManager

def test_compute_trip_start():
    """Check why _compute_trip_start returns inf for all drone launches."""

    cfg = {
        "env": "MVRPBTW",
        "tasks": ["easy_N100_F10_C"],
        "n_customers": 100,
        "max_coord": 100.0,
        "capacity_truck": 200.0,
        "capacity_drone": 20.0,
        "t_max_system_h": 24.0,
        "drone_duration_h": 1.0,
        "v_truck_km_h": 40.0,
        "v_drone_km_h": 60.0,
        "truck_cost_unit": 1.0,
        "drone_cost_unit": 0.5,
        "drone_takeoff_min": 1.0,
        "drone_landing_min": 1.0,
        "service_time_min": 5.0,
    }

    env = MVRPBTWEnv(cfg)
    task_id = cfg["tasks"][0]

    # Generate one instance
    seed_mgr = SeedManager(random_seed=0, numpy_seed=0, torch_seed=0)
    seed_mgr.seed_everything()
    instance = env._generate_instance(task_id)
    env.encode_instance(instance)

    # Run episode
    obs, info = env.reset()
    action_mask = info["action_mask"]

    cases_checked = {i: 0 for i in range(1, 9)}  # track case outcomes
    cases_feasible = {i: 0 for i in range(1, 9)}

    for step in range(100):
        feasible = np.where(action_mask)[0]
        if len(feasible) == 0:
            break

        action = np.random.choice(feasible)
        obs, reward, terminated, truncated, info = env.step(action)
        action_mask = info["action_mask"]
        state = env._current_state

        # Check each drone's launch feasibility
        for k in range(env.K):
            if state.drone_active[k]:
                continue

            # Try first few unserved customers
            for j in range(1, min(10, env.n_customers + 1)):
                if state.served[j]:
                    continue

                # Check if this is feasible according to _drone_launch_feasible
                is_feasible = env._drone_launch_feasible(state, k, j)
                if not is_feasible:
                    # Check why: run _compute_trip_start to see result
                    launch_time = env._compute_trip_start(state, k, j)
                    if launch_time < float("inf"):
                        print(f"\nUnexpected: _compute_trip_start returns finite but _drone_launch_feasible returns False")
                        print(f"  Step: {step}, Truck {k}, Customer {j}")
                        print(f"  Launch time: {launch_time}")
                    else:
                        # Trace which cases fail
                        print(f"\nStep {step}, Truck {k}, Customer {j}:")
                        print(f"  Truck prev_node={int(state.truck_prev_node[k])}, node={int(state.truck_node[k])}")
                        print(f"  Truck route: {state.truck_routes[k]}")
                        print(f"  Truck depart at launch_idx: {state.truck_depart[k][state.truck_routes[k].index(int(state.truck_prev_node[k]))]}")
                        print(f"  Truck depart at landing_idx: {state.truck_depart[k][-1]}")
                        print(f"  Truck arrive at landing_idx: {state.truck_arrive[k][-1]}")
                        print(f"  Customer {j}: tw_open={env.tw_open[j]}, tw_close={env.tw_close[j]}, service_time={env.service_times[j]}")

                        # List why constraints might fail
                        launch_node = int(state.truck_prev_node[k])
                        land_node = int(state.truck_node[k])
                        launch_idx = state.truck_routes[k].index(launch_node)
                        land_idx = len(state.truck_routes[k]) - 1

                        d_launch_to_j = env.launch_time + (env.euclidean_dist[launch_node, j] / env.v_d) + env.land_time
                        d_j_to_land = env.launch_time + (env.euclidean_dist[j, land_node] / env.v_d) + env.land_time
                        t_launch_to_land = env.manhattan_dist[launch_node, land_node] / env.v_t

                        print(f"  d_launch_to_j={d_launch_to_j:.4f}, d_j_to_land={d_j_to_land:.4f}, t_launch_to_land={t_launch_to_land:.4f}")
                        print(f"  Case 1 check:")
                        min_d = max(
                            state.truck_depart[k][launch_idx] + t_launch_to_land - env.t_max,
                            state.truck_arrive[k][launch_idx],
                        )
                        max_d = min(
                            env.tw_open[j] - d_launch_to_j,
                            state.truck_depart[k][launch_idx],
                            state.truck_depart[k][launch_idx]
                            + max(env.tw_open[land_node] - state.truck_arrive[k][land_idx], 0.0),
                            env.tw_close[j] - env.service_times[j] - d_launch_to_j,
                        )
                        print(f"    min_d_launch_depart={min_d:.4f}, max_d_launch_depart={max_d:.4f}")
                        print(f"    Valid range: {min_d <= max_d}")
                        print(f"    Truck travel time check: {state.truck_depart[k][launch_idx] + t_launch_to_land} <= {state.truck_depart[k][land_idx]} = {state.truck_depart[k][launch_idx] + t_launch_to_land <= state.truck_depart[k][land_idx]}")

        if terminated or truncated:
            break

if __name__ == "__main__":
    test_compute_trip_start()
