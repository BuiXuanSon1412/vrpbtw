"""Debug why _get_feasible_land_nodes returns empty for landing at truck_node[k]."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from impl.mvrpbtw import MVRPBTWEnv
from core import SeedManager

def test_feasible_land_nodes():
    """Check which constraint fails for truck_node[k] landing."""

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

    # Run episode to step 10 (when drones should be launchable)
    obs, info = env.reset()
    action_mask = info["action_mask"]

    for step in range(100):
        feasible = np.where(action_mask)[0]
        if len(feasible) == 0:
            break

        action = np.random.choice(feasible)
        obs, reward, terminated, truncated, info = env.step(action)
        action_mask = info["action_mask"]
        state = env._current_state

        # After trucks have moved, check drone launch feasibility
        if step >= 5:
            for k in range(min(1, env.K)):  # Just check truck 0
                if state.drone_active[k]:
                    continue

                # Find an unserved customer
                for j in range(1, min(5, env.n_customers + 1)):
                    if state.served[j]:
                        continue

                    # Manually compute what _compute_trip_start would return
                    launch_node = int(state.truck_prev_node[k])
                    land_node = int(state.truck_node[k])
                    launch_idx = state.truck_routes[k].index(launch_node)
                    land_idx = len(state.truck_routes[k]) - 1

                    d_launch_to_j = (
                        env.launch_time
                        + (env.euclidean_dist[launch_node, j] / env.v_d)
                        + env.land_time
                    )
                    d_j_to_land = (
                        env.launch_time
                        + (env.euclidean_dist[j, land_node] / env.v_d)
                        + env.land_time
                    )
                    t_launch_to_land = env.manhattan_dist[launch_node, land_node] / env.v_t

                    # Check if case 1 is feasible
                    min_d_launch_depart = max(
                        state.truck_depart[k][launch_idx] + t_launch_to_land - env.t_max,
                        state.truck_arrive[k][launch_idx],
                    )
                    max_d_launch_depart = min(
                        env.tw_open[j] - d_launch_to_j,
                        state.truck_depart[k][launch_idx],
                        state.truck_depart[k][launch_idx]
                        + max(env.tw_open[land_node] - state.truck_arrive[k][land_idx], 0.0),
                        env.tw_close[j] - env.service_times[j] - d_launch_to_j,
                    )

                    case1_feasible = min_d_launch_depart <= max_d_launch_depart
                    if not case1_feasible:
                        continue  # Case 1 not applicable

                    # Case 1 is feasible, use the midpoint as departure time
                    d_launch_depart = (min_d_launch_depart + max_d_launch_depart) / 2.0

                    print(f"\nStep {step}, Truck {k}, Customer {j}")
                    print(f"  Launch node: {launch_node}, Land node: {land_node}")
                    print(f"  Case 1 feasible: {case1_feasible}")
                    print(f"  d_launch_depart: {d_launch_depart:.4f}")

                    # Now check _get_feasible_land_nodes
                    feasible_land_indices = env._get_feasible_land_nodes(state, k, j, d_launch_depart)
                    print(f"  Feasible landing indices: {feasible_land_indices}")
                    print(f"  Expected land_idx {land_idx} to be feasible")

                    if land_idx not in feasible_land_indices:
                        # Debug why land_idx failed
                        from_node = int(state.drone_node[k])
                        land_node_val = state.truck_routes[k][land_idx]
                        t_to_j = env.euclidean_dist[from_node, j] / env.v_d
                        t_j_to_land = env.euclidean_dist[j, land_node_val] / env.v_d

                        arrive_j = d_launch_depart + env.launch_time + t_to_j + env.land_time
                        service_end = max(arrive_j, env.tw_open[j]) + env.service_times[j]

                        trip_end = max(
                            service_end + env.launch_time + t_j_to_land + env.land_time,
                            state.truck_arrive[k][land_idx],
                        )

                        trip_start = d_launch_depart

                        print(f"\n  Why land_idx={land_idx} failed:")
                        print(f"    trip_start: {trip_start:.4f}")
                        print(f"    trip_end: {trip_end:.4f}")
                        print(f"    trip_duration: {trip_end - trip_start:.4f} (limit: {env.t_max:.4f})")
                        print(f"    T_max check: {trip_end:.4f} <= {env.T_max:.4f} = {trip_end <= env.T_max}")

                        # Check service_rate_feasible
                        service_rate_ok = env._service_rate_feasible(state, k, land_idx, trip_end)
                        print(f"    service_rate_feasible: {service_rate_ok}")

                        if trip_end - trip_start > env.t_max:
                            print(f"    FAIL: trip_duration exceeds t_max")
                        if trip_end > env.T_max:
                            print(f"    FAIL: system time exceeds T_max")
                        if not service_rate_ok:
                            print(f"    FAIL: truck cannot service remaining nodes")

        if terminated or truncated:
            break

if __name__ == "__main__":
    test_feasible_land_nodes()
