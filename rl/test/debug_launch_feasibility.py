"""Debug test to check why _compute_trip_start returns finite but _get_feasible_land_nodes returns empty."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from impl.mvrpbtw import MVRPBTWEnv
from core import SeedManager

def test_launch_feasibility_mismatch():
    """Check if there's a mismatch between _compute_trip_start and _get_feasible_land_nodes."""

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

    # Run episode and collect states where drones should be launchable
    obs, info = env.reset()
    action_mask = info["action_mask"]

    mismatches = []

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
                continue  # Skip active drones

            # Try each unserved customer as a launch target
            for j in range(1, env.n_customers + 1):
                if state.served[j]:
                    continue

                # Check if launch is feasible according to _compute_trip_start
                launch_time = env._compute_trip_start(state, k, j)

                if launch_time < float("inf"):
                    # Launch is supposedly feasible - check if landing nodes exist
                    feasible_land_indices = env._get_feasible_land_nodes(state, k, j, launch_time)

                    if len(feasible_land_indices) == 0:
                        # MISMATCH FOUND
                        mismatch = {
                            "step": step,
                            "k": k,
                            "j": j,
                            "launch_time": launch_time,
                            "truck_node": int(state.truck_node[k]),
                            "truck_prev_node": int(state.truck_prev_node[k]),
                            "truck_routes_k": state.truck_routes[k],
                            "landing_nodes": env._landing_nodes(state, k),
                        }
                        mismatches.append(mismatch)

                        # Detailed debugging for first mismatch
                        if len(mismatches) == 1:
                            print(f"\n{'='*80}")
                            print(f"MISMATCH FOUND at step {step}")
                            print(f"{'='*80}")
                            print(f"Truck {k}: prev_node={int(state.truck_prev_node[k])}, node={int(state.truck_node[k])}")
                            print(f"Customer {j} (launch target)")
                            print(f"Launch time from _compute_trip_start: {launch_time}")
                            print(f"Truck route: {state.truck_routes[k]}")
                            print(f"Landing node candidates from _landing_nodes: {env._landing_nodes(state, k)}")

                            # Check why _get_feasible_land_nodes returns empty
                            land_indices = env._landing_nodes(state, k)
                            print(f"\nChecking each landing node candidate:")
                            for land_idx in land_indices:
                                land_node = state.truck_routes[k][land_idx]
                                from_node = int(state.drone_node[k])

                                t_to_j = env.euclidean_dist[from_node, j] / env.v_d
                                t_j_to_land = env.euclidean_dist[j, land_node] / env.v_d
                                arrive_j = launch_time + env.launch_time + t_to_j + env.land_time
                                service_end = max(arrive_j, env.tw_open[j]) + env.service_times[j]
                                trip_end = max(
                                    service_end + env.launch_time + t_j_to_land + env.land_time,
                                    state.truck_arrive[k][land_idx],
                                )

                                print(f"\n  Land at index {land_idx} (node {land_node}):")
                                print(f"    Trip duration: {trip_end - launch_time:.2f} (limit: {env.t_max})")
                                print(f"    Trip end time: {trip_end:.2f} (limit: {env.T_max})")
                                print(f"    Trip duration check: {trip_end - launch_time <= env.t_max}")
                                print(f"    System time check: {trip_end <= env.T_max}")

                                # Check service rate feasibility
                                service_rate_ok = env._service_rate_feasible(state, k, land_idx, trip_end)
                                print(f"    Service rate feasible: {service_rate_ok}")

                        if len(mismatches) >= 5:
                            break

            if len(mismatches) >= 5:
                break

        if len(mismatches) >= 5:
            break

        if terminated or truncated:
            break

    print(f"\n{'='*80}")
    print(f"Total mismatches found: {len(mismatches)}")
    print(f"{'='*80}")

    if mismatches:
        for i, m in enumerate(mismatches[:3]):
            print(f"\nMismatch {i+1}:")
            print(f"  Step: {m['step']}, Truck: {m['k']}, Customer: {m['j']}")
            print(f"  Launch time: {m['launch_time']:.4f}")
            print(f"  Truck route: {m['truck_routes_k']}")
            print(f"  Landing candidates: {m['landing_nodes']}")

if __name__ == "__main__":
    test_launch_feasibility_mismatch()
