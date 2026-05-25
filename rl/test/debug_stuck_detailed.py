"""Detailed debugging of stuck state - analyze all vehicles and constraints."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from impl.mvrpbtw import MVRPBTWEnv
from core import SeedManager

cfg = {
    "env": "MVRPBTW",
    "tasks": ["easy_N10_F2_C"],
    "n_customers": 10,
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

stuck_found = False
for seed in range(20):
    if stuck_found:
        break

    env = MVRPBTWEnv(cfg)
    seed_mgr = SeedManager(random_seed=seed, numpy_seed=seed, torch_seed=seed)
    seed_mgr.seed_everything()
    instance = env._generate_instance(cfg["tasks"][0])
    env.encode_instance(instance)

    obs, info = env.reset()
    action_mask = info["action_mask"]

    for step in range(50):
        feasible = np.where(action_mask)[0]
        if len(feasible) == 0:
            print(f"\n{'='*80}")
            print(f"STUCK at step {step} (seed={seed})")
            print(f"{'='*80}\n")

            state = env._current_state

            # Check trucks
            print("TRUCKS:")
            for k in range(env.K):
                print(f"\nTruck {k}:")
                print(f"  current_node: {int(state.truck_node[k])}")
                print(f"  route: {state.truck_routes[k]}")
                print(f"  served: {[j for j in range(1, 11) if state.served[j]]}")
                print(f"  unserved: {[j for j in range(1, 11) if not state.served[j]]}")

                # Try truck actions
                for j in range(1, 11):
                    if state.served[j]:
                        continue
                    feasible_truck = env._truck_feasible(state, k, j)
                    print(f"    Can serve j={j}? {feasible_truck}")

            print("\nDRONES:")
            for k in range(env.K):
                print(f"\nDrone {k}:")
                print(f"  active: {state.drone_active[k]}")
                if state.drone_active[k]:
                    print(f"  drone_node: {int(state.drone_node[k])}")
                    print(f"  current_trip: {state.drone_trips[k][-1]}")
                    print(f"  Try to extend:")
                    for j in range(1, 11):
                        if state.served[j]:
                            continue
                        extend_feas = env._drone_extend_feasible(state, k, j)
                        print(f"    Extend to j={j}? {extend_feas}")

                    print(f"  Try to land:")
                    land_indices = env._landing_nodes(state, k)
                    for land_idx in land_indices:
                        land_feas = env._drone_land_feasible(state, k, land_idx)
                        print(f"    Land at idx={land_idx}? {land_feas}")
                else:
                    print(f"  Try to launch:")
                    for j in range(1, 11):
                        if state.served[j]:
                            continue
                        launch_feas = env._drone_launch_feasible(state, k, j)
                        print(f"    Launch to j={j}? {launch_feas}")

            stuck_found = True
            break

        action = np.random.choice(feasible)
        obs, reward, terminated, truncated, info = env.step(action)
        action_mask = info["action_mask"]

        if terminated or truncated:
            break

if not stuck_found:
    print("No stuck states found in seeds 0-19")
