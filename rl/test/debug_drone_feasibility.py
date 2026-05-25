"""Debug script to check if drone launch actions are feasible."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from impl.vrpbtw import VRPBTWEnv
from core import SeedManager


def check_drone_feasibility():
    """Check if any drone launch actions are feasible during rollout."""
    cfg = {
        "env": "VRPBTW",
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

    env = VRPBTWEnv(cfg)
    seed_mgr = SeedManager(random_seed=42, numpy_seed=42, torch_seed=42)
    seed_mgr.seed_everything()

    instance = env._generate_instance(cfg["tasks"][0])
    env.encode_instance(instance)

    obs, info = env.reset()
    action_mask = info["action_mask"]

    print(f"{'='*80}")
    print(f"Drone Feasibility Check")
    print(f"{'='*80}\n")

    step = 0
    drone_actions_found = False
    truck_actions_found = False

    while step < 20:
        feasible = np.where(action_mask)[0]

        if len(feasible) == 0:
            print(f"Step {step}: STUCK - No feasible actions\n")
            break

        # Count truck vs drone actions
        truck_count = 0
        drone_count = 0

        for action in feasible:
            node, v_idx = env.decode_action(action)
            fleet_idx, vtype = env.vehicle_fleet_type(v_idx)

            if vtype == 0:  # TRUCK
                truck_count += 1
            else:  # DRONE
                drone_count += 1

        truck_actions_found = truck_count > 0
        drone_actions_found = drone_count > 0

        state = env._current_state

        print(f"Step {step}:")
        print(f"  Total feasible actions: {len(feasible)}")
        print(f"  Truck actions: {truck_count}")
        print(f"  Drone actions: {drone_count}")

        if drone_count > 0:
            print(f"  ✓ Drone launch is FEASIBLE!")
        else:
            print(f"  ✗ No drone launch actions available")

            # Show why drone might not be feasible
            for k in range(env.K):
                print(f"\n  Fleet {k} status:")
                print(f"    Truck at: {state.truck_node[k]}, Previous: {state.truck_prev_node[k]}")
                print(f"    Drone active: {state.drone_active[k]}, Drone node: {state.drone_node[k]}")
                print(f"    Drone land idx: {state.drone_land_idx[k]}")
                print(f"    Truck routes: {state.truck_routes[k]}")

                # Check landing nodes
                landing_nodes = env._landing_nodes(state, k)
                print(f"    Possible landing indices: {landing_nodes}")

                # Check first customer
                for j in range(1, env.n_customers + 1):
                    if not state.served[j]:
                        launch_time = env._compute_trip_start(state, k, j)
                        print(f"    Customer {j}: launch_time={launch_time:.2f}s (inf={launch_time == float('inf')})")
                        if launch_time < float("inf"):
                            feasible_lands = env._get_feasible_land_nodes(state, k, j, launch_time)
                            print(f"               feasible_lands={feasible_lands}")
                        break

        print()

        # Take random action
        action = np.random.choice(feasible)
        obs, reward, terminated, truncated, info = env.step(action)
        action_mask = info["action_mask"]

        step += 1

        if terminated or truncated:
            print(f"Episode terminated at step {step}\n")
            break

    print(f"{'='*80}")
    print(f"Summary:")
    print(f"  Truck actions available: {truck_actions_found}")
    print(f"  Drone actions available: {drone_actions_found}")

    if not drone_actions_found:
        print(f"\n❌ No drone launch actions were feasible during the episode!")
    else:
        print(f"\n✓ Drone launch actions are feasible!")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    check_drone_feasibility()
