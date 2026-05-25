"""Trace drone_depart values as drone extends."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from impl.mvrpbtw import MVRPBTWEnv
from core import SeedManager

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
seed_mgr = SeedManager(random_seed=0, numpy_seed=0, torch_seed=0)
seed_mgr.seed_everything()
instance = env._generate_instance(cfg["tasks"][0])
env.encode_instance(instance)

obs, info = env.reset()
action_mask = info["action_mask"]

first_drone_launch = True

for step in range(100):
    feasible = np.where(action_mask)[0]
    if len(feasible) == 0:
        print(f"\nStuck at step {step}")
        state = env._current_state
        print(f"Active drones: {[k for k in range(env.K) if state.drone_active[k]]}")

        # Show drone_depart for active drones
        for k in range(env.K):
            if state.drone_active[k]:
                print(f"\nDrone {k} drone_depart: {state.drone_depart[k]}")
        break

    action = np.random.choice(feasible)
    state_before = env._current_state

    # Decode action
    action_val = action
    v_idx = action_val % env.K
    node = action_val // env.K
    k, vtype = env.vehicle_fleet_type(v_idx)

    if vtype == 1 and node > 0:  # drone serving a customer or landing
        is_launch = not state_before.drone_active[k]
        is_land = node in state_before.truck_routes[k]

        if is_launch:
            print(f"\nStep {step}: Drone {k} LAUNCH to node {node}")
            print(f"  Before: drone_depart={state_before.drone_depart[k]}")
        elif is_land:
            print(f"\nStep {step}: Drone {k} LAND at node {node}")
            print(f"  Before: drone_depart={state_before.drone_depart[k]}")
        else:
            print(f"\nStep {step}: Drone {k} EXTEND to node {node}")
            print(f"  Before: drone_depart={state_before.drone_depart[k]}")

    obs, reward, terminated, truncated, info = env.step(action)
    action_mask = info["action_mask"]

    state_after = env._current_state

    if vtype == 1 and node > 0:
        print(f"  After:  drone_depart={state_after.drone_depart[k]}")
        if state_after.drone_active[k]:
            current_t = env._drone_current_time(state_after, k)
            print(f"  current_time={current_t:.4f}")

    if terminated or truncated:
        break
