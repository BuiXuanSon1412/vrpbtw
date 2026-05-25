"""Debug why drones get stuck at step 54."""

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

stuck_step = None
for step in range(100):
    feasible = np.where(action_mask)[0]
    if len(feasible) == 0:
        stuck_step = step
        break

    action = np.random.choice(feasible)
    obs, reward, terminated, truncated, info = env.step(action)
    action_mask = info["action_mask"]

if stuck_step:
    state = env._current_state
    print(f"\nStuck at step {stuck_step}")
    print(f"\nActive drones (cannot take action):")

    for k in range(env.K):
        if state.drone_active[k]:
            print(f"\nDrone {k} (active):")
            print(f"  Current node: {int(state.drone_node[k])}")
            print(f"  Trip so far: {state.drone_trips[k][-1]}")
            print(f"  Trip start time: {state.drone_depart[k][-1][0]:.4f}")
            print(f"  Current time: {env._drone_current_time(state, k):.4f}")

            # Check landing feasibility for each possible landing node
            landing_indices = env._landing_nodes(state, k)
            print(f"  Possible landing indices: {landing_indices}")

            for land_idx in landing_indices:
                land_node = state.truck_routes[k][land_idx]
                feasible = env._drone_land_feasible(state, k, land_idx)
                print(f"    Land at index {land_idx} (node {land_node}): {feasible}")

                if not feasible:
                    # Debug why it failed
                    from_node = int(state.drone_node[k])
                    t_back = env.euclidean_dist[from_node, land_node] / env.v_d
                    trip_start = state.drone_depart[k][-1][0]
                    trip_end = (
                        env._drone_current_time(state, k)
                        + env.launch_time
                        + t_back
                        + env.land_time
                    )

                    print(f"      trip_start={trip_start:.4f}, trip_end={trip_end:.4f}")
                    print(f"      trip_duration={trip_end - trip_start:.4f} vs t_max={env.t_max:.4f}")
                    print(f"      system_time={trip_end:.4f} vs T_max={env.T_max:.4f}")

                    service_rate_ok = env._service_rate_feasible(state, k, land_idx, trip_end)
                    print(f"      service_rate_feasible={service_rate_ok}")
