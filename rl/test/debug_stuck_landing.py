"""Debug why drones can't land at stuck state."""

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

for step in range(100):
    feasible = np.where(action_mask)[0]
    if len(feasible) == 0:
        print(f"\n{'='*80}")
        print(f"STUCK at step {step}")
        print(f"{'='*80}\n")

        state = env._current_state

        # Debug each active drone
        for k in range(env.K):
            if not state.drone_active[k]:
                continue

            print(f"\nDrone {k} (ACTIVE):")
            print(f"  Current drone node: {int(state.drone_node[k])}")
            print(f"  Current trip: {state.drone_trips[k][-1]}")
            print(f"  Trip start time: {state.drone_depart[k][-1][0]:.4f}")
            print(f"  Current time: {env._drone_current_time(state, k):.4f}")
            print(f"  Drone load: {state.drone_load[k][-1]}")

            # Check landing nodes
            landing_indices = env._landing_nodes(state, k)
            print(f"  Landing candidates (indices): {landing_indices}")

            current_time = env._drone_current_time(state, k)
            print(f"\n  Checking landing feasibility from current time {current_time:.4f}:")

            for land_idx in landing_indices:
                land_node = state.truck_routes[k][land_idx]
                from_node = int(state.drone_node[k])

                # Simplified: drone is at from_node and wants to land at land_node
                t_to_land = env.euclidean_dist[from_node, land_node] / env.v_d

                trip_start = state.drone_depart[k][-1][0]
                trip_end = current_time + env.launch_time + t_to_land + env.land_time
                trip_duration = trip_end - trip_start

                truck_depart_at_land = state.truck_depart[k][land_idx]
                truck_arrive_at_land = state.truck_arrive[k][land_idx]

                print(f"\n    Land at index {land_idx} (node {land_node}):")
                print(f"      From node: {from_node}, Distance: {env.euclidean_dist[from_node, land_node]:.4f}")
                print(f"      t_to_land: {t_to_land:.4f}")
                print(f"      trip_start: {trip_start:.4f}, trip_end: {trip_end:.4f}")
                print(f"      trip_duration: {trip_duration:.4f} (limit: {env.t_max:.4f})")
                print(f"      Constraint 1 (duration): {trip_duration <= env.t_max}")
                print(f"      Constraint 2 (system time): {trip_end <= env.T_max} ({trip_end:.4f} <= {env.T_max:.4f})")
                print(f"      Constraint 3 (depart preservation): {trip_end <= truck_depart_at_land} ({trip_end:.4f} <= {truck_depart_at_land:.4f})")
                print(f"      truck_arrive_at_land: {truck_arrive_at_land:.4f}")
                print(f"      Pass all: {trip_duration <= env.t_max and trip_end <= env.T_max and trip_end <= truck_depart_at_land}")

        break

    action = np.random.choice(feasible)
    obs, reward, terminated, truncated, info = env.step(action)
    action_mask = info["action_mask"]

    if terminated or truncated:
        break
