"""Trace _compute_trip_start for the stuck state case."""

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

env = MVRPBTWEnv(cfg)
seed_mgr = SeedManager(random_seed=6, numpy_seed=6, torch_seed=6)
seed_mgr.seed_everything()
instance = env._generate_instance(cfg["tasks"][0])
env.encode_instance(instance)

obs, info = env.reset()
action_mask = info["action_mask"]

count = 0
for step in range(50):
    feasible = np.where(action_mask)[0]
    if len(feasible) == 0:
        break

    action = np.random.choice(feasible)

    # Check if this will launch a drone
    node = action // env.K
    v_idx = action % env.K
    k, vtype = env.vehicle_fleet_type(v_idx)

    state_before = env._current_state
    if vtype == 1 and not state_before.drone_active[k] and node > 0:
        count += 1
        if count == 4:  # The 4th drone launch should be the problematic one
            print(f"\n{'='*80}")
            print(f"Step {step}: About to launch Drone {k} to serve node {node}")
            print(f"{'='*80}")

            # Manually call _compute_trip_start to see what it returns
            depart_t = env._compute_trip_start(state_before, k, node)
            print(f"\n_compute_trip_start returned: {depart_t:.4f}")

            # Now check what happens during actual service
            launch_node = int(state_before.truck_prev_node[k])
            land_node = int(state_before.truck_node[k])

            d_launch_to_j = env.launch_time + (env.euclidean_dist[launch_node, node] / env.v_d) + env.land_time
            d_j_to_land = env.launch_time + (env.euclidean_dist[node, land_node] / env.v_d) + env.land_time

            arrive_j = depart_t + d_launch_to_j
            service_start = max(arrive_j, env.tw_open[node])
            service_end = service_start + env.service_times[node]

            print(f"\nDrone service calculation:")
            print(f"  depart_t: {depart_t:.4f}")
            print(f"  d_launch_to_j: {d_launch_to_j:.4f}")
            print(f"  arrive_j: {arrive_j:.4f}")
            print(f"  tw_open[{node}]: {env.tw_open[node]:.4f}")
            print(f"  service_start: {service_start:.4f}")
            print(f"  service_times[{node}]: {env.service_times[node]:.4f}")
            print(f"  service_end: {service_end:.4f}")

            # Compute trip_end to landing
            land_idx = len(state_before.truck_routes[k]) - 1
            trip_end_to_first_land = max(service_end + d_j_to_land, state_before.truck_arrive[k][land_idx])
            trip_duration = trip_end_to_first_land - depart_t

            print(f"\n  d_j_to_land: {d_j_to_land:.4f}")
            print(f"  truck_arrive[land_idx]: {state_before.truck_arrive[k][land_idx]:.4f}")
            print(f"  trip_end_to_land: {trip_end_to_first_land:.4f}")
            print(f"  trip_duration: {trip_duration:.4f}")
            print(f"  t_max: {env.t_max:.4f}")
            print(f"  VIOLATES t_max? {trip_duration > env.t_max}")

    obs, reward, terminated, truncated, info = env.step(action)
    action_mask = info["action_mask"]

    if terminated or truncated:
        break
