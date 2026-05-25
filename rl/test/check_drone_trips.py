"""Check if drone trips are being created during episodes."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from impl.vrpbtw import VRPBTWEnv
from core import SeedManager


def check_drone_trips():
    """Run episodes and check if drone trips exist."""
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

    episodes_with_drones = 0
    total_episodes = 5

    for ep in range(total_episodes):
        env = VRPBTWEnv(cfg)
        seed_mgr = SeedManager(random_seed=ep*100, numpy_seed=ep*100, torch_seed=ep*100)
        seed_mgr.seed_everything()

        instance = env._generate_instance(cfg["tasks"][0])
        env.encode_instance(instance)

        obs, info = env.reset()
        action_mask = info["action_mask"]

        step = 0
        while step < 50:
            feasible = np.where(action_mask)[0]
            if len(feasible) == 0:
                break

            action = np.random.choice(feasible)
            obs, reward, terminated, truncated, info = env.step(action)
            action_mask = info["action_mask"]

            step += 1
            if terminated or truncated:
                break

        state = env._current_state

        # Count drone trips
        total_drone_trips = sum(len(trips) for trips in state.drone_trips)

        print(f"Episode {ep + 1}:")
        print(f"  Steps: {step}")
        print(f"  Total drone trips: {total_drone_trips}")

        for k in range(env.K):
            drone_trip_count = len(state.drone_trips[k])
            print(f"  Fleet {k}: {drone_trip_count} drone trips")
            if drone_trip_count > 0:
                episodes_with_drones += 1
                for t, trip in enumerate(state.drone_trips[k]):
                    print(f"    Trip {t}: {trip}")

    print(f"\n{'='*60}")
    print(f"Summary: {episodes_with_drones}/{total_episodes} episodes had drone trips")
    if episodes_with_drones > 0:
        print(f"✓ Drones ARE being used!")
    else:
        print(f"❌ No drone trips created")
    print(f"{'='*60}")


if __name__ == "__main__":
    check_drone_trips()
