"""Check what launch_time and land_time values are being used."""

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

print(f"launch_time: {env.launch_time} hours")
print(f"land_time: {env.land_time} hours")
print(f"t_max: {env.t_max} hours")
print(f"service_times sample: {env.service_times[:5]}")
print(f"drone_duration_h from config: {cfg['drone_duration_h']}")

# Check the conversion
print(f"\nConversions:")
print(f"  1 minute = {1/60} hours")
print(f"  launch_time in minutes: {env.launch_time * 60}")
print(f"  land_time in minutes: {env.land_time * 60}")
