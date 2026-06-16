"""Comprehensive test to verify all bug fixes."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'baselines'))

from pathlib import Path
from baselines.problem import Problem, Individual
from baselines.utils import (
    decode, cal_fitness, init_population,
    cal_drone_route_distance, cal_truck_route_distance, cal_service_rate, cal_cost
)

def test_all_fixes():
    """Test all three bug fixes."""
    test_dir = Path(__file__).parent / "data" / "generated" / "data" / "N10"
    test_file = list(test_dir.glob("*.json"))[0]

    problem = Problem(str(test_file))
    n_nodes = len(problem.nodes)

    print("=" * 80)
    print("VERIFICATION: All Bug Fixes")
    print("=" * 80)

    # Create multiple individuals to test
    indi_list = init_population(20, 42, problem)

    all_passes = True

    for i, indi in enumerate(indi_list):
        solution = decode(indi, problem)

        print(f"\n[Individual {i}]")
        print(f"Routes: {len(solution.routes)}")

        for route_idx, (truck_route, drone_trips) in enumerate(solution.routes):
            # BUG #1 CHECK: All truck route nodes must be valid indices
            max_node = max(truck_route.nodes)
            if max_node >= n_nodes:
                print(f"  ❌ BUG #1 NOT FIXED: Node {max_node} >= {n_nodes}")
                all_passes = False
            else:
                print(f"  ✅ BUG #1: Valid depot (max node {max_node} < {n_nodes})")

            # BUG #2 CHECK: All drone trips must have exactly 3 nodes [L, S, L]
            for trip_idx, trip in enumerate(drone_trips):
                if len(trip.nodes) != 3:
                    print(f"  ❌ BUG #2 NOT FIXED: Trip {trip_idx} has {len(trip.nodes)} nodes, expected 3")
                    all_passes = False
                else:
                    launch, service, land = trip.nodes
                    if launch not in truck_route.nodes:
                        print(f"  ❌ BUG #2: Trip {trip_idx} launch {launch} not in truck route")
                        all_passes = False
                    elif land not in truck_route.nodes:
                        print(f"  ❌ BUG #2: Trip {trip_idx} land {land} not in truck route")
                        all_passes = False
                    else:
                        print(f"  ✅ BUG #2: Trip {trip_idx} has correct structure [L={launch}, S={service}, L={land}]")

        # BUG #3 CHECK: Service rate calculation
        service_rate = cal_service_rate(solution, problem)
        manual_count = 0

        for truck_route, drone_trips in solution.routes:
            manual_count += len(truck_route.nodes) - 2
            for trip in drone_trips:
                manual_count += len(trip.nodes) - 2

        if service_rate != manual_count:
            print(f"  ❌ BUG #3 NOT FIXED: Service rate {service_rate} != manual count {manual_count}")
            all_passes = False
        else:
            print(f"  ✅ BUG #3: Service rate counting correct ({service_rate})")

    print("\n" + "=" * 80)
    if all_passes:
        print("✅ ALL BUGS FIXED: Solution decoder is now correct!")
    else:
        print("❌ SOME BUGS REMAIN: Check output above")
    print("=" * 80)

    # Test distance calculations don't error
    print("\nDistance Calculation Tests:")
    indi = indi_list[0]
    solution = decode(indi, problem)

    try:
        for truck_route, drone_trips in solution.routes:
            truck_dist = cal_truck_route_distance(truck_route.nodes, problem)
            print(f"  ✅ Truck distance calculated: {truck_dist:.2f}")

            for trip in drone_trips:
                drone_dist = cal_drone_route_distance(trip.nodes, problem)
                print(f"  ✅ Drone distance calculated: {drone_dist:.2f}")

        cost = cal_cost(solution, problem)
        print(f"  ✅ Total cost calculated: {cost:.2f}")
    except Exception as e:
        print(f"  ❌ Distance calculation failed: {e}")
        all_passes = False

    print("\n" + "=" * 80)
    print("Testing fitness calculation on multiple individuals:")
    errors = 0
    for i, indi in enumerate(indi_list[:5]):
        try:
            fitness, sr, cost = cal_fitness(problem, indi)
            print(f"  Individual {i}: fitness={fitness:.2f}, service_rate={sr:.4f}, cost={cost:.2f}")
        except Exception as e:
            print(f"  ❌ Individual {i} failed: {e}")
            errors += 1

    if errors == 0:
        print(f"  ✅ All {min(5, len(indi_list))} fitness calculations succeeded!")
    else:
        print(f"  ❌ {errors} fitness calculations failed")
        all_passes = False

    print("=" * 80)
    return all_passes

if __name__ == "__main__":
    success = test_all_fixes()
    sys.exit(0 if success else 1)
