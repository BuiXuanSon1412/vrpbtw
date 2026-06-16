"""Detailed analysis of potential bugs in the decode function."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'baselines'))

from pathlib import Path
from baselines.problem import Problem, Individual
from baselines.utils import (
    decode, cal_service_rate, cal_cost,
    cal_drone_route_distance, cal_truck_route_distance
)

def analyze_service_rate_bug():
    """
    Analyze potential bug in cal_service_rate calculation.

    The function assumes that all routes and trips have depot nodes at start/end,
    but drone trips actually have [launch_node, service_node(s), land_node]
    where launch_node and land_node are on the truck route, not depots.

    This means the calculation:
        served += len(route.nodes) - 2
        served += len(trip.nodes) - 2

    May be incorrect because:
    - Truck route: [0, n1, n2, ..., 0] → len - 2 = number of customers ✓
    - Drone trip: [truck_node1, customer1, truck_node2] → len - 2 = 1 customer ✓

    Wait, this might actually be correct... let me think about drone trips with multiple stops.
    """

    print("=" * 80)
    print("ISSUE 1: Drone Trip Service Rate Counting")
    print("=" * 80)

    print("""
Analysis of cal_service_rate():
  Current code:
    for route, trips in routes:
        served += len(route.nodes) - 2  # Assume start and end are depots
        for trip in trips:
            served += len(trip.nodes) - 2

Issue: What is the structure of drone trip nodes?
  From routing() function:
  - A single drone trip starts with [launch_node, service_node, land_node]
  - Multiple services on one trip: [launch, svc1, land, svc2, land, ...]
    (This looks WRONG - see Issue 2)

Expected: Each drone trip should count the number of unique customer nodes served,
not all nodes in the trip.

Example drone trip: [truck_node_1, customer_5, truck_node_2]
  len = 3, subtract 2 = 1 (correct, only customer_5 is served by drone)

Example (if bug exists): [truck_node_1, cust5, truck_node_2, cust6, truck_node_3]
  len = 5, subtract 2 = 3, but only 2 customers (cust5, cust6) are served
  This would OVERCOUNT by 1.

Conclusion: If drone trips have the structure [L, S1, L, S2, L, ...], then
the service rate calculation is INCORRECT because it counts intermediate
landing points.
""")

    test_dir = Path(__file__).parent / "data" / "generated" / "data" / "N10"
    test_file = list(test_dir.glob("*.json"))[0]
    problem = Problem(str(test_file))

    # Create an individual with forced drone masks to test
    from baselines.utils import init_population, repair, partition, routing
    from copy import deepcopy

    indi = init_population(1, 42, problem)[0]

    # Force some drone masks
    n_nodes = len(problem.nodes)
    if len(indi.chromosome[1]) > 3:
        indi.chromosome[1][2] = 1  # Force drone delivery
        if len(indi.chromosome[1]) > 4:
            indi.chromosome[1][3] = 1  # Another drone delivery

    solution = decode(indi, problem)

    print(f"\nTest case: Problem with {n_nodes} nodes, {problem.num_fleet} vehicles")
    print(f"Chromosome mask: {indi.chromosome[1]}")
    print(f"Number of routes: {len(solution.routes)}")

    total_served = 0
    for i, (truck_route, drone_trips) in enumerate(solution.routes):
        print(f"\nRoute {i}:")
        print(f"  Truck nodes: {truck_route.nodes} (len={len(truck_route.nodes)})")

        # Count truck-served nodes
        truck_served = len(truck_route.nodes) - 2
        total_served += truck_served
        print(f"  Truck customers served (len - 2): {truck_served}")

        for j, trip in enumerate(drone_trips):
            print(f"  Trip {j}: {trip.nodes} (len={len(trip.nodes)})")
            # This is the problematic calculation
            trip_served = len(trip.nodes) - 2
            total_served += trip_served
            print(f"    Customers served (len - 2): {trip_served}")

            # Check for the problematic pattern [L, S, L, S, L, ...]
            expected_pattern_issue = False
            if len(trip.nodes) > 3:
                # If we have more than 3 nodes and multiple land points
                land_count = 0
                for node in trip.nodes:
                    if node in truck_route.nodes:
                        land_count += 1

                print(f"    Truck route connection points in trip: {land_count}")
                if land_count > 2:
                    expected_pattern_issue = True
                    print(f"    WARNING: This pattern suggests potential bug!")
                    print(f"    Structure appears to be [L, S1, L, S2, L, ...] instead of [L, S1, S2, ..., L]")

    sr = cal_service_rate(solution, problem)
    print(f"\nTotal service rate: {sr} (normalized: {sr / n_nodes:.4f})")

def analyze_drone_route_distance_bug():
    """
    Analyze potential bug in cal_drone_route_distance calculation.

    The function sums distances between consecutive nodes in the drone trip.
    But if the trip has structure [L, S1, L, S2, L, ...], then:
      distance = dist[L, S1] + dist[S1, L] + dist[L, S2] + dist[S2, L] + ...

    The term dist[L, S2] doesn't make physical sense - the drone isn't flying
    from the landing point to S2 directly.
    """

    print("\n" + "=" * 80)
    print("ISSUE 2: Drone Trip Structure Bug")
    print("=" * 80)

    print("""
Analysis of routing() function and drone trip construction:

Current implementation (lines 391-398, 468-469):
  For first drone service node:
    drone_trip.extend([launch_node, service_node, land_node])

  For subsequent service nodes:
    drone_trip.extend([service_node, land_node])

Result structure: [launch, svc1, land, svc2, land, ...]

Problem: This structure doesn't make physical sense!
  - After landing at 'land', the drone is on the truck
  - To serve 'svc2', the drone needs to be launched again
  - But there's no launch node between 'land' and 'svc2'

Correct structure should be one of:
  a) [launch, svc1, svc2, ..., land] - serve multiple nodes on one trip
  b) [launch1, svc1, land1] + [launch2, svc2, land2] - separate drone instances/trips

Impact on metrics:
  - cal_drone_route_distance: computes sum of consecutive distances
    - dist[land, svc2] makes no physical sense
  - cal_service_rate: counts len - 2
    - If trip is [L, S1, L, S2, L], serves 2 customers but len=5, counts 3

Verification: Check if drone trips with >3 nodes actually exist and have
the problematic pattern.
""")

def analyze_truck_route_distance_bug():
    """
    Check if there's an issue with how depot indices are handled.
    """

    print("\n" + "=" * 80)
    print("ISSUE 3: Depot Index Handling")
    print("=" * 80)

    print("""
Analysis of depot handling:

In the Problem class:
  - nodes[0] is the depot
  - nodes[1..n] are customers

In routing():
  - Truck route ends with depot index (len(problem.nodes))
  - But the Route object stores this as truck_route = [0, ..., end_depot]

Looking at line 220: end_depot = len(problem.nodes)

But in the nodes array, we only have indices 0 to n-1 (n = len(problem.nodes)).
So end_depot = n is OUT OF BOUNDS!

In cal_truck_route_distance (line 530):
  problem.truck_distance_matrix[route[i] % n_node][route[i-1] % n_node]

The modulo operation tries to wrap the out-of-bounds index back to 0,
which is correct (depot is at both index 0 and index n with modulo).

But is this intentional or accidental? Let me check how the route is built...
""")

    test_dir = Path(__file__).parent / "data" / "generated" / "data" / "N10"
    test_file = list(test_dir.glob("*.json"))[0]
    problem = Problem(str(test_file))

    print(f"\nProblem has {len(problem.nodes)} nodes")
    print(f"Valid indices: 0 to {len(problem.nodes)-1}")
    print(f"In routing(), end_depot is set to {len(problem.nodes)}")
    print(f"This is OUTSIDE the valid range!")

    # Check the actual routes
    from baselines.utils import init_population
    indi = init_population(1, 42, problem)[0]
    solution = decode(indi, problem)

    for i, (truck_route, _) in enumerate(solution.routes):
        max_node = max(truck_route.nodes)
        print(f"\nRoute {i} max node index: {max_node}")
        if max_node >= len(problem.nodes):
            print(f"  ERROR: Node {max_node} >= {len(problem.nodes)} (out of bounds!)")
        print(f"  Truck route nodes: {truck_route.nodes}")

if __name__ == "__main__":
    try:
        analyze_service_rate_bug()
        analyze_drone_route_distance_bug()
        analyze_truck_route_distance_bug()

        print("\n" + "=" * 80)
        print("SUMMARY")
        print("=" * 80)
        print("""
Three potential bugs identified in decode() and related functions:

1. SERVICE RATE BUG: Incorrect counting of served customers if drone trips
   have the structure [L, S1, L, S2, L, ...] instead of [L, S1, S2, ..., L]

2. DRONE TRIP STRUCTURE BUG: The routing() function creates drone trips with
   intermediate landing points, which doesn't match the multi-service pattern
   they're trying to implement. This causes incorrect distance calculations.

3. DEPOT INDEX BUG: The routing() function uses end_depot = len(problem.nodes)
   which is out of bounds (valid range is 0 to n-1). The cal_truck_route_distance
   uses modulo to work around this, but it's unclear if this is intentional.

Recommendation: Review the routing() function to verify the intended behavior
for multi-service drone trips.
""")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
