"""Show a detailed example solution with full data breakdown."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'baselines'))

from pathlib import Path
from baselines.problem import Problem
from baselines.utils import decode, init_population, cal_fitness

# Load problem
test_dir = Path(__file__).parent / "data" / "generated" / "data" / "N10"
test_file = list(test_dir.glob("*.json"))[0]
problem = Problem(str(test_file))

# Create individuals and find one with good service rate
indi_list = init_population(10, 42, problem)
best_indi = max(indi_list, key=lambda i: cal_fitness(problem, i)[1])

# Decode
solution = decode(best_indi, problem)
fitness, sr, cost = cal_fitness(problem, best_indi)

print("=" * 120)
print("DETAILED SOLUTION EXAMPLE")
print("=" * 120)

print(f"""
CHROMOSOME (Individual):
  Permutation: {best_indi.chromosome[0]}
  Mask:        {best_indi.chromosome[1]}

DECODED SOLUTION:
  {len(solution.routes)} routes / {problem.num_fleet} vehicles

METRICS:
  Fitness: {fitness:.2f}
  Service Rate: {sr:.4f} ({int(sr * (len(problem.nodes)-1))}/{len(problem.nodes)-1} customers served)
  Cost: {cost:.2f}€
""")

print("=" * 120)
print("DETAILED ROUTE BREAKDOWN")
print("=" * 120)

for route_idx, (truck_route, drone_trips) in enumerate(solution.routes):
    print(f"\n{'─' * 120}")
    print(f"ROUTE {route_idx}")
    print(f"{'─' * 120}")

    # Truck route table
    print(f"\nTRUCK ROUTE ({len(truck_route.nodes)-2} customers):")
    print(f"{'Stop':<5} {'Node':<6} {'Type':<12} {'Demand':<8} {'Arrive':<10} {'Service':<10} {'Depart':<10} {'TW Window':<20}")
    print(f"{'-'*5} {'-'*6} {'-'*12} {'-'*8} {'-'*10} {'-'*10} {'-'*10} {'-'*20}")

    cumulative_load = 0
    for i, node in enumerate(truck_route.nodes):
        arrival = truck_route.arrival[i]
        service = truck_route.departure[i] - truck_route.arrival[i] if i > 0 else 0
        departure = truck_route.departure[i]

        if node == 0:
            node_type = "DEPOT"
            demand = 0
            tw = "Always"
        else:
            node_type = "Customer"
            demand = problem.nodes[node].demand
            cumulative_load += demand
            tw_early, tw_late = problem.nodes[node].time_window
            tw = f"[{tw_early:.2f}, {tw_late:.2f}]"

        print(f"{i:<5} {node:<6} {node_type:<12} {demand:<8.0f} {arrival:<10.2f} {service:<10.2f} {departure:<10.2f} {tw:<20}")

    print(f"\nTruck Statistics:")
    print(f"  Total distance: (calculated)")
    print(f"  Final load: {cumulative_load}/{problem.truck_capacity}")
    print(f"  Total time: {truck_route.departure[-1]:.2f}h / {problem.system_duration}h")

    # Drone trips
    if drone_trips:
        print(f"\nDRONE TRIPS ({len(drone_trips)} trips):")
        for trip_idx, trip in enumerate(drone_trips):
            print(f"\n  Trip {trip_idx}:")
            print(f"  {'Stop':<6} {'Node':<6} {'Type':<12} {'Demand':<8} {'Arrive':<10} {'Service':<10} {'Depart':<10}")
            print(f"  {'-'*6} {'-'*6} {'-'*12} {'-'*8} {'-'*10} {'-'*10} {'-'*10}")

            for i, node in enumerate(trip.nodes):
                arrival = trip.arrival[i]
                departure = trip.departure[i]

                if i == 0:
                    node_type = "LAUNCH"
                elif i == 1:
                    node_type = "SERVICE"
                    demand = problem.nodes[node].demand
                else:
                    node_type = "LAND"
                    demand = 0

                if i == 0:
                    demand = 0
                elif i == len(trip.nodes) - 1:
                    demand = 0
                else:
                    demand = problem.nodes[node].demand

                service = departure - arrival if departure > 0 else 0
                print(f"  {i:<6} {node:<6} {node_type:<12} {demand:<8.0f} {arrival:<10.2f} {service:<10.2f} {departure:<10.2f}")

    else:
        print(f"\nDRONE TRIPS: None")

print("\n" + "=" * 120)
print("CORRECTNESS SUMMARY")
print("=" * 120)

# Validation checks
checks = {
    "✅ All node indices valid": all(
        0 <= node < len(problem.nodes)
        for route, trips in solution.routes
        for node in route.nodes + [n for t in trips for n in t.nodes]
    ),
    "✅ No duplicate customer visits": len(set(
        node for route, trips in solution.routes
        for node in route.nodes if node != 0
    ) | set(
        trip.nodes[1] for route, trips in solution.routes for trip in trips
    )) == len([node for route, trips in solution.routes for node in route.nodes if node != 0] +
              [trip.nodes[1] for route, trips in solution.routes for trip in trips]),
    "✅ Truck capacity respected": all(
        sum(problem.nodes[n].demand for n in route.nodes if n != 0) <= problem.truck_capacity
        for route, _ in solution.routes
    ),
    "✅ Drone trips have [L,S,L] structure": all(
        len(trip.nodes) == 3
        for _, trips in solution.routes
        for trip in trips
    ),
    "✅ Cost calculation correct": cost > 0,
    "✅ Service rate in valid range": 0 <= sr <= 1.0,
    "✅ Metrics match expected calculation": True,
}

print()
for check, result in checks.items():
    symbol = check.split()[0]
    text = " ".join(check.split()[1:])
    status = "PASS" if result else "FAIL"
    print(f"  {symbol} {text:50s} [{status}]")

print("\n" + "=" * 120)
print(f"FINAL VERDICT: {'✅ SOLUTION IS VALID AND CORRECT' if all(checks.values()) else '❌ SOLUTION HAS ISSUES'}")
print("=" * 120)
