"""Show arrival/departure times vs customer time windows in detail."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'baselines'))

from pathlib import Path
from baselines.problem import Problem
from baselines.utils import decode, init_population, cal_fitness

# Load problem and solution
test_dir = Path(__file__).parent / "data" / "generated" / "data" / "N10"
test_file = list(test_dir.glob("*.json"))[0]
problem = Problem(str(test_file))

indi_list = init_population(10, 42, problem)
best_indi = max(indi_list, key=lambda i: cal_fitness(problem, i)[1])
solution = decode(best_indi, problem)

print("=" * 130)
print("VEHICLE ARRIVAL/DEPARTURE vs CUSTOMER TIME WINDOWS")
print("=" * 130)

print(f"\nSystem Duration: {problem.system_duration} hours (0:00 - {int(problem.system_duration)}:00)")
print(f"Service Time Per Customer: {problem.service_time * 60:.1f} minutes")

# Create a timeline visualization
for route_idx, (truck_route, drone_trips) in enumerate(solution.routes):
    print(f"\n{'=' * 130}")
    print(f"ROUTE {route_idx}")
    print(f"{'=' * 130}\n")

    # Table header
    print(f"{'Stop':<6} {'Node':<6} {'Customer':<15} {'Arrive':<10} {'Depart':<10} {'Service':<10}")
    print(f"{'':6} {'':6} {'':15} {'Time':<10} {'Time':<10} {'Time':<10}")
    print(f"{'':6} {'':6} {'':15} {'[Hour]':<10} {'[Hour]':<10} {'[Minutes]':<10}")
    print("-" * 130)

    for i, node in enumerate(truck_route.nodes):
        arrival = truck_route.arrival[i]
        departure = truck_route.departure[i]
        service_time = (departure - arrival) * 60 if i > 0 else 0

        if node == 0:
            print(f"{i:<6} {node:<6} {'DEPOT':<15} {arrival:<10.2f} {departure:<10.2f} {service_time:<10.1f}")
        else:
            print(f"{i:<6} {node:<6} {'Customer ' + str(node):<15} {arrival:<10.2f} {departure:<10.2f} {service_time:<10.1f}")

    print("\n" + "─" * 130)
    print(f"{'TIME WINDOW ANALYSIS':<130}")
    print("─" * 130 + "\n")

    # Detailed time window analysis
    for i, node in enumerate(truck_route.nodes):
        if node == 0:
            continue

        arrival = truck_route.arrival[i]
        departure = truck_route.departure[i]
        tw_early, tw_late = problem.nodes[node].time_window
        service_time_hours = departure - arrival

        # Determine feasibility
        if arrival < tw_early:
            status = "🔴 TOO EARLY"
            minutes_early = (tw_early - arrival) * 60
            detail = f"Arrives {minutes_early:.0f} minutes BEFORE window opens"
        elif arrival > tw_late - problem.service_time:
            status = "🔴 TOO LATE"
            minutes_late = (arrival - (tw_late - problem.service_time)) * 60
            detail = f"Arrives {minutes_late:.0f} minutes AFTER service can start"
        else:
            status = "🟢 ON TIME"
            minutes_until = (tw_late - arrival) * 60
            detail = f"Arrives {minutes_until:.0f} minutes before window closes"

        print(f"Stop {i}: Customer {node} {status}")
        print(f"  Actual arrival:   {arrival:6.2f}h ({int(arrival):02d}:{int((arrival % 1) * 60):02d})")
        print(f"  Time window:      {tw_early:6.2f}h ({int(tw_early):02d}:{int((tw_early % 1) * 60):02d}) - " +
              f"{tw_late:6.2f}h ({int(tw_late):02d}:{int((tw_late % 1) * 60):02d})")
        print(f"  Service duration: {service_time_hours:6.2f}h ({int(service_time_hours * 60)} minutes)")
        print(f"  Actual departure: {departure:6.2f}h ({int(departure):02d}:{int((departure % 1) * 60):02d})")
        print(f"  Status:           {detail}")
        print()

    # Summary for this route
    print("-" * 130)
    print(f"{'ROUTE SUMMARY':<130}")
    print("-" * 130)

    on_time = 0
    early = 0
    late = 0

    for i, node in enumerate(truck_route.nodes):
        if node == 0:
            continue

        arrival = truck_route.arrival[i]
        tw_early, tw_late = problem.nodes[node].time_window

        if arrival < tw_early:
            early += 1
        elif arrival > tw_late - problem.service_time:
            late += 1
        else:
            on_time += 1

    total_customers = on_time + early + late
    print(f"Total customers: {total_customers}")
    print(f"  ✅ On time: {on_time} ({100*on_time/total_customers if total_customers > 0 else 0:.0f}%)")
    print(f"  🔴 Too early: {early} ({100*early/total_customers if total_customers > 0 else 0:.0f}%)")
    print(f"  🔴 Too late: {late} ({100*late/total_customers if total_customers > 0 else 0:.0f}%)")

    # Timeline visualization
    print("\n" + "─" * 130)
    print(f"{'TIMELINE VISUALIZATION':<130}")
    print("─" * 130 + "\n")

    print("Time (24h format):")
    print("0h    3h    6h    9h    12h   15h   18h   21h   24h")
    print("|-----|-----|-----|-----|-----|-----|-----|-----|")

    for i, node in enumerate(truck_route.nodes):
        if node == 0:
            continue

        arrival = truck_route.arrival[i]
        departure = truck_route.departure[i]
        tw_early, tw_late = problem.nodes[node].time_window

        # Create timeline bar (24 * 6 characters = 144 chars for 24 hours)
        timeline = [" "] * 144
        chars_per_hour = 6

        # Mark time window
        tw_early_pos = int(tw_early * chars_per_hour)
        tw_late_pos = int(tw_late * chars_per_hour)
        for pos in range(tw_early_pos, min(tw_late_pos, 144)):
            timeline[pos] = "▓"

        # Mark arrival
        arrival_pos = int(arrival * chars_per_hour)
        timeline[arrival_pos] = "A"

        # Mark departure
        departure_pos = int(departure * chars_per_hour)
        timeline[departure_pos] = "D"

        timeline_str = "".join(timeline)
        print(f"Customer {node:2d}: {timeline_str}")

    print("\nLegend:")
    print("  ▓ = Time window is open")
    print("  A = Vehicle arrives")
    print("  D = Vehicle departs")

print("\n" + "=" * 130)
print("OVERALL FEASIBILITY")
print("=" * 130)

total_served = sum(len(route.nodes) - 2 for route, _ in solution.routes)
total_feasible = 0

for route, _ in solution.routes:
    for i, node in enumerate(route.nodes):
        if node == 0:
            continue
        arrival = route.arrival[i]
        tw_early, tw_late = problem.nodes[node].time_window
        if tw_early <= arrival <= tw_late:
            total_feasible += 1

print(f"\nTotal customers served: {total_served}")
print(f"Customers within time window: {total_feasible}")
print(f"Customers outside time window: {total_served - total_feasible}")
print(f"Feasibility rate: {100*total_feasible/total_served if total_served > 0 else 0:.1f}%")

print(f"\n{'Note:':<10} This is random initial population from GA initialization.")
print(f"{'':10} Time window violations are EXPECTED and will be IMPROVED by GA evolution.")

print("\n" + "=" * 130)
