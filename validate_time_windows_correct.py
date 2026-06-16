"""Correct time window validation: vehicles can arrive early and wait."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'baselines'))

from pathlib import Path
from baselines.problem import Problem
from baselines.utils import decode, init_population, cal_fitness

test_dir = Path(__file__).parent / "data" / "generated" / "data" / "N10"
test_file = list(test_dir.glob("*.json"))[0]
problem = Problem(str(test_file))

indi_list = init_population(10, 42, problem)
best_indi = max(indi_list, key=lambda i: cal_fitness(problem, i)[1])
solution = decode(best_indi, problem)

print("=" * 130)
print("TIME WINDOW FEASIBILITY CHECK (CORRECT)")
print("=" * 130)

print(f"""
CONSTRAINT: Service must START before window closes
  arrival_time ≤ tw_close[i] - service_time[i]

This means:
  ✅ Vehicle CAN arrive early and wait
  ✅ Vehicle MUST start service by deadline
  ✅ Service completes at: arrival + service_time
""")

print(f"Service time per customer: {problem.service_time * 60:.1f} minutes ({problem.service_time:.4f} hours)")
print(f"System duration: {problem.system_duration} hours\n")

all_feasible = True
summary = {"on_time": 0, "early_but_feasible": 0, "infeasible": 0}

for route_idx, (truck_route, drone_trips) in enumerate(solution.routes):
    print(f"\n{'=' * 130}")
    print(f"ROUTE {route_idx}")
    print(f"{'=' * 130}\n")

    print(f"{'Stop':<6} {'Node':<6} {'Arrival':<12} {'Service Deadline':<20} {'Feasible?':<12}")
    print(f"{'':6} {'':6} {'[Hour]':<12} {'[Hour]':<20} {'[Y/N]':<12}")
    print("-" * 130)

    for i, node in enumerate(truck_route.nodes):
        if node == 0:
            print(f"{i:<6} {node:<6} {'DEPOT':<12} {'-':<20} {'N/A':<12}")
            continue

        arrival = truck_route.arrival[i]
        tw_early, tw_late = problem.nodes[node].time_window
        service_deadline = tw_late - problem.service_time

        # Check feasibility: arrival <= service_deadline
        is_feasible = arrival <= service_deadline

        if is_feasible:
            # Check if early
            if arrival < tw_early:
                status = "✅ EARLY (waits)"
                summary["early_but_feasible"] += 1
                wait_time = tw_early - arrival
            else:
                status = "✅ ON TIME"
                summary["on_time"] += 1
                wait_time = 0
        else:
            status = "❌ INFEASIBLE"
            summary["infeasible"] += 1
            all_feasible = False
            wait_time = None

        print(f"{i:<6} {node:<6} {arrival:<12.2f} {service_deadline:<20.2f} {status:<12}")

    print("\n" + "-" * 130)
    print("DETAILED ANALYSIS:\n")

    for i, node in enumerate(truck_route.nodes):
        if node == 0:
            continue

        arrival = truck_route.arrival[i]
        departure = truck_route.departure[i]
        tw_early, tw_late = problem.nodes[node].time_window
        service_deadline = tw_late - problem.service_time
        service_start = min(arrival, service_deadline) if arrival <= service_deadline else arrival
        service_end = service_start + problem.service_time

        print(f"Customer {node}:")
        print(f"  Arrival time:        {arrival:.2f}h ({int(arrival):02d}:{int((arrival % 1) * 60):02d})")
        print(f"  Time window:         [{tw_early:.2f}h, {tw_late:.2f}h]")
        print(f"  Service deadline:    {service_deadline:.2f}h ({int(service_deadline):02d}:{int((service_deadline % 1) * 60):02d})")
        print(f"  Service time:        {problem.service_time * 60:.0f} minutes")

        if arrival <= service_deadline:
            print(f"  Service starts at:   {max(arrival, tw_early):.2f}h ({int(max(arrival, tw_early)):02d}:{int((max(arrival, tw_early) % 1) * 60):02d})")
            print(f"  Service ends at:     {max(arrival, tw_early) + problem.service_time:.2f}h")

            if arrival < tw_early:
                wait_time = (tw_early - arrival) * 60
                print(f"  Wait time:           {wait_time:.0f} minutes (arrives early)")
            else:
                print(f"  Wait time:           0 minutes (arrives within window)")

            print(f"  Status:              ✅ FEASIBLE (service can start by deadline)")
        else:
            print(f"  Status:              ❌ INFEASIBLE (arrives too late to start service)")
            print(f"  Too late by:         {(arrival - service_deadline) * 60:.0f} minutes")

        print()

    # Route summary
    served = len(truck_route.nodes) - 2
    feasible_on_route = sum(
        1 for j, node in enumerate(truck_route.nodes)
        if node != 0 and truck_route.arrival[j] <= problem.nodes[node].time_window[1] - problem.service_time
    )

    print(f"Route {route_idx} Summary:")
    print(f"  Customers served: {served}")
    print(f"  Feasible deliveries: {feasible_on_route}/{served}")
    print(f"  Feasibility rate: {100 * feasible_on_route / served if served > 0 else 0:.0f}%")

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
        tw_late = problem.nodes[node].time_window[1]
        service_deadline = tw_late - problem.service_time
        if arrival <= service_deadline:
            total_feasible += 1

print(f"\nTotal customers served: {total_served}")
print(f"Customers with feasible time windows: {total_feasible}")
print(f"Customers infeasible: {total_served - total_feasible}")
print(f"Overall feasibility: {100 * total_feasible / total_served if total_served > 0 else 0:.1f}%")

print(f"\nBreakdown:")
print(f"  On time (no wait): {summary['on_time']}")
print(f"  Early (with wait): {summary['early_but_feasible']}")
print(f"  Infeasible: {summary['infeasible']}")

print("\n" + "=" * 130)
if all_feasible:
    print("✅ ALL CUSTOMERS ARE TIME-FEASIBLE!")
    print("   Vehicles may arrive early and wait, but all can serve within deadlines.")
else:
    print(f"⚠️  {summary['infeasible']} customers are INFEASIBLE")
    print("   (arrive too late to start service before window closes)")
print("=" * 130)
