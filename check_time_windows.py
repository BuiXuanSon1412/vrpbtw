"""Check if customers are served within their time windows."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'baselines'))

from pathlib import Path
from baselines.problem import Problem
from baselines.utils import decode, init_population

def check_time_windows():
    """Validate time window feasibility for decoded solutions."""

    test_dir = Path(__file__).parent / "data" / "generated" / "data" / "N10"
    test_file = list(test_dir.glob("*.json"))[0]
    problem = Problem(str(test_file))

    print("=" * 100)
    print("TIME WINDOW FEASIBILITY CHECK")
    print("=" * 100)

    print(f"\nProblem: {test_file.name}")
    print(f"Nodes: {len(problem.nodes)} (1 depot + {len(problem.nodes)-1} customers)")
    print(f"System duration: {problem.system_duration} hours")

    # Create test individuals
    indi_list = init_population(10, 42, problem)

    all_feasible = True
    violations = []

    for indi_idx, indi in enumerate(indi_list):
        solution = decode(indi, problem)

        print(f"\n{'─' * 100}")
        print(f"Individual {indi_idx}")
        print(f"{'─' * 100}")

        for route_idx, (truck_route, drone_trips) in enumerate(solution.routes):
            print(f"\nRoute {route_idx}:")

            # Check truck route time windows
            print(f"  Truck Route: {truck_route.nodes}")
            route_feasible = True

            for i, node in enumerate(truck_route.nodes):
                if node == 0:
                    continue  # Skip depot

                arrival = truck_route.arrival[i]

                # Get customer info
                customer = problem.nodes[node]
                tw_early, tw_late = customer.time_window

                # Check if within time window
                in_window = tw_early <= arrival <= tw_late

                status = "✅" if in_window else "❌"
                print(f"    {status} Customer {node:2d}: Arrive {arrival:7.2f}h | TW [{tw_early:7.2f}, {tw_late:7.2f}]")

                if not in_window:
                    route_feasible = False
                    all_feasible = False
                    violations.append({
                        'type': 'truck',
                        'individual': indi_idx,
                        'route': route_idx,
                        'customer': node,
                        'arrival': arrival,
                        'tw_early': tw_early,
                        'tw_late': tw_late,
                        'violation': f"Outside time window [{tw_early:.2f}, {tw_late:.2f}]"
                    })

            # Check drone trips
            for trip_idx, trip in enumerate(drone_trips):
                launch, service, land = trip.nodes
                service_arrival = trip.arrival[1]  # Service node arrival

                # Get customer info
                customer = problem.nodes[service]
                tw_early, tw_late = customer.time_window

                in_window = tw_early <= service_arrival <= tw_late

                status = "✅" if in_window else "❌"
                print(f"    {status} Drone Trip {trip_idx}: Customer {service:2d} Arrive {service_arrival:7.2f}h | TW [{tw_early:7.2f}, {tw_late:7.2f}]")

                if not in_window:
                    route_feasible = False
                    all_feasible = False
                    violations.append({
                        'type': 'drone',
                        'individual': indi_idx,
                        'route': route_idx,
                        'trip': trip_idx,
                        'customer': service,
                        'arrival': service_arrival,
                        'tw_early': tw_early,
                        'tw_late': tw_late,
                        'violation': f"Outside time window [{tw_early:.2f}, {tw_late:.2f}]"
                    })

            # Summary for this route
            served_customers = (len(truck_route.nodes) - 2) + sum(len(t.nodes) - 2 for t in drone_trips)
            print(f"  Route feasible: {'✅ YES' if route_feasible else '❌ NO'}")
            print(f"  Customers served: {served_customers}")

    print("\n" + "=" * 100)
    print("SUMMARY")
    print("=" * 100)

    if all_feasible:
        print("✅ ALL TIME WINDOWS SATISFIED")
        print(f"   Checked {len(indi_list)} individuals")
        print(f"   All decoded solutions respect customer time windows")
    else:
        print(f"❌ TIME WINDOW VIOLATIONS FOUND: {len(violations)}")
        print(f"\nViolations:")
        for v in violations:
            if v['type'] == 'truck':
                print(f"  - Individual {v['individual']}, Route {v['route']}")
                print(f"    Truck → Customer {v['customer']}")
                print(f"    Arrival: {v['arrival']:.2f}h, Window: [{v['tw_early']:.2f}, {v['tw_late']:.2f}]")
                print(f"    {v['violation']}\n")
            else:
                print(f"  - Individual {v['individual']}, Route {v['route']}, Trip {v['trip']}")
                print(f"    Drone → Customer {v['customer']}")
                print(f"    Arrival: {v['arrival']:.2f}h, Window: [{v['tw_early']:.2f}, {v['tw_late']:.2f}]")
                print(f"    {v['violation']}\n")

    print("=" * 100)

    # Additional analysis: time window usage
    print("\nTIME WINDOW ANALYSIS")
    print("=" * 100)

    for node_idx in range(1, len(problem.nodes)):
        customer = problem.nodes[node_idx]
        tw_early, tw_late = customer.time_window
        tw_width = tw_late - tw_early

        print(f"Customer {node_idx:2d}: TW width = {tw_width:7.2f}h [{tw_early:7.2f}, {tw_late:7.2f}]")

if __name__ == "__main__":
    check_time_windows()
