"""Display a detailed Individual and its decoded Solution."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'baselines'))

from pathlib import Path
from baselines.problem import Problem, Individual
from baselines.utils import (
    decode, init_population, cal_fitness, cal_cost, cal_service_rate
)

def show_individual_solution():
    """Display an Individual chromosome and its decoded Solution."""

    # Load problem
    test_dir = Path(__file__).parent / "data" / "generated" / "data" / "N10"
    test_file = list(test_dir.glob("*.json"))[0]
    problem = Problem(str(test_file))

    print("=" * 100)
    print("INDIVIDUAL AND SOLUTION EXAMPLE")
    print("=" * 100)

    print(f"\nPROBLEM INSTANCE: {test_file.name}")
    print(f"  Nodes: {len(problem.nodes)} (1 depot + {len(problem.nodes)-1} customers)")
    print(f"  Fleet: {problem.num_fleet} vehicles (trucks)")
    print(f"  Truck capacity: {problem.truck_capacity} units")
    print(f"  Drone capacity: {problem.drone_capacity} units")

    # Create individuals
    indi_list = init_population(30, 42, problem)

    # Find one with drone trips for better visualization
    indi_with_drones = None
    for indi in indi_list:
        solution = decode(indi, problem)
        has_drones = any(len(trips) > 0 for _, trips in solution.routes)
        if has_drones:
            indi_with_drones = indi
            break

    if indi_with_drones is None:
        indi_with_drones = indi_list[0]

    print("\n" + "=" * 100)
    print("1. INDIVIDUAL (Chromosome Representation)")
    print("=" * 100)

    n_nodes = len(problem.nodes)
    perm = indi_with_drones.chromosome[0]
    mask = indi_with_drones.chromosome[1]

    print(f"\nChromosome length: {len(perm)}")
    print(f"  Permutation: {perm}")
    print(f"  Mask:        {mask}")

    print(f"\nChromosome interpretation:")
    print(f"  Valid node indices:    1-{n_nodes-1} (customers)")
    print(f"  Fleet delimiters:      {n_nodes}-{n_nodes + problem.num_fleet - 1} (vehicle boundaries)")
    print()
    for i, (node, m) in enumerate(zip(perm, mask)):
        if node >= n_nodes:
            node_type = f"DELIMITER (vehicle {node - n_nodes})"
        else:
            node_type = f"Customer {node}"

        if m == 0:
            service = "TRUCK DELIVERY"
        elif m == 1:
            service = "DRONE DELIVERY (linehaul)"
        elif m == -1:
            service = "DRONE PICKUP (backhaul)"
        else:
            service = f"Unknown mask {m}"

        print(f"  Position {i:2d}: Node {node:2d} ({node_type:30s}) → {service}")

    print("\n" + "=" * 100)
    print("2. SOLUTION (Decoded Routes)")
    print("=" * 100)

    solution = decode(indi_with_drones, problem)

    fitness, service_rate, cost = cal_fitness(problem, indi_with_drones)

    print(f"\nMetrics:")
    print(f"  Fitness: {fitness:.2f}")
    print(f"  Service rate: {service_rate:.4f} ({int(service_rate * len(problem.nodes))}/{len(problem.nodes)} customers served)")
    print(f"  Cost: {cost:.2f}")

    print(f"\nNumber of routes: {len(solution.routes)}")

    for route_idx, (truck_route, drone_trips) in enumerate(solution.routes):
        print(f"\n{'─' * 100}")
        print(f"ROUTE {route_idx}")
        print(f"{'─' * 100}")

        # Truck route
        print(f"\nTruck Route (vehicle {route_idx}):")
        print(f"  Nodes: {truck_route.nodes}")
        print(f"  Sequence breakdown:")
        for i, node in enumerate(truck_route.nodes):
            if node == 0:
                node_name = "DEPOT"
            else:
                node_name = f"Customer {node}"

            arrival = truck_route.arrival[i]
            departure = truck_route.departure[i]
            print(f"    {i}: {node_name:20s} | Arrive: {arrival:7.2f}h | Depart: {departure:7.2f}h")

        # Drone trips
        if drone_trips:
            print(f"\n  Associated Drone Trips: {len(drone_trips)}")
            for trip_idx, drone_trip in enumerate(drone_trips):
                print(f"\n  Drone Trip {trip_idx}:")
                print(f"    Nodes: {drone_trip.nodes}")

                launch_node = drone_trip.nodes[0]
                service_node = drone_trip.nodes[1]
                land_node = drone_trip.nodes[2]

                print(f"    Structure: [Launch={launch_node} → Service={service_node} → Land={land_node}]")
                print(f"    Details:")
                for i, node in enumerate(drone_trip.nodes):
                    if node == 0:
                        node_name = "DEPOT"
                    else:
                        node_name = f"Customer {node}"

                    arrival = drone_trip.arrival[i]
                    departure = drone_trip.departure[i]

                    if i == 0:
                        event = "LAUNCH"
                    elif i == 1:
                        event = "SERVICE"
                    else:
                        event = "LAND"

                    print(f"      {event:7s} at {node_name:20s} | Arrive: {arrival:7.2f}h | Depart: {departure:7.2f}h")

        else:
            print(f"\n  Associated Drone Trips: 0 (all deliveries by truck)")

        # Route statistics
        customers_served_truck = len(truck_route.nodes) - 2
        customers_served_drone = sum(len(trip.nodes) - 2 for trip in drone_trips)
        total_served = customers_served_truck + customers_served_drone

        print(f"\n  Route statistics:")
        print(f"    Customers served by truck: {customers_served_truck}")
        print(f"    Customers served by drone: {customers_served_drone}")
        print(f"    Total served: {total_served}")

    print("\n" + "=" * 100)
    print("3. NODE INFORMATION")
    print("=" * 100)

    print(f"\nCustomer Details (Depot + Customers):")
    for node in problem.nodes:
        print(f"  Node {node.id}: Demand={node.demand:3d} | Coord={node.coord} | Time window={node.time_window}")

    print("\n" + "=" * 100)

if __name__ == "__main__":
    show_individual_solution()
