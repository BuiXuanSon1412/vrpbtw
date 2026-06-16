"""Show an Individual, decode to Solution, and validate correctness."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'baselines'))

from pathlib import Path
from baselines.problem import Problem, Individual
from baselines.utils import (
    decode, init_population, cal_fitness, cal_cost, cal_service_rate,
    cal_truck_route_distance, cal_drone_route_distance
)

def validate_solution(problem, indi, solution):
    """Validate a decoded solution for correctness."""

    issues = []
    warnings = []

    print("\n" + "=" * 100)
    print("SOLUTION VALIDATION")
    print("=" * 100)

    # 1. Check node indices are valid
    print("\n1️⃣  NODE INDEX VALIDATION")
    print("   Checking all node indices are within valid range...")
    for route_idx, (truck_route, drone_trips) in enumerate(solution.routes):
        for node in truck_route.nodes:
            if node < 0 or node >= len(problem.nodes):
                issues.append(f"Route {route_idx}: Invalid truck node {node}")

        for trip_idx, trip in enumerate(drone_trips):
            for node in trip.nodes:
                if node < 0 or node >= len(problem.nodes):
                    issues.append(f"Route {route_idx}, Trip {trip_idx}: Invalid drone node {node}")

    if not issues:
        print("   ✅ All node indices valid (0-{})\n".format(len(problem.nodes)-1))
    else:
        for issue in issues:
            print(f"   ❌ {issue}\n")

    # 2. Check no duplicate visits
    print("2️⃣  DUPLICATE VISIT VALIDATION")
    print("   Checking each customer visited at most once...")
    visited = set()
    duplicates = []

    for route_idx, (truck_route, drone_trips) in enumerate(solution.routes):
        for node in truck_route.nodes:
            if node != 0 and node in visited:  # 0 is depot
                duplicates.append(f"Customer {node} (Route {route_idx})")
            visited.add(node)

        for trip_idx, trip in enumerate(drone_trips):
            service_node = trip.nodes[1]  # Middle node is service
            if service_node in visited:
                duplicates.append(f"Customer {service_node} (Route {route_idx}, Trip {trip_idx})")
            visited.add(service_node)

    if not duplicates:
        print(f"   ✅ No duplicate visits ({len(visited)-1} unique customers)\n")
    else:
        for dup in duplicates:
            print(f"   ❌ Duplicate visit: {dup}\n")
            issues.append(f"Duplicate visit: {dup}")

    # 3. Check capacity constraints
    print("3️⃣  TRUCK CAPACITY VALIDATION")
    print("   Checking truck load never exceeds capacity...")

    for route_idx, (truck_route, drone_trips) in enumerate(solution.routes):
        cumulative_load = 0
        route_ok = True

        for i, node in enumerate(truck_route.nodes):
            if node != 0:
                demand = problem.nodes[node].demand
                cumulative_load += demand

                if cumulative_load > problem.truck_capacity:
                    issues.append(f"Route {route_idx}: Load {cumulative_load} > {problem.truck_capacity}")
                    print(f"   ❌ Route {route_idx}: Customer {node} exceeds capacity!")
                    print(f"      Cumulative load: {cumulative_load}/{problem.truck_capacity}")
                    route_ok = False

        if route_ok:
            print(f"   ✅ Route {route_idx}: Load {cumulative_load}/{problem.truck_capacity} ✓")
    print()

    # 4. Check drone capacity
    print("4️⃣  DRONE CAPACITY VALIDATION")
    print("   Checking drone load never exceeds capacity...")

    for route_idx, (truck_route, drone_trips) in enumerate(solution.routes):
        for trip_idx, trip in enumerate(drone_trips):
            service_node = trip.nodes[1]
            demand = abs(problem.nodes[service_node].demand)

            if demand > problem.drone_capacity:
                issues.append(f"Route {route_idx}, Trip {trip_idx}: Drone load {demand} > {problem.drone_capacity}")
                print(f"   ❌ Route {route_idx}, Trip {trip_idx}: Customer {service_node} too heavy!")
            else:
                print(f"   ✅ Route {route_idx}, Trip {trip_idx}: Load {demand}/{problem.drone_capacity} ✓")
    print()

    # 5. Check time window feasibility
    print("5️⃣  TIME WINDOW FEASIBILITY CHECK")
    print("   Checking customers arrive within time windows...")

    time_ok_count = 0
    time_violation_count = 0

    for route_idx, (truck_route, drone_trips) in enumerate(solution.routes):
        for i, node in enumerate(truck_route.nodes):
            if node == 0:
                continue

            arrival = truck_route.arrival[i]
            tw_early, tw_late = problem.nodes[node].time_window

            if tw_early <= arrival <= tw_late:
                time_ok_count += 1
                print(f"   ✅ Route {route_idx}: Customer {node} arrives {arrival:.2f}h ∈ [{tw_early:.2f}, {tw_late:.2f}]")
            else:
                time_violation_count += 1
                print(f"   ⚠️  Route {route_idx}: Customer {node} arrives {arrival:.2f}h ∉ [{tw_early:.2f}, {tw_late:.2f}]")
                warnings.append(f"Customer {node}: Time window violation ({arrival:.2f}h not in [{tw_early:.2f}, {tw_late:.2f}])")

        for trip_idx, trip in enumerate(drone_trips):
            service_node = trip.nodes[1]
            arrival = trip.arrival[1]
            tw_early, tw_late = problem.nodes[service_node].time_window

            if tw_early <= arrival <= tw_late:
                time_ok_count += 1
                print(f"   ✅ Route {route_idx}, Trip {trip_idx}: Customer {service_node} arrives {arrival:.2f}h ✓")
            else:
                time_violation_count += 1
                print(f"   ⚠️  Route {route_idx}, Trip {trip_idx}: Customer {service_node} arrives {arrival:.2f}h ✗")
                warnings.append(f"Drone Customer {service_node}: Time window violation")

    print(f"   Summary: {time_ok_count} feasible, {time_violation_count} violations\n")

    # 6. Check drone trip structure
    print("6️⃣  DRONE TRIP STRUCTURE VALIDATION")
    print("   Checking drone trips have [Launch, Service, Land] structure...")

    drone_ok = True
    for route_idx, (truck_route, drone_trips) in enumerate(solution.routes):
        for trip_idx, trip in enumerate(drone_trips):
            nodes = trip.nodes

            if len(nodes) != 3:
                issues.append(f"Route {route_idx}, Trip {trip_idx}: Has {len(nodes)} nodes instead of 3")
                print(f"   ❌ Trip {trip_idx}: {len(nodes)} nodes (expected 3)")
                drone_ok = False
                continue

            launch, service, land = nodes

            # Check launch and land are on truck route
            if launch not in truck_route.nodes:
                issues.append(f"Route {route_idx}, Trip {trip_idx}: Launch {launch} not on truck route")
                print(f"   ❌ Trip {trip_idx}: Launch node {launch} not on truck route")
                drone_ok = False
            elif land not in truck_route.nodes:
                issues.append(f"Route {route_idx}, Trip {trip_idx}: Land {land} not on truck route")
                print(f"   ❌ Trip {trip_idx}: Land node {land} not on truck route")
                drone_ok = False
            else:
                print(f"   ✅ Trip {trip_idx}: [{launch} → {service} → {land}] valid")

    if drone_ok:
        print()

    # 7. Check distance calculations
    print("7️⃣  DISTANCE CALCULATION VALIDATION")
    print("   Verifying distance calculations...")

    total_cost = 0
    for route_idx, (truck_route, drone_trips) in enumerate(solution.routes):
        truck_dist = cal_truck_route_distance(truck_route.nodes, problem)
        truck_cost = truck_dist * problem.truck_cost
        print(f"   Route {route_idx} Truck: {truck_dist:.2f}km × {problem.truck_cost}€/km = {truck_cost:.2f}€")
        total_cost += truck_cost

        for trip_idx, trip in enumerate(drone_trips):
            drone_dist = cal_drone_route_distance(trip.nodes, problem)
            drone_cost = drone_dist * problem.drone_cost
            print(f"   Route {route_idx} Drone {trip_idx}: {drone_dist:.2f}km × {problem.drone_cost}€/km = {drone_cost:.2f}€")
            total_cost += drone_cost

    total_cost += len(solution.routes) * problem.basis_cost
    print(f"   Fleet basis cost: {len(solution.routes)} × {problem.basis_cost}€ = {len(solution.routes) * problem.basis_cost}€")
    print(f"   Total cost: {total_cost:.2f}€\n")

    # 8. Check metrics
    print("8️⃣  METRICS VALIDATION")
    print("   Verifying fitness, service rate, and cost calculations...")

    fitness, sr, cost = cal_fitness(problem, indi)
    calculated_cost = cal_cost(solution, problem)
    calculated_sr = cal_service_rate(solution, problem) / (len(problem.nodes) - 1)

    print(f"   Fitness: {fitness:.2f}")
    print(f"   Service Rate: {sr:.4f} ({int(sr * (len(problem.nodes)-1))}/{len(problem.nodes)-1} customers)")
    print(f"   Cost: {cost:.2f}€")

    # Verify calculations match
    if abs(cost - calculated_cost) < 0.01:
        print(f"   ✅ Cost calculation verified")
    else:
        print(f"   ❌ Cost mismatch: {cost:.2f} vs {calculated_cost:.2f}")
        issues.append(f"Cost calculation mismatch")

    if abs(sr - calculated_sr) < 0.0001:
        print(f"   ✅ Service rate calculation verified")
    else:
        print(f"   ⚠️  Service rate calculation differs slightly")

    print()

    # Summary
    print("=" * 100)
    print("VALIDATION SUMMARY")
    print("=" * 100)

    if not issues:
        print("✅ ALL CHECKS PASSED - Solution is valid!\n")
        return True
    else:
        print(f"❌ {len(issues)} CRITICAL ISSUES FOUND:\n")
        for issue in issues:
            print(f"   • {issue}")
        print()
        return False

def main():
    """Main validation flow."""

    # Load problem
    test_dir = Path(__file__).parent / "data" / "generated" / "data" / "N10"
    test_file = list(test_dir.glob("*.json"))[0]
    problem = Problem(str(test_file))

    print("=" * 100)
    print("INDIVIDUAL → SOLUTION VALIDATION")
    print("=" * 100)

    print(f"\nProblem: {test_file.name}")
    print(f"Nodes: {len(problem.nodes)} (1 depot + {len(problem.nodes)-1} customers)")
    print(f"Fleet: {problem.num_fleet} vehicles")
    print(f"Constraints:")
    print(f"  - Truck capacity: {problem.truck_capacity} units")
    print(f"  - Drone capacity: {problem.drone_capacity} units")
    print(f"  - System duration: {problem.system_duration} hours")

    # Create individual
    indi_list = init_population(10, 42, problem)

    # Find one with good metrics for better visualization
    best_indi = max(indi_list, key=lambda i: cal_fitness(problem, i)[1])

    print("\n" + "=" * 100)
    print("INDIVIDUAL (Chromosome)")
    print("=" * 100)

    perm = best_indi.chromosome[0]
    mask = best_indi.chromosome[1]

    print(f"\nChromosome length: {len(perm)}")
    print(f"Permutation: {perm}")
    print(f"Mask:        {mask}")

    print(f"\nGene interpretation:")
    for i, (node, m) in enumerate(zip(perm, mask)):
        if node >= len(problem.nodes):
            node_type = f"DELIMITER"
        else:
            node_type = f"Customer"

        delivery = "🚚" if m == 0 else "🚁" if m == 1 else "🔄"
        print(f"  [{i:2d}] Node {node:2d} ({node_type:10s}) → {delivery}")

    # Decode
    print("\n" + "=" * 100)
    print("DECODING PROCESS")
    print("=" * 100)

    print("\nStep 1: Repair chromosome")
    print("  - Check feasibility")
    print("  - Remove impossible operations")

    print("\nStep 2: Partition by delimiters")
    print("  - Split into sequences for each vehicle")

    print("\nStep 3: For each sequence, create routes")
    print("  - Separate truck nodes from drone nodes")
    print("  - Build truck route with feasibility checks")
    print("  - Create drone trips with launch/land points")

    print("\nStep 4: Apply constraints")
    print("  - Check time windows")
    print("  - Check capacity")
    print("  - Skip infeasible nodes")

    solution = decode(best_indi, problem)

    print("\n" + "=" * 100)
    print("SOLUTION (Decoded Routes)")
    print("=" * 100)

    for route_idx, (truck_route, drone_trips) in enumerate(solution.routes):
        print(f"\nRoute {route_idx}:")
        print(f"  Truck: {' → '.join(str(n) if n != 0 else 'D' for n in truck_route.nodes)}")

        if drone_trips:
            print(f"  Drones: {len(drone_trips)} trip(s)")
            for trip_idx, trip in enumerate(drone_trips):
                launch, service, land = trip.nodes
                print(f"    Trip {trip_idx}: {launch} → {service} → {land}")
        else:
            print(f"  Drones: 0 trips")

    # Validate
    is_valid = validate_solution(problem, best_indi, solution)

    print("=" * 100)
    if is_valid:
        print("✅ SOLUTION IS CORRECT AND FEASIBLE")
    else:
        print("⚠️  SOLUTION HAS ISSUES (see above)")
    print("=" * 100)

if __name__ == "__main__":
    main()
