"""Show a simple, easy-to-understand Individual and Solution example."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'baselines'))

from pathlib import Path
from baselines.problem import Problem, Individual
from baselines.utils import decode, init_population, cal_fitness

def show_simple_example():
    """Display a simple Individual and Solution with visual diagrams."""

    test_dir = Path(__file__).parent / "data" / "generated" / "data" / "N10"
    test_file = list(test_dir.glob("*.json"))[0]
    problem = Problem(str(test_file))

    print("=" * 110)
    print("SIMPLE EXAMPLE: Individual → Solution")
    print("=" * 110)

    # Create a simple individual
    indi = init_population(1, 99, problem)[0]

    # Modify it to have a simpler structure (for clarity)
    # Reset to mostly truck deliveries
    indi.chromosome[1] = [0] * len(indi.chromosome[1])
    # Add just one drone delivery for illustration
    indi.chromosome[1][2] = 1

    solution = decode(indi, problem)
    fitness, service_rate, cost = cal_fitness(problem, indi)

    n_nodes = len(problem.nodes)
    perm = indi.chromosome[0]
    mask = indi.chromosome[1]

    print("\n" + "=" * 110)
    print("STEP 1: CHROMOSOME (Individual)")
    print("=" * 110)
    print(f"""
The chromosome is the solution representation for the genetic algorithm.
It consists of TWO layers:

  Layer 1 (Permutation): Order of customers and vehicle delimiters
  Layer 2 (Mask):        0=truck delivery, 1=drone delivery, -1=drone pickup

Problem:
  - 11 nodes: 1 depot (node 0) + 10 customers (nodes 1-10)
  - 2 vehicles: Delimiters at nodes 11 and 12 to mark vehicle boundaries
  - Chromosome length: 11 (10 customers + 1 delimiter shown)

""")

    print("┌─ PERMUTATION ─────────────────────────────────────────────────────────────┐")
    print(f"│ {perm}")
    print("└────────────────────────────────────────────────────────────────────────────┘")

    print("\n┌─ DELIVERY MASK (0=truck, 1=drone, -1=pickup) ──────────────────────────────┐")
    print(f"│ {mask}")
    print("└────────────────────────────────────────────────────────────────────────────┘")

    print("\n┌─ INTERPRETATION ───────────────────────────────────────────────────────────┐")
    for i, (node, m) in enumerate(zip(perm, mask)):
        if node >= n_nodes:
            print(f"│ [{i:2d}] Node {node:2d}  → FLEET DELIMITER (vehicle boundary)")
        else:
            delivery = "🚚 TRUCK" if m == 0 else "🚁 DRONE"
            print(f"│ [{i:2d}] Node {node:2d}  → {delivery}")
    print("└────────────────────────────────────────────────────────────────────────────┘")

    print("\n" + "=" * 110)
    print("STEP 2: SOLUTION (Decoded Routes)")
    print("=" * 110)
    print(f"""
The decode() function converts the chromosome into a Solution:
  1. Repairs the chromosome (constraint checking)
  2. Partitions by delimiters (separates routes by vehicle)
  3. Creates truck routes and drone trips for each partition
  4. Returns valid routes with timing information

Metrics:
  - Fitness: {fitness:.2f}
  - Service Rate: {service_rate:.4f} ({int(service_rate * len(problem.nodes))}/{len(problem.nodes)} customers)
  - Cost: {cost:.2f}

""")

    print(f"Number of routes: {len(solution.routes)}\n")

    for route_idx, (truck_route, drone_trips) in enumerate(solution.routes):
        print(f"┌─ ROUTE {route_idx} ─────────────────────────────────────────────────────────────┐")

        # Draw truck route as a visual path
        truck_nodes_str = " → ".join(
            "DEPOT" if n == 0 else str(n) for n in truck_route.nodes
        )
        print(f"│ Truck Route (Vehicle {route_idx}): {truck_nodes_str}")
        print(f"│   Customers served by truck: {len(truck_route.nodes) - 2}")

        if drone_trips:
            print(f"│   Drone trips: {len(drone_trips)}")
            for trip_idx, trip in enumerate(drone_trips):
                launch, service, land = trip.nodes
                print(f"│     Trip {trip_idx}: LAUNCH at {launch} → SERVICE at {service} → LAND at {land}")
                print(f"│             Customers served by drone: 1")
        else:
            print(f"│   Drone trips: 0")

        total = (len(truck_route.nodes) - 2) + sum(len(t.nodes) - 2 for t in drone_trips)
        print(f"│   Total customers served: {total}")
        print("└────────────────────────────────────────────────────────────────────────────┘")

    print("\n" + "=" * 110)
    print("KEY CONCEPTS")
    print("=" * 110)
    print("""
1. INDIVIDUAL (Chromosome):
   - What: Permutation + mask representing a solution
   - Purpose: Genetic algorithm operates on chromosomes
   - Encoding: Order of visits + delivery method per customer

2. SOLUTION (Decoded Routes):
   - What: Actual routes with timing, feasibility checks applied
   - Purpose: Evaluate quality of a solution
   - Includes: Truck routes + drone trips with arrival/departure times

3. ROUTE:
   - Truck Route: [DEPOT → C₁ → C₂ → ... → DEPOT]
   - Drone Trip: [Launch point → Customer → Land point]
   - Each trip respects time windows, capacity, and system constraints

4. METRICS:
   - Fitness: Overall objective value (lower is better)
   - Service Rate: % of customers successfully served
   - Cost: Total distance/time cost of the solution

5. DRONE TRIP STRUCTURE (FIXED):
   ✅ CORRECT: [Launch=10, Service=4, Land=2]
     - Drone launches from truck at node 10
     - Delivers to customer 4
     - Lands back on truck at node 2

   ❌ WRONG: [Launch=10, Svc=4, Land=2, Svc=8, Land=6]
     - Physically impossible: drone can't fly from land point to next service
     - (This bug has been fixed in our version)

""")
    print("=" * 110)

if __name__ == "__main__":
    show_simple_example()
