import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + "/..")

from copy import deepcopy
from problem import Problem, Solution
from utils import init_population, decode
from population import Individual


def test_decode():
    """Test decode function using init_population to create an individual."""
    # Load a test problem
    print("\n" + "="*60)
    print("TEST: decode function")
    print("="*60)

    data_path = "/home/bxs/thesis/vrpbtw/data/generated/data/N10/S042_N10_C_R50.json"
    problem = Problem(data_path)
    print(f"\n1. Problem loaded:")
    print(f"   - Nodes: {len(problem.nodes)} (1 depot + {len(problem.nodes)-1} customers)")
    print(f"   - Fleet size: {problem.num_fleet}")
    print(f"   - Truck capacity: {problem.truck_capacity}")
    print(f"   - Drone capacity: {problem.drone_capacity}")

    # Initialize a population with 1 individual
    population = init_population(num_indi=1, seed=42, problem=problem)
    assert len(population) == 1, "Population should have 1 individual"
    individual = population[0]

    print(f"\n2. Population initialized:")
    print(f"   - Individuals: 1")
    print(f"   - Chromosome structure: [permutation, mask]")
    print(f"   - Permutation length: {len(individual.chromosome[0])}")
    print(f"   - Mask length: {len(individual.chromosome[1])}")

    # Decode the individual
    print(f"\n3. Decoding individual...")

    # Debug: Add tracing to decode
    from utils import partition, repair
    chro = individual.chromosome
    repaired = deepcopy(chro)
    repair(repaired, problem)
    seqs = partition(repaired, problem)

    print(f"   - Partitioned sequences: {len(seqs)}")
    for i, seq in enumerate(seqs):
        print(f"   - Seq {i+1}: {seq}")

    solution = decode(individual, problem)
    print(f"   ✓ Decode completed")

    # Verify the solution is valid
    assert isinstance(solution, Solution), "decode should return a Solution object"
    assert hasattr(solution, 'routes'), "Solution should have routes attribute"
    assert isinstance(solution.routes, list), "routes should be a list"

    print(f"\n4. Solution structure validation:")
    print(f"   - Solution type: {type(solution).__name__}")
    print(f"   - Routes: {len(solution.routes)}")

    # Verify routes structure
    total_served = 0
    for i, route in enumerate(solution.routes):
        assert len(route) == 2, "Each route should be a tuple (truck_route, drone_trips)"
        truck_route, drone_trips = route

        # Verify truck route
        assert hasattr(truck_route, 'nodes'), "truck_route should have nodes"
        assert hasattr(truck_route, 'arrival'), "truck_route should have arrival"
        assert hasattr(truck_route, 'departure'), "truck_route should have departure"
        assert len(truck_route.nodes) > 0, "truck_route should have at least depot"
        assert truck_route.nodes[0] == 0, "First node should be depot (0)"

        # Verify drone trips
        assert isinstance(drone_trips, list), "drone_trips should be a list"
        truck_nodes = len(truck_route.nodes) - 2 if len(truck_route.nodes) > 1 else 0
        drone_nodes = sum(len(trip.nodes) - 2 for trip in drone_trips)
        total_served += truck_nodes + drone_nodes

        print(f"\n   Route {i+1}:")
        print(f"   - Truck nodes: {truck_nodes}")
        print(f"   - Drone trips: {len(drone_trips)}")
        print(f"   - Drone nodes: {drone_nodes}")

        for trip in drone_trips:
            assert hasattr(trip, 'nodes'), "each trip should have nodes"
            assert hasattr(trip, 'arrival'), "each trip should have arrival"
            assert hasattr(trip, 'departure'), "each trip should have departure"

    print(f"\n5. Detailed Routes & Trips:")
    for i, route in enumerate(solution.routes):
        truck_route, drone_trips = route

        print(f"\n   ═══ Route {i+1} ═══")
        print(f"\n   TRUCK ROUTE:")
        print(f"   Nodes:     {truck_route.nodes}")
        print(f"   Arrival:   {[f'{t:.2f}' for t in truck_route.arrival]}")
        print(f"   Departure: {[f'{t:.2f}' for t in truck_route.departure]}")

        if drone_trips:
            print(f"\n   DRONE TRIPS: {len(drone_trips)}")
            for trip_idx, trip in enumerate(drone_trips):
                print(f"\n      Trip {trip_idx+1}:")
                print(f"      Nodes:     {trip.nodes}")
                print(f"      Arrival:   {[f'{t:.2f}' for t in trip.arrival]}")
                print(f"      Departure: {[f'{t:.2f}' for t in trip.departure]}")
        else:
            print(f"\n   DRONE TRIPS: None")

    print(f"\n6. Service Summary:")
    print(f"   - Total served: {total_served}")
    print(f"   - Total customers: {len(problem.nodes) - 1}")
    print(f"   - Service rate: {total_served}/{len(problem.nodes)-1}")

    print("\n" + "="*60)
    print("✓ decode test PASSED")
    print("="*60 + "\n")


if __name__ == "__main__":
    test_decode()
