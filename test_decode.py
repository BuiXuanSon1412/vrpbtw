"""Test suite for the decode function to check correctness."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'baselines'))

from pathlib import Path
from baselines.problem import Problem, Individual
from baselines.utils import (
    decode, partition, routing, cal_fitness,
    cal_service_rate, cal_cost, init_population
)

def test_decode_basic():
    """Test decode on a simple problem."""
    test_dir = Path(__file__).parent / "data" / "generated" / "data" / "N10"
    test_file = list(test_dir.glob("*.json"))[0]

    print(f"Loading problem from {test_file.name}")
    problem = Problem(str(test_file))

    print(f"Problem: {len(problem.nodes)} nodes, {problem.num_fleet} vehicles")
    print(f"Nodes: {[(n.id, n.demand) for n in problem.nodes[:3]]}...")

    # Create a simple individual
    indi_list = init_population(5, 42, problem)
    indi = indi_list[0]

    print(f"\nIndividual chromosome:")
    print(f"  Permutation: {indi.chromosome[0]}")
    print(f"  Mask: {indi.chromosome[1]}")

    # Decode
    solution = decode(indi, problem)
    print(f"\nDecoded solution:")
    print(f"  Number of routes: {len(solution.routes)}")

    for i, (truck_route, drone_trips) in enumerate(solution.routes):
        print(f"\n  Route {i}:")
        print(f"    Truck nodes: {truck_route.nodes}")
        print(f"    Truck arrivals: {[f'{t:.2f}' for t in truck_route.arrival]}")
        print(f"    Number of drone trips: {len(drone_trips)}")
        for j, trip in enumerate(drone_trips):
            print(f"      Trip {j}: {trip.nodes}")
            print(f"        Arrivals: {[f'{t:.2f}' for t in trip.arrival]}")
            print(f"        Departures: {[f'{t:.2f}' for t in trip.departure]}")

    # Check service rate calculation
    service_rate_raw = cal_service_rate(solution, problem)
    service_rate_norm = service_rate_raw / len(problem.nodes)
    cost = cal_cost(solution, problem)

    print(f"\nMetrics:")
    print(f"  Service rate (raw): {service_rate_raw}")
    print(f"  Service rate (normalized): {service_rate_norm:.4f}")
    print(f"  Cost: {cost:.2f}")

    # Check fitness calculation
    fitness, sr, c = cal_fitness(problem, indi)
    print(f"  Fitness: {fitness:.2f}")
    print(f"  SR from fitness: {sr:.4f}")
    print(f"  Cost from fitness: {c:.2f}")

def test_partition():
    """Test partition function for edge cases."""
    test_dir = Path(__file__).parent / "data" / "generated" / "data" / "N10"
    test_file = list(test_dir.glob("*.json"))[0]

    problem = Problem(str(test_file))
    n_node = len(problem.nodes)

    print(f"\nTesting partition function:")
    print(f"  Problem has {n_node} nodes")

    # Test case 1: Simple partition
    chro = ([1, 2, n_node, 3, 4], [0, 0, 0, 0, 0])
    seqs = partition(chro, problem)
    print(f"\n  Test 1: Simple partition")
    print(f"    Input: {chro}")
    print(f"    Output: {seqs}")
    assert len(seqs) == 2, f"Expected 2 sequences, got {len(seqs)}"
    assert seqs[0] == [(1, 0), (2, 0)], f"First sequence incorrect: {seqs[0]}"
    assert seqs[1] == [(3, 0), (4, 0)], f"Second sequence incorrect: {seqs[1]}"

    # Test case 2: Ending with delimiter
    chro = ([1, 2, 3, n_node], [0, 0, 0, 0])
    seqs = partition(chro, problem)
    print(f"\n  Test 2: Ending with delimiter")
    print(f"    Input: {chro}")
    print(f"    Output: {seqs}")
    assert len(seqs) == 1, f"Expected 1 sequence, got {len(seqs)}"

    # Test case 3: Multiple consecutive delimiters
    chro = ([1, 2, n_node, n_node+1, 3, 4], [0, 0, 0, 0, 0, 0])
    seqs = partition(chro, problem)
    print(f"\n  Test 3: Multiple consecutive delimiters")
    print(f"    Input: {chro}")
    print(f"    Output: {seqs}")
    print(f"    Number of sequences: {len(seqs)}")
    # Should skip empty sequences between consecutive delimiters

    print(f"\n  Partition tests passed!")

def test_drone_trip_structure():
    """Test drone trip node structure for correctness."""
    test_dir = Path(__file__).parent / "data" / "generated" / "data" / "N10"
    test_file = list(test_dir.glob("*.json"))[0]

    problem = Problem(str(test_file))

    print(f"\nTesting drone trip structure:")

    # Create multiple individuals and check
    indi_list = init_population(10, 42, problem)

    for idx, indi in enumerate(indi_list):
        solution = decode(indi, problem)

        for route_idx, (truck_route, drone_trips) in enumerate(solution.routes):
            for trip_idx, trip in enumerate(drone_trips):
                nodes = trip.nodes

                # Check if nodes make sense
                if len(nodes) > 1:
                    # First node should be launch point (on truck route)
                    if nodes[0] not in truck_route.nodes:
                        print(f"  WARNING: Drone trip {route_idx}.{trip_idx} launches from {nodes[0]} which is not in truck route {truck_route.nodes}")

                    # Check for problematic patterns like [launch, service, land, service, land]
                    # which would indicate incorrect re-launch handling
                    for i in range(1, len(nodes)-1):
                        if nodes[i] == nodes[i+1]:
                            print(f"  WARNING: Drone trip {route_idx}.{trip_idx} has consecutive duplicate nodes: {nodes}")

                    # Last node should be land point
                    if nodes[-1] not in truck_route.nodes:
                        print(f"  WARNING: Drone trip {route_idx}.{trip_idx} lands at {nodes[-1]} which is not in truck route {truck_route.nodes}")


if __name__ == "__main__":
    print("=" * 80)
    print("Testing decode function")
    print("=" * 80)

    try:
        test_partition()
        test_decode_basic()
        test_drone_trip_structure()
        print("\n" + "=" * 80)
        print("All tests completed!")
        print("=" * 80)
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
