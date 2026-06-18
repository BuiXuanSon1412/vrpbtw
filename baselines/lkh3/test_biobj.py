"""
Test script for bi-objective LKH3 solver.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from biobj_solver import BiObjLKH3Solver


def test_solver():
    """Test bi-objective solver on a small instance."""
    # Use a small test instance
    data_path = "../data/generated/data/N10/S042_N10_C_R50.json"

    print("="*60)
    print("TEST: BiObjLKH3Solver")
    print("="*60)

    # Initialize solver
    try:
        solver = BiObjLKH3Solver(data_path)
        print(f"✓ Solver initialized")
        print(f"  Problem: {os.path.basename(data_path)}")
        print(f"  Customers: {solver.num_customers}")
        print(f"  Vehicles: {solver.num_vehicles}")
        print(f"  Max coord: {solver.max_coord}")
    except Exception as e:
        print(f"✗ Failed to initialize solver: {e}")
        return False

    # Test problem conversion
    try:
        problem_file, param_file, metadata_file = solver.convert_to_lkh()
        print(f"✓ Problem converted to TSPLIB format")
        print(f"  Problem file: {problem_file}")

        # Verify files exist
        assert os.path.exists(problem_file), f"Problem file not created: {problem_file}"
        assert os.path.exists(param_file), f"Param file not created: {param_file}"
        assert os.path.exists(metadata_file), f"Metadata file not created: {metadata_file}"
        print(f"✓ All files created successfully")
    except Exception as e:
        print(f"✗ Failed to convert problem: {e}")
        return False

    # Test objective computation
    try:
        # Test with some served customers
        served_count_tests = [5, 8, 10]
        cost_tests = [100.0, 150.0, 200.0]

        print(f"\n✓ Testing objective computation (service-rate-first scalarization):")
        for k, cost in zip(served_count_tests, cost_tests):
            obj = solver.compute_objective(k, cost)
            service_rate = k / solver.num_customers
            print(f"  k={k:2d} (SR={service_rate:.0%}), cost={cost:6.1f} → obj={obj:10.2f}")

        # Verify that increasing k increases objective (for same cost)
        obj1 = solver.compute_objective(5, 100.0)
        obj2 = solver.compute_objective(6, 100.0)
        assert obj2 > obj1, "Objective should increase with more served customers"
        print(f"✓ Objective properly prioritizes service rate")

        # Verify that lower cost is better (for same k)
        obj1 = solver.compute_objective(10, 100.0)
        obj2 = solver.compute_objective(10, 150.0)
        assert obj1 > obj2, "Objective should decrease with higher cost"
        print(f"✓ Objective properly prioritizes lower cost secondarily")

    except Exception as e:
        print(f"✗ Failed objective tests: {e}")
        return False

    print("\n" + "="*60)
    print("All tests passed! ✓")
    print("="*60)
    print("\nNext steps:")
    print("  1. Run LKH3 solver: python biobj_solver.py --filename S042_N10_C_R50.json")
    print("  2. Compare results with other baselines")
    print("="*60)
    return True


if __name__ == "__main__":
    success = test_solver()
    sys.exit(0 if success else 1)
