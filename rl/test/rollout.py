"""Test script with detailed episode rollout and solution output."""

import sys
from pathlib import Path

# Add parent directory to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import numpy as np
from typing import Dict, Any, List
from impl.mvrpbtw import MVRPBTWEnv
from core import SeedManager


def run_detailed_episode(
    env: MVRPBTWEnv, episode_id: int, seed: int = 42
) -> Dict[str, Any]:
    """Run a single episode with detailed output."""
    results = {
        "episode": episode_id,
        "seed": seed,
        "success": True,
        "errors": [],
        "steps": 0,
        "terminal_reason": None,
        "total_reward": 0.0,
        "final_cost": 0.0,
        "served_count": 0,
        "truck_routes": [],
        "drone_trips": [],
    }

    try:
        obs, info = env.reset()
        action_mask = info["action_mask"]

        if action_mask.sum() == 0:
            results["errors"].append("Initial state has no feasible actions")
            results["success"] = False
            return results

        step = 0
        max_steps = 500

        while step < max_steps:
            step += 1
            results["steps"] = step

            feasible = np.where(action_mask)[0]
            if len(feasible) == 0:
                results["errors"].append(f"Empty action mask at step {step}")
                results["success"] = False
                results["terminal_reason"] = "stuck_state"
                break

            # Random action from feasible set
            action = np.random.choice(feasible)
            obs, reward, terminated, truncated, info = env.step(action)

            results["total_reward"] += float(reward) if reward is not None else 0.0
            action_mask = info["action_mask"]

            results["served_count"] = info.get("served_count", 0)
            results["final_cost"] = info.get("current_cost", 0.0)

            if terminated or truncated:
                results["terminal_reason"] = "episode_end"
                break

        if step >= max_steps:
            results["warnings"] = f"Episode reached max steps ({max_steps})"
            results["terminal_reason"] = "max_steps"

    except Exception as e:
        results["errors"].append(f"Exception: {str(e)}")
        results["success"] = False
        results["terminal_reason"] = "exception"

    # Get solution details
    try:
        state = env._current_state
        results["truck_routes"] = [
            [int(n) for n in state.truck_routes[k]] for k in range(env.K)
        ]
        results["truck_arrive"] = state.truck_arrive
        results["truck_depart"] = state.truck_depart
        results["drone_trips"] = [
            [[int(n) for n in trip] for trip in trips]
            for trips in state.drone_trips_node
        ]
        results["drone_arrive"] = state.drone_arrive
        results["drone_depart"] = state.drone_depart
    except:
        pass

    return results


def print_detailed_result(results: Dict[str, Any], env: MVRPBTWEnv) -> None:
    """Print detailed information for a single rollout."""
    status = "✓ PASS" if results["success"] else "✗ FAIL"
    print(f"\n  Rollout: {status}")
    print(
        f"    Steps: {results['steps']}, Served: {results['served_count']}/{env.n_customers}"
    )
    print(
        f"    Cost: {results['final_cost']:.2f}, Reward: {results['total_reward']:.4f}"
    )
    print(f"    Reason: {results['terminal_reason']}")

    if results["truck_routes"]:
        print("    Truck routes:")
        for k, route in enumerate(results["truck_routes"]):
            print(f"      Truck {k}:")
            if "truck_arrive" in results and "truck_depart" in results:
                arrive = results["truck_arrive"][k]
                depart = results["truck_depart"][k]
                node_str = " ".join(f"{node:>8}" for node in route)
                print(f"        Node   {node_str}")
                arrive_str = " ".join(
                    f"{arrive[i]:>8.4f}" if i < len(arrive) else "      N/A"
                    for i in range(len(route))
                )
                print(f"        Arrive {arrive_str}")
                depart_str = " ".join(
                    f"{depart[i]:>8.4f}" if i < len(depart) else "      N/A"
                    for i in range(len(route))
                )
                print(f"        Depart {depart_str}")
            else:
                print(f"        {route}")

    if results["drone_trips"]:
        print("    Drone trips:")
        for k, trips in enumerate(results["drone_trips"]):
            if trips:
                print(f"      Drone {k}:")
                for t_idx, trip in enumerate(trips):
                    print(f"        Trip {t_idx}:")
                    if "drone_arrive" in results and "drone_depart" in results:
                        arrive = (
                            results["drone_arrive"][k][t_idx]
                            if t_idx < len(results["drone_arrive"][k])
                            else []
                        )
                        depart = (
                            results["drone_depart"][k][t_idx]
                            if t_idx < len(results["drone_depart"][k])
                            else []
                        )
                        node_str = " ".join(f"{node:>8}" for node in trip)
                        print(f"          Node   {node_str}")
                        arrive_str = " ".join(
                            f"{arrive[i]:>8.4f}" if i < len(arrive) else "      N/A"
                            for i in range(len(trip))
                        )
                        print(f"          Arrive {arrive_str}")
                        depart_str = " ".join(
                            f"{depart[i]:>8.4f}" if i < len(depart) else "      N/A"
                            for i in range(len(trip))
                        )
                        print(f"          Depart {depart_str}")
                    else:
                        print(f"          {trip}")

    if results["errors"]:
        for error in results["errors"]:
            print(f"    ERROR: {error}")


def print_summary_stats(results_list: List[Dict[str, Any]], n_customers: int) -> None:
    """Print summary statistics for all rollouts."""
    passed = sum(1 for r in results_list if r["success"])
    total = len(results_list)

    rewards = [r["total_reward"] for r in results_list]
    costs = [r["final_cost"] for r in results_list]
    served = [r["served_count"] for r in results_list]
    steps = [r["steps"] for r in results_list]

    print(f"\n{'=' * 80}")
    print("SUMMARY STATISTICS")
    print(f"{'=' * 80}")

    print(f"\nSuccess Rate: {passed}/{total} ({100 * passed / total:.1f}%)")

    print(f"\nReward per episode:")
    print(f"  Mean:     {np.mean(rewards):>8.4f}")
    print(f"  Std:      {np.std(rewards):>8.4f}")
    print(f"  Var:      {np.var(rewards):>8.4f}")
    print(f"  Min:      {np.min(rewards):>8.4f}")
    print(f"  Max:      {np.max(rewards):>8.4f}")
    print(f"  Median:   {np.median(rewards):>8.4f}")

    print(f"\nCost per episode:")
    print(f"  Mean:     {np.mean(costs):>8.2f}")
    print(f"  Std:      {np.std(costs):>8.2f}")
    print(f"  Var:      {np.var(costs):>8.2f}")
    print(f"  Min:      {np.min(costs):>8.2f}")
    print(f"  Max:      {np.max(costs):>8.2f}")
    print(f"  Median:   {np.median(costs):>8.2f}")

    print(f"\nCustomers served per episode:")
    print(f"  Mean:     {np.mean(served):>8.2f} / {n_customers}")
    print(f"  Std:      {np.std(served):>8.2f}")
    print(f"  Var:      {np.var(served):>8.2f}")
    print(f"  Min:      {np.min(served):>8.0f} / {n_customers}")
    print(f"  Max:      {np.max(served):>8.0f} / {n_customers}")
    print(f"  Median:   {np.median(served):>8.1f} / {n_customers}")

    print(f"\nSteps per episode:")
    print(f"  Mean:     {np.mean(steps):>8.1f}")
    print(f"  Std:      {np.std(steps):>8.1f}")
    print(f"  Var:      {np.var(steps):>8.1f}")
    print(f"  Min:      {np.min(steps):>8.0f}")
    print(f"  Max:      {np.max(steps):>8.0f}")
    print(f"  Median:   {np.median(steps):>8.1f}")

    print(f"\n{'=' * 80}")


def run_tests(
    n_instances: int = 5,
    n_rollouts: int = 3,
    detailed: bool = False,
) -> None:
    """Run multiple instances with multiple rollouts per instance."""

    cfg = {
        "env": "MVRPBTW",
        "tasks": ["easy_N100_F10_C"],
        "n_customers": 100,
        "max_coord": 100.0,
        "capacity_truck": 200.0,
        "capacity_drone": 20.0,
        "t_max_system_h": 24.0,
        "drone_duration_h": 1.0,
        "v_truck_km_h": 40.0,
        "v_drone_km_h": 60.0,
        "truck_cost_unit": 1.0,
        "drone_cost_unit": 0.5,
        "drone_takeoff_min": 1.0,
        "drone_landing_min": 1.0,
        "service_time_min": 5.0,
    }

    env = MVRPBTWEnv(cfg)
    task_id = cfg["tasks"][0]
    total_rollouts = n_instances * n_rollouts

    if detailed:
        print(
            f"Running {total_rollouts} rollouts ({n_instances} instances × {n_rollouts} rollouts)"
        )
        print(f"Task: {task_id}")
        print(f"Detailed output: {detailed}\n")

    all_results = []

    for inst_id in range(n_instances):
        # Generate instance once
        seed_mgr = SeedManager(
            random_seed=inst_id, numpy_seed=inst_id, torch_seed=inst_id
        )
        seed_mgr.seed_everything()
        instance = env._generate_instance(task_id)
        env.encode_instance(instance)

        if detailed:
            print(f"\nInstance {inst_id + 1}/{n_instances}:")

        for rollout_id in range(n_rollouts):
            # Run rollout on same instance
            seed_mgr = SeedManager(
                random_seed=inst_id * 1000 + rollout_id,
                numpy_seed=inst_id * 1000 + rollout_id,
                torch_seed=inst_id * 1000 + rollout_id,
            )
            seed_mgr.seed_everything()

            results = run_detailed_episode(
                env,
                episode_id=inst_id * n_rollouts + rollout_id + 1,
                seed=inst_id * 1000 + rollout_id,
            )
            all_results.append(results)

            if detailed:
                print_detailed_result(results, env)

    print_summary_stats(all_results, env.n_customers)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Test VRPBTW environment with rollout episodes"
    )
    parser.add_argument(
        "--detailed",
        action="store_true",
        help="Print detailed output for each rollout (default: False)",
    )
    parser.add_argument(
        "--n-instances",
        type=int,
        default=5,
        help="Number of instances to generate (default: 5)",
    )
    parser.add_argument(
        "--n-rollouts",
        type=int,
        default=3,
        help="Number of rollouts per instance (default: 3)",
    )

    args = parser.parse_args()

    run_tests(
        n_instances=args.n_instances,
        n_rollouts=args.n_rollouts,
        detailed=args.detailed,
    )
