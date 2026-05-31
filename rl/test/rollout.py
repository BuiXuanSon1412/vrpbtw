#!/usr/bin/env python3
"""
Test script with detailed episode rollout and CPU usage measurement.

Tests environment correctness and measures CPU usage per episode.
Uses PyTorch sampling (matches production agent code).

CLI Usage
---------
  python test/rollout.py                           # All tasks, default settings
  python test/rollout.py --task 001_N10_F2_RC      # Specific task
  python test/rollout.py --n-episodes 5            # Override episodes per task
  python test/rollout.py --detailed                # Print detailed routes/timing
  python test/rollout.py --config configs/train.yaml  # Custom config

Arguments
---------
  --config CONFIG      Training config (default: configs/train.yaml)
  --task TASK          Specific task (if not set, test all tasks)
  --n-episodes N       Episodes per task (default: 3)
  --detailed           Print truck routes, drone trips, timings
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import re
import time
import numpy as np
import torch
import psutil
from typing import Dict, Any, List

import globals
from config import load_config
from core import SeedManager
from core.registry import build_environment


def get_test_tasks() -> List[str]:
    """Get hardcoded list of tasks for testing.

    Independent of mvrpbtw.yaml to ensure test reproducibility.
    Format: {difficulty:003d}_N{customers}_F{fleets}_{distribution}
    """
    return [
        # Small problems (N=10, 2 fleets)
        "001_N10_F2_RC",  # Random-Clustered
        "002_N10_F2_R",  # Random
        "003_N10_F2_C",  # Clustered
        # Medium problems (N=20, 3 fleets)
        "004_N20_F3_RC",  # Random-Clustered
        "005_N20_F3_R",  # Random
        "006_N20_F3_C",  # Clustered
        # Large problems (N=50, 5 fleets)
        "007_N50_F5_RC",  # Random-Clustered
        "008_N50_F5_R",  # Random
        "009_N50_F5_C",  # Clustered
        # Very large problems (N=100, 10 fleets)
        "010_N100_F10_RC",  # Random-Clustered
        "011_N100_F10_R",  # Random
        "012_N100_F10_C",  # Clustered
    ]


def run_detailed_episode(
    env: Any,
    task_id: str,
    episode_id: int,
    verbose: bool = False,
) -> Dict[str, Any]:
    """Run a single episode with uniform action sampling and CPU measurement."""
    process = psutil.Process()

    results = {
        "episode": episode_id,
        "task_id": task_id,
        "success": True,
        "errors": [],
        "steps": 0,
        "terminal_reason": None,
        "total_reward": 0.0,
        "final_cost": 0.0,
        "served_count": 0,
        "truck_routes": [],
        "drone_trips": [],
        "duration_s": 0.0,
        "cpu_percent": 0.0,
    }

    cpu_samples = []
    start_memory = process.memory_info().rss / 1024 / 1024
    start_time = time.time()

    try:
        env.retask(task_id)
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

            # Sample uniformly from feasible actions using Categorical with uniform logits
            logits = torch.ones(len(feasible), device=globals.DEVICE)
            dist = torch.distributions.Categorical(logits=logits)
            idx = dist.sample().item()
            action = int(feasible[int(idx)])

            # Sample CPU during environment step only
            cpu_percent = process.cpu_percent(interval=0.001)
            obs, reward, terminated, truncated, info = env.step(action)
            cpu_samples.append(cpu_percent)
            action_mask = info["action_mask"]

            results["total_reward"] += float(reward) if reward is not None else 0.0
            results["served_count"] = info.get("served_count", 0)
            results["final_cost"] = info.get("current_cost", 0.0)

            if terminated or truncated:
                results["terminal_reason"] = "episode_end"
                break

        if step >= max_steps:
            results["terminal_reason"] = "max_steps"

    except Exception as e:
        results["errors"].append(f"Exception: {str(e)}")
        results["success"] = False
        results["terminal_reason"] = "exception"

    finally:
        duration = time.time() - start_time
        peak_memory = process.memory_info().rss / 1024 / 1024
        memory_delta = peak_memory - start_memory

        results["duration_s"] = float(duration)
        results["cpu_percent"] = float(np.mean(cpu_samples)) if cpu_samples else 0.0
        results["memory_mb"] = float(memory_delta)

    # Get solution details
    try:
        state = env._current_state
        results["truck_routes"] = [
            [int(n) for n in state.truck_routes[k]] for k in range(env.K)
        ]
        results["truck_arrive"] = state.truck_arrive
        results["truck_depart"] = state.truck_depart
        results["drone_trips"] = [
            [[int(n) for n in trip] for trip in trips] for trips in state.drone_trips
        ]
        results["drone_arrive"] = state.drone_arrive
        results["drone_depart"] = state.drone_depart
    except:
        pass

    return results


def print_detailed_result(results: Dict[str, Any], env: Any) -> None:
    """Print detailed information for a single rollout."""
    status = "✓ PASS" if results["success"] else "✗ FAIL"
    print(f"\n  Rollout {results['episode']}: {status}")
    print(
        f"    Steps: {results['steps']}, Served: {results['served_count']}/{env.n_customers}"
    )
    print(
        f"    Cost: {results['final_cost']:.2f}, Reward: {results['total_reward']:.4f}"
    )
    print(
        f"    Duration: {results['duration_s']:.3f}s, CPU: {results['cpu_percent']:.1f}%, Memory: {results['memory_mb']:.1f}MB"
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


def get_n_customers_from_task_id(task_id: str) -> int:
    """Extract customer count from task ID (e.g., '001_N10_F2_RC' -> 10)."""
    match = re.search(r"_N(\d+)_", task_id)
    return int(match.group(1)) if match else 0


def print_task_summary(results_list: List[Dict[str, Any]], task_id: str):
    """Return single-line summary for a task."""
    n_customers = get_n_customers_from_task_id(task_id)
    passed = sum(1 for r in results_list if r["success"])
    total = len(results_list)

    rewards = [r["total_reward"] for r in results_list]
    costs = [r["final_cost"] for r in results_list]
    served = [r["served_count"] for r in results_list]
    steps = [r["steps"] for r in results_list]
    durations = [r["duration_s"] for r in results_list]
    cpu_percents = [r["cpu_percent"] for r in results_list]
    memory_deltas = [r["memory_mb"] for r in results_list]

    success = f"{passed}/{total}" if total > 1 else ("✓" if passed else "✗")
    reward_str = f"{np.mean(rewards):.2f}±{np.std(rewards):.2f}"
    cost_str = f"{np.mean(costs):.1f}±{np.std(costs):.1f}"
    served_str = f"{np.mean(served):.0f}±{np.std(served):.0f}/{n_customers}"
    steps_str = f"{np.mean(steps):.0f}±{np.std(steps):.0f}"
    dur_str = f"{np.mean(durations):.3f}±{np.std(durations):.3f}"
    cpu_str = f"{np.mean(cpu_percents):.1f}±{np.std(cpu_percents):.1f}"
    mem_str = f"{np.mean(memory_deltas):.1f}±{np.std(memory_deltas):.1f}"

    return (
        success,
        reward_str,
        cost_str,
        served_str,
        steps_str,
        dur_str,
        cpu_str,
        mem_str,
    )


def run_tests(
    cfg: Dict,
    tasks: List[str],
    n_episodes: int = 10,
    detailed: bool = False,
) -> None:
    """Run multiple tasks with multiple episodes per task."""

    # Setup
    device = cfg.get("device", "cpu")
    globals.DEVICE = device

    reproducibility_cfg = cfg.get("reproducibility", {})
    seed_cfg = reproducibility_cfg.get("seed", {})
    seed_mgr = SeedManager(
        random_seed=seed_cfg.get("random_seed", 42),
        numpy_seed=seed_cfg.get("numpy_seed", 42),
        torch_seed=seed_cfg.get("torch_seed", 42),
    )
    seed_mgr.seed_everything()

    # Build environment only
    env = build_environment(cfg)

    print("=" * 100)
    print("ENVIRONMENT CORRECTNESS & CPU USAGE TEST")
    print("=" * 100)
    print(f"Config: {cfg.get('experiment', {}).get('name', 'default')}")
    print(f"Device: {device}")
    print(f"Sampler: PyTorch Categorical (uniform logits over feasible actions)")
    print(f"Tasks: {len(tasks)}")
    print(f"Episodes per task: {n_episodes}")
    print()

    all_results = {}

    for task_id in tasks:
        print(f"Testing: {task_id}")
        task_results = []

        for ep_idx in range(n_episodes):
            results = run_detailed_episode(env, task_id, ep_idx + 1, verbose=False)
            task_results.append(results)

            if detailed:
                print_detailed_result(results, env)

        all_results[task_id] = task_results
        print()

    # Print per-task summary (all metrics in one table)
    print("=" * 180)
    print("PER-TASK SUMMARY")
    print("=" * 180)
    print()

    headers = [
        "Task",
        "Success",
        "Reward",
        "Cost",
        "Served",
        "Steps",
        "Duration(s)",
        "CPU(%)",
        "Memory(MB)",
    ]
    print(
        f"{'Task':<20} {'Success':<10} {'Reward':<18} {'Cost':<16} {'Served':<18} {'Steps':<16} {'Dur(s)':<14} {'CPU(%)':<14} {'Mem(MB)':<14}"
    )
    print("-" * 180)

    for task_id in tasks:
        results = all_results[task_id]
        success, reward, cost, served, steps, dur, cpu, mem = print_task_summary(
            results, task_id
        )
        print(
            f"{task_id:<20} {success:<10} {reward:<18} {cost:<16} {served:<18} {steps:<16} {dur:<14} {cpu:<14} {mem:<14}"
        )

    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Test environment correctness and measure CPU usage with PyTorch sampling"
    )
    parser.add_argument(
        "--config",
        default="configs/train.yaml",
        help="Training config (default: configs/train.yaml)",
    )
    parser.add_argument(
        "--task",
        default=None,
        help="Specific task (if not set, test all tasks from config)",
    )
    parser.add_argument(
        "--n-episodes",
        type=int,
        default=100,
        help="Episodes per task (default: 100)",
    )
    parser.add_argument(
        "--detailed",
        action="store_true",
        help="Print detailed output for each rollout (default: False)",
    )

    args = parser.parse_args()

    # Load config
    cfg = load_config(args.config)

    # Get tasks (hardcoded, independent of config)
    tasks = get_test_tasks()

    if args.task:
        if args.task not in tasks:
            print(f"Error: Task {args.task} not found in test suite")
            print(f"   Available tasks: {tasks}")
            sys.exit(1)
        tasks = [args.task]

    run_tests(cfg, tasks, n_episodes=args.n_episodes, detailed=args.detailed)
