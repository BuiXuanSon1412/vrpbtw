#!/usr/bin/env python3
"""
Generalize trained checkpoint to different problem instances.

Loads the best checkpoint from a training experiment and evaluates it on
different task instances (size, fleet count, distribution).
Available tasks are defined in configs/environment/mvrpbtw.yaml
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).parent))

import globals
from config import load_config
from core import Evaluator, SeedManager
from core.registry import build_agents, build_environment

# Available tasks (format: {difficulty:03d}_N{customers}_F{fleets}_{distribution})
AVAILABLE_TASKS = [
    "001_N10_F2_RC",
    "002_N10_F2_C",
    "003_N10_F2_R",
    "004_N20_F3_RC",
    "005_N20_F3_R",
    "006_N20_F3_C",
    "007_N50_F5_RC",
    "008_N50_F5_R",
    "009_N50_F5_C",
    "010_N100_F10_RC",
    "011_N100_F10_R",
    "012_N100_F10_C",
    "013_N150_F15_RC",
    "014_N150_F15_R",
    "015_N150_F15_C",
]


def _load_available_tasks() -> List[str]:
    """Return predefined list of available tasks."""
    return AVAILABLE_TASKS


def _is_valid_task_format(task_id: str) -> bool:
    """Check if task_id follows valid format: {difficulty:03d}_N{N}_F{F}_{dist}"""
    parts = task_id.split("_")
    if len(parts) != 4:
        return False

    # Check difficulty (3 digits)
    if not (parts[0].isdigit() and len(parts[0]) == 3):
        return False

    # Check N{customers}
    if not parts[1].startswith("N") or not parts[1][1:].isdigit():
        return False

    # Check F{fleets}
    if not parts[2].startswith("F") or not parts[2][1:].isdigit():
        return False

    # Check distribution
    if parts[3] not in ["R", "C", "RC"]:
        return False

    return True


def _build_parser() -> argparse.ArgumentParser:
    available_tasks = _load_available_tasks()
    tasks_help = f"Task ID to evaluate (e.g., 010_N100_F10_C). Use --list-tasks to see all available tasks."

    p = argparse.ArgumentParser(
        description="Evaluate trained checkpoint on different task instances",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Evaluate on N100 clustered with 10 fleets
  python generalize.py --experiment GCN_PPO_N10_C --task-id 010_N100_F10_C

  # Evaluate with beam search
  python generalize.py --experiment GCN_PPO_N10_C --task-id 009_N50_F5_C --beam 5

  # Evaluate with more episodes
  python generalize.py --experiment GCN_PPO_N10_C --task-id 012_N100_F10_C --episodes 50
        """,
    )
    p.add_argument(
        "--experiment",
        required=False,
        help="Training experiment name (e.g., GCN_PPO_N10_C)",
    )
    p.add_argument(
        "--task-id",
        required=False,
        default=None,
        help=f"{tasks_help} If not specified, will evaluate on all available tasks.",
    )
    p.add_argument(
        "--episodes",
        type=int,
        default=10,
        help="Number of evaluation episodes (default: 10)",
    )
    p.add_argument(
        "--beam",
        type=int,
        default=1,
        help="Beam width for decoding (default: 1)",
    )
    p.add_argument(
        "--device",
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device to use (default: cpu)",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("experiment/generalize"),
        help="Output directory for results (default: experiment/generalize)",
    )
    p.add_argument(
        "--list-tasks",
        action="store_true",
        help="List available tasks and exit",
    )
    return p


def main() -> None:
    args = _build_parser().parse_args()

    # List available tasks if requested
    if args.list_tasks:
        available_tasks = _load_available_tasks()
        print("\nAvailable tasks:")
        for task in available_tasks:
            print(f"  {task}")
        sys.exit(0)

    # Validate required arguments
    if not args.experiment:
        print("\nError: --experiment is required")
        sys.exit(1)

    # Determine tasks to evaluate
    if args.task_id:
        # Single task
        if not _is_valid_task_format(args.task_id):
            print(f"\nError: Invalid task ID format: '{args.task_id}'")
            print("Valid format: {{difficulty:03d}}_N{{customers}}_F{{fleets}}_{{distribution}}")
            print("Example: 010_N100_F10_C")
            print("Distribution: R (Random), C (Clustered), RC (Random-Clustered)")
            sys.exit(1)
        tasks_to_eval = [args.task_id]
    else:
        # All tasks
        tasks_to_eval = _load_available_tasks()
        print(f"\nNo task specified. Will evaluate on all {len(tasks_to_eval)} available tasks.")

    # Setup output directory
    output_dir = args.output_dir / args.experiment
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'=' * 70}")
    print(f"  Generalization Evaluation")
    print(f"{'=' * 70}")
    print(f"  Training Experiment: {args.experiment}")
    print(f"  Tasks to Evaluate  : {len(tasks_to_eval)}")
    print(f"  Episodes           : {args.episodes}")
    print(f"  Beam Width         : {args.beam}")
    print(f"  Device             : {args.device}")
    print(f"  Output Dir         : {output_dir}")

    # Load training config
    experiment_dir = Path("experiment/train") / args.experiment
    if not experiment_dir.exists():
        print(f"  Error: Experiment not found: {experiment_dir}")
        sys.exit(1)

    config_path = experiment_dir / "config.yaml"
    if not config_path.exists():
        print(f"  Error: Config not found: {config_path}")
        sys.exit(1)

    cfg = load_config(str(config_path))
    cfg["device"] = args.device
    globals.DEVICE = args.device

    # Find and load best checkpoint (once)
    checkpoint_dir = experiment_dir / "checkpoints"
    best_checkpoints = list(checkpoint_dir.glob("*tune_best*.pt"))

    if not best_checkpoints:
        print(f"  Error: No best checkpoint found in {checkpoint_dir}")
        sys.exit(1)

    best_checkpoint = best_checkpoints[0]
    print(f"  Checkpoint         : {best_checkpoint.name}\n")

    # Load checkpoint once
    checkpoint = torch.load(best_checkpoint, map_location=args.device)

    # Evaluate on all tasks
    all_results = {}
    for task_idx, task_id in enumerate(tasks_to_eval, 1):
        print(f"\n{'=' * 70}")
        print(f"  Task {task_idx}/{len(tasks_to_eval)}: {task_id}")
        print(f"{'=' * 70}")

        # Set reproducibility for each task
        seed_cfg = cfg.get("reproducibility", {}).get("seed", {})
        seed_mgr = SeedManager(
            random_seed=seed_cfg.get("random_seed", 42),
            numpy_seed=seed_cfg.get("numpy_seed", 42),
            torch_seed=seed_cfg.get("torch_seed", 42),
        )
        seed_mgr.seed_everything()

        # Update environment config for this task
        cfg["environment"]["properties"]["tasks"] = [task_id]

        # Build environment and agents
        env = build_environment(cfg)
        agents = build_agents(cfg=cfg)
        agent = next(iter(agents.values()))

        # Load checkpoint into agent
        if isinstance(checkpoint, dict) and "network_state" in checkpoint:
            agent.network.load_state_dict(checkpoint["network_state"])
        elif isinstance(checkpoint, dict) and "agent" in checkpoint:
            agent.network.load_state_dict(checkpoint["agent"])
        elif isinstance(checkpoint, dict):
            agent.network.load_state_dict(checkpoint)
        else:
            agent.network.load_state_dict(checkpoint)

        # Evaluate
        print(f"  Evaluating on {args.episodes} instances...")

        evaluator = Evaluator(
            agent=agent,
            env=env,
            n_episodes=args.episodes,
            deterministic=True,
            n_samples=1,
            beam_width=args.beam,
        )

        results = evaluator.evaluate(task_id)
        all_results[task_id] = results

        # Print results
        print(f"\n  Results:")
        print(f"  {'-' * 66}")
        print(f"    mean_objective        : {results.get('mean_objective', 0):>12.2f}")
        print(f"    mean_service_rate     : {results.get('mean_service_rate', 0):>12.4f}")
        print(f"    mean_cost             : {results.get('mean_cost', 0):>12.2f}")
        print(f"    mean_time_s           : {results.get('mean_time_s', 0):>12.2f}")
        print(f"    best_objective        : {results.get('best_objective', 0):>12.2f}")
        print(f"    best_service_rate     : {results.get('best_service_rate', 0):>12.4f}")

        # Save results
        results_file = output_dir / f"{task_id}_results.json"
        with open(results_file, "w") as f:
            json.dump(results, f, indent=2)

        # Save config
        config_out = output_dir / f"{task_id}_config.json"
        config_data = {
            "training_experiment": args.experiment,
            "task_id": task_id,
            "n_episodes": args.episodes,
            "beam_width": args.beam,
            "device": args.device,
            "checkpoint": best_checkpoint.name,
        }
        with open(config_out, "w") as f:
            json.dump(config_data, f, indent=2)

    # Summary
    print(f"\n{'=' * 70}")
    print(f"  Summary - All Tasks")
    print(f"{'=' * 70}")
    for task_id, results in all_results.items():
        print(f"  {task_id:<25} | SR: {results.get('mean_service_rate', 0):>6.2%} | Cost: {results.get('mean_cost', 0):>8.2f}")

    print(f"\n  Results saved to: {output_dir}")
    print(f"  Total tasks evaluated: {len(all_results)}\n")


if __name__ == "__main__":
    main()
