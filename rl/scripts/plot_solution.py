"""
scripts/plot_solution.py
------------------------
Generate problem instances for specific tasks and plot solutions from trained checkpoints.

Generates fresh instances for a task and runs trained agent to solve them,
then visualizes the resulting solutions (truck routes and drone trips).

Task IDs are read from config.yaml of each experiment.

Usage:
  # Generate and plot 3 instances for all tasks across all experiments
  python scripts/plot_solution.py --num-instances 3

  # Plot 5 instances for specific experiment
  python scripts/plot_solution.py --num-instances 5 --exp-name PPO_N10_RC

  # Plot specific task from specific experiment
  python scripts/plot_solution.py --num-instances 3 --exp-name PPO_N10_RC --task-id 001_N10_F2_RC

  # Plot specific task across all experiments
  python scripts/plot_solution.py --num-instances 3 --task-id 001_N10_F2_RC

  # Use GPU and custom seed
  python scripts/plot_solution.py --num-instances 3 --device cuda --seed 42
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

import globals
from config import load_config
from core import SeedManager
from core.registry import build_agents, build_environment
from core.utils import obs_to_tensor


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Generate and plot solutions from trained checkpoints",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--num-instances",
        type=int,
        default=3,
        metavar="N",
        help="Number of instances to generate and plot per task (default: 3)",
    )
    p.add_argument(
        "--exp-name",
        type=str,
        default=None,
        metavar="NAME",
        help="Specific experiment directory name (e.g., PPO_N10_RC). If not specified, use all experiments.",
    )
    p.add_argument(
        "--task-id",
        type=str,
        default=None,
        metavar="TASK",
        help="Specific task ID to plot (e.g., 001_N10_F2_RC). If not specified, plot all tasks.",
    )
    p.add_argument(
        "--checkpoint-type",
        type=str,
        default="best",
        choices=["best", "final"],
        metavar="TYPE",
        help="Which checkpoint to use (best or final, default: best)",
    )
    p.add_argument(
        "--device",
        default="cpu",
        metavar="DEVICE",
        help="Device to use (cpu or cuda, default: cpu)",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        metavar="SEED",
        help="Random seed (default: 42)",
    )
    p.add_argument(
        "--map-size",
        type=float,
        default=100.0,
        metavar="SIZE",
        help="Map size for visualization (default: 100.0)",
    )
    p.add_argument(
        "--dpi",
        type=int,
        default=150,
        metavar="DPI",
        help="DPI for saved figures (default: 150)",
    )
    return p


def _find_experiments() -> List[Path]:
    """Find all experiments in experiment/train/."""
    train_dir = Path("experiment/train")
    if not train_dir.exists():
        return []

    experiments = []
    for exp_dir in sorted(train_dir.iterdir()):
        if exp_dir.is_dir() and (exp_dir / "config.yaml").exists():
            experiments.append(exp_dir)

    return experiments


def _filter_experiments(
    experiments: List[Path],
    exp_name: Optional[str] = None,
) -> List[Path]:
    """Filter experiments by exact name."""
    if exp_name is None:
        return experiments

    filtered = []
    for exp_dir in experiments:
        if exp_dir.name == exp_name:
            filtered.append(exp_dir)

    return filtered


def _get_tasks_from_config(config_path: Path) -> List[str]:
    """Extract task IDs from config.yaml."""
    try:
        cfg = load_config(str(config_path))
        tasks = cfg.get("environment", {}).get("properties", {}).get("tasks", [])
        return list(tasks) if tasks else []
    except Exception as e:
        print(f"    Warning: Could not load config from {config_path}: {e}")
        return []


def _get_checkpoint(experiment_dir: Path, task_id: str, checkpoint_type: str) -> Optional[Path]:
    """Get checkpoint for a specific task."""
    checkpoint_dir = experiment_dir / "checkpoints"
    if not checkpoint_dir.exists():
        return None

    # Find checkpoint matching task_id and type
    pattern = f"*{checkpoint_type}*{task_id}*"
    matches = list(checkpoint_dir.glob(pattern))

    if matches:
        return sorted(matches)[-1]

    return None


def _plot_solution(
    solution: Any,
    output_path: Path,
    env: Any = None,
    map_size: float = 100.0,
    dpi: int = 150,
) -> None:
    """Plot truck routes and drone trips on a 2D map."""
    metadata = solution.metadata
    truck_routes = metadata.get("truck_routes", [])
    drone_trips = metadata.get("drone_trips", [])

    # Get coordinates from environment
    coords = None
    if env is not None and hasattr(env, "coords"):
        coords = env.coords
    elif hasattr(solution, "coords"):
        coords = solution.coords

    if coords is None:
        print(f"    Warning: Could not extract coordinates")
        return

    fig, ax = plt.subplots(figsize=(8, 8))

    # Plot map bounds
    ax.set_xlim(-5, map_size + 5)
    ax.set_ylim(-5, map_size + 5)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)

    # Plot truck routes (blue edges)
    for route in truck_routes:
        if len(route) > 1:
            for i in range(len(route) - 1):
                x1, y1 = coords[route[i]]
                x2, y2 = coords[route[i + 1]]
                ax.plot([x1, x2], [y1, y2], "b-", linewidth=1.5, alpha=0.7)

    # Plot drone trips (red edges)
    for vehicle_trips in drone_trips:
        for trip in vehicle_trips:
            if len(trip) > 1:
                for i in range(len(trip) - 1):
                    x1, y1 = coords[trip[i]]
                    x2, y2 = coords[trip[i + 1]]
                    ax.plot([x1, x2], [y1, y2], "r-", linewidth=1.5, alpha=0.7)

    # Plot all nodes
    ax.scatter(coords[1:, 0], coords[1:, 1], c="black", s=20, zorder=3, label="Customers")
    ax.scatter(coords[0, 0], coords[0, 1], c="green", s=50, zorder=4, marker="s", label="Depot")

    objective = solution.objective
    served = metadata.get("served_count", 0)
    total = metadata.get("n_customers", 0)

    ax.set_title(f"Objective: {objective:.2f} | Served: {served}/{total}")
    ax.legend(loc="upper right")

    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    print(f"      Saved: {output_path}")
    plt.close(fig)


def _run_inference(
    agent: Any,
    env: Any,
    task_id: str,
    device: str,
) -> Any:
    """Run inference on environment and return solution."""
    env.retask(task_id)
    obs, info = env.reset()

    with torch.no_grad():
        mask = info["action_mask"]
        done = False

        while not done:
            obs_tensor = obs_to_tensor(obs, device=device)
            mask_tensor = torch.tensor(mask, dtype=torch.bool, device=device).unsqueeze(0)

            action, _, _, _ = agent.act(obs_tensor, mask_tensor, deterministic=False)
            action = int(action.item())

            obs, reward, terminated, truncated, info = env.step(action)
            mask = info["action_mask"]
            done = terminated or truncated

    return env.current_solution()


def main() -> None:
    args = _build_parser().parse_args()

    print(f"\n{'=' * 80}")
    print(f"  Generating and Plotting Solutions")
    print(f"{'=' * 80}")

    # Find experiments
    experiments = _find_experiments()
    if not experiments:
        print("  No experiments found in experiment/train/")
        return

    # Filter by exp_name if specified
    if args.exp_name:
        experiments = _filter_experiments(experiments, exp_name=args.exp_name)
        if not experiments:
            print(f"  No experiments found matching: {args.exp_name}")
            return

    print(f"  Found {len(experiments)} experiment(s)")
    print(f"  Num instances per task: {args.num_instances}")
    print(f"  Checkpoint type: {args.checkpoint_type}")
    print(f"  Device: {args.device}")
    if args.task_id:
        print(f"  Task filter: {args.task_id}")
    if args.exp_name:
        print(f"  Experiment filter: {args.exp_name}")
    print()

    total_plots = 0
    globals.DEVICE = args.device

    # Process each experiment
    for exp_idx, exp_dir in enumerate(experiments, 1):
        exp_name = exp_dir.name
        config_path = exp_dir / "config.yaml"

        if not config_path.exists():
            print(f"  {exp_idx}. {exp_name}")
            print(f"     Warning: config.yaml not found, skipping")
            continue

        # Load config and setup reproducibility
        try:
            cfg = load_config(str(config_path))
        except Exception as e:
            print(f"  {exp_idx}. {exp_name}")
            print(f"     Warning: Could not load config: {e}")
            continue

        reproducibility_cfg = cfg.get("reproducibility", {})
        seed_cfg = reproducibility_cfg.get("seed", {})
        seed_mgr = SeedManager(
            random_seed=seed_cfg.get("random_seed", args.seed),
            numpy_seed=seed_cfg.get("numpy_seed", args.seed),
            torch_seed=seed_cfg.get("torch_seed", args.seed),
        )
        seed_mgr.seed_everything()

        # Get tasks from config
        tasks = _get_tasks_from_config(config_path)
        if not tasks:
            print(f"  {exp_idx}. {exp_name}")
            print(f"     Warning: No tasks found in config, skipping")
            continue

        # Filter tasks if specified - exact match
        if args.task_id:
            tasks = [t for t in tasks if t == args.task_id]
            if not tasks:
                print(f"  {exp_idx}. {exp_name}")
                print(f"     Task '{args.task_id}' not found in config")
                continue

        print(f"  {exp_idx}. {exp_name} ({len(tasks)} task(s))")

        # Build environment and agents
        try:
            env = build_environment(cfg)
            agents = build_agents(cfg=cfg)
            agent = next(iter(agents.values()))
        except Exception as e:
            print(f"     Warning: Could not build environment/agents: {e}")
            continue

        # Process each task
        for task_id in tasks:
            # Get checkpoint
            checkpoint_path = _get_checkpoint(exp_dir, task_id, args.checkpoint_type)
            if not checkpoint_path or not checkpoint_path.exists():
                print(f"     {task_id}: No {args.checkpoint_type} checkpoint found")
                continue

            # Load checkpoint
            try:
                checkpoint = torch.load(checkpoint_path, map_location=args.device)
                if isinstance(checkpoint, dict) and "network_state" in checkpoint:
                    agent.network.load_state_dict(checkpoint["network_state"])
                elif isinstance(checkpoint, dict) and "agent" in checkpoint:
                    agent.network.load_state_dict(checkpoint["agent"])
                elif isinstance(checkpoint, dict):
                    agent.network.load_state_dict(checkpoint)
            except Exception as e:
                print(f"     {task_id}: Could not load checkpoint: {e}")
                continue

            agent.network.eval()

            print(f"     {task_id} ({args.num_instances} instance(s))")

            # Create output directory
            output_dir = exp_dir / "artifacts" / "solutions" / task_id
            output_dir.mkdir(parents=True, exist_ok=True)

            # Generate and plot instances
            for instance_idx in range(args.num_instances):
                try:
                    solution = _run_inference(agent, env, task_id, args.device)
                    output_path = output_dir / f"instance_{instance_idx:02d}_solution.png"
                    _plot_solution(solution, output_path, env=env, map_size=args.map_size, dpi=args.dpi)
                    total_plots += 1
                except Exception as e:
                    print(f"       instance {instance_idx}: Error - {e}")
                    continue

    print()
    print(f"{'=' * 80}")
    print(f"  Total plots generated: {total_plots}")
    print(f"{'=' * 80}\n")


if __name__ == "__main__":
    main()
