"""
scripts/plot_solution.py
------------------------
Visualize truck routes and drone trips from trained checkpoints.

Loads checkpoints for a specific task and plots solution routes on a 100x100 map.
Generates two plots: one from best checkpoint and one from final checkpoint.

Usage:
  # Plot solutions for default task (009_N50_F5_C)
  python scripts/plot_solution.py

  # Plot solutions for specific task
  python scripts/plot_solution.py 003_N10_F2_C

  # Use specific experiment (if multiple experiments have the same task)
  python scripts/plot_solution.py 009_N50_F5_C --experiment 202_PPO_N50_C
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

import globals
from config import load_config
from core import SeedManager
from core.environment import Solution
from core.registry import build_agents, build_environment
from core.utils import obs_to_tensor


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Plot truck routes and drone trips from trained checkpoints.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "task_id",
        nargs="?",
        default="009_N50_F5_C",
        help="Task ID (default: 009_N50_F5_C). Format: {difficulty}_N{customers}_F{fleets}_{distribution}",
    )
    p.add_argument(
        "--experiment",
        default=None,
        metavar="NAME",
        help="Specific experiment name. If not provided, uses first experiment with the task.",
    )
    p.add_argument(
        "--output-dir",
        default=None,
        metavar="PATH",
        help="Output directory for plots (default: experiment/train/{experiment_name}/artifacts/)",
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
    return p


def _find_experiments_with_task(task_id: str) -> list[Path]:
    """Find all experiments in experiment/train/ that have the given task and checkpoints."""
    train_dir = Path("experiment/train")
    if not train_dir.exists():
        raise FileNotFoundError(f"Training directory not found: {train_dir}")

    experiments = []
    for exp_dir in sorted(train_dir.iterdir()):
        if not exp_dir.is_dir():
            continue

        config_path = exp_dir / "config.yaml"
        if not config_path.exists():
            continue

        # Check if checkpoints directory exists
        checkpoint_dir = exp_dir / "checkpoints"
        if not checkpoint_dir.exists():
            continue

        try:
            cfg = load_config(str(config_path))
            tasks = cfg.get("environment", {}).get("properties", {}).get("tasks", [])
            if task_id in tasks:
                experiments.append(exp_dir)
        except Exception:
            continue

    return experiments


def _get_checkpoints(experiment_dir: Path) -> tuple[Path, Path]:
    """Get best and final checkpoints from experiment."""
    checkpoint_dir = experiment_dir / "checkpoints"
    if not checkpoint_dir.exists():
        raise FileNotFoundError(f"Checkpoints directory not found: {checkpoint_dir}")

    best_checkpoints = sorted(checkpoint_dir.glob("*tune_best*.pt"))
    final_checkpoints = sorted(checkpoint_dir.glob("*tune_final*.pt"))

    if not best_checkpoints:
        raise FileNotFoundError(f"No best checkpoints found in {checkpoint_dir}")
    if not final_checkpoints:
        raise FileNotFoundError(f"No final checkpoints found in {checkpoint_dir}")

    return best_checkpoints[-1], final_checkpoints[-1]


def _plot_routes(
    solution: Solution,
    output_path: Path,
    env: Any = None,
    map_size: float = 100.0,
) -> None:
    """Plot truck routes and drone trips on a 2D map."""
    metadata = solution.metadata
    truck_routes = metadata.get("truck_routes", [])
    drone_trips = metadata.get("drone_trips", [])

    # Get coordinates from environment
    if env is None or not hasattr(env, "coords"):
        print(f"Warning: Could not extract coordinates from environment")
        return

    coords = env.coords

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
                ax.plot([x1, x2], [y1, y2], "b-", linewidth=1.5)

    # Plot drone trips (red edges)
    for vehicle_trips in drone_trips:
        for trip in vehicle_trips:
            if len(trip) > 1:
                for i in range(len(trip) - 1):
                    x1, y1 = coords[trip[i]]
                    x2, y2 = coords[trip[i + 1]]
                    ax.plot([x1, x2], [y1, y2], "r-", linewidth=1.5)

    # Plot all nodes (black dots)
    ax.scatter(coords[:, 0], coords[:, 1], c="black", s=10, zorder=3)

    ax.set_title(
        f"Objective: {solution.objective:.2f} | "
        f"Served: {metadata.get('served_count', 0)}/{metadata.get('n_customers', 0)}"
    )

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"  Saved: {output_path}")
    plt.close(fig)


def _evaluate_checkpoint(
    checkpoint_path: Path,
    cfg: Dict[str, Any],
    device: str,
    seed: int,
    output_dir: Path,
    task_id: str,
    checkpoint_type: str,
) -> None:
    """Evaluate checkpoint and generate plot."""

    print(f"\n  Checkpoint ({checkpoint_type}): {checkpoint_path.name}")

    # Set reproducibility
    reproducibility_cfg = cfg.get("reproducibility", {})
    seed_cfg = reproducibility_cfg.get("seed", {})
    seed_mgr = SeedManager(
        random_seed=seed_cfg.get("random_seed", seed),
        numpy_seed=seed_cfg.get("numpy_seed", seed),
        torch_seed=seed_cfg.get("torch_seed", seed),
    )
    seed_mgr.seed_everything()

    globals.DEVICE = device

    # Build environment and agents
    env = build_environment(cfg)
    agents = build_agents(cfg=cfg)
    agent = next(iter(agents.values()))

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if isinstance(checkpoint, dict) and "network_state" in checkpoint:
        agent.network.load_state_dict(checkpoint["network_state"])
    elif isinstance(checkpoint, dict) and "agent" in checkpoint:
        agent.network.load_state_dict(checkpoint["agent"])
    elif isinstance(checkpoint, dict):
        agent.network.load_state_dict(checkpoint)

    agent.network.eval()

    # Retask environment for the specific task
    env.retask(task_id)

    # Reset environment
    obs, info = env.reset()

    # Greedy decoding
    with torch.no_grad():
        mask = info["action_mask"]
        done = False

        while not done:
            obs_tensor = obs_to_tensor(obs, device=device)
            mask_tensor = torch.tensor(mask, dtype=torch.bool, device=device).unsqueeze(0)

            action, _, _, _ = agent.act(obs_tensor, mask_tensor, deterministic=True)
            action = int(action.item())

            obs, reward, terminated, truncated, info = env.step(action)
            mask = info["action_mask"]
            done = terminated or truncated

    # Get current solution
    solution = env.current_solution()

    # Plot solution
    output_path = output_dir / f"solution_{checkpoint_type}.png"
    _plot_routes(solution, output_path, env=env)

    print(
        f"    Objective: {solution.objective:.2f} | "
        f"Served: {solution.metadata.get('served_count', 0)}/{solution.metadata.get('n_customers', 0)}"
    )


def main() -> None:
    args = _build_parser().parse_args()

    task_id = args.task_id

    print(f"\n{'=' * 70}")
    print(f"  Task ID: {task_id}")

    # Find experiments with this task
    experiments = _find_experiments_with_task(task_id)
    if not experiments:
        raise FileNotFoundError(f"No experiments found with task {task_id}")

    # Select experiment
    if args.experiment:
        exp_dir = Path("experiment/train") / args.experiment
        if not exp_dir.exists():
            raise FileNotFoundError(f"Experiment not found: {exp_dir}")

        # Verify experiment has the requested task
        config_path = exp_dir / "config.yaml"
        cfg_check = load_config(str(config_path))
        tasks = cfg_check.get("environment", {}).get("properties", {}).get("tasks", [])
        if task_id not in tasks:
            raise ValueError(f"Experiment {args.experiment} does not have task {task_id}. Available tasks: {tasks}")
    else:
        exp_dir = experiments[0]

    print(f"  Experiment: {exp_dir.name}")
    print(f"  Found {len(experiments)} experiment(s) with this task")

    # Load training config
    config_path = exp_dir / "config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    cfg = load_config(str(config_path))

    # Setup output directory
    output_dir = Path(args.output_dir or exp_dir / "artifacts")
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"  Output Directory: {output_dir}")

    # Get checkpoints
    best_checkpoint, final_checkpoint = _get_checkpoints(exp_dir)
    print(f"  Found checkpoints: best and final")

    # Generate plots
    print(f"\n{'=' * 70}")
    print(f"  Generating plots...")
    _evaluate_checkpoint(
        checkpoint_path=best_checkpoint,
        cfg=cfg,
        device=args.device,
        seed=args.seed,
        output_dir=output_dir,
        task_id=task_id,
        checkpoint_type="best",
    )

    _evaluate_checkpoint(
        checkpoint_path=final_checkpoint,
        cfg=cfg,
        device=args.device,
        seed=args.seed,
        output_dir=output_dir,
        task_id=task_id,
        checkpoint_type="final",
    )

    print(f"\n{'=' * 70}")
    print(f"  Plots saved to: {output_dir}\n")


if __name__ == "__main__":
    main()
