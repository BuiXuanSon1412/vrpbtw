"""
scripts/plot_convergence.py
---------------------------
Plot convergence curves from training experiments.

Supports both MetaTrainer and POMOTrainer with the new phase+task logging structure.
Generates visualizations for:
- Per-task convergence
- Aggregate convergence across tasks
- Loss curves
- Objective curves
- Service rate curves
- Gradient norm stability

Usage
-----
  # Plot all experiments
  python scripts/plot_convergence.py

  # Plot specific experiments
  python scripts/plot_convergence.py --exp-ids 100 102 106

  # Plot specific phase
  python scripts/plot_convergence.py --exp-ids 100 --phase fine_tuning

  # Compare multiple experiments
  python scripts/plot_convergence.py --exp-ids 100 102 --compare

  # High resolution output
  python scripts/plot_convergence.py --exp-ids 100 --dpi 150
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np


def load_metrics_jsonl(jsonl_path: Path) -> List[Dict]:
    """Load metrics from JSONL file."""
    if not jsonl_path.exists():
        return []

    metrics = []
    with open(jsonl_path) as f:
        for line in f:
            try:
                metrics.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return metrics


def find_exp_dirs(exp_ids: Optional[List[int]] = None) -> List[Path]:
    """Find experiment directories."""
    base_dir = Path("experiment/train")
    if not base_dir.exists():
        return []

    exp_dirs = []
    for d in sorted(base_dir.iterdir()):
        if not d.is_dir():
            continue

        # Extract exp ID from directory name (e.g., "100_PPO_N10_C" -> 100)
        try:
            exp_id = int(d.name.split("_")[0])
            if exp_ids is None or exp_id in exp_ids:
                exp_dirs.append(d)
        except (ValueError, IndexError):
            continue

    return exp_dirs


def get_phase_tasks(exp_dir: Path) -> Dict[str, List[str]]:
    """Find all phases in experiment (simplified for single metrics.jsonl).

    Returns:
        {"training": []} - Single phase since all logs in one file
    """
    logs_dir = exp_dir / "logs"
    metrics_path = logs_dir / "metrics.jsonl"

    if not metrics_path.exists():
        return {}

    # With single metrics.jsonl, we treat entire training as one phase
    return {"training": []}


def plot_convergence_per_task(
    exp_dir: Path,
    phase: str,
    task_id: Optional[str] = None,
    output_dir: Optional[Path] = None,
    dpi: int = 100,
) -> None:
    """Plot convergence for experiment (single metrics.jsonl file) - generates 4 separate images."""
    metrics_path = exp_dir / "logs" / "metrics.jsonl"

    if not metrics_path.exists():
        return

    metrics = load_metrics_jsonl(metrics_path)
    if not metrics:
        return

    # Extract metric arrays
    steps = []
    loss_means = []
    loss_stds = []
    grad_norms = []
    objectives = []
    service_rates = []

    for m in metrics:
        if "step" not in m:
            continue

        steps.append(m["step"])

        # Handle both tune/ and train/ prefixes
        loss_key = None
        for key in ["tune/loss_mean", "train/loss_mean"]:
            if key in m:
                loss_key = key
                break
        if loss_key:
            loss_means.append(m[loss_key])

        loss_std_key = None
        for key in ["tune/loss_std", "train/loss_std"]:
            if key in m:
                loss_std_key = key
                break
        if loss_std_key:
            loss_stds.append(m[loss_std_key])

        grad_key = None
        for key in ["tune/grad_norm_mean", "train/grad_norm_mean"]:
            if key in m:
                grad_key = key
                break
        if grad_key:
            grad_norms.append(m[grad_key])

        if "eval/mean_objective" in m:
            objectives.append(m["eval/mean_objective"])

        if "eval/mean_service_rate" in m:
            service_rates.append(m["eval/mean_service_rate"])

    if not steps:
        return

    if output_dir is None:
        output_dir = exp_dir / "artifacts"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Plot 1: Loss convergence
    if loss_means:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(steps[:len(loss_means)], loss_means, "b-", linewidth=2, label="loss_mean")
        if loss_stds:
            loss_stds_trimmed = loss_stds[:len(loss_means)]
            ax.fill_between(
                steps[:len(loss_means)],
                np.array(loss_means) - np.array(loss_stds_trimmed),
                np.array(loss_means) + np.array(loss_stds_trimmed),
                alpha=0.2,
            )
        ax.set_xlabel("Step")
        ax.set_ylabel("Loss")
        ax.set_title(f"Training Loss Convergence - {exp_dir.name}")
        ax.grid(True, alpha=0.3)
        ax.legend()
        plt.tight_layout()

        filename = f"convergence_{phase}_loss.png"
        filepath = output_dir / filename
        plt.savefig(filepath, dpi=dpi, bbox_inches="tight")
        print(f"  Saved: {filepath}")
        plt.close()

    # Plot 2: Objective convergence
    if objectives:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(
            steps[:len(objectives)],
            objectives,
            "g-",
            linewidth=2,
            label="mean_objective",
        )
        ax.axhline(
            y=min(objectives),
            color="r",
            linestyle="--",
            alpha=0.5,
            label=f"best={min(objectives):.2f}",
        )
        ax.set_xlabel("Step")
        ax.set_ylabel("Objective (Cost)")
        ax.set_title(f"Objective Convergence - {exp_dir.name}")
        ax.grid(True, alpha=0.3)
        ax.legend()
        plt.tight_layout()

        filename = f"convergence_{phase}_objective.png"
        filepath = output_dir / filename
        plt.savefig(filepath, dpi=dpi, bbox_inches="tight")
        print(f"  Saved: {filepath}")
        plt.close()

    # Plot 3: Service rate convergence
    if service_rates:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(
            steps[:len(service_rates)],
            service_rates,
            "m-",
            linewidth=2,
            label="service_rate",
        )
        ax.axhline(
            y=max(service_rates),
            color="g",
            linestyle="--",
            alpha=0.5,
            label=f"best={max(service_rates):.3f}",
        )
        ax.set_xlabel("Step")
        ax.set_ylabel("Service Rate")
        ax.set_title(f"Service Rate Convergence - {exp_dir.name}")
        ax.set_ylim([0, 1.05])
        ax.grid(True, alpha=0.3)
        ax.legend()
        plt.tight_layout()

        filename = f"convergence_{phase}_service_rate.png"
        filepath = output_dir / filename
        plt.savefig(filepath, dpi=dpi, bbox_inches="tight")
        print(f"  Saved: {filepath}")
        plt.close()

    # Plot 4: Gradient norm stability
    if grad_norms:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(steps[:len(grad_norms)], grad_norms, "orange", linewidth=2, label="grad_norm_mean")
        ax.axhline(y=1.0, color="r", linestyle="--", alpha=0.5, label="safe_bound=1.0")
        ax.set_xlabel("Step")
        ax.set_ylabel("Gradient Norm")
        ax.set_title(f"Gradient Norm Stability - {exp_dir.name}")
        ax.grid(True, alpha=0.3)
        ax.legend()
        plt.tight_layout()

        filename = f"convergence_{phase}_grad_norm.png"
        filepath = output_dir / filename
        plt.savefig(filepath, dpi=dpi, bbox_inches="tight")
        print(f"  Saved: {filepath}")
        plt.close()


def plot_convergence_comparison(
    exp_dirs: List[Path],
    phase: str,
    output_dir: Optional[Path] = None,
    dpi: int = 100,
) -> None:
    """Compare convergence across multiple experiments."""
    if not exp_dirs or not all(d.exists() for d in exp_dirs):
        return

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f"Convergence Comparison (Experiments)", fontsize=14, fontweight="bold")

    colors = plt.cm.tab10(np.linspace(0, 1, len(exp_dirs)))

    for idx, exp_dir in enumerate(exp_dirs):
        color = colors[idx]
        exp_name = exp_dir.name

        # Load metrics from single metrics.jsonl file
        metrics_path = exp_dir / "logs" / "metrics.jsonl"
        if not metrics_path.exists():
            continue

        metrics = load_metrics_jsonl(metrics_path)
        if not metrics:
            continue

        # Extract data
        steps = []
        objectives = []
        service_rates = []

        for m in metrics:
            if "step" in m:
                steps.append(m["step"])
            if "eval/mean_objective" in m:
                objectives.append(m["eval/mean_objective"])
            if "eval/mean_service_rate" in m:
                service_rates.append(m["eval/mean_service_rate"])

        # Plot objective
        ax = axes[0, 0]
        if objectives:
            ax.plot(
                steps[:len(objectives)],
                objectives,
                color=color,
                linewidth=2,
                marker="o",
                markersize=4,
                label=exp_name,
            )

        # Plot service rate
        ax = axes[0, 1]
        if service_rates:
            ax.plot(
                steps[:len(service_rates)],
                service_rates,
                color=color,
                linewidth=2,
                marker="s",
                markersize=4,
                label=exp_name,
            )

        # Plot final objective
        ax = axes[1, 0]
        if objectives:
            final_obj = objectives[-1]
            best_obj = min(objectives)
            ax.bar(
                idx,
                best_obj,
                color=color,
                alpha=0.7,
                label=f"{exp_name}: {best_obj:.1f}",
            )

        # Plot final service rate
        ax = axes[1, 1]
        if service_rates:
            final_sr = service_rates[-1]
            best_sr = max(service_rates)
            ax.bar(
                idx,
                best_sr,
                color=color,
                alpha=0.7,
                label=f"{exp_name}: {best_sr:.3f}",
            )

    # Configure subplots
    axes[0, 0].set_xlabel("Step")
    axes[0, 0].set_ylabel("Objective")
    axes[0, 0].set_title("Objective Convergence Comparison")
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend()

    axes[0, 1].set_xlabel("Step")
    axes[0, 1].set_ylabel("Service Rate")
    axes[0, 1].set_title("Service Rate Convergence Comparison")
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_ylim([0, 1.05])
    axes[0, 1].legend()

    axes[1, 0].set_ylabel("Best Objective")
    axes[1, 0].set_title("Best Objective by Experiment")
    axes[1, 0].grid(True, alpha=0.3, axis="y")

    axes[1, 1].set_ylabel("Best Service Rate")
    axes[1, 1].set_title("Best Service Rate by Experiment")
    axes[1, 1].set_ylim([0, 1.05])
    axes[1, 1].grid(True, alpha=0.3, axis="y")

    plt.tight_layout()

    # Save figure
    if output_dir is None:
        output_dir = Path("experiment/train") / "_comparison"
    output_dir.mkdir(parents=True, exist_ok=True)

    filename = f"convergence_comparison_{phase}.png"
    filepath = output_dir / filename
    plt.savefig(filepath, dpi=dpi, bbox_inches="tight")
    print(f"  Saved: {filepath}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Plot convergence curves from training experiments"
    )
    parser.add_argument(
        "--exp-ids",
        type=int,
        nargs="+",
        help="Experiment IDs to plot (e.g., 100 102 106)",
    )
    parser.add_argument(
        "--phase",
        type=str,
        default=None,
        help="Specific phase to plot (meta_learning, fine_tuning, training)",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Generate comparison plots across experiments",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=100,
        help="DPI for output images",
    )
    args = parser.parse_args()

    # Find experiments
    exp_dirs = find_exp_dirs(args.exp_ids)
    if not exp_dirs:
        print("No experiments found.")
        return

    print(f"Found {len(exp_dirs)} experiment(s)")

    # Plot per-experiment convergence
    for exp_dir in exp_dirs:
        print(f"\n{exp_dir.name}:")

        # Get phases (simplified - always "training")
        phases_tasks = get_phase_tasks(exp_dir)
        if not phases_tasks:
            print("  No metrics found")
            continue

        for phase in phases_tasks.keys():
            print(f"  Plotting convergence...")

            # Plot convergence
            plot_convergence_per_task(exp_dir, phase, dpi=args.dpi)

    # Generate comparison plots if requested
    if args.compare and len(exp_dirs) > 1:
        print(f"\nGenerating comparison plots across {len(exp_dirs)} experiments:")
        plot_convergence_comparison(exp_dirs, "training", dpi=args.dpi)

    print("\nDone!")


if __name__ == "__main__":
    main()
