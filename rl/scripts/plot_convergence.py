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
import matplotlib.cm as cm
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


def find_exp_dirs(exp_specs: Optional[List[str]] = None) -> List[Path]:
    """Find experiment directories by ID or name.

    Supports both formats:
    - Numeric IDs: "100" matches "100_PPO_N10_RC"
    - Official names: "PPO_N10_RC" matches exact directory "PPO_N10_RC"

    Args:
        exp_specs: List of experiment identifiers (numeric IDs or exact names)
    """
    base_dir = Path("experiment/train")
    if not base_dir.exists():
        return []

    # Parse specs into numeric IDs and names
    numeric_ids = []
    names = []
    if exp_specs:
        for spec in exp_specs:
            try:
                numeric_ids.append(int(spec))
            except ValueError:
                names.append(spec)

    exp_dirs = []
    for d in sorted(base_dir.iterdir()):
        if not d.is_dir():
            continue

        if exp_specs is None:
            # No filter, include all
            exp_dirs.append(d)
        else:
            # Try to extract numeric ID from start of directory name
            try:
                dir_id = int(d.name.split("_")[0])
                if dir_id in numeric_ids:
                    exp_dirs.append(d)
                    continue
            except (ValueError, IndexError):
                pass

            # Try exact directory name match
            if d.name in names:
                exp_dirs.append(d)

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

    # Extract metric arrays (aligned by step)
    steps_loss = []
    loss_totals = []
    steps_value = []
    value_losses = []
    steps_grad = []
    grad_norms = []
    steps_entropy = []
    entropy_losses = []
    steps_obj = []
    objectives = []
    steps_sr = []
    service_rates = []

    for m in metrics:
        if "step" not in m:
            continue

        # Handle loss_total_mean from fine-tuning and loss_mean from meta-learning
        loss_key = None
        for key in ["tune/loss_total_mean", "train/loss_total_mean", "meta/loss_mean", "tune/loss_mean", "train/loss_mean"]:
            if key in m:
                loss_key = key
                break
        if loss_key:
            steps_loss.append(m["step"])
            loss_totals.append(m[loss_key])

        # Extract value loss
        value_key = None
        for key in ["tune/loss_value_mean", "train/loss_value_mean", "meta/loss_value_mean"]:
            if key in m:
                value_key = key
                break
        if value_key:
            steps_value.append(m["step"])
            value_losses.append(m[value_key])

        grad_key = None
        for key in ["tune/grad_norm_mean", "train/grad_norm_mean"]:
            if key in m:
                grad_key = key
                break
        if grad_key:
            steps_grad.append(m["step"])
            grad_norms.append(m[grad_key])

        # Extract entropy loss
        entropy_key = None
        for key in ["tune/loss_entropy_mean", "train/loss_entropy_mean", "meta/loss_entropy_mean"]:
            if key in m:
                entropy_key = key
                break
        if entropy_key:
            steps_entropy.append(m["step"])
            entropy_losses.append(m[entropy_key])

        if "eval/mean_objective" in m:
            steps_obj.append(m["step"])
            objectives.append(m["eval/mean_objective"])

        if "eval/mean_service_rate" in m:
            steps_sr.append(m["step"])
            service_rates.append(m["eval/mean_service_rate"])

    if not (steps_loss or steps_value or steps_obj or steps_sr or steps_grad or steps_entropy):
        return

    if output_dir is None:
        output_dir = exp_dir / "artifacts"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Plot 1: Loss convergence (total loss per update)
    if loss_totals:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(steps_loss, loss_totals, "b-", linewidth=1.5, label="loss_total")
        ax.set_xlabel("Epoch")
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

    # Plot 1b: Value loss convergence
    if value_losses:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(steps_value, value_losses, "c-", linewidth=1.5, label="loss_value")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Value Loss")
        ax.set_title(f"Value Loss Convergence - {exp_dir.name}")
        ax.grid(True, alpha=0.3)
        ax.legend()
        plt.tight_layout()

        filename = f"convergence_{phase}_value_loss.png"
        filepath = output_dir / filename
        plt.savefig(filepath, dpi=dpi, bbox_inches="tight")
        print(f"  Saved: {filepath}")
        plt.close()

    # Plot 2: Objective convergence
    if objectives:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(
            steps_obj,
            objectives,
            "g-",
            linewidth=2,
            label="mean_objective",
        )
        ax.axhline(
            y=max(objectives),
            color="r",
            linestyle="--",
            alpha=0.5,
            label=f"best={max(objectives):.2f}",
        )
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Objective (Service Value - Cost)")
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
            steps_sr,
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
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Service Rate")
        ax.set_title(f"Service Rate Convergence - {exp_dir.name}")
        ax.set_ylim((0, 1.05))
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
        ax.plot(steps_grad, grad_norms, "orange", linewidth=2, label="grad_norm_mean")
        ax.axhline(y=1.0, color="r", linestyle="--", alpha=0.5, label="safe_bound=1.0")
        ax.set_xlabel("Epoch")
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

    # Plot 5: Entropy convergence (policy learning/convergence indicator)
    if entropy_losses:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(steps_entropy, entropy_losses, "purple", linewidth=2, label="entropy_loss")
        ax.axhline(
            y=0.0,
            color="g",
            linestyle="--",
            alpha=0.5,
            label="zero_entropy (full convergence)",
        )
        ax.fill_between(
            steps_entropy,
            entropy_losses,
            0,
            alpha=0.2,
            color="purple",
        )
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Entropy Loss (−mean entropy)")
        ax.set_title(f"Policy Entropy Convergence - {exp_dir.name}")
        ax.grid(True, alpha=0.3)
        ax.legend()
        plt.tight_layout()

        filename = f"convergence_{phase}_entropy.png"
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

    fig, axes = plt.subplots(2, 3, figsize=(20, 10))
    fig.suptitle(f"Convergence Comparison (Experiments)", fontsize=14, fontweight="bold")

    cmap = cm.get_cmap('tab10')
    colors = cmap(np.linspace(0, 1, len(exp_dirs)))

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

        # Extract data (aligned by metric type)
        steps_obj = []
        objectives = []
        steps_sr = []
        service_rates = []
        steps_entropy = []
        entropy_losses = []

        for m in metrics:
            if "eval/mean_objective" in m:
                steps_obj.append(m["step"])
                objectives.append(m["eval/mean_objective"])
            if "eval/mean_service_rate" in m:
                steps_sr.append(m["step"])
                service_rates.append(m["eval/mean_service_rate"])
            # Extract entropy loss
            entropy_key = None
            for key in ["tune/loss_entropy_mean", "train/loss_entropy_mean", "meta/loss_entropy_mean"]:
                if key in m:
                    entropy_key = key
                    break
            if entropy_key:
                steps_entropy.append(m["step"])
                entropy_losses.append(m[entropy_key])

        # Plot objective
        ax = axes[0, 0]
        if objectives:
            ax.plot(
                steps_obj,
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
                steps_sr,
                service_rates,
                color=color,
                linewidth=2,
                marker="s",
                markersize=4,
                label=exp_name,
            )

        # Plot best objective
        ax = axes[1, 0]
        if objectives:
            best_obj = max(objectives)
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

        # Plot entropy convergence
        ax = axes[0, 2]
        if entropy_losses:
            ax.plot(
                steps_entropy,
                entropy_losses,
                color=color,
                linewidth=2,
                marker="^",
                markersize=4,
                label=exp_name,
            )

        # Plot final entropy
        ax = axes[1, 2]
        if entropy_losses:
            final_entropy = entropy_losses[-1]
            min_entropy = min(entropy_losses)
            ax.bar(
                idx,
                min_entropy,
                color=color,
                alpha=0.7,
                label=f"{exp_name}: {min_entropy:.4f}",
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
    axes[0, 1].set_ylim((0, 1.05))
    axes[0, 1].legend()

    axes[0, 2].set_xlabel("Step")
    axes[0, 2].set_ylabel("Entropy Loss (−mean entropy)")
    axes[0, 2].set_title("Policy Entropy Convergence Comparison")
    axes[0, 2].grid(True, alpha=0.3)
    axes[0, 2].legend()

    axes[1, 0].set_ylabel("Best Objective")
    axes[1, 0].set_title("Best Objective by Experiment")
    axes[1, 0].grid(True, alpha=0.3, axis="y")

    axes[1, 1].set_ylabel("Best Service Rate")
    axes[1, 1].set_title("Best Service Rate by Experiment")
    axes[1, 1].set_ylim((0, 1.05))
    axes[1, 1].grid(True, alpha=0.3, axis="y")

    axes[1, 2].set_ylabel("Min Entropy Loss")
    axes[1, 2].set_title("Best (Lowest) Entropy by Experiment")
    axes[1, 2].grid(True, alpha=0.3, axis="y")

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
        nargs="+",
        help="Experiment IDs (numeric, e.g., 100 102) or names (e.g., PPO_N10_RC MetaTrainer_NoMetaLearning)",
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
