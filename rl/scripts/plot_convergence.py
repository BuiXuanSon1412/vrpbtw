"""
scripts/plot_convergence.py
---------------------------
Plot convergence curves from training experiments.

Generates visualizations for:
- Per-experiment convergence
- Comparison across experiments (grouped by problem size)
- Loss curves, objective curves, service rate curves, gradient norm stability

Usage
-----
  # Plot all experiments
  python scripts/plot_convergence.py

  # Plot specific experiments
  python scripts/plot_convergence.py 100 102 106

  # Plot by name
  python scripts/plot_convergence.py GCN_PPO_N10_R GCN_PPO_N10_C

  # High resolution output
  python scripts/plot_convergence.py 100 102 --dpi 150

  # Output comparison plots as PNG
  python scripts/plot_convergence.py --formats png

  # Output in multiple formats
  python scripts/plot_convergence.py --formats pdf,png
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.ticker as ticker
import numpy as np

plt.rcParams.update(
    {
        "text.usetex": False,
        "font.family": "serif",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


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


def extract_problem_size(exp_name: str) -> Optional[str]:
    """Extract problem size (N10, N20, etc.) from experiment name.

    Examples:
        "GCN_PPO_N10_R" -> "N10"
        "100_PPO_N10_RC" -> "N10"
        "test_experiment" -> None
    """
    import re
    match = re.search(r"(N\d+)", exp_name)
    return match.group(1) if match else None


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

        # Only process GCN_PPO_ experiments
        if not d.name.startswith("GCN_PPO_"):
            continue

        if exp_specs is None:
            # No filter, include all GCN_PPO_ experiments
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
        for key in [
            "tune/loss_total_mean",
            "train/loss_total_mean",
            "meta/loss_mean",
            "tune/loss_mean",
            "train/loss_mean",
        ]:
            if key in m:
                loss_key = key
                break
        if loss_key:
            steps_loss.append(m["step"])
            loss_totals.append(m[loss_key])

        # Extract value loss
        value_key = None
        for key in [
            "tune/loss_value_mean",
            "train/loss_value_mean",
            "meta/loss_value_mean",
        ]:
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
        for key in [
            "tune/loss_entropy_mean",
            "train/loss_entropy_mean",
            "meta/loss_entropy_mean",
        ]:
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

    if not (
        steps_loss
        or steps_value
        or steps_obj
        or steps_sr
        or steps_grad
        or steps_entropy
    ):
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
        ax.plot(
            steps_entropy, entropy_losses, "purple", linewidth=2, label="entropy_loss"
        )
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


def extract_distribution(exp_name: str) -> Optional[str]:
    """Extract distribution type (C, RC, R) from experiment name.

    Examples:
        "GCN_PPO_N150_C" -> "C"
        "GCN_PPO_N100_RC" -> "RC"
        "GCN_PPO_N50_R" -> "R"
    """
    match = re.search(r"_(C|RC|R)$", exp_name)
    return match.group(1) if match else None


def get_dist_color_and_label(dist: Optional[str]) -> Tuple[str, str]:
    """Get color and legend label for distribution type.

    Returns:
        (color, label) tuple
    """
    dist_map = {
        "C": ("#1f77b4", "Clustered (C)"),      # blue
        "RC": "#2ca02c",  # green
        "R": "#ff7f0e",   # orange
    }

    if dist == "C":
        return "#1f77b4", "Clustered (C)"
    elif dist == "RC":
        return "#2ca02c", "Mixed (RC)"
    elif dist == "R":
        return "#ff7f0e", "Random (R)"
    else:
        return "#808080", "Unknown"  # gray fallback


def plot_convergence_comparison(
    exp_dirs: List[Path],
    phase: str,
    problem_size: Optional[str] = None,
    output_dir: Optional[Path] = None,
    dpi: int = 100,
    formats: Optional[List[str]] = None,
) -> None:
    """Compare service rate convergence across experiments of same problem size.

    Generates visualizations with colors: blue=C, green=RC, orange=R
    Legend shows: Clustered (C), Mixed (RC), Random (R)

    Args:
        formats: List of output formats ('pdf', 'png'). Defaults to ['pdf'].
    """
    if formats is None:
        formats = ['pdf']
    if not exp_dirs or not all(d.exists() for d in exp_dirs):
        return

    # Create figure with 4:3 ratio (12:9)
    fig, ax = plt.subplots(figsize=(12, 9))

    # Track which distributions we've already added to legend
    legend_added = set()

    # First pass: calculate min/max objective for this size group
    min_obj = float('inf')
    max_obj = float('-inf')

    for exp_dir in sorted(exp_dirs):
        metrics_path = exp_dir / "logs" / "metrics.jsonl"
        if not metrics_path.exists():
            continue

        metrics = load_metrics_jsonl(metrics_path)
        if not metrics:
            continue

        for m in metrics:
            if "eval/mean_objective" in m:
                obj = m["eval/mean_objective"]
                min_obj = min(min_obj, obj)
                max_obj = max(max_obj, obj)

    # Calculate ylim with 10% padding
    if min_obj != float('inf') and max_obj != float('-inf'):
        range_val = max_obj - min_obj
        padding = range_val * 0.10
        ylim_min = max(0, min_obj - padding)
        ylim_max = max_obj + padding
    else:
        ylim_min, ylim_max = 0, 1000000

    # Second pass: plot data
    for exp_dir in sorted(exp_dirs):
        exp_name = exp_dir.name
        dist = extract_distribution(exp_name)

        if dist is None:
            continue

        # Load metrics from single metrics.jsonl file
        metrics_path = exp_dir / "logs" / "metrics.jsonl"
        if not metrics_path.exists():
            continue

        metrics = load_metrics_jsonl(metrics_path)
        if not metrics:
            continue

        # Extract objective data
        steps_obj = []
        objectives = []

        for m in metrics:
            if "eval/mean_objective" in m:
                steps_obj.append(m["step"])
                objectives.append(m["eval/mean_objective"])

        # Plot objective convergence
        if objectives:
            color, label = get_dist_color_and_label(dist)

            # Only add to legend if this distribution hasn't been added yet
            if dist not in legend_added:
                ax.plot(
                    steps_obj,
                    objectives,
                    color=color,
                    linewidth=2.5,
                    label=label,
                    alpha=0.85,
                )
                legend_added.add(dist)
            else:
                # Plot without label to avoid duplicate legend entries
                ax.plot(
                    steps_obj,
                    objectives,
                    color=color,
                    linewidth=2.5,
                    alpha=0.85,
                )

    # Configure subplot with quadrupled font sizes (doubled twice)
    ax.set_xlabel("Epoch", fontsize=48)
    ax.set_ylabel("Objective (×1e6)", fontsize=48)
    ax.tick_params(axis='both', which='major', labelsize=40)
    # Set y-axis limits based on actual data range
    ax.set_ylim((ylim_min, ylim_max))
    # Format y-axis to show values in millions (divide by 1e6)
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: f'{x/1e6:.1f}'))
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=44, loc="lower right")

    plt.tight_layout()

    # Save figure in requested formats
    if output_dir is None:
        if problem_size:
            output_dir = Path("experiment/train") / "_comparison"
        else:
            output_dir = Path("experiment/train") / "_comparison"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save in each requested format
    for fmt in formats:
        filename = f"convergence_GCN_PPO_{problem_size}.{fmt}" if problem_size else f"convergence_comparison.{fmt}"
        filepath = output_dir / filename
        plt.savefig(filepath, dpi=dpi, bbox_inches="tight", format=fmt)
        print(f"  Saved: {filepath}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Plot convergence curves from training experiments"
    )
    parser.add_argument(
        "experiments",
        nargs="*",
        help="Experiment IDs (numeric, e.g., 100 102) or names (e.g., GCN_PPO_N10_R). If empty, plot all.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=100,
        help="DPI for output images",
    )
    parser.add_argument(
        "--formats",
        type=str,
        default="pdf",
        help="Output formats for comparison plots: 'pdf', 'png', or comma-separated list (e.g., 'pdf,png'). Default: pdf",
    )
    args = parser.parse_args()

    # Find experiments
    exp_dirs = find_exp_dirs(args.experiments if args.experiments else None)
    if not exp_dirs:
        print("No experiments found.")
        return

    print(f"Found {len(exp_dirs)} experiment(s)\n")

    # Plot per-experiment convergence
    print("Plotting convergence curves:")
    for exp_dir in exp_dirs:
        print(f"  {exp_dir.name}")

        # Get phases (simplified - always "training")
        phases_tasks = get_phase_tasks(exp_dir)
        if not phases_tasks:
            print("    No metrics found")
            continue

        for phase in phases_tasks.keys():
            # Plot convergence
            plot_convergence_per_task(exp_dir, phase, dpi=args.dpi)

    # Parse formats argument
    formats = [fmt.strip().lower() for fmt in args.formats.split(",")]
    valid_formats = {"pdf", "png"}
    formats = [fmt for fmt in formats if fmt in valid_formats]
    if not formats:
        formats = ["pdf"]

    # Generate comparison plots (grouped by problem size)
    if len(exp_dirs) > 1:
        print(f"\nGrouping by problem size and generating comparisons:")

        # Group experiments by problem size
        size_groups: Dict[Optional[str], List[Path]] = {}
        for exp_dir in exp_dirs:
            size = extract_problem_size(exp_dir.name)
            if size not in size_groups:
                size_groups[size] = []
            size_groups[size].append(exp_dir)

        # Generate comparison plots for each size group
        for size in sorted(size_groups.keys(), key=lambda x: (x is None, x)):
            exp_list = size_groups[size]
            if len(exp_list) > 1:
                size_label = size if size else "unknown"
                print(f"  {size_label}: {len(exp_list)} experiments")
                plot_convergence_comparison(exp_list, "training", problem_size=size, dpi=args.dpi, formats=formats)
            else:
                size_label = size if size else "unknown"
                print(f"  {size_label}: {len(exp_list)} experiment (skipped, need >1 for comparison)")

    print("\nDone!")


if __name__ == "__main__":
    main()
