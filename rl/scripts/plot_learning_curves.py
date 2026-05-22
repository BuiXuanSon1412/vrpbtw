"""
scripts/plot_learning_curves.py
-------------------------------
Plot learning curves from training experiments.
Saves plots to experiment/train/{exp_id}/artifacts/

Reads from:
  - summary.json: final results per task (always available)
  - logs/metrics.jsonl: per-epoch training metrics (when available)

Usage
-----
  # Plot all experiments
  python scripts/plot_learning_curves.py

  # Plot specific experiments
  python scripts/plot_learning_curves.py --exp-ids 101 102 103

  # Regenerate plots for a single experiment
  python scripts/plot_learning_curves.py --exp-ids 101
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np


def load_experiment_summary(exp_dir: Path) -> Optional[Dict]:
    """Load summary.json from an experiment directory."""
    summary_path = exp_dir / "summary.json"
    if not summary_path.exists():
        return None
    with open(summary_path) as f:
        return json.load(f)


def load_metrics_jsonl(exp_dir: Path) -> Tuple[List[Dict], Dict[int, List[Dict]]]:
    """Load metrics from metrics.jsonl.

    Returns:
        (all_entries, metrics_by_step) where:
        - all_entries: list of all entries in order
        - metrics_by_step: dict mapping step -> list of metric entries at that step
    """
    metrics_path = exp_dir / "logs" / "metrics.jsonl"
    if not metrics_path.exists():
        return [], {}

    all_entries = []
    metrics_by_step = {}

    try:
        with open(metrics_path) as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    entry = json.loads(line)
                    all_entries.append(entry)

                    # Index by step for quick lookup
                    if "step" in entry:
                        step = int(entry["step"])
                        if step not in metrics_by_step:
                            metrics_by_step[step] = []
                        metrics_by_step[step].append(entry)
                except json.JSONDecodeError:
                    continue
    except Exception as e:
        print(f"  Warning: error reading metrics.jsonl: {e}")
        return [], {}

    return all_entries, metrics_by_step


def extract_metric_timeline(metrics_by_step: Dict[int, List[Dict]], key: str) -> Tuple[List[int], List[float]]:
    """Extract a specific metric across steps.

    Args:
        metrics_by_step: dict from load_metrics_jsonl
        key: metric key like "train/loss_mean" or "eval/mean_objective"

    Returns:
        (steps, values): sorted by step
    """
    steps = []
    values = []

    for step in sorted(metrics_by_step.keys()):
        entries = metrics_by_step[step]
        # Find entry with this metric key
        for entry in entries:
            if key in entry:
                steps.append(step)
                values.append(float(entry[key]))
                break

    return steps, values


def plot_per_epoch_metrics(exp_id: int, artifacts_dir: Path, metrics_by_step: Dict[int, List[Dict]]) -> None:
    """Plot per-epoch training curves from metrics.jsonl."""
    if not metrics_by_step:
        return

    artifacts_dir.mkdir(parents=True, exist_ok=True)

    # Extract key metrics over time
    steps_loss, loss_means = extract_metric_timeline(metrics_by_step, "train/loss_mean")
    steps_return, return_means = extract_metric_timeline(metrics_by_step, "train/return_mean")
    steps_return_std, return_stds = extract_metric_timeline(metrics_by_step, "train/return_std")
    steps_eval_obj, eval_objs = extract_metric_timeline(metrics_by_step, "eval/mean_objective")
    steps_eval_cost, eval_costs = extract_metric_timeline(metrics_by_step, "eval/mean_cost")
    steps_eval_sr, eval_srs = extract_metric_timeline(metrics_by_step, "eval/mean_service_rate")

    # Plot 1: Training loss over time
    if steps_loss:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(steps_loss, loss_means, "o-", linewidth=2, markersize=6, color="tab:red")
        ax.set_xlabel("Update Step", fontsize=12)
        ax.set_ylabel("Loss", fontsize=12)
        ax.set_title(f"Experiment {exp_id}: Training Loss Over Time", fontsize=14)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        output_path = artifacts_dir / "01_training_loss.png"
        plt.savefig(output_path, dpi=150)
        print(f"  Saved: {output_path.relative_to(artifacts_dir.parent.parent)}")
        plt.close()

    # Plot 2: Training return (with std band)
    if steps_return:
        fig, ax = plt.subplots(figsize=(10, 6))
        return_means_arr = np.array(return_means)
        return_stds_arr = np.array(return_stds) if return_stds else np.zeros_like(return_means_arr)
        ax.plot(
            steps_return,
            return_means_arr,
            "o-",
            linewidth=2,
            markersize=6,
            color="tab:blue",
            label="Mean",
        )
        ax.fill_between(
            steps_return,
            return_means_arr - return_stds_arr,
            return_means_arr + return_stds_arr,
            alpha=0.2,
            color="tab:blue",
            label="±1 Std",
        )
        ax.set_xlabel("Update Step", fontsize=12)
        ax.set_ylabel("Return", fontsize=12)
        ax.set_title(f"Experiment {exp_id}: Training Return Over Time", fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        output_path = artifacts_dir / "02_training_return.png"
        plt.savefig(output_path, dpi=150)
        print(f"  Saved: {output_path.relative_to(artifacts_dir.parent.parent)}")
        plt.close()

    # Plot 3: Evaluation metrics (objective)
    if steps_eval_obj:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(steps_eval_obj, eval_objs, "s-", linewidth=2, markersize=8, color="tab:green")
        ax.set_xlabel("Update Step", fontsize=12)
        ax.set_ylabel("Mean Objective", fontsize=12)
        ax.set_title(f"Experiment {exp_id}: Evaluation Objective Over Time", fontsize=14)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        output_path = artifacts_dir / "03_eval_objective.png"
        plt.savefig(output_path, dpi=150)
        print(f"  Saved: {output_path.relative_to(artifacts_dir.parent.parent)}")
        plt.close()

    # Plot 4: Multi-metric comparison
    if steps_eval_cost or steps_eval_sr:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        if steps_eval_cost:
            axes[0].plot(
                steps_eval_cost,
                eval_costs,
                "^-",
                linewidth=2,
                markersize=8,
                color="tab:orange",
            )
            axes[0].set_xlabel("Update Step", fontsize=11)
            axes[0].set_ylabel("Mean Cost", fontsize=11)
            axes[0].set_title("Evaluation Cost Over Time")
            axes[0].grid(True, alpha=0.3)
        else:
            axes[0].axis("off")

        if steps_eval_sr:
            axes[1].plot(
                steps_eval_sr,
                eval_srs,
                "d-",
                linewidth=2,
                markersize=8,
                color="tab:purple",
            )
            axes[1].set_xlabel("Update Step", fontsize=11)
            axes[1].set_ylabel("Mean Service Rate", fontsize=11)
            axes[1].set_title("Evaluation Service Rate Over Time")
            axes[1].grid(True, alpha=0.3)
        else:
            axes[1].axis("off")

        plt.tight_layout()
        output_path = artifacts_dir / "04_eval_metrics.png"
        plt.savefig(output_path, dpi=150)
        print(f"  Saved: {output_path.relative_to(artifacts_dir.parent.parent)}")
        plt.close()


def plot_experiment_summary(exp_id: int, summary: Dict, artifacts_dir: Path) -> None:
    """Create plots for a single experiment."""
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    task_summaries = summary.get("task_summaries", {})
    if not task_summaries:
        print(f"  No task summaries found for experiment {exp_id}")
        return

    # Extract data for each task
    task_names = list(task_summaries.keys())
    best_objectives = [task.get("best_objective", 0) for task in task_summaries.values()]
    best_epochs = [task.get("best_epoch", 0) for task in task_summaries.values()]
    final_epochs = [task.get("final_epoch", 0) for task in task_summaries.values()]

    # Plot 1: Best objectives by task
    fig, ax = plt.subplots(figsize=(12, 6))
    x_pos = np.arange(len(task_names))
    bars = ax.bar(x_pos, best_objectives, color="steelblue", alpha=0.7, edgecolor="black")
    ax.set_xlabel("Task", fontsize=12)
    ax.set_ylabel("Best Objective Value", fontsize=12)
    ax.set_title(f"Experiment {exp_id}: Best Objectives by Task", fontsize=14)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(task_names, rotation=45, ha="right")
    ax.grid(True, alpha=0.3, axis="y")

    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height,
            f"{height:.1f}",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    plt.tight_layout()
    output_path = artifacts_dir / "best_objectives_by_task.png"
    plt.savefig(output_path, dpi=150)
    print(f"  Saved: {output_path.relative_to(artifacts_dir.parent.parent)}")
    plt.close()

    # Plot 2: Best epoch achieved per task
    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.bar(x_pos, best_epochs, color="darkgreen", alpha=0.7, edgecolor="black")
    ax.set_xlabel("Task", fontsize=12)
    ax.set_ylabel("Epoch", fontsize=12)
    ax.set_title(f"Experiment {exp_id}: Epoch at Best Performance", fontsize=14)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(task_names, rotation=45, ha="right")
    ax.grid(True, alpha=0.3, axis="y")

    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height,
            f"{int(height)}",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    plt.tight_layout()
    output_path = artifacts_dir / "best_epochs_by_task.png"
    plt.savefig(output_path, dpi=150)
    print(f"  Saved: {output_path.relative_to(artifacts_dir.parent.parent)}")
    plt.close()

    # Plot 3: Training duration (final epoch) per task
    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.bar(x_pos, final_epochs, color="darkorange", alpha=0.7, edgecolor="black")
    ax.set_xlabel("Task", fontsize=12)
    ax.set_ylabel("Epoch", fontsize=12)
    ax.set_title(f"Experiment {exp_id}: Training Duration (Final Epoch)", fontsize=14)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(task_names, rotation=45, ha="right")
    ax.grid(True, alpha=0.3, axis="y")

    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height,
            f"{int(height)}",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    plt.tight_layout()
    output_path = artifacts_dir / "final_epochs_by_task.png"
    plt.savefig(output_path, dpi=150)
    print(f"  Saved: {output_path.relative_to(artifacts_dir.parent.parent)}")
    plt.close()

    # Plot 4: Summary metrics (2x2)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Best objectives
    axes[0, 0].bar(x_pos, best_objectives, color="steelblue", alpha=0.7, edgecolor="black")
    axes[0, 0].set_title("Best Objectives")
    axes[0, 0].set_xticks(x_pos)
    axes[0, 0].set_xticklabels(task_names, rotation=45, ha="right", fontsize=9)
    axes[0, 0].grid(True, alpha=0.3, axis="y")

    # Best epochs
    axes[0, 1].bar(x_pos, best_epochs, color="darkgreen", alpha=0.7, edgecolor="black")
    axes[0, 1].set_title("Epoch at Best Performance")
    axes[0, 1].set_xticks(x_pos)
    axes[0, 1].set_xticklabels(task_names, rotation=45, ha="right", fontsize=9)
    axes[0, 1].grid(True, alpha=0.3, axis="y")

    # Final epochs
    axes[1, 0].bar(x_pos, final_epochs, color="darkorange", alpha=0.7, edgecolor="black")
    axes[1, 0].set_title("Training Duration (Final Epoch)")
    axes[1, 0].set_xticks(x_pos)
    axes[1, 0].set_xticklabels(task_names, rotation=45, ha="right", fontsize=9)
    axes[1, 0].grid(True, alpha=0.3, axis="y")

    # Summary stats
    axes[1, 1].axis("off")
    stats_text = f"""
Experiment {exp_id} Summary
{'=' * 40}

Training Time: {summary.get('training_time_s', 0):.1f}s
Total Updates: {summary.get('total_updates', 0):,}
Total Instances: {summary.get('total_instances', 0):,}
Stop Reason: {summary.get('stop_reason', 'unknown')}

Avg Best Objective: {np.mean(best_objectives):.2f}
Avg Best Epoch: {np.mean(best_epochs):.1f}
Avg Final Epoch: {np.mean(final_epochs):.1f}
Number of Tasks: {len(task_names)}
"""
    axes[1, 1].text(0.1, 0.5, stats_text, fontsize=11, family="monospace", va="center")

    plt.tight_layout()
    output_path = artifacts_dir / "summary.png"
    plt.savefig(output_path, dpi=150)
    print(f"  Saved: {output_path.relative_to(artifacts_dir.parent.parent)}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Plot learning curves from training experiments to artifacts/ directories."
    )
    parser.add_argument(
        "--exp-ids",
        type=int,
        nargs="+",
        default=None,
        help="Specific experiment IDs to plot (default: all)",
    )

    args = parser.parse_args()

    # Determine paths
    script_dir = Path(__file__).parent
    project_dir = script_dir.parent
    train_dir = project_dir / "experiment" / "train"

    if not train_dir.exists():
        print(f"No experiments directory found at {train_dir}")
        return

    # Find experiments
    experiments = {}
    for exp_dir in sorted(train_dir.iterdir()):
        if not exp_dir.is_dir():
            continue
        try:
            exp_id = int(exp_dir.name)
            summary = load_experiment_summary(exp_dir)
            if summary:
                experiments[exp_id] = (exp_id, summary, exp_dir)
        except ValueError:
            continue

    # Filter by exp_ids if specified
    if args.exp_ids:
        experiments = {
            k: v for k, v in experiments.items() if k in args.exp_ids
        }

    if not experiments:
        print("No experiments found")
        return

    print(f"Found {len(experiments)} experiments: {sorted(experiments.keys())}\n")

    # Generate plots for each experiment
    for exp_id, (_, summary, exp_dir) in sorted(experiments.items()):
        print(f"Plotting experiment {exp_id}...")
        artifacts_dir = exp_dir / "artifacts"

        # Try to load per-epoch metrics from jsonl
        _, metrics_by_step = load_metrics_jsonl(exp_dir)
        if metrics_by_step:
            print(f"  Found {len(metrics_by_step)} metric steps in metrics.jsonl")
            plot_per_epoch_metrics(exp_id, artifacts_dir, metrics_by_step)
        else:
            print(f"  No detailed metrics.jsonl (or empty)")

        # Always plot summary-based plots
        plot_experiment_summary(exp_id, summary, artifacts_dir)

    print("\nDone!")


if __name__ == "__main__":
    main()
