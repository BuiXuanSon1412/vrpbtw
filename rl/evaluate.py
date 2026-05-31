"""
evaluate.py
-----------
Standalone evaluation script for trained checkpoints.

Loads a trained checkpoint from a training experiment and evaluates it on fresh instances.
Configuration is read from configs/evaluate.yaml

Usage
-----
  # Evaluate all checkpoints using default config
  python evaluate.py

  # Evaluate specific checkpoint
  python evaluate.py --checkpoint tune_best_009_N50_F5_C.pt

  # Override settings
  python evaluate.py --episodes 1000 --beam 5 --device cuda
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))

import globals
from config import load_config, merge_configs, save_config
from core import Evaluator, SeedManager
from core.registry import build_agents, build_environment


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Evaluate trained RL checkpoints using evaluate.yaml config.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--config",
        default="configs/evaluate.yaml",
        metavar="PATH",
        help="Evaluation config file (default: configs/evaluate.yaml)",
    )
    p.add_argument(
        "--checkpoint",
        default=None,
        metavar="NAME",
        help="Specific checkpoint to evaluate (e.g., tune_best_009_N50_F5_C.pt). If not provided, evaluates all.",
    )
    p.add_argument(
        "--override",
        default=None,
        metavar="PATH",
        help="Optional config override YAML.",
    )
    p.add_argument(
        "--device",
        default=None,
        metavar="DEVICE",
        help="Override device from config (cpu or cuda)",
    )
    p.add_argument(
        "--episodes",
        type=int,
        default=None,
        metavar="N",
        help="Override number of evaluation instances from config",
    )
    p.add_argument(
        "--beam",
        type=int,
        default=None,
        metavar="WIDTH",
        help="Override beam width for decoding",
    )
    p.add_argument(
        "--samples",
        type=int,
        default=1,
        metavar="N",
        help="Best-of-N sampling (default: 1)",
    )
    return p


def _find_checkpoints(experiment_dir: Path) -> List[Path]:
    """Find all .pt checkpoint files in experiment directory."""
    checkpoint_dir = experiment_dir / "checkpoints"
    if not checkpoint_dir.exists():
        return []
    return sorted(checkpoint_dir.glob("*.pt"))


def _extract_task_id(checkpoint_name: str) -> str | None:
    """Extract task_id from checkpoint filename.

    Format: {exp_name}_tune_best_{task_id}.pt or {exp_name}_tune_final_{task_id}.pt
    Returns the task_id part.
    """
    # Remove .pt extension
    stem = checkpoint_name[:-3] if checkpoint_name.endswith(".pt") else checkpoint_name

    for prefix in ["tune_best_", "tune_final_"]:
        if prefix in stem:
            return stem.split(prefix, 1)[1]

    return None


def _evaluate_checkpoint(
    checkpoint_path: Path,
    cfg: Dict[str, Any],
    args: argparse.Namespace,
    output_dir: Path,
) -> Dict[str, Any]:
    """Evaluate a single checkpoint with batch progress output."""

    checkpoint_name = checkpoint_path.stem
    task_id = _extract_task_id(checkpoint_path.name)

    print(f"\n{'=' * 70}")
    print(f"  Checkpoint: {checkpoint_path.name}")
    if task_id:
        print(f"  Task ID   : {task_id}")

    # Set reproducibility
    reproducibility_cfg = cfg.get("reproducibility", {})
    seed_cfg = reproducibility_cfg.get("seed", {})
    seed_mgr = SeedManager(
        random_seed=seed_cfg.get("random_seed", 42),
        numpy_seed=seed_cfg.get("numpy_seed", 42),
        torch_seed=seed_cfg.get("torch_seed", 42),
    )
    seed_mgr.seed_everything()

    # Get device and evaluation settings
    device = args.device or cfg.get("device", "cpu")
    globals.DEVICE = device

    # Get evaluation config
    eval_cfg = cfg.get("evaluation", {})
    n_episodes = args.episodes or eval_cfg.get("n_episodes", 10000)
    beam_width = args.beam or eval_cfg.get("decoding", {}).get("beam_width", 1)
    deterministic = eval_cfg.get("deterministic", True)

    # Build environment and agents
    env = build_environment(cfg)
    agents = build_agents(cfg=cfg)

    # Get first agent (typically the eval agent)
    agent = next(iter(agents.values()))

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if isinstance(checkpoint, dict) and "network_state" in checkpoint:
        agent.network.load_state_dict(checkpoint["network_state"])
    elif isinstance(checkpoint, dict) and "agent" in checkpoint:
        agent.network.load_state_dict(checkpoint["agent"])
    elif isinstance(checkpoint, dict):
        agent.network.load_state_dict(checkpoint)

    # Evaluate in batches with progress output
    batch_size = 100
    num_batches = max(1, n_episodes // batch_size)
    actual_batch_size = n_episodes // num_batches

    print(f"  Evaluating {n_episodes} instances in {num_batches} batches...")

    all_batch_stats = []

    for batch_idx in range(num_batches):
        evaluator = Evaluator(
            agent=agent,
            env=env,
            n_episodes=actual_batch_size,
            deterministic=deterministic,
            n_samples=args.samples,
            beam_width=beam_width,
        )
        batch_stats = evaluator.evaluate(task_id)
        all_batch_stats.append(batch_stats)

        # Progress output
        mean_obj = batch_stats.get("mean_objective", 0)
        mean_cost = batch_stats.get("mean_cost", 0)
        mean_sr = batch_stats.get("mean_service_rate", 0)

        print(
            f"    Batch {batch_idx + 1:3d}/{num_batches} | "
            f"mean_objective: {mean_obj:8.2f} | "
            f"mean_cost: {mean_cost:7.2f} | "
            f"mean_service_rate: {mean_sr:.4f}"
        )

    # Aggregate statistics across batches
    aggregated_stats = {}
    for key in all_batch_stats[0].keys():
        values = [s[key] for s in all_batch_stats if isinstance(s[key], (int, float))]
        if values:
            aggregated_stats[key] = float(np.mean(values))
        else:
            aggregated_stats[key] = all_batch_stats[0][key]

    # Print final results
    print("\n  Results:")
    for k, v in aggregated_stats.items():
        if isinstance(v, float):
            print(f"    {k:<26}: {v:.4f}")
        else:
            print(f"    {k:<26}: {v}")

    # Save to CSV
    csv_path = output_dir / f"{checkpoint_name}.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        for k, v in aggregated_stats.items():
            writer.writerow([k, v])

    print(f"  CSV saved: {csv_path}")

    return aggregated_stats


def main() -> None:
    args = _build_parser().parse_args()

    # Load evaluate config
    config_path = args.config
    if not Path(config_path).exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    eval_cfg = (
        merge_configs(config_path, args.override)
        if args.override
        else load_config(config_path)
    )

    print(f"\n  Evaluate Config: {config_path}")

    # Get experiment names
    eval_exp_name = eval_cfg.get("experiment", {}).get("name")
    if not eval_exp_name:
        raise ValueError("experiment.name must be specified in evaluate config")

    dirs_cfg = eval_cfg.get("directories", {})
    training_exp_name = dirs_cfg.get("source")
    if not training_exp_name:
        raise ValueError("directories.source must be specified in evaluate config")

    # Load training config
    experiment_dir = Path("experiment/train") / training_exp_name
    if not experiment_dir.exists():
        raise FileNotFoundError(
            f"Training experiment not found: {experiment_dir}\n"
            f"Make sure 'directories.source' in {config_path} points to a valid experiment."
        )

    training_config_path = experiment_dir / "config.yaml"
    if not training_config_path.exists():
        raise FileNotFoundError(
            f"Training config not found: {training_config_path}\n"
            f"Expected to find config.yaml in training experiment directory."
        )

    # Load training config and merge with evaluate config
    cfg = load_config(str(training_config_path))

    # Merge evaluate settings
    if "device" in eval_cfg:
        cfg["device"] = eval_cfg["device"]
    if "evaluation" in eval_cfg:
        cfg["evaluation"] = eval_cfg["evaluation"]
    if "directories" in eval_cfg:
        cfg["directories"] = eval_cfg["directories"]
    if "reproducibility" in eval_cfg:
        cfg["reproducibility"] = eval_cfg["reproducibility"]

    # Setup output directory
    base_dir = dirs_cfg.get("base_dir", "experiment/evaluate")
    output_dir = Path(base_dir) / training_exp_name / eval_exp_name
    artifacts_dir = output_dir / dirs_cfg.get("artifacts", "artifacts")
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    # Save evaluate config
    eval_config_path = output_dir / "evaluate.yaml"
    save_config(eval_cfg, str(eval_config_path))

    print(f"  Training Experiment: {training_exp_name}")
    print(f"  Evaluation Experiment: {eval_exp_name}")
    print(f"  Training Config: {training_config_path}")
    print(f"  Output Directory: {output_dir}")

    # Find checkpoints
    if args.checkpoint:
        checkpoint_path = experiment_dir / "checkpoints" / f"{args.checkpoint}.pt"
        if not checkpoint_path.exists():
            # Try without .pt extension if provided
            checkpoint_path = experiment_dir / "checkpoints" / args.checkpoint
            if not checkpoint_path.exists():
                raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
        checkpoints = [checkpoint_path]
    else:
        checkpoints = _find_checkpoints(experiment_dir)
        if not checkpoints:
            raise FileNotFoundError(
                f"No checkpoints found in {experiment_dir}/checkpoints/"
            )

    print(f"  Found {len(checkpoints)} checkpoint(s)")

    # Evaluate each checkpoint
    results = {}
    for checkpoint_path in checkpoints:
        results[checkpoint_path.name] = _evaluate_checkpoint(
            checkpoint_path, cfg, args, artifacts_dir
        )

    # Summary for multiple checkpoints
    if len(checkpoints) > 1:
        print(f"\n{'=' * 70}")
        print("\nSummary of all checkpoints:")

        summary_csv = artifacts_dir / "summary.csv"
        metrics = list(next(iter(results.values())).keys())

        with open(summary_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["checkpoint"] + metrics)
            for ckpt_name, stats in results.items():
                row = [ckpt_name] + [stats.get(m, "") for m in metrics]
                writer.writerow(row)

        print(f"  Summary CSV saved: {summary_csv}\n")

        for ckpt_name, stats in results.items():
            print(f"\n  {ckpt_name}:")
            for k, v in stats.items():
                if isinstance(v, float):
                    print(f"    {k:<26}: {v:.4f}")
                else:
                    print(f"    {k:<26}: {v}")

    print(f"\n  Evaluation results saved to: {output_dir}\n")


if __name__ == "__main__":
    main()
