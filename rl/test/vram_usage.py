#!/usr/bin/env python3
"""
test/vram_usage.py
------------------
Measure GPU VRAM usage for trainer updates on each task.

Metrics:
- Peak VRAM: Maximum GPU memory during update step
- Allocated VRAM: GPU memory allocated (not necessarily used)
- Reserved VRAM: GPU memory reserved by PyTorch

Measures update step VRAM for:
- MetaTrainer (meta-learning)
- POMOTrainer (POMO algorithm)

Usage:
  python test/vram_usage.py                    # All tasks, all trainers
  python test/vram_usage.py --trainer meta     # Specific trainer
  python test/vram_usage.py --task 001_N10_F2_RC   # Specific task
  python test/vram_usage.py --batch-size 32   # Custom batch size
"""

import argparse
import copy
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

import globals
from config import load_config
from core import SeedManager
from core.registry import build_agents, build_environment, build_trainer


def get_active_tasks(cfg: Dict) -> List[str]:
    """Extract active tasks from mvrpbtw.yaml config."""
    env_cfg = cfg.get("environment", {})
    properties = env_cfg.get("properties", {})
    tasks = properties.get("tasks", [])
    return tasks


def get_available_trainers() -> List[str]:
    """Get list of available trainer types."""
    return ["meta", "pomo"]


def measure_vram_before_after() -> Tuple[float, float, float]:
    """
    Measure current GPU memory usage.

    Returns:
        (allocated_mb, reserved_mb, cached_mb)
    """
    if not torch.cuda.is_available():
        return 0.0, 0.0, 0.0

    torch.cuda.synchronize()
    allocated = torch.cuda.memory_allocated() / 1024 / 1024  # MB
    reserved = torch.cuda.memory_reserved() / 1024 / 1024  # MB
    cached = torch.cuda.memory_reserved() - torch.cuda.memory_allocated()
    cached = cached / 1024 / 1024  # MB

    return allocated, reserved, cached


def collect_sample_batch(
    trainer: Any,
    task_id: str,
    batch_size: int = 4,
) -> Dict[str, Any]:
    """
    Collect a sample batch for training using the trainer's collector.

    Returns:
        batch dict suitable for trainer.update()
    """
    collector = trainer.collector

    # Collect episodes
    episodes = []
    for _ in range(batch_size):
        episode = collector.collect_episode(task_id=task_id)
        episodes.append(episode)

    # Create batch
    batch = collector.make_batch(episodes)

    return batch


def measure_trainer_update(
    cfg: Dict,
    task_id: str,
    trainer_type: str,
    batch_size: int = 4,
    verbose: bool = False,
) -> Dict[str, Any]:
    """
    Measure VRAM usage during trainer update step.

    Returns:
        dict with vram metrics
    """
    if not torch.cuda.is_available():
        return {
            "task_id": task_id,
            "trainer_type": trainer_type,
            "vram_allocated_mb": 0.0,
            "vram_reserved_mb": 0.0,
            "peak_vram_mb": 0.0,
            "batch_size": batch_size,
            "error": "CUDA not available",
        }

    # Setup
    device = cfg.get("device", "cuda")
    globals.DEVICE = device

    reproducibility_cfg = cfg.get("reproducibility", {})
    seed_cfg = reproducibility_cfg.get("seed", {})
    seed_mgr = SeedManager(
        random_seed=seed_cfg.get("random_seed", 42),
        numpy_seed=seed_cfg.get("numpy_seed", 42),
        torch_seed=seed_cfg.get("torch_seed", 42),
    )
    seed_mgr.seed_everything()

    try:
        # Build environment and agents
        env = build_environment(cfg)
        agents = build_agents(cfg=cfg)

        # Override trainer type in config
        cfg_copy = copy.deepcopy(cfg)
        if "trainer" not in cfg_copy:
            cfg_copy["trainer"] = {}
        cfg_copy["trainer"]["name"] = trainer_type

        # Build trainer (no logger needed)
        trainer = build_trainer(cfg_copy, agents, env, logger=None)

        if verbose:
            print(f"  Collecting batch ({batch_size} episodes)...")

        # Collect batch
        batch = collect_sample_batch(trainer, task_id, batch_size)

        if verbose:
            print(f"  Performing update step...")

        # Clear cache before measurement
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        # Measure before
        vram_before_alloc, vram_before_resv, _ = measure_vram_before_after()

        # Update step (this is where VRAM is consumed)
        start_time = time.time()
        update_metrics = trainer.update(batch)
        update_time = time.time() - start_time

        # Synchronize and measure peak
        torch.cuda.synchronize()
        peak_vram = torch.cuda.max_memory_allocated() / 1024 / 1024  # MB

        # Measure after
        vram_after_alloc, vram_after_resv, _ = measure_vram_before_after()

        # Cleanup
        del trainer
        del agents
        del env
        torch.cuda.empty_cache()

        return {
            "task_id": task_id,
            "trainer_type": trainer_type,
            "batch_size": batch_size,
            "vram_allocated_before_mb": float(vram_before_alloc),
            "vram_allocated_after_mb": float(vram_after_alloc),
            "vram_allocated_delta_mb": float(vram_after_alloc - vram_before_alloc),
            "vram_reserved_before_mb": float(vram_before_resv),
            "vram_reserved_after_mb": float(vram_after_resv),
            "vram_reserved_delta_mb": float(vram_after_resv - vram_before_resv),
            "peak_vram_mb": float(peak_vram),
            "update_time_s": float(update_time),
            "success": True,
        }

    except Exception as e:
        return {
            "task_id": task_id,
            "trainer_type": trainer_type,
            "batch_size": batch_size,
            "peak_vram_mb": 0.0,
            "success": False,
            "error": str(e),
        }


def main():
    parser = argparse.ArgumentParser(
        description="Measure GPU VRAM usage for trainer updates"
    )
    parser.add_argument(
        "--config",
        default="configs/train.yaml",
        help="Training config (default: configs/train.yaml)",
    )
    parser.add_argument(
        "--trainer",
        default=None,
        help="Specific trainer to measure (meta, pomo)",
    )
    parser.add_argument(
        "--task",
        default=None,
        help="Specific task to measure",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Batch size for collecting episodes (default: 4)",
    )
    args = parser.parse_args()

    # Check CUDA
    if not torch.cuda.is_available():
        print("Error: CUDA not available. This test requires GPU.")
        sys.exit(1)

    # Load config
    cfg = load_config(args.config)
    cfg["device"] = "cuda"
    globals.DEVICE = "cuda"

    # Get tasks and trainers
    tasks = get_active_tasks(cfg)
    trainers = get_available_trainers()

    if args.task:
        if args.task not in tasks:
            print(f"Error: Task {args.task} not found")
            print(f"   Available: {tasks}")
            sys.exit(1)
        tasks = [args.task]

    if args.trainer:
        if args.trainer not in trainers:
            print(f"Error: Trainer {args.trainer} not found")
            print(f"   Available: {trainers}")
            sys.exit(1)
        trainers = [args.trainer]

    print("=" * 100)
    print("GPU VRAM USAGE FOR TRAINER UPDATES")
    print("=" * 100)
    print(f"Config: {args.config}")
    print(f"Device: cuda")
    print(f"Batch size: {args.batch_size}")
    print(f"Trainers: {trainers}")
    print(f"Tasks: {len(tasks)}")
    print()

    results = []

    for task_id in tasks:
        for trainer_type in trainers:
            print(f"Measuring: {task_id} + {trainer_type}")

            result = measure_trainer_update(
                cfg,
                task_id,
                trainer_type,
                batch_size=args.batch_size,
                verbose=True,
            )

            if result.get("success", False):
                peak_vram = result["peak_vram_mb"]
                alloc_delta = result["vram_allocated_delta_mb"]
                print(
                    f"    Peak VRAM: {peak_vram:.1f}MB, "
                    f"Allocated Δ: {alloc_delta:.1f}MB, "
                    f"Update time: {result['update_time_s']:.3f}s"
                )
            else:
                print(f"    Error: {result.get('error', 'Unknown')}")

            results.append(result)
            print()

    # Print summary table
    print("=" * 100)
    print("SUMMARY TABLE")
    print("=" * 100)
    print()

    # Filter successful results
    successful = [r for r in results if r.get("success", False)]

    if not successful:
        print("Error: No successful measurements")
        sys.exit(1)

    print("PEAK VRAM USAGE:")
    print("-" * 100)
    print(
        f"{'Task':<20} {'Trainer':<15} {'Peak VRAM (MB)':<20} {'Alloc Δ (MB)':<20} {'Time (s)':<15}"
    )
    print("-" * 100)

    for r in successful:
        task = r["task_id"]
        trainer = r["trainer_type"]
        peak = r["peak_vram_mb"]
        delta = r["vram_allocated_delta_mb"]
        time_s = r["update_time_s"]
        print(f"{task:<20} {trainer:<15} {peak:>18.1f} {delta:>18.1f} {time_s:>13.3f}")

    print()
    print("VRAM ALLOCATION COMPARISON:")
    print("-" * 100)
    print(
        f"{'Task':<20} {'Trainer':<15} {'Before (MB)':<20} {'After (MB)':<20} {'Delta (MB)':<15}"
    )
    print("-" * 100)

    for r in successful:
        task = r["task_id"]
        trainer = r["trainer_type"]
        before = r["vram_allocated_before_mb"]
        after = r["vram_allocated_after_mb"]
        delta = r["vram_allocated_delta_mb"]
        print(f"{task:<20} {trainer:<15} {before:>18.1f} {after:>18.1f} {delta:>13.1f}")

    print()
    print("DETAILS:")
    print("-" * 100)

    for r in successful:
        print(f"\n{r['task_id']} + {r['trainer_type']}:")
        print(f"  Batch size:           {r['batch_size']}")
        print(f"  Peak VRAM:            {r['peak_vram_mb']:.1f}MB")
        print(f"  Allocated (before):   {r['vram_allocated_before_mb']:.1f}MB")
        print(f"  Allocated (after):    {r['vram_allocated_after_mb']:.1f}MB")
        print(f"  Allocated Δ:          {r['vram_allocated_delta_mb']:.1f}MB")
        print(f"  Reserved (before):    {r['vram_reserved_before_mb']:.1f}MB")
        print(f"  Reserved (after):     {r['vram_reserved_after_mb']:.1f}MB")
        print(f"  Reserved Δ:           {r['vram_reserved_delta_mb']:.1f}MB")
        print(f"  Update time:          {r['update_time_s']:.3f}s")

    print()
    print("=" * 100)

    # Summary statistics
    print("\nSUMMARY STATISTICS:")
    print("-" * 100)

    by_trainer = {}
    for r in successful:
        trainer = r["trainer_type"]
        if trainer not in by_trainer:
            by_trainer[trainer] = []
        by_trainer[trainer].append(r["peak_vram_mb"])

    for trainer, vram_values in sorted(by_trainer.items()):
        mean_vram = float(np.mean(vram_values))
        max_vram = float(np.max(vram_values))
        print(
            f"{trainer:<15} | Mean peak VRAM: {mean_vram:6.1f}MB | Max peak VRAM: {max_vram:6.1f}MB"
        )

    print()


if __name__ == "__main__":
    main()
