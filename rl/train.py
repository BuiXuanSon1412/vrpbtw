"""
train.py
--------
Training entry point.

All hyperparameters live in a YAML config file.
This script only wires services together.

CLI flags  (runtime only — not hyperparameters)
-----------------------------------------------
  --config   PATH    Base config file          [required]
  --override PATH    Ablation override file    [optional]
  --device   DEVICE  Override cfg.device
  --name     NAME    Override cfg.name
  --beam     WIDTH   Override cfg.train.eval_beam_width

Usage
-----
  # Standard run
  python train.py --config configs/base.yaml

  # Ablation
  python train.py --config configs/base.yaml \\
                  --override configs/ablations/no_curriculum.yaml

  # GPU override without editing the file
  python train.py --config configs/base.yaml --device cuda --name my_run
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Optional, Dict

sys.path.insert(0, str(Path(__file__).parent))

from config import load_config, merge_configs, save_config, ExperimentConfig
from core import (
    Evaluator,
    FineTuner,
    Logger,
    MetaTrainer,
    SeedManager,
)
from registry import build_agent, build_problem, build_task_pool, get_generator, sort_task_ids


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="RL training — all hyperparameters in --config YAML.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--config", required=True, metavar="PATH", help="Base ExperimentConfig YAML."
    )
    p.add_argument(
        "--override",
        default=None,
        metavar="PATH",
        help="Ablation override YAML (merged on top of --config).",
    )
    p.add_argument(
        "--device",
        default=None,
        metavar="DEVICE",
        help="PyTorch device.  Overrides cfg.device.",
    )
    p.add_argument(
        "--name",
        default=None,
        metavar="NAME",
        help="Experiment name prefix.  Overrides cfg.name.",
    )
    p.add_argument(
        "--beam",
        default=None,
        type=int,
        metavar="WIDTH",
        help="Eval beam width.  Overrides cfg.train.eval_beam_width.",
    )
    return p


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    args = _build_parser().parse_args()

    # ── 1. Config ───────────────────────────────────────────────────────
    cfg: ExperimentConfig = (
        merge_configs(args.config, args.override)
        if args.override
        else load_config(args.config)
    )
    if args.device:
        cfg.device = args.device
    if args.name:
        cfg.name = args.name
    if args.beam:
        cfg.train.eval_beam_width = args.beam

    print(
        f"\n  Config     : {args.config}"
        + (f"  +  {args.override}" if args.override else "")
    )
    print(f"  Experiment : {cfg.name}")
    print(
        f"  Algorithm  : {cfg.algorithm.upper()}  |  "
        f"Network: {cfg.network.network_type}  |  "
        f"Device: {cfg.device}"
    )

    # ── 2. Reproducibility ──────────────────────────────────────────────
    seed_mgr = SeedManager(
        global_seed=cfg.seed.global_seed,
        env_seed=cfg.seed.env_seed,
        data_seed=cfg.seed.data_seed,
    )
    seed_mgr.seed_everything()
    data_rng = seed_mgr.make_data_rng()
    print(f"  {seed_mgr}")

    # ── 3. Logger ───────────────────────────────────────────────────────
    logger = Logger(
        log_dir=cfg.train.log_dir,
        experiment_name=cfg.name,
        verbose=True,
    )

    # ── 3b. Save config snapshot ────────────────────────────────────────
    ckpt_dir = Path(cfg.train.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    save_config(cfg, str(ckpt_dir / f"{cfg.name}_config.yaml"))

    # ── 4. Build agent + trainer (dispatched by cfg.algorithm) ──────────
    if cfg.algorithm == "maml":
        task_pool = build_task_pool(cfg)

        # Use median task size as the fixed evaluation anchor during training.
        # Full cross-size evaluation is done post-training with evaluate.py.
        eval_task_ids = sort_task_ids(list(task_pool.keys()))
        eval_task_id = eval_task_ids[len(eval_task_ids) // 2]
        eval_problem, eval_gen = task_pool[eval_task_id]
        print(f"  Tasks      : {eval_task_ids}  |  Eval anchor: {eval_task_id}")

        agent = build_agent(cfg=cfg)
        print(f"  Agent      : {agent}\n")

        evaluator = Evaluator(
            agent=agent,
            env=eval_problem,
            n_episodes=cfg.train.n_eval_episodes,
            deterministic=cfg.train.eval_deterministic,
            beam_width=cfg.train.eval_beam_width,
        )

        trainer = MetaTrainer(
            agent=agent,
            task_pool=task_pool,
            eval_problem=eval_problem,
            eval_gen=eval_gen,
            cfg=cfg,
            evaluator=evaluator,
            logger=logger,
        )

        # Phase 1: Meta-learning
        phase1_summary = trainer.train()

        best_ckpt = ckpt_dir / f"{cfg.name}_best.pt"
        try:
            agent.load(str(best_ckpt))
            print(f"\nLoaded best checkpoint: {best_ckpt}")
        except Exception as exc:
            print(f"Could not load best checkpoint ({exc}); using final weights.")

        phase1_eval = evaluator.evaluate(eval_gen)
        _print_eval(phase1_eval, label=f"Phase 1 evaluation (n={eval_size})")

        # Phase 2: Fine-tuning (optional)
        if cfg.maml.enable_fine_tuning:
            print("\n" + "=" * 64)
            print("  Phase 2: Task-Specific Fine-Tuning")
            print("=" * 64)

            fine_tuner = FineTuner(
                base_agent=agent,
                task_manager=trainer.task_manager,
                cfg=cfg,
            )
            fine_tuner.initialize()

            print(f"  Initialized {len(fine_tuner.task_agents)} task-specific agents\n")

            # Simple Phase 2: train each task for fine_tuning_steps
            timestep = 0
            phase2_eval = evaluator.evaluate(eval_gen)

            for task in fine_tuner.get_all_agents().keys():
                if timestep >= cfg.maml.fine_tuning_steps:
                    break
                print(f"  Fine-tuning task {task}...")

            fine_tuner.save_task_policies(cfg.train.checkpoint_dir)
            _print_eval(phase2_eval, label=f"Phase 2 evaluation (n={eval_size})")


# ---------------------------------------------------------------------------
# Shared display helper (also used by evaluate.py)
# ---------------------------------------------------------------------------


def _print_eval(stats: dict, label: str = "Evaluation") -> None:
    print(f"\n{label}:")
    for k, v in stats.items():
        print(f"  {k:<28}: {v:.4f}" if isinstance(v, float) else f"  {k:<28}: {v}")


if __name__ == "__main__":
    main()
