"""
infer.py
--------
Inference script for trained VRPBTW checkpoints.

Loads instances from data/generated/data and stores results in rl/result/N{size}/{filename}.json

Usage
-----
  python infer.py --checkpoint experiment/train/exp_name/checkpoints/checkpoint.pt
  python infer.py --checkpoint checkpoint.pt --device cuda
  python infer.py --checkpoint checkpoint.pt --beam 5 --beam-width 5
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))

import globals
from config import load_config
from core import SeedManager
from core.registry import build_agents, build_environment


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Run inference on VRPBTW instances using trained checkpoint",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python infer.py --exp my_experiment --checkpoint best_checkpoint
  python infer.py --exp my_experiment --checkpoint tune_best_checkpoint --device cuda
  python infer.py --exp my_experiment --checkpoint best_checkpoint --sizes 10,20,50 --limit 100
  python infer.py --checkpoint /path/to/checkpoint.pt  # Direct path (legacy)
        """,
    )

    # Either --exp + --checkpoint, or just --checkpoint (full path)
    p.add_argument(
        "--exp",
        default=None,
        metavar="NAME",
        help="Experiment name (in experiment/train/{exp_name}). Use with --checkpoint for checkpoint name.",
    )
    p.add_argument(
        "--checkpoint",
        default=None,
        metavar="NAME_OR_PATH",
        help="Checkpoint name (e.g., best_checkpoint) or full path to .pt file",
    )
    p.add_argument(
        "--exp-dir",
        default="experiment/train",
        metavar="PATH",
        help="Base experiment directory (default: experiment/train)",
    )
    p.add_argument(
        "--config",
        default=None,
        metavar="PATH",
        help="Training config (auto-detected if not provided)",
    )
    p.add_argument(
        "--device",
        default=None,
        metavar="DEVICE",
        help="Device to use (cpu or cuda)",
    )
    p.add_argument(
        "--deterministic",
        action="store_true",
        default=True,
        help="Use deterministic (greedy) decoding",
    )
    p.add_argument(
        "--beam-width",
        type=int,
        default=1,
        metavar="WIDTH",
        help="Beam search width (default: 1 = greedy)",
    )
    p.add_argument(
        "--data-dir",
        default="data/generated/data",
        metavar="PATH",
        help="Directory containing instance data (default: data/generated/data)",
    )
    p.add_argument(
        "--output-dir",
        default="rl/result",
        metavar="PATH",
        help="Output directory for results (default: rl/result)",
    )
    p.add_argument(
        "--sizes",
        default=None,
        metavar="N1,N2,...",
        help="Problem sizes to process (e.g., 10,20,50). If not set, process all.",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=None,
        metavar="N",
        help="Limit number of instances per size",
    )
    return p


def _find_config(checkpoint_path: Path) -> Path:
    """Find config.yaml in checkpoint parent directories."""
    current = checkpoint_path.parent
    while current != current.parent:  # Stop at root
        config_path = current / "config.yaml"
        if config_path.exists():
            return config_path
        current = current.parent
    raise FileNotFoundError(
        f"Could not find config.yaml for checkpoint {checkpoint_path}"
    )


def _convert_json_to_raw_instance(json_data: Dict[str, Any]) -> Dict[str, Any]:
    """Convert JSON instance format to raw_instance format for encode_instance.

    JSON format has:
    - Config.Depot: {id, coord, time_window_h}
    - Config.Vehicles.NUM_TRUCKS, NUM_DRONES
    - Nodes: [{id, coord, demand, type, tw_h}, ...]

    raw_instance format (for encode_instance):
    - depot: [x, y]
    - customers: [[x, y, tw_open, tw_close, demand], ...]
    - n_fleets: int
    """
    config = json_data["Config"]
    depot = config["Depot"]["coord"]
    n_fleets = config["Vehicles"]["NUM_TRUCKS"] + config["Vehicles"]["NUM_DRONES"]

    customers = []
    for node in json_data["Nodes"]:
        # node: {id, coord, demand, type, tw_h}
        x, y = node["coord"]
        tw_open, tw_close = node["tw_h"]
        demand = node["demand"]
        customers.append([x, y, tw_open, tw_close, demand])

    return {
        "depot": depot,
        "customers": customers,
        "n_fleets": n_fleets,
    }


def _run_inference(
    env: Any,
    agent: Any,
    raw_instance: Dict[str, Any],
    beam_width: int = 1,
    deterministic: bool = True,
) -> Dict[str, Any]:
    """Run inference on a single instance.

    Returns dict with solution metrics and metadata.
    """
    # Encode instance directly
    env.encode_instance(raw_instance)

    # Reset environment
    obs, info = env.reset()
    mask = info["action_mask"]

    start_time = time.time()
    ep_reward = 0.0
    actions: List[int] = []

    done = False
    while not done:
        feasible = np.where(mask)[0]
        if len(feasible) == 0:
            break

        # Convert obs to tensor
        from core.utils import obs_to_tensor

        obs_t = obs_to_tensor(obs, device=globals.DEVICE)
        mask_t = torch.tensor(mask, dtype=torch.bool, device=globals.DEVICE).unsqueeze(0)

        # Get action from agent
        action, _, _, _ = agent.act(obs_t, mask_t, deterministic=deterministic)
        action = int(action.item())

        obs, reward, terminated, truncated, info = env.step(action)
        mask = info["action_mask"]
        ep_reward += reward if reward is not None else 0.0
        actions.append(action)
        done = terminated or truncated

    elapsed = time.time() - start_time

    # Get solution
    sol = env.current_solution()

    return {
        "objective": sol.objective,
        "episode_reward": ep_reward,
        "total_cost": sol.metadata.get("total_cost", 0.0),
        "served_count": sol.metadata.get("served_count", 0),
        "n_customers": sol.metadata.get("n_customers", 0),
        "n_unserved": sol.metadata.get("unserved", 0),
        "time_s": elapsed,
        "num_actions": len(actions),
        "truck_routes": sol.metadata.get("truck_routes", []),
        "drone_trips": sol.metadata.get("drone_trips", []),
        "decision_sequence": actions,
    }


def main() -> None:
    args = _build_parser().parse_args()

    # Resolve checkpoint path and experiment
    if args.exp:
        # New syntax: --exp name --checkpoint checkpoint_name
        if not args.checkpoint:
            raise ValueError("--checkpoint is required when using --exp")

        exp_dir = Path(args.exp_dir) / args.exp
        if not exp_dir.exists():
            raise FileNotFoundError(f"Experiment not found: {exp_dir}")

        # Try to find checkpoint file
        checkpoint_name = args.checkpoint
        checkpoint_path = exp_dir / "checkpoints" / f"{checkpoint_name}.pt"

        if not checkpoint_path.exists():
            # Try alternate naming: {exp_name}_{checkpoint_name}.pt
            checkpoint_path = exp_dir / "checkpoints" / f"{args.exp}_{checkpoint_name}.pt"

        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"Checkpoint not found: {checkpoint_name}\n"
                f"  Tried: {exp_dir}/checkpoints/{args.checkpoint}.pt\n"
                f"  Tried: {exp_dir}/checkpoints/{args.exp}_{args.checkpoint}.pt"
            )

        # Auto-detect config path
        config_path = args.config or (exp_dir / "config.yaml")
        experiment_name = args.exp

    else:
        # Legacy syntax: --checkpoint /path/to/checkpoint.pt
        if not args.checkpoint:
            raise ValueError("Either --exp + --checkpoint or full --checkpoint path is required")

        checkpoint_path = Path(args.checkpoint)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        # Auto-detect config if not provided
        config_path = args.config or _find_config(checkpoint_path)
        experiment_name = checkpoint_path.stem.split("_")[0]

    if not Path(config_path).exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    print(f"\n{'=' * 70}")
    print(f"  Experiment: {experiment_name}")
    print(f"  Checkpoint: {checkpoint_path}")
    print(f"  Config: {config_path}")

    cfg = load_config(str(config_path))

    # Setup device
    device = args.device or cfg.get("device", "cpu")
    globals.DEVICE = device
    print(f"  Device: {device}")

    # Setup reproducibility
    reproducibility_cfg = cfg.get("reproducibility", {})
    seed_cfg = reproducibility_cfg.get("seed", {})
    seed_mgr = SeedManager(
        random_seed=seed_cfg.get("random_seed", 42),
        numpy_seed=seed_cfg.get("numpy_seed", 42),
        torch_seed=seed_cfg.get("torch_seed", 42),
    )
    seed_mgr.seed_everything()

    # Build environment and agent
    env = build_environment(cfg)
    agents = build_agents(cfg=cfg)
    agent = next(iter(agents.values()))

    # Load checkpoint weights
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if isinstance(checkpoint, dict) and "network_state" in checkpoint:
        agent.network.load_state_dict(checkpoint["network_state"])
    elif isinstance(checkpoint, dict) and "agent" in checkpoint:
        agent.network.load_state_dict(checkpoint["agent"])
    elif isinstance(checkpoint, dict):
        agent.network.load_state_dict(checkpoint)

    agent.network.eval()

    # Setup data directory and output directory
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)

    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    # Determine problem sizes to process
    if args.sizes:
        sizes = [f"N{s}" for s in args.sizes.split(",")]
    else:
        sizes = sorted([d.name for d in data_dir.iterdir() if d.is_dir()])

    print(f"  Processing {len(sizes)} problem size(s): {', '.join(sizes)}")
    print(f"  Output directory: {output_dir}\n")

    total_instances = 0
    total_success = 0

    for size_dir_name in sizes:
        size_path = data_dir / size_dir_name
        if not size_path.exists():
            print(f"  Warning: {size_path} not found, skipping")
            continue

        # Get all JSON files in this size directory
        json_files = sorted(size_path.glob("*.json"))
        if args.limit:
            json_files = json_files[: args.limit]

        print(f"  {size_dir_name}: {len(json_files)} instances")

        # Create output directory for this size
        size_output_dir = output_dir / size_dir_name
        size_output_dir.mkdir(parents=True, exist_ok=True)

        for json_file in json_files:
            total_instances += 1
            result_file = size_output_dir / f"{json_file.stem}.json"

            try:
                # Load instance from JSON
                with open(json_file) as f:
                    json_data = json.load(f)

                # Convert to raw_instance format
                raw_instance = _convert_json_to_raw_instance(json_data)

                # Run inference
                with torch.no_grad():
                    result = _run_inference(
                        env=env,
                        agent=agent,
                        raw_instance=raw_instance,
                        beam_width=args.beam_width,
                        deterministic=args.deterministic,
                    )

                # Save result
                with open(result_file, "w") as f:
                    json.dump(result, f, indent=2)

                total_success += 1

                # Progress output
                if total_success % 10 == 0:
                    print(
                        f"    [{total_success:4d}] {json_file.name} "
                        f"objective={result['objective']:8.2f} "
                        f"time={result['time_s']:.3f}s"
                    )

            except Exception as e:
                print(f"    Error processing {json_file.name}: {e}")
                import traceback

                traceback.print_exc()

    print(f"\n{'=' * 70}")
    print(f"  Results:")
    print(f"    Total instances: {total_instances}")
    print(f"    Successful: {total_success}")
    print(f"    Failed: {total_instances - total_success}")
    print(f"    Output directory: {output_dir}\n")


if __name__ == "__main__":
    main()
