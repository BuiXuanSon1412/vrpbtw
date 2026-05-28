"""Test that instances_per_batch=2 optimization doesn't cause memory issues."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np
from impl.mvrpbtw import MVRPBTWEnv
from core.trainer import POMOTrainer
from core.collector import POMOSampler
from core.agent import POMOAgent
from impl.geman import GEMANActorCritic
import globals


def test_pomo_memory_with_increased_batch():
    """Verify instances_per_batch=2 doesn't cause memory issues."""

    # Config with instances_per_batch=2
    cfg = {
        "env": "MVRPBTW",
        "tasks": ["easy_N50_F5_C"],
        "n_customers": 50,
        "max_coord": 100.0,
        "capacity_truck": 200.0,
        "capacity_drone": 20.0,
        "t_max_system_h": 24.0,
        "drone_duration_h": 1.0,
        "v_truck_km_h": 40.0,
        "v_drone_km_h": 60.0,
        "truck_cost_unit": 1.0,
        "drone_cost_unit": 0.5,
        "drone_takeoff_min": 1.0,
        "drone_land_min": 1.0,
        "service_time_min": 5.0,
    }

    # Initialize environment
    env = MVRPBTWEnv(cfg)

    # Initialize network and agent
    network_cfg = {
        "encoder": {
            "node_encoder": {"embedding_dim": 128, "n_heads": 4, "dropout": 0.1, "use_instance_norm": True},
            "vehicle_encoder": {"embedding_dim": 128, "n_heads": 4, "dropout": 0.1, "use_instance_norm": True},
            "graph_encoder": {"embedding_dim": 128, "n_layers": 3, "dropout": 0.1},
        },
        "decoder": {"node_decoder": {"clip": 10.0}},
    }

    globals.DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    network = GEMANActorCritic(network_cfg).to(globals.DEVICE)
    optimizer = torch.optim.Adam(network.parameters(), lr=0.001)
    agent = POMOAgent.from_config(
        {"entropy_coef": 0.1, "max_grad_norm": 0.5},
        network,
        optimizer
    )
    collector = POMOSampler.from_config({})

    # Test 5 batches with 2 instances each (simulating increased batch size)
    print("Testing POMO with instances_per_batch=2")
    print(f"Device: {globals.DEVICE}")

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        initial_memory = torch.cuda.memory_allocated() / 1024**2
        print(f"Initial GPU memory: {initial_memory:.2f} MB")

    for batch_idx in range(5):
        batch_log_probs = []
        batch_rewards = []
        batch_entropies = []

        # Collect 2 instances
        for instance_idx in range(2):
            env.retask("easy_N50_F5_C")
            batch_data = collector.collect(agent, env)

            if batch_data["log_probs"]:
                instance_log_probs = torch.stack(
                    [torch.as_tensor(lp, device=globals.DEVICE).squeeze(0) for lp in batch_data["log_probs"]]
                )
                instance_rewards = torch.tensor(
                    batch_data["rewards"], dtype=torch.float32, device=globals.DEVICE
                )
                instance_entropies = torch.stack(
                    [torch.as_tensor(ent, device=globals.DEVICE).squeeze(0) for ent in batch_data.get("entropies", [])]
                )
                batch_log_probs.append(instance_log_probs)
                batch_rewards.append(instance_rewards)
                batch_entropies.append(instance_entropies)

        # Update with batch of 2 instances
        if batch_log_probs:
            batch = {
                "log_probs": batch_log_probs,
                "rewards": batch_rewards,
                "entropies": batch_entropies,
            }
            metrics = agent.update(batch)

            if torch.cuda.is_available():
                current_memory = torch.cuda.memory_allocated() / 1024**2
                peak_memory = torch.cuda.max_memory_allocated() / 1024**2
                print(f"Batch {batch_idx+1}: Current={current_memory:.2f} MB, Peak={peak_memory:.2f} MB")

    if torch.cuda.is_available():
        peak_memory_total = torch.cuda.max_memory_allocated() / 1024**2
        print(f"\n✓ Peak GPU memory used: {peak_memory_total:.2f} MB")
        print(f"✓ No OOM errors with instances_per_batch=2")
    else:
        print(f"\n✓ Test passed on CPU (no CUDA available)")


if __name__ == "__main__":
    test_pomo_memory_with_increased_batch()
