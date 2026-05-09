#!/usr/bin/env python3
"""
Estimate GEMAN model size and memory requirements for different batch sizes.
This is the proper way to estimate - by actually instantiating the model.
"""

import torch
import torch.nn as nn
from impl.geman import GEMANActorCritic
import json

# Load the configuration
cfg = {
    "encoder": {
        "node_encoder": {
            "embedding_dim": 128,
            "n_heads": 4,
            "n_layers": 3,
            "dropout": 0.1,
            "use_instance_norm": True,
        },
        "vehicle_encoder": {
            "embedding_dim": 128,
            "n_heads": 4,
            "n_layers": 3,
            "dropout": 0.1,
            "use_instance_norm": True,
        },
        "graph_encoder": {
            "embedding_dim": 128,
            "n_layers": 3,
            "dropout": 0.1,
        },
    },
    "decoder": {
        "node_decoder": {
            "embedding_dim": 128,
            "clip": 10.0,
            "activation": "relu",
        },
        "vehicle_decoder": {
            "embedding_dim": 128,
            "clip": 10.0,
            "activation": "relu",
        },
    },
}

def count_parameters(model):
    """Count total trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def estimate_memory(model, batch_size, num_nodes, num_vehicles, device="cuda"):
    """
    Estimate memory usage for forward and backward pass.

    Args:
        model: neural network
        batch_size: number of instances per batch
        num_nodes: number of nodes per instance (N+1)
        num_vehicles: number of vehicles per instance (2K)
        device: 'cuda' or 'cpu'
    """
    model = model.to(device)

    # Create dummy input
    node_features = torch.randn(batch_size, num_nodes, 5, device=device)
    vehicle_features = torch.randn(batch_size, num_vehicles, 5, device=device)

    # Simple edge indices (same for all batch items)
    num_edges = min(num_nodes * num_nodes // 2, 100)  # reasonable estimate
    truck_edge_index = torch.randint(0, num_nodes, (2, num_edges), device=device)
    truck_edge_attr = torch.randn(batch_size, num_edges, 2, device=device)
    drone_edge_index = torch.randint(0, num_nodes, (2, num_edges), device=device)
    drone_edge_attr = torch.randn(batch_size, num_edges, 2, device=device)

    action_mask = torch.ones(batch_size, num_nodes, dtype=torch.bool, device=device)

    obs = {
        "node_features": node_features,
        "vehicle_features": vehicle_features,
        "truck_edge_index": truck_edge_index,
        "truck_edge_attr": truck_edge_attr,
        "drone_edge_index": drone_edge_index,
        "drone_edge_attr": drone_edge_attr,
    }

    # Forward pass
    try:
        logits, values = model.forward(obs, action_mask=action_mask)
        print(f"✓ Forward pass successful")
        print(f"  Logits shape: {logits.shape}")
        print(f"  Values shape: {values.shape}")
    except Exception as e:
        print(f"✗ Forward pass failed: {e}")
        return None

    # Backward pass (compute gradients)
    try:
        loss = (logits.sum() + values.sum())
        loss.backward()
        print(f"✓ Backward pass successful")
    except Exception as e:
        print(f"✗ Backward pass failed: {e}")
        return None

    return obs


def main():
    print("=" * 80)
    print("GEMAN Model Memory Estimation")
    print("=" * 80)

    # Instantiate model
    print("\n1. Creating model...")
    model = GEMANActorCritic(cfg)

    # Count parameters
    total_params = count_parameters(model)
    print(f"\n2. Parameter Count:")
    print(f"   Total trainable parameters: {total_params:,}")
    print(f"   Model size (float32): {total_params * 4 / (1024**2):.2f} MB")
    print(f"   Optimizer state (Adam, 2x): {total_params * 8 / (1024**2):.2f} MB")

    # Break down by module
    print(f"\n3. Parameter breakdown by module:")
    for name, module in model.named_modules():
        if isinstance(module, (nn.Linear, nn.MultiheadAttention)):
            params = count_parameters(module)
            if params > 0:
                print(f"   {name}: {params:,} params ({params*4/(1024**2):.2f} MB)")

    # Estimate activation memory for different batch sizes
    print(f"\n4. GPU Memory Usage for different batch sizes:")
    print(f"   (Assuming typical VRP: 50 nodes, 2 vehicles)")
    print()

    num_nodes = 50
    num_vehicles = 2

    results = []
    for batch_size in [4, 8, 16, 20, 32, 64]:
        print(f"   Testing batch_size={batch_size}...")
        try:
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.empty_cache()

            obs = estimate_memory(model, batch_size, num_nodes, num_vehicles, device="cuda")

            if obs is not None:
                peak_memory_mb = torch.cuda.max_memory_allocated() / (1024**2)
                peak_memory_gb = peak_memory_mb / 1024

                result = {
                    "batch_size": batch_size,
                    "peak_memory_mb": round(peak_memory_mb, 2),
                    "peak_memory_gb": round(peak_memory_gb, 2),
                }
                results.append(result)

                print(f"      Peak GPU memory: {peak_memory_gb:.2f} GB ({peak_memory_mb:.0f} MB)")
            print()
        except Exception as e:
            print(f"      Error: {e}\n")

    # Print summary table
    print(f"\n5. Summary Table:")
    print()
    print(f"{'Batch Size':<15} {'Peak Memory (MB)':<20} {'Peak Memory (GB)':<20}")
    print("-" * 55)
    for r in results:
        print(f"{r['batch_size']:<15} {r['peak_memory_mb']:<20.1f} {r['peak_memory_gb']:<20.2f}")

    # GPU recommendations
    print(f"\n6. GPU Recommendations:")
    print()
    vram_by_gpu = {
        "RTX 3060": 12,
        "RTX 4060": 8,
        "RTX 4070": 12,
        "RTX 4080": 16,
        "A100 40GB": 40,
        "H100 80GB": 80,
    }

    for gpu, vram_gb in vram_by_gpu.items():
        recommended_batch = None
        for r in results:
            if r['peak_memory_gb'] * 1.2 < vram_gb:  # 20% safety margin
                recommended_batch = r['batch_size']

        if recommended_batch:
            print(f"   {gpu:<20} ({vram_gb:>2}GB VRAM): instances_per_batch = {recommended_batch}")
        else:
            print(f"   {gpu:<20} ({vram_gb:>2}GB VRAM): batch too small to test")

    print()


if __name__ == "__main__":
    main()
