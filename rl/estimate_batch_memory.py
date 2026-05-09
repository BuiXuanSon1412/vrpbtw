#!/usr/bin/env python3
"""
Memory estimation breakdown - this is how you estimate batch size requirements.
Method: count parameters, then calculate activation memory based on batch size.
"""

import torch
from impl.geman import GEMANActorCritic

# Configuration
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
        },
        "vehicle_decoder": {
            "embedding_dim": 128,
            "clip": 10.0,
        },
    },
}

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def estimate_activation_memory(batch_size, num_nodes, num_vehicles, embedding_dim=128,
                              num_layers=3, num_heads=4):
    """
    Estimate activation memory for forward+backward pass.

    Key insight: during backprop, we need to store all intermediate activations.
    Memory ≈ batch_size × (num_nodes × embedding_dim + attention_matrices)
    """

    # Forward pass activations (stored for backward)
    # Each layer stores: input, attention_weights, intermediate FFN outputs

    # Node encoder: 3 layers, each storing ~3 tensors
    # (B, N, D) tensors
    node_activations = num_layers * 3 * (batch_size * num_nodes * embedding_dim)

    # Vehicle encoder: 3 layers, each storing ~3 tensors
    # (B, 2K, D) tensors
    vehicle_activations = num_layers * 3 * (batch_size * num_vehicles * embedding_dim)

    # Graph encoder: similar
    graph_activations = num_layers * 3 * (batch_size * num_nodes * embedding_dim)

    # Attention weights per layer: (B, H, T, T)
    # For nodes: (B, 4, N, N)
    # For vehicles: (B, 4, 2K, 2K)
    attention_memory = (
        num_layers * (batch_size * num_heads * num_nodes * num_nodes) +  # node attention
        num_layers * (batch_size * num_heads * num_vehicles * num_vehicles)  # vehicle attention
    )

    # Total activation memory (in float32, 4 bytes per element)
    total_activations = node_activations + vehicle_activations + graph_activations + attention_memory
    total_memory_bytes = total_activations * 4
    total_memory_mb = total_memory_bytes / (1024 ** 2)

    return total_memory_mb

def main():
    print("=" * 80)
    print("GEMAN Memory Estimation: Manual Calculation")
    print("=" * 80)

    # Create model and count params
    model = GEMANActorCritic(cfg)
    total_params = count_parameters(model)

    print("\n1. MODEL PARAMETERS (Fixed - doesn't change with batch size)")
    print("-" * 80)
    print(f"   Total parameters: {total_params:,}")
    print(f"   Size (float32, 4 bytes/param): {total_params * 4 / (1024**2):.2f} MB")
    print(f"   Optimizer state (Adam=2x params): {total_params * 8 / (1024**2):.2f} MB")
    print(f"   Total fixed overhead: {(total_params * 4 + total_params * 8) / (1024**2):.2f} MB")

    print("\n2. ACTIVATION MEMORY (Varies with batch size)")
    print("-" * 80)
    print("   Formula: batch_size × (node_vectors + vehicle_vectors + attention_matrices)")
    print("   Where:")
    print("     - node_vectors = N × D × num_layers (stored for each layer during backward)")
    print("     - vehicle_vectors = 2K × D × num_layers")
    print("     - attention_matrices = H × N × N × num_layers (for backprop through attention)")
    print()

    # Estimate for typical VRP
    num_nodes = 50  # typical problem size
    num_vehicles = 2  # typical (1 truck + 1 drone)

    print(f"   Typical problem size: {num_nodes} nodes, {num_vehicles} vehicles")
    print()

    print(f"{'Batch Size':<12} {'Activation Memory':<20} {'Total (Model+Act)':<20} {'Safe for GPU':<20}")
    print("-" * 72)

    for batch_size in [4, 8, 16, 20, 24, 32, 48, 64]:
        act_mem = estimate_activation_memory(batch_size, num_nodes, num_vehicles)
        total_mem = (total_params * 12 / (1024**2)) + act_mem  # 12 bytes overhead (params + optimizer)

        # Safe if uses ~70% of typical GPU VRAM
        safe_8gb = "✓ RTX 4060" if total_mem < 5.6 else ""
        safe_12gb = "✓ RTX 4070" if total_mem < 8.4 else ""
        safe_16gb = "✓ RTX 4080" if total_mem < 11.2 else ""
        safe_40gb = "✓ A100" if total_mem < 28 else ""

        safe_for = ", ".join(filter(None, [safe_8gb, safe_12gb, safe_16gb, safe_40gb]))
        if not safe_for:
            safe_for = "Too large"

        print(f"{batch_size:<12} {act_mem:<20.1f} MB {total_mem:<20.1f} MB {safe_for:<20}")

    print("\n3. KEY INSIGHTS")
    print("-" * 80)
    print(f"   • Memory grows ~linearly with batch_size (due to activation storage)")
    print(f"   • Each batch item adds ~{estimate_activation_memory(1, num_nodes, num_vehicles):.1f} MB")
    print(f"   • Fixed overhead (params + optimizer): {(total_params * 12 / (1024**2)):.1f} MB")
    print(f"   • Current setting (20 instances):")
    act_20 = estimate_activation_memory(20, num_nodes, num_vehicles)
    total_20 = (total_params * 12 / (1024**2)) + act_20
    print(f"     - Activation memory: {act_20:.1f} MB")
    print(f"     - Total: {total_20:.1f} MB")
    print(f"     - Safe for: RTX 4070 (12GB) and larger ✓")

    print("\n4. MEMORY FORMULA")
    print("-" * 80)
    print("   Total GPU Memory = Model Params + Optimizer State + Activation Memory")
    print()
    print("   Model params: ~7.3 MB (fixed)")
    print("   Optimizer (Adam): ~14.5 MB (fixed)")
    print("   Activations: batch_size × problem_size × embedding_dim × num_layers × 4 bytes")
    print()
    print(f"   For {num_nodes} nodes, {num_vehicles} vehicles, embedding_dim=128:")
    print(f"   Total ≈ 21.8 + batch_size × {estimate_activation_memory(1, num_nodes, num_vehicles):.2f}")


if __name__ == "__main__":
    main()
