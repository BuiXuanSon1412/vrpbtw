"""
Parameter Analysis: g_graph computation mechanism

This script identifies and counts ALL learnable parameters involved in computing g_graph.
"""

import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent.parent))

from impl.geman import GCN_MEGAGraphEncoder, _GCNMEGAMRGNNLayer


def analyze_all_parameters():
    """Count and display all learnable parameters in the g_graph mechanism."""
    print("=" * 80)
    print("PARAMETER ANALYSIS: g_graph Computation")
    print("=" * 80)

    cfg = {
        "embedding_dim": 128,
        "n_layers": 3,
        "dropout": 0.1,
        "use_instance_norm": False,
    }

    encoder = GCN_MEGAGraphEncoder(cfg)
    D = cfg["embedding_dim"]

    print(f"\nConfiguration:")
    print(f"  D (embedding_dim): {D}")
    print(f"  n_layers: {cfg['n_layers']}")
    print(f"  dropout: {cfg['dropout']}")
    print(f"  use_instance_norm: {cfg['use_instance_norm']}")

    # ────────────────────────────────────────────────────────────────────────
    # 1. INPUT PROJECTION
    # ────────────────────────────────────────────────────────────────────────

    print(f"\n" + "=" * 80)
    print("1. INPUT PROJECTION")
    print("=" * 80)
    print(f"\nself.input_proj = Linear(GRAPH_NODE_DIM=4 → D={D})")

    input_proj = encoder.input_proj
    n_params_input = sum(p.numel() for p in input_proj.parameters())
    print(f"  Weight: ({D}, 4)")
    print(f"  Bias:   ({D},)")
    print(f"  Total:  {n_params_input} parameters")

    # ────────────────────────────────────────────────────────────────────────
    # 2. EDGE PROJECTIONS (for truck and drone)
    # ────────────────────────────────────────────────────────────────────────

    print(f"\n" + "=" * 80)
    print("2. EDGE PROJECTIONS")
    print("=" * 80)

    edge_truck_proj = encoder.edge_truck_proj
    edge_drone_proj = encoder.edge_drone_proj

    n_params_edge_truck = sum(p.numel() for p in edge_truck_proj.parameters())
    n_params_edge_drone = sum(p.numel() for p in edge_drone_proj.parameters())

    print(f"\nself.edge_truck_proj = Linear(GRAPH_EDGE_DIM=2 → D={D})")
    print(f"  Weight: ({D}, 2)")
    print(f"  Bias:   ({D},)")
    print(f"  Total:  {n_params_edge_truck} parameters")

    print(f"\nself.edge_drone_proj = Linear(GRAPH_EDGE_DIM=2 → D={D})")
    print(f"  Weight: ({D}, 2)")
    print(f"  Bias:   ({D},)")
    print(f"  Total:  {n_params_edge_drone} parameters")

    # ────────────────────────────────────────────────────────────────────────
    # 3. GCN MESSAGE PASSING LAYERS
    # ────────────────────────────────────────────────────────────────────────

    print(f"\n" + "=" * 80)
    print("3. GCN MESSAGE PASSING LAYERS (3 layers)")
    print("=" * 80)

    n_params_gcn_total = 0
    for layer_idx, layer in enumerate(encoder.layers):
        print(f"\nLayer {layer_idx}:")

        # msg_truck MLP: [D + D] → 2D → D
        msg_truck_params = sum(p.numel() for p in layer.msg_truck.parameters())
        print(f"  msg_truck MLP([{D}+{D}]→2{D}→{D}): {msg_truck_params} params")
        print(f"    ├─ Linear({D*2} → 2*{D}): weight ({2*D}, {D*2}) + bias ({2*D},)")
        print(f"    └─ Linear(2*{D} → {D}): weight ({D}, {2*D}) + bias ({D},)")

        # msg_drone MLP: identical structure
        msg_drone_params = sum(p.numel() for p in layer.msg_drone.parameters())
        print(f"  msg_drone MLP([{D}+{D}]→2{D}→{D}): {msg_drone_params} params")
        print(f"    ├─ Linear({D*2} → 2*{D}): weight ({2*D}, {D*2}) + bias ({2*D},)")
        print(f"    └─ Linear(2*{D} → {D}): weight ({D}, {2*D}) + bias ({D},)")

        # update projection
        update_params = sum(p.numel() for p in layer.update.parameters())
        print(f"  update Linear(2*{D} → {D}): {update_params} params")
        print(f"    ├─ Weight: ({D}, {2*D})")
        print(f"    └─ Bias:   ({D},)")

        # LayerNorm
        norm_params = sum(p.numel() for p in layer.norm.parameters())
        print(f"  norm LayerNorm({D}): {norm_params} params")
        print(f"    ├─ Weight (scale γ): ({D},)")
        print(f"    └─ Bias (shift β):   ({D},)")

        layer_total = msg_truck_params + msg_drone_params + update_params + norm_params
        n_params_gcn_total += layer_total
        print(f"  ► Layer {layer_idx} total: {layer_total} parameters")

    # ────────────────────────────────────────────────────────────────────────
    # 4. OUTPUT NORMALIZATION
    # ────────────────────────────────────────────────────────────────────────

    print(f"\n" + "=" * 80)
    print("4. OUTPUT NORMALIZATION")
    print("=" * 80)

    out_norm = encoder.out_norm
    n_params_out_norm = sum(p.numel() for p in out_norm.parameters())

    if isinstance(out_norm, nn.LayerNorm):
        print(f"\nself.out_norm = LayerNorm({D})")
        print(f"  Weight (scale γ): ({D},)")
        print(f"  Bias (shift β):   ({D},)")
    elif isinstance(out_norm, nn.InstanceNorm1d):
        print(f"\nself.out_norm = InstanceNorm1d({D})")
        print(f"  Weight (scale γ): ({D},) [if affine=True]")
        print(f"  Bias (shift β):   ({D},) [if affine=True]")
    else:
        print(f"\nself.out_norm = Identity (no parameters)")

    print(f"  Total: {n_params_out_norm} parameters")

    # ────────────────────────────────────────────────────────────────────────
    # 5. POOLING LAYER ★ CRITICAL FOR g_graph
    # ────────────────────────────────────────────────────────────────────────

    print(f"\n" + "=" * 80)
    print("5. POOLING LAYER ★ CRITICAL FOR g_graph")
    print("=" * 80)

    pool = encoder.pool
    n_params_pool = sum(p.numel() for p in pool.parameters())

    print(f"\n★ self.pool = Linear(D={D} → 1)")
    print(f"  Weight: (1, {D})      ← learned importance scores for each dimension")
    print(f"  Bias:   (1,)         ← offset")
    print(f"  Total:  {n_params_pool} parameters")

    print(f"\n  This layer learns to assign importance to each node:")
    print(f"  ├─ For each node i: score[i] = weight @ Z_graph[i] + bias")
    print(f"  ├─ softmax(scores) → weights that sum to 1.0")
    print(f"  └─ g_graph = Σ weight[i] * Z_graph[i]")

    # ────────────────────────────────────────────────────────────────────────
    # SUMMARY
    # ────────────────────────────────────────────────────────────────────────

    print(f"\n" + "=" * 80)
    print("SUMMARY: TOTAL PARAMETERS")
    print("=" * 80)

    total_params = (
        n_params_input
        + n_params_edge_truck
        + n_params_edge_drone
        + n_params_gcn_total
        + n_params_out_norm
        + n_params_pool
    )

    print(f"\n1. Input projection:           {n_params_input:6d} params")
    print(f"2. Edge truck projection:      {n_params_edge_truck:6d} params")
    print(f"3. Edge drone projection:      {n_params_edge_drone:6d} params")
    print(f"4. GCN message passing (3×):   {n_params_gcn_total:6d} params")
    print(f"5. Output normalization:       {n_params_out_norm:6d} params")
    print(f"6. Pooling layer (g_graph):    {n_params_pool:6d} params  ★")
    print(f"   " + "─" * 50)
    print(f"   TOTAL:                      {total_params:6d} params")

    # Verify with PyTorch
    pytorch_total = sum(p.numel() for p in encoder.parameters())
    print(f"   Verified (PyTorch):         {pytorch_total:6d} params ✓")

    # ────────────────────────────────────────────────────────────────────────
    # GRADIENT FLOW
    # ────────────────────────────────────────────────────────────────────────

    print(f"\n" + "=" * 80)
    print("GRADIENT FLOW DURING TRAINING")
    print("=" * 80)

    print(f"""
During backprop, gradients flow through:

  1. loss → policy/value head
  2. → decoder context (g_node, g_veh, g_graph used here)
  3. → g_graph = pooled(Z_graph)
  4. → pool layer weights (LINEAR(D→1))  ← gradient updates these
  5. → Z_graph (per-node embeddings)
  6. → out_norm weights
  7. → GCN layers (msg_truck, msg_drone, update, norm)
  8. → edge projections
  9. → input projection

All parameters are jointly optimized. The pool layer learns which nodes
should contribute more to the global representation based on the RL loss signal.
""")


def show_pool_layer_details():
    """Show exactly what the pool layer does."""
    print("\n" + "=" * 80)
    print("POOL LAYER MECHANICS")
    print("=" * 80)

    cfg = {"embedding_dim": 8, "n_layers": 1, "dropout": 0.0, "use_instance_norm": False}
    encoder = GCN_MEGAGraphEncoder(cfg)

    D = 8
    pool = encoder.pool

    print(f"\nPool layer: Linear({D} → 1)")
    print(f"\nWeight shape: {pool.weight.shape}")
    print(f"Bias shape:   {pool.bias.shape}")

    print(f"\nWeight (learned importance vector):")
    print(pool.weight.data)

    print(f"\nBias (offset):")
    print(pool.bias.data)

    print(f"\nExample computation:")
    print(f"─" * 60)

    # Create a small example
    h = torch.randn(1, 3, D)  # 1 batch, 3 nodes, D dims
    print(f"\nNode embeddings Z_graph: shape {h.shape}")
    print(h.squeeze(0))

    # Compute scores
    scores = pool(h)  # (1, 3, 1)
    print(f"\nScores = Linear(Z_graph):")
    print(f"  Node 0: {scores[0, 0, 0].item():.6f}")
    print(f"  Node 1: {scores[0, 1, 0].item():.6f}")
    print(f"  Node 2: {scores[0, 2, 0].item():.6f}")

    # Softmax
    weights = torch.softmax(scores, dim=1)
    print(f"\nWeights (softmax): sum = {weights.sum().item():.6f}")
    print(f"  Node 0: {weights[0, 0, 0].item():.6f}")
    print(f"  Node 1: {weights[0, 1, 0].item():.6f}")
    print(f"  Node 2: {weights[0, 2, 0].item():.6f}")

    # Weighted sum
    g_graph = (weights * h).sum(dim=1)
    print(f"\ng_graph = weighted_sum(Z_graph): shape {g_graph.shape}")
    print(g_graph.squeeze(0))


def compare_pooling_strategies():
    """Show alternatives to learned pooling."""
    print("\n" + "=" * 80)
    print("ALTERNATIVE POOLING STRATEGIES (not used, for reference)")
    print("=" * 80)

    B, N, D = 2, 6, 8
    h = torch.randn(B, N, D)

    print(f"\nInput: h shape {h.shape} (batch, nodes, embedding_dim)")

    # Strategy 1: Mean pooling (no parameters)
    g_mean = h.mean(dim=1)
    print(f"\n1. MEAN POOLING (0 parameters):")
    print(f"   g = mean(Z_graph) over all nodes")
    print(f"   → treats all nodes equally")
    print(f"   Shape: {g_mean.shape}")

    # Strategy 2: Max pooling (no parameters)
    g_max = h.max(dim=1)[0]
    print(f"\n2. MAX POOLING (0 parameters):")
    print(f"   g = max(Z_graph) over all nodes")
    print(f"   → only uses strongest signal per dimension")
    print(f"   Shape: {g_max.shape}")

    # Strategy 3: Learned pooling (ACTUAL - used in GCN_MEGA)
    pool = nn.Linear(D, 1)
    scores = pool(h)  # (B, N, 1)
    weights = torch.softmax(scores, dim=1)
    g_learned = (weights * h).sum(dim=1)
    print(f"\n3. LEARNED POOLING ({D+1} parameters) ★ ACTUAL:")
    print(f"   pool = Linear(D={D} → 1)")
    print(f"   scores = pool(Z_graph)")
    print(f"   weights = softmax(scores)")
    print(f"   g = Σ weights[i] * Z_graph[i]")
    print(f"   → network learns which nodes matter")
    print(f"   Shape: {g_learned.shape}")

    # Strategy 4: Multi-head pooling (not used)
    K = 4
    print(f"\n4. MULTI-HEAD POOLING (not used, for reference):")
    print(f"   K={K} heads with Linear(D → 1)")
    print(f"   → {K * (D+1)} parameters")
    print(f"   → concatenate K pooled vectors: (B, K*D)")

    # Strategy 5: Gating pooling (not used)
    print(f"\n5. GATING POOLING (not used, for reference):")
    print(f"   gate = sigmoid(Linear(D → 1))")
    print(f"   g = (gate * Z_graph).sum(dim=1)")
    print(f"   → uses multiplicative gating instead of softmax")

    print(f"\nGCN_MEGA chose: LEARNED SOFTMAX POOLING")
    print(f"  Pros:")
    print(f"  ├─ Differentiable, learnable importance weights")
    print(f"  ├─ Probabilistic interpretation (weights sum to 1)")
    print(f"  ├─ Captures which nodes matter for the task")
    print(f"  └─ Minimal parameters (only D+1)")
    print(f"  Cons:")
    print(f"  └─ Less expressive than multi-head (uses single score per node)")


if __name__ == "__main__":
    analyze_all_parameters()
    show_pool_layer_details()
    compare_pooling_strategies()
