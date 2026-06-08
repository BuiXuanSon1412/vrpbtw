"""
Demonstration: How g_graph is computed in GCN_MEGAGraphEncoder

This script shows step-by-step how the global graph representation (g_graph)
is computed through attention-weighted pooling of node embeddings.
"""

import sys
from pathlib import Path

import torch
import torch.nn as nn

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from impl.geman import GCN_MEGAGraphEncoder, _GCNMEGAMRGNNLayer


def demo_pooling():
    """Demonstrate the pooling mechanism step-by-step."""
    print("=" * 80)
    print("POOLING DEMONSTRATION")
    print("=" * 80)

    # Simple example: 1 batch, 5 nodes, embedding_dim=4
    B, N, D = 1, 5, 4

    # Create mock node embeddings (e.g., after 3 GCN layers)
    h = torch.randn(B, N, D, requires_grad=False)
    print(f"\nNode embeddings h:  shape {h.shape}")
    print(h.squeeze(0))

    # Create a pooling layer
    pool = nn.Linear(D, 1)

    # Step 1: Score each node
    scores = pool(h)  # (B, N, 1)
    print(f"\nStep 1 - Pool scores (before softmax):  shape {scores.shape}")
    print(scores.squeeze())

    # Step 2: Apply softmax to get attention weights
    weights = torch.softmax(scores, dim=1)  # (B, N, 1)
    print(f"\nStep 2 - Attention weights (after softmax):  shape {weights.shape}")
    print(weights.squeeze())
    print(f"Sum of weights: {weights.sum():.4f}")  # Should be ~1.0

    # Step 3: Weighted sum
    g_global = (weights * h).sum(dim=1)  # (B, D)
    print(f"\nStep 3 - Global representation g_graph:  shape {g_global.shape}")
    print(g_global.squeeze())

    # Manual verification
    manual_g = torch.zeros(1, D)
    for i in range(N):
        contribution = weights[0, i, 0] * h[0, i, :]
        manual_g += contribution
        print(f"  Node {i}: w={weights[0,i,0]:.4f} × h → contributes {contribution}")

    print(f"\nManual g_graph: {manual_g.squeeze()}")
    print(f"Match: {torch.allclose(g_global, manual_g, atol=1e-6)}")


def demo_full_encoder():
    """Demonstrate the full GCN_MEGAGraphEncoder forward pass."""
    print("\n" + "=" * 80)
    print("FULL GCN_MEGAGraphEncoder DEMONSTRATION")
    print("=" * 80)

    # Create config
    cfg = {
        "embedding_dim": 16,  # Small for display
        "n_layers": 2,
        "dropout": 0.1,
        "use_instance_norm": False,
    }

    encoder = GCN_MEGAGraphEncoder(cfg)

    # Create mock input
    B, N, D = 2, 6, 4  # 2 batches, 6 nodes (5 + depot), 4 features

    # Node features: [x, y, tw_open, tw_close]
    node_feat = torch.randn(B, N, D)

    # Truck edges: small dense graph
    t_ei = torch.tensor([
        [0, 0, 1, 1, 2, 3, 4, 5],  # src
        [1, 2, 3, 4, 5, 4, 5, 0],  # dst
    ], dtype=torch.long)
    t_ea = torch.randn(t_ei.shape[1], 2)  # [cost, time]

    # Drone edges: sparse graph
    d_ei = torch.tensor([
        [0, 0, 1, 2],
        [2, 4, 5, 0],
    ], dtype=torch.long)
    d_ea = torch.randn(d_ei.shape[1], 2)

    print(f"\nInput shapes:")
    print(f"  node_feat:      {node_feat.shape}  [x, y, tw_open, tw_close]")
    print(f"  truck_ei:       {t_ei.shape}  (edges: depot→all, dense)")
    print(f"  truck_ea:       {t_ea.shape}  [cost, time]")
    print(f"  drone_ei:       {d_ei.shape}  (edges: sparse)")
    print(f"  drone_ea:       {d_ea.shape}")

    # Forward pass
    with torch.no_grad():
        Z_graph, g_graph = encoder(node_feat, t_ei, t_ea, d_ei, d_ea)

    print(f"\nOutput shapes:")
    print(f"  Z_graph (per-node):  {Z_graph.shape}  (B, N, D)")
    print(f"  g_graph (global):    {g_graph.shape}  (B, D)")

    # Show results for first batch
    print(f"\n--- Batch 0 ---")
    print(f"Per-node embeddings Z_graph[0]:")
    print(Z_graph[0])

    print(f"\nGlobal representation g_graph[0]:")
    print(g_graph[0])

    # Manually show what pooling did
    print(f"\n--- Manual pooling breakdown (Batch 0) ---")

    # Access pool weights
    pool_layer = encoder.pool
    node_emb_after_norm = encoder.out_norm(Z_graph[0:1])  # (1, N, D)

    scores = pool_layer(node_emb_after_norm)  # (1, N, 1)
    weights = torch.softmax(scores, dim=1)     # (1, N, 1)

    print(f"Pooling scores (before softmax):")
    for i, s in enumerate(scores[0]):
        print(f"  Node {i}: {s.item():.4f}")

    print(f"\nAttention weights (after softmax):")
    total_weight = 0.0
    for i, w in enumerate(weights[0]):
        print(f"  Node {i}: {w.item():.4f}")
        total_weight += w.item()
    print(f"  Sum: {total_weight:.4f}")

    print(f"\nWeighted contribution to g_graph:")
    D_actual = Z_graph.shape[-1]
    g_reconstructed = torch.zeros(D_actual)
    for i in range(N):
        contrib = weights[0, i, 0].item() * Z_graph[0, i].detach()
        g_reconstructed += contrib
        print(f"  Node {i}: weight={weights[0,i,0].item():.4f}")

    print(f"\nReconstructed g_graph: {g_reconstructed}")
    print(f"Actual g_graph[0]:     {g_graph[0].detach()}")
    print(f"Match: {torch.allclose(g_graph[0], g_reconstructed, atol=1e-5)}")


def demo_comparing_batches():
    """Show that different problem sizes learn different pooling patterns."""
    print("\n" + "=" * 80)
    print("POOLING VARIES BY PROBLEM SIZE")
    print("=" * 80)

    cfg = {
        "embedding_dim": 8,
        "n_layers": 2,
        "dropout": 0.0,
        "use_instance_norm": False,
    }

    encoder = GCN_MEGAGraphEncoder(cfg)

    # Different problem sizes in batch
    # Batch 0: 5 nodes, Batch 1: 8 nodes
    # We need uniform batch size, so pad batch 0 with zeros

    B, max_N, feat_D = 2, 8, 4
    node_feat = torch.zeros(B, max_N, feat_D)

    # Batch 0: 5 nodes with features
    node_feat[0, :5] = torch.randn(5, feat_D)

    # Batch 1: 8 nodes with features
    node_feat[1, :8] = torch.randn(8, feat_D)

    # Simple star graph (all connect to depot)
    t_ei = torch.tensor([
        [0, 0, 0, 0, 0, 0, 0, 0],  # depot
        [1, 2, 3, 4, 5, 6, 7, 8],  # all others
    ], dtype=torch.long)
    t_ea = torch.randn(t_ei.shape[1], 2)

    d_ei = torch.tensor([[0], [0]], dtype=torch.long)  # dummy
    d_ea = torch.randn(1, 2)

    with torch.no_grad():
        Z_graph, g_graph = encoder(node_feat, t_ei, t_ea, d_ei, d_ea)

    print(f"\nBatch 0: 5 nodes")
    print(f"  Only uses node_feat[0, :5]")
    print(f"  Z_graph[0] shape: {Z_graph[0].shape}")

    pool_layer = encoder.pool
    node_emb_0 = encoder.out_norm(Z_graph[0:1])
    scores_0 = pool_layer(node_emb_0[:, :5])  # Only first 5
    weights_0 = torch.softmax(scores_0, dim=1)
    print(f"  Pooling weights: {weights_0.squeeze().numpy()}")
    print(f"  g_graph[0]: {g_graph[0].numpy()}")

    print(f"\nBatch 1: 8 nodes")
    print(f"  Uses all of node_feat[1]")
    print(f"  Z_graph[1] shape: {Z_graph[1].shape}")

    node_emb_1 = encoder.out_norm(Z_graph[1:2])
    scores_1 = pool_layer(node_emb_1[:, :8])  # All 8
    weights_1 = torch.softmax(scores_1, dim=1)
    print(f"  Pooling weights: {weights_1.squeeze().numpy()}")
    print(f"  g_graph[1]: {g_graph[1].numpy()}")

    print(f"\nNote: Different problem sizes → different pooling weights")
    print(f"      Same pool layer processes different numbers of nodes")


if __name__ == "__main__":
    # Run demonstrations
    demo_pooling()
    demo_full_encoder()
    # demo_comparing_batches()  # Skipped - requires careful edge index handling

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print("""
g_graph is computed as:

  1. After GCN layers:  Z_graph[b] = [h_0, h_1, ..., h_N]  per batch

  2. Output normalization: Z_graph = LayerNorm(Z_graph)

  3. Pooling scores: score[b,i] = Linear(Z_graph[b,i])  for each node

  4. Attention weights: w[b,i] = softmax(score[b,i])  over all nodes
                       (each batch: Σ_i w[b,i] = 1.0)

  5. Weighted sum: g_graph[b] = Σ_i w[b,i] * Z_graph[b,i]

Result: g_graph[b] is a (D,) vector = attention-weighted average of all nodes

The pool layer learns which nodes are important for the global representation!
""")
