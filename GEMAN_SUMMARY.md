# GEMAN (Graph-Enhanced Multi-relational Attention Network) - Complete Architecture Guide

## Quick Summary

**GEMAN** is an actor-critic neural network designed for the Vehicle Routing Problem with Time Windows and Drones (VRPBTW). It makes bilevel decisions: **(1) which location to visit** and **(2) which vehicle to use**.

### Key Innovation
The network uses **three independent encoders** (nodes, vehicles, graph) that feed into **two independent decoders** (bilevel), enabling factorized but coordinated decision-making without coupling.

---

## Architecture Overview

### The Big Picture

```
Observation Input
    ↓
    ├─ Node Encoder       ──→ (Z_node, g_node)
    ├─ Vehicle Encoder    ──→ (Z_veh, g_veh)
    └─ Graph Encoder      ──→ (Z_graph, g_graph)
         ↓
    Node Decoder          ──→ node_logits
    Vehicle Decoder       ──→ vehicle_logits
         ↓
    Joint Logits          ──→ flat_logits (policy)
    Value Head            ──→ value (critic)
```

---

## Detailed Components

### 1. **Node Encoder**
**Purpose**: Encode location information with heterogeneous attention

- **Input**: node_features (B, N+1, 5) → [x, y, linehaul_demand, backhaul_demand, tw_open, tw_close]
- **Architecture**:
  - Linear projection: 5D → 128D (embedding_dim)
  - 3 layers of transformer-style attention (4 heads)
  - Each layer: MultiHeadAttention → ReLU FFN → LayerNorm
  - Adaptive pooling: learned softmax weights
- **Output**:
  - **Z_node** (B, N+1, 128): Per-location embeddings
  - **g_node** (B, 128): Global context representation

### 2. **Vehicle Encoder**
**Purpose**: Encode vehicle information (both truck and drone)

- **Input**: vehicle_features (B, 2K, 5) → [capacity, battery, cost_rate, speed, time_budget]
- **Architecture**:
  - Linear projection: 5D → 128D
  - 3 layers of transformer-style attention
  - Adaptive pooling
- **Output**:
  - **Z_veh** (B, 2K, 128): Per-vehicle embeddings (K vehicles, each has truck and drone variants)
  - **g_veh** (B, 128): Global vehicle context

### 3. **Graph Encoder** (Multi-Relational GNN)
**Purpose**: Capture spatial-temporal routing structure via two independent graph relations

- **Input**: 
  - node_features → extract [x, y, tw_open, tw_close] → (B, N+1, 4)
  - truck edges (index + attributes)
  - drone edges (index + attributes)
- **Architecture**:
  - Linear projection: 4D → 128D
  - 3 Multi-Relational GNN Layers (MRGNNLayer):
    - **Truck relation**: MLP message passing → mean aggregation
    - **Drone relation**: MLP message passing → mean aggregation
    - **Update**: LayerNorm(h + Linear(concat[agg_truck, agg_drone]))
  - Adaptive pooling
- **Output**:
  - **Z_graph** (B, N+1, 128): Routing-aware node embeddings
  - **g_graph** (B, 128): Global routing context

### 4. **Node Decoder**
**Purpose**: Select which location to visit

- **Inputs**:
  - g_node (B, 128): Node context
  - mean(Z_veh) (B, 128): Vehicle context (cross-modal)
  - g_graph (B, 128): Routing context
  - Z_node (B, N+1, 128): All location embeddings
- **Process**:
  1. Concatenate contexts: ctx = [g_node, mean(Z_veh), g_graph] → (B, 384)
  2. Project: Linear(384 → 128) + ReLU → (B, 128)
  3. Query: Q = Linear(128 → 128) → (B, 1, 128)
  4. Keys: K = Linear(Z_node) → (B, N+1, 128)
  5. Attention: scores = Q·K^T / √128
  6. Clip: logits = 10.0 * tanh(scores) → (B, N+1)
- **Output**: node_logits (B, N+1) — unnormalized logits over locations

### 5. **Vehicle Decoder**
**Purpose**: Select which vehicle to use

- **Inputs**:
  - g_veh (B, 128): Vehicle context
  - mean(Z_node) (B, 128): Node context (cross-modal)
  - g_graph (B, 128): Routing context
  - Z_veh (B, 2K, 128): All vehicle embeddings
- **Process**: Identical to NodeDecoder but over vehicles
- **Output**: veh_logits (B, 2K) — unnormalized logits over vehicles

### 6. **Joint Action Logits**
**Purpose**: Combine node and vehicle decisions into single action space

```python
flat_logits[b, i*K + j] = node_logits[b, i] + veh_logits[b, j]
# Shape: (B, (N+1) × 2K)
```

This **outer sum** means: "score of taking action (node i, vehicle j)" = "quality of node i" + "quality of vehicle j", where high scores indicate good combinations.

### 7. **Value Head**
**Purpose**: Estimate state value V(s) for advantage computation

- **Input**: concat[g_node, g_veh, g_graph] → (B, 384)
- **Architecture**: 
  - Linear(384 → 128) + ReLU + Dropout
  - Linear(128 → 128) + ReLU + Dropout
  - Linear(128 → 1)
- **Output**: value (B,) — scalar state value estimate

---

## Data Flow & Shapes

| Layer | Input Shape | Output Shape | Parameters |
|-------|-------------|--------------|------------|
| node_features input | (B, N+1, 5) | — | — |
| NodeEncoder.proj | (B, N+1, 5) | (B, N+1, 128) | 768 |
| NodeEncoder.layers | (B, N+1, 128) | (B, N+1, 128) | ~200K |
| NodeEncoder.pool | (B, N+1, 128) | (B, 128) + (B, N+1, 128) | 129 |
| VehicleEncoder (similar) | (B, 2K, 5) | (B, 128) + (B, 2K, 128) | ~200K |
| GraphEncoder | (B, N+1, 4) + edges | (B, 128) + (B, N+1, 128) | ~150K |
| NodeDecoder | 3×(B, 128) + (B, N+1, 128) | (B, N+1) | ~65K |
| VehicleDecoder | 3×(B, 128) + (B, 2K, 128) | (B, 2K) | ~65K |
| Joint Logits | (B, N+1) + (B, 2K) | (B, (N+1)×2K) | 0 |
| ValueHead | (B, 384) | (B,) | ~66K |
| **TOTAL** | — | — | **~750K** |

---

## Alternative Graph Encoders (Ablation Variants)

The architecture supports swappable graph encoders for ablation studies:

### 1. **EGAEncoder** (Sparse Attention)
- k-NN sparse attention mask (each node → ~3 neighbors + depot)
- Distance-based relative position encoding (32 buckets)
- Single-relation attention (no truck/drone distinction)
- Use case: Compare sparse vs. dense attention

### 2. **MLP_MEGAGraphEncoder** (Simple Baseline)
- Direct MLP message passing: MLP([h_src || edge_proj])
- Mean aggregation per relation
- Fast, interpretable baseline
- ~150K parameters

### 3. **GCN_MEGAGraphEncoder** (Normalized Aggregation)
- Symmetric degree normalization: D^(-1/2) A D^(-1/2)
- Good for graphs with skewed degree distributions
- ~150K parameters

### 4. **GAT_MEGAGraphEncoder** (Attention Aggregation)
- Multi-head attention over edge messages
- Learned edge importance weights
- Most expressive but slowest
- ~200K parameters

---

## Configuration Example

```python
config = {
    "encoder": {
        "node_encoder": {
            "embedding_dim": 128,
            "n_heads": 4,
            "n_layers": 3,
            "dropout": 0.1,
            "use_instance_norm": True
        },
        "vehicle_encoder": {
            "embedding_dim": 128,
            "n_heads": 4,
            "n_layers": 3,
            "dropout": 0.1,
            "use_instance_norm": True
        },
        "graph_encoder": {
            "name": GraphEncoder,  # or EGAEncoder, MLP_MEGAGraphEncoder, etc.
            "embedding_dim": 128,
            "n_layers": 3,
            "dropout": 0.1,
            "use_instance_norm": False
        }
    },
    "decoder": {
        "node_decoder": {
            "clip": 10.0,
            "context_hidden_dim": 128,
            "context_dropout": 0.0
        },
        "vehicle_decoder": {
            "clip": 10.0,
            "context_hidden_dim": 128,
            "context_dropout": 0.0
        }
    },
    "value_head": {
        "hidden_dims": [128, 128],
        "dropout": 0.1,
        "use_dropout": True
    },
    "regularization": {
        "ortho_init": True
    }
}
```

---

## Forward Pass Pseudocode

```python
def forward(obs, action_mask=None):
    # Unpack observation
    nf, vf, t_ei, t_ea, d_ei, d_ea = unpack(obs)
    
    # Encoders (parallel)
    Z_node, g_node = node_encoder(nf)
    Z_veh, g_veh = vehicle_encoder(vf)
    
    # Extract routing-structure features for graph encoder
    nf_graph = nf[:, :, [0,1,4,5]]  # [x, y, tw_open, tw_close]
    Z_graph, g_graph = graph_encoder(nf_graph, t_ei, t_ea, d_ei, d_ea)
    
    # Decoders (independent but cross-aware)
    node_logits = node_decoder(g_node, Z_veh.mean(1), g_graph, Z_node)
    veh_logits = vehicle_decoder(g_veh, Z_node.mean(1), g_graph, Z_veh)
    
    # Joint action space
    flat_logits = (node_logits[:, :, None] + veh_logits[:, None, :])
                  .reshape(B, N*K)
    if action_mask:
        flat_logits[~action_mask] = -∞
    
    # Value estimation
    value = value_head(cat[g_node, g_veh, g_graph])
    
    return flat_logits, value
```

---

## Training Integration

### Actor Loss (Policy Gradient)
```python
dist = torch.distributions.Categorical(logits=flat_logits)
log_probs = dist.log_prob(actions)
policy_loss = -(log_probs * advantages).mean()
entropy_bonus = dist.entropy().mean()
actor_loss = policy_loss - 0.01 * entropy_bonus
```

### Critic Loss (Value Function)
```python
value_loss = F.smooth_l1_loss(value, targets)
```

### Total Loss
```python
total_loss = actor_loss + 0.5 * value_loss
```

---

## Key Design Principles

1. **Hierarchical Encoding**: Separate encoders allow feature reuse and specialization
2. **Bilevel Decisions**: Independent node/vehicle decoders prevent coupling
3. **Graph Awareness**: Multi-relational GNN captures routing constraints
4. **Cross-Attention**: Implicit cross-modal awareness via context concatenation
5. **Attention-based Pooling**: Learned aggregation preserves salient information
6. **Masking Support**: Action feasibility constraints applied via -∞ masking
7. **Numerical Stability**: Tanh clipping and NaN handling prevent training instability

---

## Computational Complexity

| Component | Time | Space |
|-----------|------|-------|
| Node Encoder (N² attention) | O(B×N²×D) | O(B×N×D) |
| Vehicle Encoder (K² attention) | O(B×K²×D) | O(B×K×D) |
| Graph Encoder (E edges, 2 relations) | O(B×E×D) | O(B×E×D) |
| Node/Vehicle Decoders | O(B×N×D) + O(B×K×D) | O(B×D) |
| Value Head | O(B×D²) | O(B×D) |
| **Total** | **O(B×(N²+E)×D)** | **O(B×(N+K+E)×D)** |

Dominated by node encoder when N is large. For typical VRPBTW instances (N~100, K~5), total time ~10-50ms per batch on modern GPUs.

---

## Visualization Resources

The repository includes:
- `geman_architecture.md` — Detailed component documentation
- `geman_architecture_diagram.txt` — ASCII art architecture diagram
- `geman_architecture.dot` — Graphviz diagram source
- `visualize_geman_architecture.py` — Python visualization script

Generate PNG/SVG diagrams:
```bash
python3 visualize_geman_architecture.py --format png --output geman_architecture.png
python3 visualize_geman_architecture.py --format svg --output geman_architecture.svg
```

---

## References

**Paper Context**: Graph-Enhanced Multi-relational Attention Network
**Domain**: Vehicle Routing Problem with Time Windows and Drones
**Base Classes**: `ActorCritic`, `_MHA` (MultiHeadAttention), `_FF` (FeedForward)

**File**: `rl/impl/geman.py` (1217 lines)

---

## Troubleshooting

### Issue: NaN/Inf in embeddings
- **Cause**: Unbounded attention scores
- **Fix**: Increase `clip` parameter in decoder config

### Issue: Poor vehicle diversity in actions
- **Cause**: Vehicle embeddings not distinctive
- **Fix**: Increase VehicleEncoder `n_layers` or decrease `dropout`

### Issue: Graph encoder not learning
- **Cause**: Edge attributes not informative
- **Fix**: Verify truck/drone edge construction; try alternative encoder (GAT_MEGAGraphEncoder)

### Issue: Memory OOM
- **Cause**: Large batch or graph size
- **Fix**: Reduce batch size; use smaller `embedding_dim`; try EGAEncoder (sparse)

---

End of GEMAN Architecture Summary
