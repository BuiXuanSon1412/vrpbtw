# GEMAN Actor-Critic Network with GCN_MEGA Graph Encoder

## Overview
GEMAN (Graph-Enhanced Multi-relational Attention Network) is a bilevel policy network for VRPBTW that processes problem structure through three parallel encoder streams and produces joint (node, vehicle) action logits.

---

## Network Architecture

```
OBSERVATION
     ↓
┌────────────────────────────────────────────────────────────────┐
│  Input Dictionary                                              │
│  ├─ node_features       (B, N+1, 5)  [x, y, lh_d, bh_d, tw]   │
│  ├─ vehicle_features    (B, 2K, 5)   [capacity, time, ...]     │
│  ├─ truck_edge_index    (2, E_t)     edges for trucks          │
│  ├─ truck_edge_attr     (E_t, 2)     [cost, time]              │
│  ├─ drone_edge_index    (2, E_d)     edges for drones          │
│  └─ drone_edge_attr     (E_d, 2)     [cost, time]              │
└────────────────────────────────────────────────────────────────┘
     ↓
┌────────────────────────────────────────────────────────────────┐
│  ENCODER STAGE (3 parallel streams)                            │
│                                                                │
│  ┌─────────────────────┐  ┌─────────────────────┐             │
│  │  NodeEncoder        │  │  VehicleEncoder     │             │
│  │  ───────────────    │  │  ───────────────    │             │
│  │  Linear(5 → D)      │  │  Linear(5 → D)      │             │
│  │  [3 × TransLayer]   │  │  [3 × TransLayer]   │             │
│  │  Softmax Pool       │  │  Softmax Pool       │             │
│  │                     │  │                     │             │
│  │  Z_node: (B,N+1,D)  │  │  Z_veh: (B,2K,D)    │             │
│  │  g_node: (B,D)      │  │  g_veh: (B,D)       │             │
│  └─────────────────────┘  └─────────────────────┘             │
│           ↑                         ↑                          │
│           │                         │                          │
│    node_features              vehicle_features                │
│                                                                │
│  ┌──────────────────────────────────────────────────────────┐ │
│  │  GCN_MEGAGraphEncoder (GCN-based Multi-Edge Aggregation) │ │
│  │  ────────────────────────────────────────────────────────│ │
│  │  Input Projection:                                       │ │
│  │    Linear(4 → D)  [drops demands, keeps x,y,tw_open/cl] │ │
│  │                                                          │ │
│  │  [3 × _GCNMEGAMRGNNLayer]:                               │ │
│  │  ─────────────────────────────                           │ │
│  │    For each relation (truck, drone):                     │ │
│  │      1. Edge Projection: [cost, time] → D                │ │
│  │      2. Message Creation: MLP([h_src || e_proj])         │ │
│  │      3. GCN Normalization: D^(-1/2) A D^(-1/2)           │ │
│  │         - Compute in/out degree per node                 │ │
│  │         - Apply symmetric normalization to messages      │ │
│  │      4. Scatter-add to destination nodes                 │ │
│  │      5. Concatenate truck + drone aggregations           │ │
│  │      6. Update: Linear(2D → D)                           │ │
│  │      7. LayerNorm residual: h' = LN(h + update)          │ │
│  │                                                          │ │
│  │  Output Normalization & Pooling:                         │ │
│  │    LayerNorm(h)                                          │ │
│  │    Softmax pool → (B, D) global embedding                │ │
│  │                                                          │ │
│  │  Z_graph: (B, N+1, D)                                    │ │
│  │  g_graph: (B, D)                                         │ │
│  └──────────────────────────────────────────────────────────┘ │
│           ↑                                                    │
│           │                                                    │
│    nf_gnn, t_ei, t_ea, d_ei, d_ea                             │
│    (graph topology via edge indices/attributes)               │
└────────────────────────────────────────────────────────────────┘
     ↓
     ┌──────────────────┐  ┌──────────────────┐
     │  NodeDecoder     │  │ VehicleDecoder   │
     │  ──────────────  │  │ ──────────────── │
     │  Context:        │  │  Context:        │
     │  f(g_node,       │  │  f(g_veh,        │
     │   mean(Z_veh),   │  │   mean(Z_node),  │
     │   g_graph)       │  │   g_graph)       │
     │                  │  │                  │
     │  Attention over  │  │  Attention over  │
     │  Z_node → ln     │  │  Z_veh → lv      │
     │  (B, N+1)        │  │  (B, 2K)         │
     └──────────────────┘  └──────────────────┘
         ↓                       ↓
         └───────────┬───────────┘
                     ↓
              ┌──────────────────┐
              │ Joint Logits     │
              │ ──────────────   │
              │ flat_logits =    │
              │ ln[:, :, None] + │
              │ lv[:, None, :]   │
              │ .flatten()       │
              │                  │
              │ shape: (B, N*2K) │
              └──────────────────┘
                     ↓
    ┌────────────────┬─────────────────────┐
    ↓                ↓                      ↓
┌─────────┐    ┌──────────────┐    ┌──────────────┐
│ Policy  │    │ Value Head   │    │ Entropy      │
│ Logits  │    │ ──────────── │    │ (Categorical)
│         │    │ MLP          │    │              │
│ (B,     │    │ [g_node ||   │    │              │
│  N*2K)  │    │  g_veh ||    │    │              │
└─────────┘    │  g_graph]    │    └──────────────┘
               │    → D*3 → D → 1 │
               │ (B, 1)           │
               └──────────────────┘
```

---

## GCN_MEGAGraphEncoder: Detailed Message Passing

### _GCNMEGAMRGNNLayer (per layer)

```
Input: h (B, N, D), truck & drone edge indices/attributes

┌─────────────────────────────────────────────────────┐
│ For TRUCK relation:                                 │
│ ────────────────────                                │
│  h_src = h[:, truck_src_idx, :]      (B, E_t, D)   │
│  e_proj = MLP([cost, time])          (E_t, D)      │
│                                                     │
│  msg = MLP_truck([h_src || e_proj])  (B, E_t, D)   │
│                                                     │
│  Degree Normalization:                              │
│  ─────────────────                                  │
│    in_deg[dst] = count of edges → dst   (N,)       │
│    out_deg[src] = count of edges ← src  (N,)       │
│    d_inv_sqrt = 1 / sqrt(in_deg)        (N,)       │
│    d_inv_sqrt_out = 1 / sqrt(out_deg)   (N,)       │
│                                                     │
│  norm_coeff = d_inv_sqrt_out[src] *                │
│               d_inv_sqrt[dst]           (E_t,)     │
│                                                     │
│  msg = msg * norm_coeff                (B, E_t, D) │
│                                                     │
│  scatter_add → agg_truck               (B, N, D)   │
└─────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────┐
│ For DRONE relation: (same pattern)                  │
│ ────────────────────────────────                    │
│  → agg_drone                           (B, N, D)    │
└─────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────┐
│ Update & Residual:                                  │
│ ──────────────────                                  │
│  upd = MLP_update([agg_truck || agg_drone]) (B,N,D) │
│  h' = LayerNorm(h + Dropout(upd))          (B,N,D) │
└─────────────────────────────────────────────────────┘
```

### Normalization Strategy

GCN_MEGA uses **symmetric degree normalization** (like Graph Convolutional Networks):

$$
\tilde{A} = D^{-1/2} A D^{-1/2}
$$

Where:
- $A$ is the adjacency matrix (implicit in scatter-add)
- $D_{ii} = \text{degree of node } i$ (in-degree or out-degree)
- Applied **per relation** (truck & drone separately)
- Handles sparse/skewed degree distributions better than mean-aggregation

**Why GCN over others?**
- **MLP_MEGA**: Simple mean-aggregation, fast, works for uniform degree graphs
- **GCN_MEGA**: Degree-normalized, handles hubs & sparse regions better
- **GAT_MEGA**: Full multi-head attention per edge, most expressive but slowest

---

## Input/Output Dimensions

### Encoder Inputs
| Component | Shape | Description |
|-----------|-------|-------------|
| `node_features` | `(B, N+1, 5)` | [x, y, linehaul_demand, backhaul_demand, tw_open, tw_close] |
| `vehicle_features` | `(B, 2K, 5)` | [capacity, remaining_time, vehicle_type_bin, ...] |
| `truck_edge_index` | `(2, E_truck)` | [src, dst] for truck compatibility graph |
| `truck_edge_attr` | `(E_truck, 2)` | [cost, time] for each truck edge |
| `drone_edge_index` | `(2, E_drone)` | [src, dst] for drone compatibility graph |
| `drone_edge_attr` | `(E_drone, 2)` | [cost, time] for each drone edge |

### Encoder Outputs
| Component | Shape | Description |
|-----------|-------|-------------|
| `Z_node` | `(B, N+1, D)` | Per-node embeddings from node encoder |
| `g_node` | `(B, D)` | Global node representation (attention-pooled) |
| `Z_veh` | `(B, 2K, D)` | Per-vehicle embeddings |
| `g_veh` | `(B, D)` | Global vehicle representation |
| `Z_graph` | `(B, N+1, D)` | Graph-aware node embeddings (GCN output) |
| `g_graph` | `(B, D)` | Global graph representation |

### Decoder Outputs
| Component | Shape | Description |
|-----------|-------|-------------|
| `ln` | `(B, N+1)` | Node selection logits |
| `lv` | `(B, 2K)` | Vehicle selection logits |
| `flat_logits` | `(B, N+1 × 2K)` | Joint (node, vehicle) logits |
| `value` | `(B,)` | State value estimate |

---

## Configuration Example

```yaml
encoder:
  node_encoder:
    embedding_dim: 128
    n_heads: 4
    n_layers: 3
    dropout: 0.1
    use_instance_norm: true
  
  vehicle_encoder:
    embedding_dim: 128
    n_heads: 4
    n_layers: 3
    dropout: 0.1
    use_instance_norm: true
  
  graph_encoder:
    name: GCN_MEGAGraphEncoder  # or MLP_MEGA, GAT_MEGA
    embedding_dim: 128
    n_layers: 3
    dropout: 0.1
    use_instance_norm: false

decoder:
  node_decoder:
    clip: 10.0
    context_hidden_dim: 128
    context_dropout: 0.0
  
  vehicle_decoder:
    clip: 10.0
    context_hidden_dim: 128
    context_dropout: 0.0

value_head:
  hidden_dims: [128, 128]
  dropout: 0.1
  use_dropout: true

regularization:
  ortho_init: true
```

---

## Key Design Choices

### 1. **Bilevel Decoding**
- Two independent decoders: one for nodes, one for vehicles
- Outputs combined as element-wise sum: `logits[n,v] = ln[n] + lv[v]`
- Ensures valid (node, vehicle) pairs while learning independent preferences

### 2. **Graph Encoder Separation**
- Only uses positional + time-window features (drops demand)
- Graph encoder focuses on **routing structure**, not delivery semantics
- Reduces confounding between topology and customer types

### 3. **Three Aggregation Variants**
- **MLP_MEGA**: Message MLP + mean pooling (baseline)
- **GCN_MEGA**: Message MLP + degree-normalized pooling (robust)
- **GAT_MEGA**: Message + multi-head attention weights (expressive)

### 4. **Attention Pooling**
- NodeEncoder & VehicleEncoder: softmax-weighted pool
- GraphEncoder: learnable scalar pool weights
- Reduces sequence information to global context efficiently

### 5. **NaN Handling**
- Explicit `torch.nan_to_num()` in encoder layers
- Guards against numerical instability in attention

---

## Training Pipeline Integration

```python
# In trainer.train() or agent.update():

1. Collect trajectories → obs dict
2. obs → GEMANActorCritic.forward()
3. → flat_logits (batch action space)
   → value (baseline)
4. Sample actions | compute log-probs
5. Compute advantages (GAE)
6. PPO optimization loop:
   - recompute logits/values via .evaluate()
   - compute policy loss
   - compute value loss
   - backprop through all encoders + decoders + value head
```

---

## File References

- **Main architecture**: `/home/bxs/thesis/vrpbtw/rl/impl/geman.py`
  - `GEMANActorCritic` (main class)
  - `GCN_MEGAGraphEncoder` + `_GCNMEGAMRGNNLayer`
  - `NodeEncoder`, `VehicleEncoder`
  - `NodeDecoder`, `VehicleDecoder`
  - `ValueHead`

- **Registry** (instantiation): `/home/bxs/thesis/vrpbtw/rl/core/registry.py`
  - Maps `"gcn_mega"` → `GCN_MEGAGraphEncoder`

- **Base classes**: `/home/bxs/thesis/vrpbtw/rl/core/network.py`
  - `ActorCritic` parent
  - `_MHA` (multi-head attention)
  - `_FF` (feed-forward)

---

## Computational Complexity

| Component | Complexity | Notes |
|-----------|-----------|-------|
| NodeEncoder | O(L × B × N² × D) | L layers, quadratic in N (all-pairs attention) |
| VehicleEncoder | O(L × B × K² × D) | K vehicles (2K total) |
| GCN_MEGA | O(L × B × (E_t + E_d) × D) | Linear in edges, not quadratic |
| NodeDecoder | O(B × N × D) | Single dot-product attention |
| VehicleDecoder | O(B × K × D) | Single dot-product attention |
| ValueHead | O(B × D²) | Small MLP |
| **Total** | O(B × (N² + K² + E) × D × L) | Dominated by encoder quadratic terms |

For typical VRP: N~100, K~5 → encoder cost dominates.
