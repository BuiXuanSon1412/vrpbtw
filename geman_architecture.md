# GEMAN Actor-Critic Network Architecture

## Overview
GEMAN is a Graph-Enhanced Multi-relational Attention Network for VRPBTW (Vehicle Routing Problem with Time Windows and Drones). It combines hierarchical node/vehicle encoding with multi-relational graph processing for bilevel decision-making (route node selection and vehicle assignment).

---

## Input Specification

```
Observation Dictionary (all pre-normalized by state_to_obs):
├── node_features       (B, N+1, 5)    [x, y, linehaul_demand, backhaul_demand, tw_open, tw_close]
├── vehicle_features    (B, 2K, 5)     [capacity, battery, cost_rate, speed, time_budget]
├── truck_edge_index    (2, E)         [src, dst] indices
├── truck_edge_attr     (E, 2)         [cost, time]
├── drone_edge_index    (2, E)         [src, dst] indices
└── drone_edge_attr     (E, 2)         [cost, time]
```

---

## Architecture Flow

### **ENCODER STAGE** (Feature Extraction)

```
┌─────────────────────────────────────────────────────────────────────┐
│                      THREE PARALLEL ENCODERS                        │
└─────────────────────────────────────────────────────────────────────┘

1. NODE ENCODER
   ────────────────────────────────────────────────────
   node_features (B, N+1, 5)
          │
          ├─→ Linear Projection: D (default 128)
          │
          ├─→ [N_LAYERS × NodeEncoderLayer]
          │    ├─ MultiHeadAttention (self-attention)
          │    ├─ ReLU FFN
          │    └─ LayerNorm (or InstanceNorm if use_instance_norm=True)
          │
          ├─→ Adaptive Pooling: softmax(Linear(D→1))
          │
          └─→ OUTPUT:
              Z_node: (B, N+1, D)     [per-node embeddings]
              g_node: (B, D)          [global node context]


2. VEHICLE ENCODER
   ────────────────────────────────────────────────────
   vehicle_features (B, 2K, 5)
          │
          ├─→ Linear Projection: D
          │
          ├─→ [N_LAYERS × VehicleEncoderLayer]
          │    ├─ MultiHeadAttention (self-attention)
          │    ├─ ReLU FFN
          │    └─ LayerNorm (or InstanceNorm)
          │
          ├─→ Adaptive Pooling: softmax(Linear(D→1))
          │
          └─→ OUTPUT:
              Z_veh: (B, 2K, D)      [per-vehicle embeddings]
              g_veh: (B, D)          [global vehicle context]


3. GRAPH ENCODER (Multi-Relational GNN)
   ────────────────────────────────────────────────────
   node_features (B, N+1, 5)  →  Extract [x, y, tw_open, tw_close] → (B, N+1, 4)
   
   + truck_edge_index, truck_edge_attr  (truck relations)
   + drone_edge_index, drone_edge_attr  (drone relations)
          │
          ├─→ Linear Projection: D
          │
          ├─→ [3 × MRGNNLayer]
          │    For each relation (truck, drone):
          │    ├─ Message Passing:  msg = MLP(h_src || edge_feat)
          │    ├─ Aggregation:      agg = MEAN(incoming messages)
          │    └─ Update:           h' = LayerNorm(h + Linear(cat[agg_truck, agg_drone]))
          │
          ├─→ Adaptive Pooling: softmax(Linear(D→1))
          │
          └─→ OUTPUT:
              Z_graph: (B, N+1, D)   [graph-aware node embeddings]
              g_graph: (B, D)        [global routing context]
```

---

### **DECODER STAGE** (Policy Decoding)

```
┌─────────────────────────────────────────────────────────────────────┐
│                    BILEVEL DECISION DECODING                        │
│                  (Independent Node & Vehicle Heads)                 │
└─────────────────────────────────────────────────────────────────────┘

1. NODE DECODER (Which location to visit?)
   ────────────────────────────────────────────────────
   Inputs:  g_node (B, D)
            mean(Z_veh) (B, D)
            g_graph (B, D)
            Z_node (B, N+1, D)
   
   Step 1: Context Projection
           ctx = ReLU(Linear(D*3 → context_hidden_dim))
                 └─ applies to cat[g_node, mean(Z_veh), g_graph]
   
   Step 2: Query Generation
           Q = Linear(context_hidden_dim → D)  [shape: (B, 1, D)]
   
   Step 3: Dot-Product Attention
           scores = (Q · K^T) / sqrt(D)
           where K = Linear(Z_node) [shape: (B, N+1, D)]
   
   Step 4: Clipping & Masking
           logits = clip * tanh(scores)           [shape: (B, N+1)]
           if action_mask: logits[~mask] = -∞
   
   └─→ OUTPUT:  node_logits (B, N+1)


2. VEHICLE DECODER (Which vehicle to use?)
   ────────────────────────────────────────────────────
   Inputs:  g_veh (B, D)
            mean(Z_node) (B, D)
            g_graph (B, D)
            Z_veh (B, 2K, D)
   
   Step 1: Context Projection
           ctx = ReLU(Linear(D*3 → context_hidden_dim))
                 └─ applies to cat[g_veh, mean(Z_node), g_graph]
   
   Step 2: Query Generation
           Q = Linear(context_hidden_dim → D)  [shape: (B, 1, D)]
   
   Step 3: Dot-Product Attention
           scores = (Q · K^T) / sqrt(D)
           where K = Linear(Z_veh) [shape: (B, 2K, D)]
   
   Step 4: Clipping & Masking
           logits = clip * tanh(scores)           [shape: (B, 2K)]
           if action_mask: logits[~mask] = -∞
   
   └─→ OUTPUT:  vehicle_logits (B, 2K)


3. JOINT ACTION LOGITS (Bilevel Combination)
   ────────────────────────────────────────────────────
   Combined action space: (N+1) × (2K) = N+1 node choices × 2K vehicle choices
   
   flat_logits = (node_logits[:, :, None] + vehicle_logits[:, None, :])
                 .reshape(B, (N+1)*2K)
   
   with action_mask applied: flat_logits[~mask] = -∞
   
   └─→ OUTPUT:  flat_logits (B, N+1 × 2K)
```

---

### **VALUE HEAD** (State Value Estimation)

```
┌─────────────────────────────────────────────────────────────────────┐
│                         VALUE FUNCTION HEAD                         │
└─────────────────────────────────────────────────────────────────────┘

Inputs:  g_node (B, D)
         g_veh (B, D)
         g_graph (B, D)

Step 1: Concatenate Global Contexts
        x = cat[g_node, g_veh, g_graph]  [shape: (B, D*3)]

Step 2: Multi-Layer MLP
        ├─ Layer 1:  ReLU(Linear(D*3 → hidden_dim[0]))
        ├─ Dropout (if enabled)
        │
        ├─ Layer 2..N-1:  ReLU(Linear(hidden_dim[i] → hidden_dim[i+1]))
        ├─ Dropout (if enabled)
        │
        └─ Output:  Linear(hidden_dim[-1] → 1)

└─→ OUTPUT:  value (B,)  [scalar state value estimate]
```

---

## Alternative Encoders (for Ablation Studies)

### **EGAEncoder** - Sparse Attention with Dynamic Graphs
- **Purpose**: Compare sparse attention vs. dense multi-relational GNN
- **Mechanism**:
  - k-NN sparse attention mask (each node attends to ~3 neighbors + depot)
  - Distance-based relative position encoding (32 distance buckets)
  - Single-relation attention (no truck/drone distinction)

### **MLP_MEGAGraphEncoder** - Multi-Edge Graph Aggregation (MLP Variant)
- **Purpose**: Simple baseline using direct MLP message passing
- **Mechanism**:
  - Project edge attributes separately (truck_edge_proj, drone_edge_proj)
  - MLP([h_src || edge_proj]) → mean aggregation per relation
  - Concatenate truck & drone aggregations for update

### **GCN_MEGAGraphEncoder** - Graph Convolution Network Variant
- **Purpose**: Normalized aggregation for graphs with skewed degree distribution
- **Mechanism**:
  - Symmetric degree normalization: D^(-1/2) A D^(-1/2)
  - Otherwise identical message passing to MLP variant

### **GAT_MEGAGraphEncoder** - Graph Attention Network Variant
- **Purpose**: Learned attention weights over edge messages
- **Mechanism**:
  - Multi-head attention over edge messages
  - Edge attributes modulate attention weights
  - Most expressive but computationally slower

---

## Configuration Structure

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
    }
}
```

---

## Data Flow Summary

```
Input Observation
       │
       ├─────────────┬─────────────┬─────────────┐
       │             │             │             │
       ▼             ▼             ▼             ▼
   NodeEnc      VehicleEnc    GraphEnc     (parallel)
   (Z_node,     (Z_veh,       (Z_graph,
    g_node)      g_veh)        g_graph)
       │             │             │
       └─────────────┼─────────────┘
                     │
       ┌─────────────┼─────────────┐
       │             │             │
       ▼             ▼             ▼
   NodeDec      VehicleDec    ValueHead
   (logits)     (logits)      (scalar)
       │             │             │
       └─────────────┼─────────────┘
                     │
       ┌─────────────┴─────────────┐
       │                           │
       ▼                           ▼
  Bilevel Action              State Value
  Logits (N+1)×(2K)          Estimate (B,)
```

---

## Key Design Principles

1. **Hierarchical Encoding**: Separate encoders for nodes and vehicles enable feature reuse
2. **Graph Awareness**: Multi-relational GNN captures spatial-temporal routing structure
3. **Bilevel Decoding**: Independent decoders for nodes and vehicles prevent coupling
4. **Cross-Attention**: Each decoder sees mean of the other modality (implicit cross-attention)
5. **Attention-based Pooling**: Learned adaptive pooling preserves rich contextual information
6. **Masking Support**: Action masks enforce feasibility constraints at decode time

---

## Complexity Analysis

| Component | Complexity | Notes |
|-----------|-----------|-------|
| NodeEncoder | O(B × N² × D × H) | H attention heads per layer |
| VehicleEncoder | O(B × K² × D × H) | Typically K ≪ N |
| GraphEncoder (MRNN) | O(B × E × D × 2) | E edges, 2 relations |
| NodeDecoder | O(B × N × D) | Dot-product attention |
| VehicleDecoder | O(B × K × D) | Dot-product attention |
| ValueHead | O(B × D × hidden) | Linear MLP operations |
| **Total** | **O(B × (N² + E) × D)** | Dominated by encoder |

Memory: ~O(B × (N + K + E) × D)

---

## Architecture Advantages for VRPBTW

1. **Multi-relational reasoning**: Separate truck/drone graphs capture mode-specific constraints
2. **Hierarchical structure**: Nodes and vehicles processed independently prevents cross-talk
3. **Scalability**: Attention mechanisms handle variable N and K efficiently
4. **Interpretability**: Attention weights reveal which nodes/vehicles are most relevant
5. **Flexibility**: Modular encoder/decoder design supports easy swapping (e.g., GraphEncoder variants)
6. **Robustness**: Masking and normalization prevent NaN/Inf issues in training

