# GEMAN vs HGNN Architecture Comparison

Both networks share the same overall structure but differ in **node encoding strategy**.

---

## Shared Components

### Architecture Structure (Identical)
```
Encoder:
  ├─ NodeEncoder      → Z_node (B, N+1, D), g_node (B, D)
  ├─ VehicleEncoder   → Z_veh (B, 2K, D), g_veh (B, D)
  └─ GraphEncoder     → Z_graph (B, N+1, D), g_graph (B, D)

Decoder:
  ├─ NodeDecoder      (dot-product attention + rejection sampling)
  ├─ VehicleDecoder   (dot-product attention + rejection sampling)
  └─ Value head       MLP(g_node ‖ g_veh ‖ g_graph)
```

### GraphEncoder
Both use identical **multi-relational GNN** with truck/drone edge types.

---

## Key Difference: Node Encoding

### HGNN - Heterogeneous Node Encoder

**Strategy:** Explicitly model linehaul ↔ backhaul cross-attention

```python
class _HeteroNodeLayer(nn.Module):
    def __init__(self, D, H, dropout):
        self.sa = _MHA(D, H, dropout)           # self-attention
        self.xl2b = _MHA(D, H, dropout)         # linehaul → backhaul
        self.xb2l = _MHA(D, H, dropout)         # backhaul → linehaul
    
    def forward(self, h, l_idx, b_idx):
        h_sa = self.sa(h, h, h)                 # all nodes attend to all
        
        # Cross-attention between linehaul and backhaul
        if l_idx.numel() > 0 and b_idx.numel() > 0:
            h_l = h[:, l_idx]
            h_b = h[:, b_idx]
            h_het[:, l_idx] = self.xl2b(h_l, h_b, h_b)  # linehaul nodes look at backhaul
            h_het[:, b_idx] = self.xb2l(h_b, h_l, h_l)  # backhaul nodes look at linehaul
        
        h = norm(h + h_sa + h_het)              # combine both attention paths
```

**Pros:**
- ✓ Explicitly captures linehaul/backhaul semantics
- ✓ Allows nodes to attend to complementary node types
- ✓ Leverages problem structure (phase transitions matter in VRPBTW)
- ✓ More parameters in each layer (3 attention heads instead of 1)

**Cons:**
- ✗ More complex (more parameters to train)
- ✗ Requires identifying linehaul/backhaul node indices each forward pass
- ✗ Can get stuck if indices empty (edge case handling needed)

---

### GEMAN - Homogeneous Node Encoder

**Strategy:** Standard self-attention, treat all nodes equally

```python
class _NodeEncoderLayer(nn.Module):
    def __init__(self, D, H, dropout):
        self.sa = _MHA(D, H, dropout)           # self-attention only
    
    def forward(self, h):
        h = norm(h + self.sa(h, h, h))          # all nodes attend to all nodes
```

**Pros:**
- ✓ Simpler architecture (fewer parameters)
- ✓ Easier to train (no demand-based indexing needed)
- ✓ More robust (no edge case handling for empty linehaul/backhaul sets)
- ✓ Demand signal is already in node features (network can learn distinctions)

**Cons:**
- ✗ Doesn't explicitly leverage linehaul/backhaul structure
- ✗ Fewer dedicated parameters for capturing phase transitions
- ✗ Network must learn to distinguish demand types implicitly

---

## Current State

**Training is using GEMAN** (see `configs/network/geman.yaml`):
- Simpler, more robust
- Works with current refactored features
- Successfully training (no dimension issues)

---

## Recommendation: **GEMAN** (for this implementation)

### Why GEMAN is better here:

1. **Feature-Rich Input**: New node features already include `[linehaul_demand, backhaul_demand]` separately
   - Network can learn linehaul/backhaul distinctions from features alone
   - Heterogeneous attention is redundant

2. **Training Stability**: 
   - GEMAN requires no node indexing logic → fewer failure modes
   - No edge cases with empty linehaul/backhaul sets
   - Simpler backprop path

3. **Computational Efficiency**:
   - GEMAN: 1 MHA per layer
   - HGNN: 3 MHA per layer (33% more compute)
   - With CPUs, this matters

4. **Recent Architecture Trends**:
   - Modern GNNs increasingly use **implicit** semantics (learned from features)
   - Rather than **explicit** structure (hand-crafted cross-attention)
   - Reason: Learned representations are more flexible

### When HGNN would be better:

- If node features were sparse/incomplete (only x, y, time window)
- If you wanted to enforce hard phase constraints
- If demand signal wasn't explicitly available to the network
- If you had strong evidence that explicit structure helps convergence

---

## Actionable Insight

**Keep GEMAN for production.** The refactored feature set (6-dim with explicit demand) makes the heterogeneous architecture unnecessary complexity. The network can learn what matters.

If training plateaus, try HGNN only as a last resort—first explore:
- Hyperparameter tuning (learning rate, hidden dim, dropout)
- More training epochs
- Better curriculum learning
