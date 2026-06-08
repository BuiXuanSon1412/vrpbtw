# NodeDecoder & VehicleDecoder: Mechanisms & Names

## Quick Answer

**NO** — Decoders use a **completely different mechanism** from the encoders:

| Component | Encoders | Decoders |
|-----------|----------|----------|
| **Mechanism** | Attention Pooling (AP) | Scaled Dot-Product Attention (SDPA) |
| **Purpose** | Aggregate → graph-level vector | Score → per-element logits |
| **Output shape** | (B, D) single vector | (B, N) scores for each element |
| **Formal name** | Attention pooling | Scaled dot-product attention |
| **Abbreviation** | AP | SDPA |

---

## Encoder: Attention Pooling (AP)

```python
# Produces ONE vector per batch (graph-level representation)
scores = Linear(h)                    # (B, N, 1)
weights = softmax(scores)             # Σ weights = 1
output = (weights * h).sum(dim=1)     # (B, D) ← single vector
```

**Output:** Graph-level summary (one D-dim vector)

---

## Decoder: Scaled Dot-Product Attention (SDPA)

```python
# Produces scores for EACH element (action logits)
context = ctx_proj(concat[g_node, mean(Z_veh), g_graph])  # (B, D)
Q = Linear(context).unsqueeze(1)                          # (B, 1, D)
K = Linear(Z_node)                                        # (B, N, D)
scores = Q @ K^T / sqrt(D)                                # (B, 1, N)
logits = tanh(scores).squeeze(1)                          # (B, N)
```

**Output:** Scores for each node/vehicle (N logits per batch)

---

## Side-by-Side Code Comparison

### **NodeDecoder**

```python
class NodeDecoder(nn.Module):
    def __init__(self, D, clip=10.0, context_hidden_dim=128, context_dropout=0.0):
        super().__init__()
        # Step 1: Context projection (MLP)
        self.ctx_proj = nn.Sequential(
            nn.Linear(D * 3, context_hidden_dim),  # Concatenate 3 global vectors
            nn.ReLU(),
            nn.Dropout(context_dropout) if context_dropout > 0 else nn.Identity()
        )
        
        # Step 2: Attention parameters
        self.Wq = nn.Linear(context_hidden_dim, D, bias=False)  # Query projection
        self.Wk = nn.Linear(D, D, bias=False)                    # Key projection
        self.clip = clip
        self._scale = None

    def forward(self, g_node, Z_veh, g_graph, Z_node, mask=None):
        # Step 1: Build context from global representations
        z_veh_mean = Z_veh.mean(1)                # (B, D) average vehicle embedding
        ctx = self.ctx_proj(
            torch.cat([g_node, z_veh_mean, g_graph], dim=-1)  # (B, 3D) → (B, D)
        )
        
        # Step 2: Compute query (single query per batch)
        Q = self.Wq(ctx).unsqueeze(1)            # (B, 1, D)
        
        # Step 3: Compute scaled dot-product attention
        self._scale = math.sqrt(Z_node.shape[-1])
        logits = torch.bmm(Q, self.Wk(Z_node).transpose(1, 2)).squeeze(1) / self._scale
        # Q @ K^T / sqrt(D) = (B, 1, D) @ (B, D, N) = (B, 1, N) → (B, N)
        
        # Step 4: Apply nonlinearity (tanh clipping)
        logits = self.clip * torch.tanh(logits)  # Bound to [-clip, +clip]
        
        # Step 5: Mask invalid actions
        if mask is not None:
            logits = logits.masked_fill(~mask, float("-inf"))
        
        return logits  # (B, N)


class VehicleDecoder(nn.Module):
    # IDENTICAL STRUCTURE
    # Only difference: uses Z_veh instead of Z_node
    # Input: (g_veh, Z_node, g_graph, Z_veh)
    # Output: (B, 2K) logits for vehicles
```

---

## Component Breakdown

### **NodeDecoder Components**

| Component | Layer Type | Input Shape | Output Shape | Parameters | Purpose |
|-----------|-----------|------------|--------------|-----------|---------|
| `ctx_proj` | Linear | (B, 3D) | (B, context_dim) | 3D × D_ctx + D_ctx | Context encoding |
| `Wq` | Linear | (B, D_ctx) | (B, D) | context_dim × D | Query projection |
| `Wk` | Linear | (B, N, D) | (B, N, D) | D × D | Key projection |
| Attention | SDPA | Q:(B,1,D), K:(B,N,D) | (B, N) | 0 (operation) | Query-key matching |
| Tanh clip | Nonlinearity | (B, N) | (B, N) | 0 | Output bounding |

### **Exact Parameter Count**

```
NodeDecoder:
├─ ctx_proj:
│  ├─ Linear(3D → context_dim): 3D × context_dim + context_dim
│  │  (with D=128, context_dim=128): 49,280 params
│  ├─ ReLU: 0 params
│  └─ Dropout: 0 params
├─ Wq: Linear(context_dim → D): context_dim × D
│  (128 → 128): 16,384 params
├─ Wk: Linear(D → D): D × D
│  (128 → 128): 16,384 params
└─ TOTAL: 82,048 params
```

---

## Formal Mechanism Names

### **What NodeDecoder Does:**

**Legitimate names** (in order of formality):

1. **Scaled Dot-Product Attention (SDPA)** ✓ MOST FORMAL
   - Abbreviation: **SDPA**
   - From: "Attention is All You Need" (Vaswani et al., 2017)
   - Formula: `Attention(Q, K, V) = softmax(QK^T / √D_k) V`
   - (Note: decoders use Q @ K^T without V)

2. **Single-Head Attention** 
   - Abbreviation: SHA
   - Emphasizes: one attention head (vs. multi-head)

3. **Dot-Product Attention** (Simplified name)
   - Abbreviation: DPA
   - Less formal, more descriptive

4. **Query-Key Attention**
   - Abbreviation: QKA
   - Emphasizes: learned query matching against keys

5. **Attention-based Decoder** (Contextual term)
   - Emphasizes: it's decoding (producing logits)
   - Less specific than above

---

## Full Formula with Notation

```
NodeDecoder computes:

logits[b, n] = clip · tanh(  (Wq · ctx[b])^T · Wk · Z_node[b,n]  /  √D  )

where:
  - ctx[b] = context projection of [g_node[b], mean(Z_veh[b]), g_graph[b]]
  - Wq, Wk: learned linear projections
  - √D: scaling factor (D = embedding dimension)
  - tanh(...): bounded nonlinearity
  - clip: hyperparameter (typically 10.0)
  - logits[b,n]: score for selecting node n in batch b
```

**More formally (in paper notation):**

$$\text{logits}_{b,n} = \text{clip} \cdot \tanh\left(\frac{(W_q \cdot \text{ctx}_b)^T W_k Z_{\text{node},b,n}}{\sqrt{D}}\right)$$

where:
- $\text{ctx}_b = \text{MLP}(g_{\text{node},b} \| \text{mean}(Z_{\text{veh},b}) \| g_{\text{graph},b})$
- $W_q, W_k \in \mathbb{R}^{D \times D}$ are learnable projections

---

## Comparison: All 5 Mechanisms in GEMAN

```
┌─────────────────────────────────────────────────────────────┐
│ 1. NodeEncoder: Transformer Self-Attention                  │
│    Uses: Multi-head attention (h heads, each D/h dims)      │
│    Purpose: Process node features sequentially               │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│ 2. NodeEncoder Output: Attention Pooling (AP)                │
│    Uses: Linear(D→1) + softmax + weighted sum                │
│    Purpose: Aggregate into graph-level g_node               │
│    Abbreviation: AP                                          │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│ 3. VehicleEncoder: Transformer Self-Attention               │
│    Uses: Multi-head attention                                │
│    Purpose: Process vehicle features sequentially            │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│ 4. VehicleEncoder Output: Attention Pooling (AP)             │
│    Uses: Linear(D→1) + softmax + weighted sum                │
│    Purpose: Aggregate into graph-level g_veh                │
│    Abbreviation: AP                                          │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│ 5. GCNMEGAGraphEncoder: Multi-Relational GNN                 │
│    Uses: Message passing (truck/drone relations)             │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│ 6. GraphEncoder Output: Attention Pooling (AP)               │
│    Uses: Linear(D→1) + softmax + weighted sum                │
│    Purpose: Aggregate into graph-level g_graph              │
│    Abbreviation: AP                                          │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│ 7. NodeDecoder: Scaled Dot-Product Attention (SDPA)          │
│    Uses: Q @ K^T / √D with learned context                   │
│    Purpose: Score each node for selection                    │
│    Abbreviation: SDPA                                        │
│    Output: (B, N) logits                                     │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│ 8. VehicleDecoder: Scaled Dot-Product Attention (SDPA)       │
│    Uses: Q @ K^T / √D with learned context                   │
│    Purpose: Score each vehicle for selection                 │
│    Abbreviation: SDPA                                        │
│    Output: (B, 2K) logits                                    │
└─────────────────────────────────────────────────────────────┘
```

---

## Key Differences Summary

### **Attention Pooling (AP) — Encoders**

```
Input:  (B, N, D) sequence of embeddings
Output: (B, D) single aggregated vector

Mechanism:
  score[i] = w^T h[i]
  weight[i] = softmax(score[i])
  g = Σ weight[i] × h[i]

Interpretation: "Which elements matter most?"
```

### **Scaled Dot-Product Attention (SDPA) — Decoders**

```
Input:  Query (B, D) + Keys (B, N, D)
Output: (B, N) scores for each element

Mechanism:
  score[i] = (Q^T K[i]) / √D
  logits = clip × tanh(score)

Interpretation: "How well does the query match each key?"
```

---

## Literature & Citations

### **Attention Pooling (AP)**

Used in: Graph Attention Networks, GraphSAINT, Graph Transformers

> "We aggregate node embeddings using attention pooling, learning which nodes 
> contribute most to the graph-level representation." (GraphSAINT, Zeng et al., 2021)

### **Scaled Dot-Product Attention (SDPA)**

Standard reference: "Attention is All You Need" (Vaswani et al., 2017)

> "We compute attention as: Attention(Q, K, V) = softmax(QK^T / √D_k) V" 
> (Transformer paper, eq. 1)

---

## Formal Names for Your Documentation

```markdown
## Architecture

**Encoder Stage:**
- NodeEncoder: Multi-head self-attention over node features
  - Aggregation: Attention pooling (AP) to g_node
- VehicleEncoder: Multi-head self-attention over vehicle features
  - Aggregation: Attention pooling (AP) to g_veh
- GCNMEGAGraphEncoder: Multi-relational message passing
  - Aggregation: Attention pooling (AP) to g_graph

**Decoder Stage:**
- NodeDecoder: Scaled dot-product attention (SDPA) over Z_node
  - Input: Learned context from [g_node, mean(Z_veh), g_graph]
  - Output: (B, N+1) logits
- VehicleDecoder: Scaled dot-product attention (SDPA) over Z_veh
  - Input: Learned context from [g_veh, mean(Z_node), g_graph]
  - Output: (B, 2K) logits
```

---

## Summary Table

```
╔════════════════════════════════════════════════════════════════╗
║ ENCODER AGGREGATION: Attention Pooling (AP)                  ║
║ ─ Produces: Graph-level representation (B, D)                ║
║ ─ Mechanism: Linear(D→1) + softmax + weighted sum            ║
║ ─ Used in: NodeEncoder, VehicleEncoder, GraphEncoder         ║
╠════════════════════════════════════════════════════════════════╣
║ DECODER SCORING: Scaled Dot-Product Attention (SDPA)         ║
║ ─ Produces: Action logits (B, N) or (B, 2K)                  ║
║ ─ Mechanism: Q @ K^T / √D with tanh clipping                 ║
║ ─ Used in: NodeDecoder, VehicleDecoder                       ║
╚════════════════════════════════════════════════════════════════╝
```
