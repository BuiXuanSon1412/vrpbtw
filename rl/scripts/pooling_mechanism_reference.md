# Pooling Mechanism: Formal Names & Abbreviations

## The Mechanism (Used in All 3 Encoders)

```python
# Step 1: Score each element
scores = Linear(h)              # h: (B, N, D) → scores: (B, N, 1)

# Step 2: Normalize via softmax
weights = softmax(scores)       # weights: (B, N, 1), Σ weights = 1

# Step 3: Weighted aggregation
output = (weights * h).sum()    # output: (B, D) - single vector per batch
```

---

## Legitimate Names (In Order of Formality)

### **1. Attention Pooling** ✓ MOST COMMON
- **Full name:** Attention-based pooling
- **Abbreviation:** AP
- **Used in:** GAT, GraphSAINT, Graph Transformer literature
- **Why:** Emphasizes the attention mechanism for selecting important elements

**Example citation:**
> "We aggregate node embeddings using attention pooling with learned importance weights."

---

### **2. Softmax Pooling**
- **Abbreviation:** SP
- **Used in:** Some GNN variants
- **Why:** Explicitly names the softmax normalization
- **Less formal than "Attention Pooling"**

---

### **3. Global Attention Pooling** 
- **Full name:** Global attention pooling / Global context pooling
- **Abbreviation:** GAP
- **Used in:** CNN-to-GNN literature, graph networks
- **Why:** Emphasizes producing "global" (graph-level) representation

**Related:** NOT to be confused with **Global Average Pooling (GAP)** used in CNNs (which is parameter-free mean pooling)

---

### **4. Learnable Pooling / Learned Pooling**
- **Abbreviation:** LP
- **Used in:** Generic machine learning
- **Why:** Broad term emphasizing the learnable parameters
- **Less specific** than attention pooling

---

### **5. Differentiable Pooling**
- **Abbreviation:** DiffPool
- **Used in:** Hierarchical graph pooling literature (Ying et al., 2018)
- **Why:** Emphasizes end-to-end differentiability through pooling
- **Note:** Often used for hierarchical multi-layer pooling, not just single-layer aggregation

---

### **6. Set Aggregation** (Mathematical Term)
- **Abbreviation:** N/A (very technical)
- **Used in:** DeepSets, Set2Set, permutation-invariant learning
- **Why:** Describes the mathematical property (order-independent)
- **Formula:** g = ρ(φ₁(x₁), φ₂(x₂), ..., φₙ(xₙ)) where ρ is permutation-invariant

---

## Most Accurate Choice for Your Work

### **Recommended: "Attention Pooling" (AP)**

**Why it's best:**
- ✓ Standard terminology in GNN literature
- ✓ Immediately understood by researchers
- ✓ Matches the mechanism (learned importance weights)
- ✓ Widely used in recent papers (2020+)

**How to write it:**

In paper/documentation:
```
"Node embeddings are aggregated via attention pooling:
 
 g = Σᵢ αᵢ hᵢ
 
 where αᵢ = softmax(wᵀhᵢ + b) and w, b are learnable parameters."
```

In code comments:
```python
def forward(self, h):
    # Attention pooling: learn which elements matter
    scores = self.pool(h)           # Linear projection
    weights = torch.softmax(scores, dim=1)
    return (weights * h).sum(dim=1) # Weighted aggregation
```

---

## Literature Survey: What Authors Call It

| Paper | Term | Context |
|-------|------|---------|
| **Graph Attention Networks (GAT)** | "Attention mechanism" | Multi-head attention for node aggregation |
| **GraphSAINT** | "Attention pooling" | Graph-level aggregation |
| **Differentiable Pooling (DiffPool)** | "Attention-based pooling" | Hierarchical graph pooling |
| **Graph Isomorphism Network (GIN)** | "Sum aggregation" | No learned weights (baseline) |
| **Transformer (Vaswani et al.)** | "Attention" | Query-weighted aggregation |
| **DeepSets** | "Set aggregation" | Permutation-invariant aggregation |
| **Set2Set** | "Set-to-sequence" | Sequential attention-based aggregation |

---

## Comparison with Related Mechanisms

| Mechanism | Formula | Parameters | Name |
|-----------|---------|-----------|------|
| **Attention Pooling** | Σ softmax(Wh) ⊙ h | D+1 | AP ✓ |
| **Average Pooling** | Σ h / N | 0 | Mean pooling |
| **Max Pooling** | max(h) | 0 | Max pooling |
| **Gating** | Σ sigmoid(Wh) ⊙ h | D+1 | Gating mechanism |
| **Multi-head Attention** | concat[Σ softmax(Q·K) V] | ~4D² | MHA |
| **Set Aggregation** | ρ(φ(x₁), φ(x₂), ...) | Variable | DeepSets |

---

## How to Abbreviate in Your Work

### **Option 1: Full Name (Clearest)**
```
"We use attention pooling to aggregate node embeddings."
→ First mention: full term
→ Later: "AP" (established abbreviation)
```

### **Option 2: Introduce Abbreviation**
```
"We employ attention pooling (AP) to aggregate node embeddings 
 from the encoder layers. AP learns task-specific importance 
 weights via a single linear projection followed by softmax."
```

### **Option 3: In Math/Equations**
```
g = AP(Z)  = Σᵢ αᵢ zᵢ,  where αᵢ = softmax(Linear(zᵢ))

or more formally:

g = AP(Z) = Σᵢ softmax(wᵀzᵢ + b) ⊙ zᵢ
```

---

## Official vs. Informal

| Formality Level | Name | When to Use |
|-----------------|------|------------|
| **Very Formal** | "Differentiable, learnable aggregation via softmax-weighted pooling" | Theory papers |
| **Formal** | "Attention pooling" (AP) | Publications, technical docs ✓ BEST |
| **Semi-formal** | "Softmax pooling" | Internal docs |
| **Informal** | "Learned pooling" / "Weighted average" | Code comments, discussions |

---

## Quick Reference: How to Cite This

If citing the exact mechanism:

**If used in combination with Transformer encoders:**
> "Following Vaswani et al. (2017), we use attention-based pooling to aggregate 
> encoder outputs into a single graph-level representation."

**If used generally in GNNs:**
> "We adopt attention pooling (Hamilton et al., 2017; Veličković et al., 2018) 
> to learn task-specific importance weights for node aggregation."

**If novel contribution:**
> "We propose a learnable attention pooling mechanism that adaptively weights 
> node contributions based on the RL objective, implemented as a single linear 
> projection followed by softmax normalization."

---

## Summary Table

```
╔═══════════════════════════════════════════════════════════════╗
║ MECHANISM:  Linear(D→1) + Softmax + Weighted Sum            ║
╠═══════════════════════════════════════════════════════════════╣
║ PRIMARY NAME:     Attention Pooling                          ║
║ ABBREVIATION:     AP                                         ║
║ ALTERNATIVE:      Attention-based pooling (ABP)              ║
║ FORMULA:          g = Σᵢ softmax(wᵀhᵢ) ⊙ hᵢ                  ║
║ PARAMETERS:       D + 1 (weight vector + bias)               ║
║ LITERATURE:       GAT, GraphSAINT, Graph Transformers        ║
║ USAGE:            Node aggregation, graph-level readout      ║
╚═══════════════════════════════════════════════════════════════╝
```

---

## For Your Codebase

**Recommendation:** Use **"Attention Pooling"** and abbreviate as **"AP"**

Example in your code:

```python
class NodeEncoder(nn.Module):
    def __init__(self, D: int, ...):
        # ...
        self.pool = nn.Linear(D, 1)  # Attention pooling (AP) layer

    def forward(self, node_feat):
        h = self.input_proj(node_feat)
        for layer in self.layers:
            h = layer(h)
        # Attention pooling aggregation
        g = self._attention_pool(h)
        return h, g
    
    def _attention_pool(self, h):
        """Attention pooling (AP): learn which nodes matter.
        
        Returns:
            g: (B, D) graph-level representation via learned attention weights
        """
        scores = self.pool(h)                    # (B, N, 1)
        weights = torch.softmax(scores, dim=1)   # AP weights
        return (weights * h).sum(dim=1)           # (B, D)
```
