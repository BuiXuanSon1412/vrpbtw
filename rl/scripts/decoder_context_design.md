# Decoder Context Design: Why mean(Z_veh) Instead of g_veh?

## The Question

In **NodeDecoder**, why use `mean(Z_veh)` instead of `g_veh`?

```python
# What NodeDecoder receives:
ln = self.node_decoder(
    g_node,      # ← Global node summary (from AP)
    Z_veh,       # ← Per-vehicle embeddings (NOT aggregated)
    g_graph,     # ← Global graph summary (from AP)
    Z_node       # ← Per-node embeddings (used for matching)
)

# Inside NodeDecoder:
z_veh_mean = Z_veh.mean(1)  # ← Simple mean, NOT g_veh!
ctx = MLP(concat[g_node, z_veh_mean, g_graph])
```

**Key observation:** `g_veh` is NOT passed to NodeDecoder at all!

---

## The Pattern: Cross-Modal Information Exchange

### How Decoders are Called

```python
# Line 1148-1149 in geman.py:

ln = self.node_decoder(g_node, Z_veh, g_graph, Z_node, n_mask)
#                        ↑      ↑     ↑       ↑
#                      own    cross   own    own
#                     global  modal  global  per-element
#                                    summary

lv = self.vehicle_decoder(g_veh, Z_node, g_graph, Z_veh, v_mask)
#                          ↑      ↑      ↑       ↑
#                        own    cross   own    own
#                       global  modal  global  per-element
#                                      summary
```

### Pattern Discovery

| Decoder | Receives | Uses in Context | Does NOT use |
|---------|----------|-----------------|--------------|
| **NodeDecoder** | g_node, Z_veh, g_graph, Z_node | g_node, **mean(Z_veh)**, g_graph | g_veh (not passed!) |
| **VehicleDecoder** | g_veh, Z_node, g_graph, Z_veh | g_veh, **mean(Z_node)**, g_graph | g_node (not passed!) |

---

## Why This Design? (3 Possible Reasons)

### **Reason 1: Symmetry & Simplicity**

The architecture enforces **symmetry** between decoders:

```
NodeDecoder scoring nodes:
  Context = [own_global (g_node), cross_simple (mean Z_veh), graph_global (g_graph)]
                                        ↑
                                  uniform/unbiased view

VehicleDecoder scoring vehicles:
  Context = [own_global (g_veh), cross_simple (mean Z_node), graph_global (g_graph)]
                                        ↑
                                  uniform/unbiased view
```

**Benefit:** Each decoder gets an unbiased view of the other modality (all equal importance)

---

### **Reason 2: Avoid Information Bottleneck**

**If using g_veh** (learned AP):
```
Z_veh (B, 2K, D) with N=2K separate vectors
           ↓
         [Attention Pooling]  ← learns which vehicles matter in general
           ↓
        g_veh (B, D)          ← compressed to single vector

NodeDecoder uses g_veh:
  Problem: Lost information about individual vehicles
  g_veh is biased toward "important" vehicles in general
  But for THIS specific node, we might need different vehicles!
```

**With mean(Z_veh)** (simple average):
```
Z_veh (B, 2K, D) with N=2K separate vectors
           ↓
        mean(Z_veh) (B, D)    ← simple, unbiased average
           ↓
NodeDecoder uses mean(Z_veh):
  Benefit: Preserves information about ALL vehicles equally
  NodeDecoder can decide which vehicles matter for each node
  Via the learned attention mechanism in the decoder itself
```

---

### **Reason 3: Task-Specific Attention Over Cross-Modal Features**

The decoder's **scaled dot-product attention** is the mechanism for cross-modal importance:

```
NodeDecoder decides which nodes matter via:
  
  logits[n] = SDPA(
    query = learned_context(g_node, mean(Z_veh), g_graph),
    keys = [Z_node[0], Z_node[1], ..., Z_node[N]]
  )
  
NodeDecoder effectively learns:
  "Given this context (nodes, vehicles, graph),
   which NODES are best to select?"
   
The decoder's attention mechanism does the heavy lifting.
It doesn't need g_veh's learned vehicle importance—
it can learn its own via the attention scores.
```

**Contrast with using g_veh:**
```
If we used g_veh (pre-computed importance):
  The g_veh weights are "baked in" from the encoder
  NodeDecoder can't adapt vehicle importance per node
  It just gets a fixed vehicle summary
  
With mean(Z_veh):
  NodeDecoder has full flexibility
  It can re-weight vehicles based on each node decision
```

---

## Empirical Impact

### **Option A: Using mean(Z_veh)** (Current Design)

```python
ctx = MLP(concat[g_node, mean(Z_veh), g_graph])
# Context shape: (B, 3D)
# Vehicle influence: Democratic (all equal)
# Flexibility: High (decoder adapts per node)
```

**Pros:**
- Symmetric architecture
- Flexible cross-modal attention
- Simpler computation (mean vs. learned pool)
- Each decoder learns its own cross-modal importance

**Cons:**
- Loses learned vehicle importance from encoder
- Less efficient (doesn't reuse g_veh computation)

---

### **Option B: Using g_veh (Alternative)**

```python
ctx = MLP(concat[g_node, g_veh, g_graph])
# Context shape: (B, 3D)
# Vehicle influence: Learned (from encoder AP)
# Flexibility: Low (fixed vehicle importance)
```

**Pros:**
- Reuses learned vehicle importance
- More information from encoder

**Cons:**
- Breaks symmetry
- Information bottleneck (vehicle diversity lost)
- Decoder can't adapt vehicle importance per node
- Would require passing g_veh (architectural change)

---

## Design Philosophy

**GEMAN uses:** Separate concern principle

```
Encoder layers:
  - Learn what vectors (Z_node, Z_veh, Z_graph) should look like
  - Do summary aggregation (AP) for global context

Decoder layers:
  - Use global context + per-element embeddings
  - Learn task-specific matching (SDPA) for action selection
  - Can independently attend to any vehicle/node
```

This separation means:
1. **Encoders** learn representations
2. **Global pooling (AP)** creates summaries
3. **Decoders** learn context-dependent importance

---

## What If We Changed It?

### **Alternative 1: Always Use Global Summaries**

```python
# All globals:
ctx = MLP(concat[g_node, g_veh, g_graph])
```

**Impact:**
- Simpler context (3 globals only)
- BUT: Less information about individual vehicles/nodes
- Decoder has to infer diversity from limited context
- Likely worse performance (less detail for matching)

---

### **Alternative 2: Always Use Per-Element Means**

```python
# All means:
ctx = MLP(concat[mean(Z_node), mean(Z_veh), g_graph])
```

**Impact:**
- Consistent with current NodeDecoder design
- VehicleDecoder already does this!
- Symmetric but loses learned importance from both encoders
- Simpler but less information

---

### **Alternative 3: Use Both (More Expressive)**

```python
# Concat all summaries + means:
ctx = MLP(concat[g_node, g_veh, mean(Z_node), mean(Z_veh), g_graph])
```

**Impact:**
- More information (6D input instead of 3D)
- Higher parameter count
- Richer context for matching
- Decoder can learn to weight global vs. local views
- More complex, potentially better

---

## Actual Current Design

Looking at the code:

```python
# NodeDecoder receives: g_node, Z_veh, g_graph, Z_node
ctx = MLP(concat[
    g_node,              # (B, D) - Learned encoder summary
    mean(Z_veh),         # (B, D) - Simple average (no learning)
    g_graph              # (B, D) - Learned encoder summary
])  # Total: (B, 3D) → MLP → (B, D_ctx)

# VehicleDecoder receives: g_veh, Z_node, g_graph, Z_veh
ctx = MLP(concat[
    g_veh,               # (B, D) - Learned encoder summary
    mean(Z_node),        # (B, D) - Simple average (no learning)
    g_graph              # (B, D) - Learned encoder summary
])  # Total: (B, 3D) → MLP → (B, D_ctx)
```

**Why this asymmetry?**

Actually, it's SYMMETRIC! Both follow the same pattern:
- Own global summary (learned AP)
- Cross-modal simple average (unbiased)
- Graph global summary (learned AP)

The asymmetry is in what gets passed:
- NodeDecoder doesn't receive g_veh (it's not computed; only g_node is passed)
- But it COULD receive g_veh if architecturally changed

---

## Summary: Why mean(Z_veh) Not g_veh?

| Aspect | Reason |
|--------|--------|
| **Information Preservation** | mean() keeps all vehicle diversity; g_veh is compressed summary |
| **Flexibility** | Decoder can learn its own cross-modal importance via SDPA |
| **Symmetry** | Both decoders use same pattern: own + cross_mean + graph |
| **Simplicity** | mean() doesn't require passing g_veh (architectural simplicity) |
| **Unbiased Context** | mean(Z_veh) gives equal weight to all vehicles, unbiased |
| **Task-Specificity** | Node selection shouldn't be constrained by encoder's vehicle importance |

**Final answer:** The design uses **simple means instead of learned global summaries** for cross-modal context to:
1. Provide unbiased information to each decoder
2. Let decoders learn their own task-specific attention
3. Maintain architectural symmetry
4. Avoid information bottleneck from compression

