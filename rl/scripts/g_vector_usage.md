# Where Are g_node, g_veh, g_graph Used?

## Quick Answer

**`g_veh` is used in 2 places:**

1. **VehicleDecoder** (line 1149): Context for scoring vehicles
2. **ValueHead** (line 1160): State value estimation

---

## Complete Trace Through the Forward Pass

```python
# GEMANActorCritic.forward()

def forward(self, obs, action_mask=None, context=None):
    device = globals.DEVICE
    
    # ════════════════════════════════════════════════════════════════
    # STEP 1: ENCODE (produces all g_* vectors)
    # ════════════════════════════════════════════════════════════════
    Z_node, g_node, Z_veh, g_veh, Z_graph, g_graph, N1, V2K = self._encode(obs, device)
    
    #   g_node:  (B, D=128)  ← from NodeEncoder attention pooling
    #   g_veh:   (B, D=128)  ← from VehicleEncoder attention pooling
    #   g_graph: (B, D=128)  ← from GraphEncoder attention pooling
    
    # ════════════════════════════════════════════════════════════════
    # STEP 2: DECODE NODE LOGITS
    # ════════════════════════════════════════════════════════════════
    
    ln = self.node_decoder(g_node, Z_veh, g_graph, Z_node, n_mask)  # (B, N+1)
    #                        ↑      ↑     ↑       ↑
    #                      Used   Used  Used    Used
    #   
    #   g_node is input to NodeDecoder's context MLP
    #   (g_veh is NOT used here)
    
    # ════════════════════════════════════════════════════════════════
    # STEP 3: DECODE VEHICLE LOGITS  ← g_veh USED HERE!
    # ════════════════════════════════════════════════════════════════
    
    lv = self.vehicle_decoder(g_veh, Z_node, g_graph, Z_veh, v_mask)  # (B, 2K)
    #                          ↑     ↑      ↑       ↑
    #                     Used  Used  Used  Used
    #
    #   ★ g_veh is PRIMARY input to VehicleDecoder's context MLP
    #   This is g_veh's FIRST MAJOR USE
    
    # ════════════════════════════════════════════════════════════════
    # STEP 4: COMBINE INTO JOINT ACTION LOGITS
    # ════════════════════════════════════════════════════════════════
    
    flat = (ln.unsqueeze(2) + lv.unsqueeze(1)).view(...)
    #  (B, N+1, 1) + (B, 1, 2K) → (B, N+1, 2K) → (B, N+1×2K)
    # Combine node and vehicle scores into joint logits
    
    # ════════════════════════════════════════════════════════════════
    # STEP 5: ESTIMATE STATE VALUE  ← g_veh USED AGAIN!
    # ════════════════════════════════════════════════════════════════
    
    # Prepare for ValueHead
    g_node_2d = g_node if g_node.dim() == 2 else g_node.flatten(1)      # (B, D)
    g_veh_2d = g_veh if g_veh.dim() == 2 else g_veh.flatten(1)          # (B, D) ★
    g_graph_2d = g_graph if g_graph.dim() == 2 else g_graph.flatten(1)  # (B, D)
    
    # Concatenate all global representations
    state_repr = torch.cat([g_node_2d, g_veh_2d, g_graph_2d], dim=-1)  # (B, 3D)
    #                                    ↑
    #                            g_veh USED HERE (2nd place)
    
    value = self.value_head(state_repr).squeeze(-1)  # (B,)
    
    return flat, value
```

---

## g_veh Usage Summary

### **Use #1: VehicleDecoder Context (Primary)**

```python
# Line 1149
lv = self.vehicle_decoder(g_veh, Z_node, g_graph, Z_veh, v_mask)
                           ↑
                  Main global context for scoring vehicles

# Inside VehicleDecoder.forward():
ctx = self.ctx_proj(torch.cat([
    g_veh,              # (B, D) ← "What's the vehicle situation?"
    mean(Z_node),       # (B, D) ← "What are typical nodes like?"
    g_graph             # (B, D) ← "What's the overall problem?"
], dim=-1))
# ctx: (B, 3D) → MLP → (B, context_dim) → Linear → Query for attention
```

**Meaning:** g_veh tells the VehicleDecoder "What's the overall vehicle state?" so it can score which vehicle to select next.

---

### **Use #2: ValueHead Input (State Representation)**

```python
# Lines 1157, 1159-1161
value = self.value_head(
    torch.cat([
        g_node,   # (B, D) ← "What's the node situation?"
        g_veh,    # (B, D) ← "What's the vehicle situation?"
        g_graph   # (B, D) ← "What's the overall problem?"
    ], dim=-1)
)  # (B, 3D) → MLP → (B, 1)

# Inside ValueHead:
layers = [
    nn.Linear(3*D, hidden_dims[0]), nn.ReLU(),  # (B, 3D) → (B, hidden)
    ...
    nn.Linear(hidden_dims[-1], 1)               # (B, hidden) → (B, 1)
]
```

**Meaning:** g_veh contributes to the state value estimate (baseline for advantage estimation in PPO).

---

## Comparison: Usage of All Three g_* Vectors

```
                │  NodeDecoder  │  VehicleDecoder  │  ValueHead
────────────────┼───────────────┼──────────────────┼────────────────
g_node          │      ✓        │        ✗         │      ✓
g_veh           │      ✗        │        ✓         │      ✓
g_graph         │      ✓        │        ✓         │      ✓
────────────────┼───────────────┼──────────────────┼────────────────
```

**Symmetry:**
- g_node and g_veh are NOT symmetric in decoders
  - g_node → NodeDecoder (asymmetry!)
  - g_veh → VehicleDecoder
- But they ARE symmetric in ValueHead
  - Both contribute equally to state value

---

## Why This Design?

### **Decoder Asymmetry**

**NodeDecoder:**
```
"I'm choosing a NODE. What do I need to know?"
  - g_node:      "What's important about nodes?"       ✓ Direct
  - mean(Z_veh): "What are vehicles like (average)?"   ✓ Cross-modal (simple)
  - g_graph:     "What's the overall problem?"         ✓ Problem context
```

**VehicleDecoder:**
```
"I'm choosing a VEHICLE. What do I need to know?"
  - g_veh:       "What's important about vehicles?"    ✓ Direct (vs. mean!)
  - mean(Z_node):"What are nodes like (average)?"      ✓ Cross-modal (simple)
  - g_graph:     "What's the overall problem?"         ✓ Problem context
```

**Why g_veh but mean(Z_node)?** 
- VehicleDecoder gets learned vehicle importance (g_veh)
- But simple node average (mean Z_node, no learning)
- This is... actually ASYMMETRIC!

Wait, let me reconsider. Looking more carefully:

```python
# NodeDecoder receives:
ln = self.node_decoder(g_node, Z_veh, g_graph, Z_node, n_mask)
                        ↑      ↑              ↑      ↑
                     global  per-el         global per-el

# VehicleDecoder receives:
lv = self.vehicle_decoder(g_veh, Z_node, g_graph, Z_veh, v_mask)
                           ↑     ↑               ↑     ↑
                        global per-el         global per-el

# Pattern is SYMMETRIC:
# [own_global, cross_per_element, graph_global, own_per_element]
```

So BOTH use per-element embeddings for cross-modal info:
- NodeDecoder: g_node + Z_veh (per-vehicle per-element) + g_graph + Z_node
- VehicleDecoder: g_veh + Z_node (per-node per-element) + g_graph + Z_veh

But NodeDecoder uses **mean(Z_veh)** in context, VehicleDecoder uses **g_veh**.

Actually, let me re-read the code more carefully...

Wait, NodeDecoder receives Z_veh and computes mean(Z_veh) internally.
VehicleDecoder receives Z_node and computes mean(Z_node) internally.

So both are using means of the cross-modal per-element embeddings!

Then g_veh is ONLY used in VehicleDecoder... but wait, let me check the signature again.

NodeDecoder signature:
```python
def forward(
    self,
    g_node: torch.Tensor,       # (B, D)
    Z_veh: torch.Tensor,        # (B, 2K, D)
    g_graph: torch.Tensor,      # (B, D)
    Z_node: torch.Tensor,       # (B, N+1, D)
    mask: Optional[torch.Tensor] = None,
)
```

It receives g_node (not g_veh) and Z_veh.

VehicleDecoder signature:
```python
def forward(
    self,
    g_veh: torch.Tensor,        # (B, D)
    Z_node: torch.Tensor,       # (B, N+1, D)
    g_graph: torch.Tensor,      # (B, D)
    Z_veh: torch.Tensor,        # (B, 2K, D)
    mask: Optional[torch.Tensor] = None,
)
```

It receives g_veh (not g_node) and Z_node.

So:
- NodeDecoder gets: g_node, Z_veh, g_graph, Z_node
- VehicleDecoder gets: g_veh, Z_node, g_graph, Z_veh

This IS symmetric! Each gets its own global (g_node/g_veh) and the cross-modal per-element (Z_veh/Z_node).

So g_veh is used:
1. As primary input to VehicleDecoder (for context)
2. In ValueHead (for value estimation)

And importantly, g_veh is NOT passed to NodeDecoder at all!

This is the architectural design - each decoder gets its own global representation, not the other's.
