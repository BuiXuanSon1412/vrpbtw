# PPO Normalization & Gradient Handling Guide

## Overview

PPO (Proximal Policy Optimization) involves three types of normalization/clipping for stable training:

1. **Advantage Normalization** ✅ (JUST ADDED)
2. **Gradient Norm Clipping** ✅ (Already implemented)
3. **Value Function Normalization** ⚠️ (Partial - returns normalized, values not)

---

## 1. Advantage Normalization (NEWLY IMPLEMENTED)

### What It Does
Rescales advantages to have mean 0 and standard deviation 1, ensuring consistent gradient magnitudes across batches.

### Implementation
```python
# Line 211 in core/agent.py
advantages_normalized = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

# Then use in loss computation (lines 232-234)
surr1 = ratio * advantages_normalized
surr2 = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * advantages_normalized
```

### Why It Matters

**Without normalization:**
```
Batch 1: advantages ∈ [-0.01, 0.01]  
        → policy_loss ≈ 0.001 (tiny update)

Batch 2: advantages ∈ [-100, 100]    
        → policy_loss ≈ 100 (huge update)
        
Result: Unstable, high variance training
```

**With normalization:**
```
Batch 1: advantages ∈ [-1.5, 1.5]    
        → policy_loss ≈ 0.5 (consistent)

Batch 2: advantages ∈ [-1.2, 1.8]    
        → policy_loss ≈ 0.5 (consistent)
        
Result: Stable, predictable training
```

### Mathematical Details

```
μ = mean(advantages)        # Center around 0
σ = std(advantages)         # Standardize spread
ε = 1e-8                    # Prevent division by zero

normalized = (A - μ) / (σ + ε)

Properties:
  • Mean: 0
  • Std: 1 (approximately)
  • Variance independent of batch
```

### When It's Useful

✅ **Helps with:**
- Variable reward scales (small/large bonuses)
- Unstable training curves
- High variance in advantage estimates
- Different task difficulties

✅ **Especially important:**
- When advantage range varies wildly across batches
- With long-horizon problems (large returns)
- With heterogeneous tasks

---

## 2. Gradient Norm Clipping (ALREADY IMPLEMENTED)

### What It Does
Clips gradient magnitude to prevent exploding gradients, protecting against unstable updates.

### Implementation
```python
# Lines 252-254 in core/agent.py
grad_norm_val = torch.nn.utils.clip_grad_norm_(
    self.network.parameters(), 
    self.max_grad_norm  # default: 0.5
)
```

### How It Works

```
Before clipping:
  ||∇|| = 150 (huge gradient)
  
After clipping with max=0.5:
  ∇_clipped = ∇ * (0.5 / 150) = ∇ * 0.0033
  ||∇_clipped|| = 0.5 (safe)
  
Direction preserved, magnitude controlled
```

### Configuration

```yaml
agents:
  tune_agent:
    max_grad_norm: 0.5  # Standard PPO value
```

**Typical ranges:**
- `0.5`: Conservative (most stable)
- `1.0`: Moderate (standard)
- `2.0`: Aggressive (faster but riskier)

### Interaction with Advantage Normalization

✅ **Complement each other:**

```
Advantage Norm  →  Stable loss magnitudes  →  Reasonable gradients
Grad Norm Clip  →  Clip remaining outliers  →  Safe parameter updates

Both together = robust training
```

---

## 3. Value Function Normalization (PARTIAL)

### Current State

**Returns ARE normalized** (in GAECollector):
```python
# collector.py: Line 105
returns = (advantages + values[:T]).detach()
# advantages already normalized by GAE formula
```

**Values are NOT normalized:**
```python
# agent.py: Line 237 (value loss)
value_loss = torch.nn.functional.mse_loss(values, returns)
# Both `values` and `returns` can have large magnitude
```

### What This Means

✅ Advantages are normalized → stable policy updates
❌ Value function sees raw returns → can be large magnitudes

### Example

```
Task 1:  returns ∈ [0, 5]        → value_loss ≈ 1-2
Task 2:  returns ∈ [0, 1000]     → value_loss ≈ 1e6
         
Result: Value loss dominates Task 2 updates
```

### Would Help If

You experience:
- Value loss much larger than policy loss
- Different tasks with vastly different return scales
- Value function dominating gradient updates

---

## Complete PPO Update Flow

```
1. COLLECT TRAJECTORIES
   ├─ Environment rollout
   ├─ Compute log_probs (old policy)
   ├─ Compute values (old policy)
   └─ Compute advantages (GAE)

2. NORMALIZE ADVANTAGES ✅ (NEW)
   └─ advantages_norm = (advantages - μ) / (σ + ε)

3. RE-EVALUATE WITH NEW POLICY
   ├─ Forward pass: new_log_probs, new_values
   ├─ Compute importance ratio
   └─ Clipped surrogate loss

4. COMPUTE LOSSES
   ├─ Policy loss (using advantages_norm)
   ├─ Value loss (MSE: new_values vs returns)
   └─ Entropy bonus

5. BACKWARD & CLIP GRADIENTS ✅ (EXISTING)
   ├─ .backward()
   └─ clip_grad_norm_(max=0.5)

6. OPTIMIZER STEP
   └─ .step()
```

---

## Configuration Recommendations

### Conservative (Stable)
```yaml
agents:
  tune_agent:
    learning_rate: 0.001
    max_grad_norm: 0.5      # Tight clipping
    clip_eps: 0.1           # Small policy change
    entropy_coef: 0.01      # Low exploration
```

**Effect:** Very stable, slower convergence
**Use when:** New task, unstable training

### Balanced (Standard)
```yaml
agents:
  tune_agent:
    learning_rate: 0.001
    max_grad_norm: 0.5      # Standard clipping
    clip_eps: 0.2           # Standard PPO
    entropy_coef: 0.01      # Standard exploration
```

**Effect:** Good stability + convergence
**Use when:** Normal training

### Aggressive (Fast)
```yaml
agents:
  tune_agent:
    learning_rate: 0.005
    max_grad_norm: 1.0      # Loose clipping
    clip_eps: 0.3           # Allow bigger changes
    entropy_coef: 0.05      # More exploration
```

**Effect:** Faster but riskier
**Use when:** Pre-trained weights, known stable

---

## Debugging Training Instability

### Symptom: Loss explodes
```
Solution:
  1. Reduce learning_rate (0.001 → 0.0005)
  2. Tighten max_grad_norm (0.5 → 0.3)
  3. Reduce clip_eps (0.2 → 0.15)
  
Check: Is advantage_norm working? Print std of advantages_normalized
```

### Symptom: No improvement
```
Solution:
  1. Increase learning_rate (0.001 → 0.005)
  2. Increase entropy_coef (0.01 → 0.05)
  3. Loosen max_grad_norm (0.5 → 1.0)
  
Check: Are advantages_normalized near-zero? May indicate bad rewards
```

### Symptom: High variance between runs
```
Solution:
  1. Increase rollout_length (256 → 512)
  2. Increase ppo_epochs (3 → 5)
  3. Tighten max_grad_norm (0.5 → 0.3)
  
Check: Is seed randomness the issue? Fix seed for debugging
```

---

## Summary Table

| Feature | Status | Purpose | Impact |
|---------|--------|---------|--------|
| **Advantage Norm** | ✅ NEW | Stable advantage scale | Consistent learning signal |
| **Grad Norm Clip** | ✅ EXISTING | Prevent explosion | Safe parameter updates |
| **Value Norm** | ❌ MISSING | Stable value learning | Could help large-return tasks |
| **Entropy** | ✅ IMPLEMENTED | Encourage exploration | Prevents early convergence |
| **Return Normalization** | ✅ IMPLICIT | Through GAE | Better variance reduction |

---

## Key Takeaways

1. **Advantage Normalization** (just added)
   - Ensures consistent gradient magnitudes across batches
   - Critical for variable-scale problems
   - Low computational cost

2. **Gradient Norm Clipping** (already here)
   - Protects against exploding gradients
   - Use standard value (0.5) unless debugging
   - Always used with advantage normalization

3. **Together They Work Best**
   - Advantage norm: soft limit on gradient size
   - Grad norm clip: hard limit on gradient size
   - Complementary safety mechanisms

---

## References

- Advantage Normalization: PPO paper (Schulman et al., 2017) § 4
- Gradient Clipping: Common practice in policy gradient methods
- Implementation: `torch.nn.utils.clip_grad_norm_` documentation

