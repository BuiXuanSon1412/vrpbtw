# Full PPO Pipeline Evaluation
## GAECollector → PPOAgent → MetaTrainer

---

## Pipeline Overview

```
GAECollector.collect()
    ↓ (collects trajectory with agent.act())
    ↓ (computes log_probs, values, entropies from network)
    ↓ (computes GAE advantages/returns)
    ↓ returns batch dict
    ↓
PPOAgent.update(batch)
    ↓ (uses batch["log_probs"] from collector)
    ↓ (recomputes log_probs from logits - FIXED BUG #1)
    ↓ (uses advantages from GAE)
    ↓
MetaTrainer.fine_tune() / meta_train()
    ↓ (calls collector.collect() → agent.update() in loop)
```

---

## Issue Analysis

### **Issue 1: GAECollector vs PPOAgent Log Probability Mismatch** 🔴 CRITICAL

**The Problem**: Dual log probability computation causing inconsistency

**GAECollector (line 162)**:
```python
action_t, lp, val, ent = agent.act(obs_t, mask_t, deterministic=False)
log_probs.append(lp)  # ← These are correct log probs
```

**PPOAgent.update() (BEFORE FIX)**:
```python
new_log_probs = torch.cat(logits_list, dim=0)  # ← LOGITS, not log probs!
# Then: ratio = exp(new_log_probs - old_log_probs)
```

**What Happens**:
1. Collector computes `log_probs` ✅ (correct)
2. Collector stores as `old_log_probs` in batch ✅
3. PPOAgent receives them ✅
4. PPOAgent recomputes from logits as `new_log_probs` ❌ (WRONG)
5. Ratio = exp(logits - correct_log_probs) → **BROKEN**

**After Fix**:
```python
logits = torch.cat(logits_list, dim=0)
new_log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
new_log_probs = new_log_probs.gather(-1, batch["actions"].unsqueeze(-1)).squeeze(-1)
# Now: ratio = exp(correct_log_probs - old_log_probs) ✅
```

---

### **Issue 2: Advantage Source Inconsistency** 🟡 MEDIUM

**The Problem**: Advantages computed in collector, normalized in agent

**GAECollector (line 208-214)**:
```python
advantages, returns = _compute_gae(
    rewards_tensor,
    values_with_bootstrap,
    dones_tensor,
    gamma=self.gamma,
    gae_lambda=self.gae_lambda,
)
# Returns unnormalized advantages
```

**PPOAgent (line 211)**:
```python
advantages_normalized = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
# Normalizes after receiving
```

**Why This Matters**:
- ✅ Normalization is correct for PPO stability
- ⚠️ But happens AFTER being computed by GAE
- ⚠️ One-episode batch: normalization may be unstable with small T
- ✅ Example: T=50 steps, advantages.std() could be tiny or zero

**Example Problem Case**:
```
Episode length: 10 steps
Advantages: [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
Mean: 0.5, Std: 0.0
Normalized: NaN (division by ~0)
```

**Better Approach**: Normalize in GAECollector before returning:
```python
# In GAECollector.collect() before returning
advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
```

---

### **Issue 3: Bootstrap Value Sign Inconsistency** 🟡 MEDIUM

**The Problem**: Bootstrap value handling at episode end

**GAECollector (line 101)**:
```python
# In _compute_gae():
delta = rewards[t] + gamma * next_value * (1 - dones[t]) - current_value
#                                           └─ This zeroes next_value when done
```

**When dones[t] = 1 (episode ended)**:
```
delta = rewards[t] + gamma * next_value * 0 - current_value
      = rewards[t] - current_value
# next_value is correctly ignored ✅
```

**But GAECollector line 188**:
```python
_, _, bootstrap_val, _ = agent.act(obs_end, mask_end, deterministic=False)
# Gets value estimate at terminal state
```

**Issue**: If episode ended due to truncation (not natural termination):
- Bootstrap value should be used for return calculation
- But if natural termination, bootstrap should be zero
- Current code doesn't distinguish between `terminated` vs `truncated`

**Check (line 180-181)**:
```python
if terminated or truncated or not action_mask.any():
    break
```

Both are treated the same, but:
- `terminated=True` (natural, constraint-based end): bootstrap_val should = 0
- `truncated=True` (step limit): bootstrap_val should be used

---

### **Issue 4: Value Loss Instability** 🟡 MEDIUM

**PPOAgent (line 235, 239)**:
```python
value_loss = torch.nn.functional.mse_loss(values, returns)
# MSE between value predictions and returns
```

**Problem**: 
- Values computed during collection (stale)
- PPOAgent recomputes values from logits in update
- But these are recomputed at **same observations** with **updated network**
- Leads to off-policy value training

**Why It Matters**:
- Collection: values computed with old policy θ_old
- Update: values recomputed with new policy θ_new
- Returns computed from old trajectory
- This is off-policy value learning → can diverge

**Better Approach**: Use bootstrapped returns, not original rewards
```python
# Returns are already computed in collector with old network
# No need to recompute values - just use collected values
```

---

### **Issue 5: Entropy Tensor Shape Issues** 🟡 MEDIUM

**GAECollector (line 197)**:
```python
entropies_tensor = torch.stack(entropies).squeeze(-1)
# entropies from agent.act() are (B,) or (1,)
```

**PPOAgent (line 207)**:
```python
entropies = batch.get("entropies", torch.tensor(0.0))
if isinstance(entropies, torch.Tensor) and entropies.numel() > 0:
    entropy_loss = -entropies.mean()
else:
    entropy_loss = torch.tensor(0.0, device=policy_loss.device)
```

**Potential Issue**:
- If entropies is (T,), mean() aggregates correctly
- But entropy is computed on action distribution, should it be normalized by episode length?

**Current Behavior**: ✅ Acceptable
- Entropy regularization is per-step penalty
- Mean over T steps is correct

---

### **Issue 6: MetaTrainer PPO Epoch Problem** 🟡 MEDIUM

**MetaTrainer.fine_tune() (line 461-462)**:
```python
for ppo_epoch in range(self.fcfg["ppo_epochs"]):
    metrics = agent.update(batch)
```

**Problem**: Runs PPO update on **same batch** multiple times

**Current Behavior**:
```
Epoch 1:
  Batch 1: collect() → update(batch, ppo_epoch=1) → update(batch, ppo_epoch=2) → update(batch, ppo_epoch=3)
  Batch 2: collect() → update(batch, ppo_epoch=1) → update(batch, ppo_epoch=2) → update(batch, ppo_epoch=3)
```

**Issue**: 
- Each ppo_epoch reuses same batch with updated network
- First epoch: ratio ≈ 1 (network unchanged)
- Second epoch: ratio may diverge (network has changed)
- Third epoch: ratio completely off (network far from data)

**Standard PPO**:
```
Epoch 1:
  Collect batch
  For ppo_epoch in 1..3:
    Shuffle batch
    For mini_batch in shuffled_batch:
      update(mini_batch)
```

**Your Implementation Missing**:
- ❌ No mini-batch shuffling
- ❌ No mini-batch splitting
- ❌ Reusing exact same batch 3 times

**Expected Effect**:
- ppo_epoch 1: ~correct
- ppo_epoch 2: gradient noise increases
- ppo_epoch 3: near-random updates

---

## Summary Table

| Issue | Component | Severity | Impact | Status |
|-------|-----------|----------|--------|--------|
| Logit→Log Prob | PPOAgent | 🔴 Critical | Training divergence | ✅ FIXED |
| Advantage Norm Consistency | GAECollector/PPOAgent | 🟡 Medium | Potential NaN in small batches | ⚠️ Needs Fix |
| Bootstrap Truncation Handling | GAECollector | 🟡 Medium | Biased returns for truncated episodes | ⚠️ Needs Fix |
| Value Off-Policy Learning | PPOAgent | 🟡 Medium | Value loss instability | ⚠️ Consider Fix |
| PPO Multi-Epoch Without Minibatch | MetaTrainer | 🟡 Medium | Divergence after epoch 2 | ⚠️ Needs Fix |
| FOMAML Advantage Norm | MetaTrainer | 🟡 Medium | Inconsistent meta-gradients | ✅ FIXED |

---

## Recommended Fixes (Priority Order)

### Priority 1: ALREADY FIXED ✅
1. Logit→Log Prob conversion (PPOAgent)
2. Advantage normalization in FOMAML (MetaTrainer)

### Priority 2: CRITICAL
3. **Move advantage normalization to GAECollector**
4. **Add mini-batch shuffling to PPO epochs** (MetaTrainer.fine_tune)

### Priority 3: RECOMMENDED
5. Fix bootstrap value handling for truncation
6. Consider on-policy value learning only

---

## Code Fixes

### Fix #3: Advantage Normalization in GAECollector

```python
# In GAECollector.collect(), after _compute_gae() call (line 208)
advantages, returns = _compute_gae(
    rewards_tensor,
    values_with_bootstrap,
    dones_tensor,
    gamma=self.gamma,
    gae_lambda=self.gae_lambda,
)

# ✅ ADD: Normalize advantages for PPO stability
advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

# Then return in batch dict
return {
    "observations": observations_tensor,
    "masks": masks_tensor,
    "actions": actions_tensor,
    "log_probs": log_probs_tensor,
    "advantages": advantages,  # ✅ Already normalized
    "returns": returns,
    "entropies": entropies_tensor,
}
```

Then **REMOVE** normalization from PPOAgent:
```python
# In PPOAgent.update(), line 211 - DELETE THIS
# advantages_normalized = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

# And change line 234-235 to use raw advantages
surr1 = ratio * advantages  # Already normalized from collector
surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * advantages
```

### Fix #4: Mini-Batch Shuffling in PPO Epochs

```python
# In MetaTrainer.fine_tune(), line 461-462
for batch_idx in range(self.fcfg["batches_per_epoch"]):
    self.env.retask(task_id)
    batch = self.collector.collect(agent, self.env)

    # ✅ ADD: Multiple gradient steps with mini-batch shuffling
    num_steps = batch["observations"].__len__()  # Get trajectory length
    indices = torch.randperm(num_steps)
    
    # Create mini-batch size (e.g., 1/4 of episode)
    mini_batch_size = max(1, num_steps // 4)
    
    for ppo_epoch in range(self.fcfg["ppo_epochs"]):
        # Shuffle trajectory
        shuffled_indices = indices[torch.randperm(len(indices))]
        
        # Process mini-batches
        for start_idx in range(0, num_steps, mini_batch_size):
            end_idx = min(start_idx + mini_batch_size, num_steps)
            mini_batch_indices = shuffled_indices[start_idx:end_idx]
            
            # Create mini-batch
            mini_batch = {
                k: v[mini_batch_indices] if isinstance(v, torch.Tensor) else [v[i] for i in mini_batch_indices]
                for k, v in batch.items()
            }
            
            metrics = agent.update(mini_batch)
            self._total_updates += 1
```

---

## Testing Strategy

```bash
# Test 1: Single task fine-tuning (test PPO)
python train.py --config experiment/train/107_PPO_N10_C/config.yaml \
  --override '{"trainer": {"phases": {"meta_learning": {"enabled": false}}}}'

# Expected: Smooth loss decrease, best ≥ final
```

```bash
# Test 2: Meta-training only (test FOMAML)
python train.py --config experiment/train/107_PPO_N10_C/config.yaml \
  --override '{"trainer": {"phases": {"fine_tuning": {"enabled": false}}}}'

# Expected: Meta-learning curves smooth
```

```bash
# Test 3: Full pipeline (both phases)
python train.py --config experiment/train/107_PPO_N10_C/config.yaml

# Expected: Meta phase stable, then fine-tune improves
```
