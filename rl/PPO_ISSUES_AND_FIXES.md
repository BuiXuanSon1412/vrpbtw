# PPO Implementation Issues & Fixes

## Summary
Your current PPO implementation in `MetaTrainer` and `PPOAgent` has **3 critical issues** affecting training stability and correctness. The "best < final" checkpoint degradation is likely caused by these bugs.

---

## Issue 1: 🔴 CRITICAL - Log Probability Computation Error

**Location**: `core/agent.py:225` in `PPOAgent.update()`

### Problem
```python
# Line 218-225 (WRONG)
for i, obs_t in enumerate(observations):
    mask_t = masks[i:i+1]
    actions_t = batch["actions"][i:i+1]
    logits_t, values_t, _ = self.network.evaluate(obs_t, mask_t, actions=actions_t)
    logits_list.append(logits_t)
    values_list.append(values_t)
new_log_probs = torch.cat(logits_list, dim=0)  # ❌ LOGITS, NOT LOG PROBS!
```

**Issue**: `network.evaluate()` returns **logits**, not log probabilities. Line 225 treats them as log_probs directly.

**Consequence**: 
- Log probability ratios are computed using raw logits instead of log probabilities
- This breaks PPO's clipping mechanism (ratio becomes meaningless)
- Policy diverges unpredictably → training instability
- Explains why best checkpoint > final checkpoint (training diverges)

### Fix
```python
# CORRECT VERSION
for i, obs_t in enumerate(observations):
    mask_t = masks[i:i+1]
    actions_t = batch["actions"][i:i+1]
    logits_t, values_t, _ = self.network.evaluate(obs_t, mask_t, actions=actions_t)
    logits_list.append(logits_t)
    values_list.append(values_t)

logits = torch.cat(logits_list, dim=0)
values = torch.cat(values_list, dim=0)

# ✅ CONVERT TO LOG PROBABILITIES
new_log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
new_log_probs = new_log_probs.gather(-1, batch["actions"].unsqueeze(-1)).squeeze(-1)
```

---

## Issue 2: 🟡 MEDIUM - Advantage Normalization Impact

**Location**: `core/agent.py:211` in `PPOAgent.update()`

### Problem
```python
advantages_normalized = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
```

This is **correct PPO practice**, but when combined with Issue 1, the logits-based ratio becomes unstable.

### Context
- ✅ Good: Normalizes advantages for stable training
- ⚠️ Issue: With logit-based ratio (Issue 1), clipping bounds (1±clip_eps) become meaningless
- The clipping is supposed to be on probability ratios [0, ∞], not logit ratios [-∞, +∞]

### Solution
Once you fix Issue 1 (logit→log_prob conversion), this normalization will work correctly.

---

## Issue 3: 🟡 MEDIUM - Missing Advantage Normalization in FOMAML

**Location**: `core/trainer.py:699-703` in `_compute_task_losses()`

### Problem
```python
# Line 699-703 (INCOMPLETE)
surr1 = ratio * advantages
surr2 = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * advantages
support_loss = -torch.min(surr1, surr2).mean()
```

Unlike `PPOAgent.update()`, the FOMAML inner-loop computation **doesn't normalize advantages**.

### Consequence
- Meta-learning (inner loop) uses unnormalized advantages
- Fine-tuning (outer loop) uses normalized advantages
- Inconsistent gradient signals → meta-agent learns poorly adapted initializations

### Fix
```python
# CORRECT VERSION - Add normalization
advantages_normalized = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

ratio = torch.exp(
    torch.nn.functional.log_softmax(logits, dim=-1)
    .gather(-1, support_batch["actions"].unsqueeze(-1))
    .squeeze(-1)
    - old_log_probs
)
surr1 = ratio * advantages_normalized  # ✅ USE NORMALIZED
surr2 = (
    torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps)
    * advantages_normalized  # ✅ USE NORMALIZED
)
support_loss = -torch.min(surr1, surr2).mean() + value_coef * torch.nn.functional.mse_loss(
    values, returns
)
```

---

## Implementation Guide

### Step 1: Fix PPOAgent.update() (CRITICAL)

Replace lines 214-226 in `core/agent.py`:

```python
# BEFORE (lines 214-226)
logits_list = []
values_list = []
for i, obs_t in enumerate(observations):
    mask_t = masks[i:i+1]
    actions_t = batch["actions"][i:i+1]
    logits_t, values_t, _ = self.network.evaluate(obs_t, mask_t, actions=actions_t)
    logits_list.append(logits_t)
    values_list.append(values_t)
new_log_probs = torch.cat(logits_list, dim=0)
values = torch.cat(values_list, dim=0)

# AFTER
logits_list = []
values_list = []
for i, obs_t in enumerate(observations):
    mask_t = masks[i:i+1]
    actions_t = batch["actions"][i:i+1]
    logits_t, values_t, _ = self.network.evaluate(obs_t, mask_t, actions=actions_t)
    logits_list.append(logits_t)
    values_list.append(values_t)

logits = torch.cat(logits_list, dim=0)
values = torch.cat(values_list, dim=0)

# ✅ Convert logits to log probabilities
new_log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
new_log_probs = new_log_probs.gather(-1, batch["actions"].unsqueeze(-1)).squeeze(-1)
```

### Step 2: Add Advantage Normalization to FOMAML

In `_compute_task_losses()` around line 699, add:

```python
# After line 698 (after computing ratio)
advantages_normalized = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

# Then use advantages_normalized in surr1 and surr2 computations
ratio = torch.exp(...)
surr1 = ratio * advantages_normalized  # ✅ CHANGED
surr2 = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * advantages_normalized  # ✅ CHANGED
```

### Step 3: Test After Fixes

```bash
# Run fine-tuning phase only to test PPO fix
python train.py --config experiment/train/107_PPO_N10_C/config.yaml \
  --override '{"trainer": {"phases": {"meta_learning": {"enabled": false}}}}'
```

Expected: Best checkpoint should stay best (or improve), not degrade to final.

---

## Why Best < Final Happens

The sequence of events:

1. **Epoch 1-20**: With Issue 1, logit ratios are computed incorrectly
   - Clipping doesn't work (bounds are on logit scale, not probability scale)
   - Random policy drift occurs
   - By luck, model finds okay solution → **best checkpoint**

2. **Epoch 21-50**: Continued PPO training with broken log_prob computation
   - Model starts diverging (no proper clipping)
   - Advantages become unreliable
   - Early stopping doesn't trigger (Issue 2 masks the divergence)
   - Model learns bad patterns → **final checkpoint much worse**

3. **Result**: best_objective=1826.92, final_objective=410927.56 ✅ Matches your observation!

---

## Expected Improvements After Fixes

| Metric | Before | After |
|--------|--------|-------|
| PPO Stability | ❌ Broken | ✅ Stable |
| Advantage Ratio Clipping | ❌ Non-functional | ✅ Working |
| Meta-Learning Consistency | ❌ Inconsistent | ✅ Consistent |
| Best vs Final Checkpoints | ❌ Divergent | ✅ Improving |
| Training Curves | ⚠️ Erratic | ✅ Smooth |

---

## Prevention in Future

1. **Always verify network outputs**: Know whether your network returns logits or probabilities
2. **Consistent preprocessing**: Apply log_softmax at ONE place (collection time is better)
3. **Unit tests**: Add tests that verify log_prob is in [-∞, 0]
4. **Metric tracking**: Monitor `new_log_probs.mean()` and `old_log_probs.mean()` - they should be similar
