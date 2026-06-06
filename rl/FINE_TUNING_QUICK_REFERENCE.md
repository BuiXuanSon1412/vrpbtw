# Fine-Tuning Phase: Quick Reference Guide

**Issue**: When meta-learning is disabled, fine-tuning objective degrades 68.7% (11,955 → 20,163)

**Root Cause**: Random network initialization → poor value estimates → failed PPO

**Minimum Fix**: Enable meta-learning or load pre-trained checkpoint

---

## Quick Diagnosis

Check these metrics in `experiment/logs/metrics.jsonl`:

```python
import json
from pathlib import Path

metrics_file = Path("experiment/train/YOUR_EXP/logs/metrics.jsonl")
metrics = [json.loads(line) for line in metrics_file.read_text().split('\n') if 'eval/mean_objective' in line]

# Check 1: Is objective degrading?
objs = [m['eval/mean_objective'] for m in metrics]
print(f"Initial: {objs[0]}, Final: {objs[-1]}, Degradation: {(objs[-1]-objs[0])/objs[0]*100:.1f}%")
# GOOD: Negative degradation, BAD: Positive degradation > 10%

# Check 2: Is value loss stuck high?
value_losses = [m['tune/loss_value_mean'] for m in metrics]
print(f"Value loss trend: {value_losses[0]:.2f} → {value_losses[-1]:.2f}")
# GOOD: Decreasing below 1.0, BAD: Staying > 1.5

# Check 3: Are gradients being clipped?
grad_norms = [m['tune/grad_norm_mean'] for m in metrics]
print(f"Gradient norm: mean={sum(grad_norms)/len(grad_norms):.1f}, max={max(grad_norms):.1f}")
# GOOD: 0.5-5.0, BAD: > 50 (indicates clipping)

# Check 4: Is policy loss decreasing despite bad objective?
policy_losses = [m['tune/loss_policy_mean'] for m in metrics]
print(f"Policy loss: {policy_losses[0]:.3f} → {policy_losses[-1]:.3f}")
# RED FLAG: Decreasing policy loss + increasing objective = overfitting
```

---

## Symptoms Checklist

| Symptom | Indicates | Fix Priority |
|---------|-----------|--------------|
| ✗ Objective oscillates 19k-21k | Poor initialization | CRITICAL |
| ✗ Value loss stays > 1.5 | Poor value function | CRITICAL |
| ✗ Grad norm > 50 | Gradient explosion | HIGH |
| ✗ Policy loss ↓ while obj ↑ | Overfitting | HIGH |
| ✓ Objective monotonically ↓ | Healthy training | GOOD |
| ✓ Value loss < 1.0 | Good value function | GOOD |

---

## One-Liner Fixes

### Fix 1: Enable Meta-Learning (RECOMMENDED)
```bash
# Edit configs/trainer/meta.yaml
# Line 69: Change `enabled: false` → `enabled: true`
```

### Fix 2: Reduce Max Grad Norm
```bash
# Edit configs/trainer/meta.yaml
# Line 43: Change `max_grad_norm: 0.5` → `max_grad_norm: 1.0`
```

### Fix 3: Tighten Early Stopping
```bash
# Edit configs/trainer/meta.yaml  
# Line 98: Change `patience: 1000` → `patience: 20`
```

---

## Config Comparison

### ❌ BROKEN (Current with meta_learning disabled)
```yaml
meta_learning:
  enabled: false
fine_tuning:
  control:
    rollout_length: 512
    ppo_epochs: 1
    batch_size: 1
  early_stopping:
    patience: 1000
```
**Result**: 68.7% objective degradation, oscillatory training

### ✓ WORKING (Meta-learning enabled)
```yaml
meta_learning:
  enabled: true
  control:
    epochs: 50  # Short meta-training
fine_tuning:
  control:
    rollout_length: 512
    ppo_epochs: 1
    batch_size: 1
  early_stopping:
    patience: 20  # Tighter
```
**Expected Result**: Stable training, objective improving/stable

### ✓ AGGRESSIVE (No meta-learning but optimized)
```yaml
meta_learning:
  enabled: false
fine_tuning:
  control:
    rollout_length: 1024  # Longer rollouts
    ppo_epochs: 3  # More data reuse
    batch_size: 4  # Parallel collection
    minibatch_size: 64
  early_stopping:
    patience: 50
```
**Expected Result**: Slower but steady improvement (no guarantee of convergence)

---

## Step-by-Step Debugging

### Step 1: Verify Meta-Learning Status
```bash
# Check config
grep "enabled:" configs/trainer/meta.yaml | head -2

# Expected output:
# meta_learning: enabled: false or true
# fine_tuning: enabled: true
```

### Step 2: Check Recent Experiment Logs
```bash
tail -1 experiment/train/son_PPO_10_RC_fake/logs/metrics.jsonl | python3 -m json.tool | grep eval
```

### Step 3: Plot Objective Trajectory
```bash
python3 << 'EOF'
import json
from pathlib import Path
import matplotlib.pyplot as plt

metrics_file = Path("experiment/train/YOUR_EXP/logs/metrics.jsonl")
metrics = [json.loads(line) for line in metrics_file.read_text().split('\n') if 'eval/mean_objective' in line]

objs = [m['eval/mean_objective'] for m in metrics]
plt.plot(objs)
plt.xlabel('Iteration')
plt.ylabel('Objective (Lower is Better)')
plt.axhline(min(objs), color='g', label='Best')
plt.axhline(objs[0], color='b', label='Initial')
plt.legend()
plt.savefig('objective_trajectory.png')
plt.show()
EOF
```

### Step 4: Implement Fix
**Option A** (Recommended): Enable meta-learning
```yaml
phases:
  meta_learning:
    enabled: true  # Changed from false
```

**Option B** (If meta-learning too slow): Increase rollout & epochs
```yaml
fine_tuning:
  control:
    rollout_length: 1024  # From 512
    ppo_epochs: 3  # From 1
    batch_size: 4  # From 1 (enable parallelism)
```

### Step 5: Test Fix
```bash
# Run quick test: 10 iterations on 1 task
python3 train.py --config configs/train.yaml --experiment test_fix --n_iteration 10

# Check output
tail -20 experiment/train/test_fix/logs/metrics.jsonl
```

---

## What Not To Do

| ❌ DON'T | Why | Better Alternative |
|----------|-----|-------------------|
| Increase `ppo_epochs` to 5-10 | Makes overfitting worse | Enable meta-learning first |
| Reduce `rollout_length` to 256 | Worsens bootstrap values | Keep at 512 or increase to 1024 |
| Increase `learning_rate` to 0.001 | Amplifies instability | Fix initialization (meta-learning) |
| Add `entropy_coef: 0.1` | Prevents convergence | Only after meta-learning working |
| Set `patience: 10000` | Allows continued training on bad policy | Use early stopping patience=20 |

---

## Expected Timelines

| Approach | Time | Objective | Confidence |
|----------|------|-----------|------------|
| Meta-learning enabled (50 epochs) | 1-2 hours | 8,000-10,000 | Very High |
| Aggressive config (1024 rollout, 4 parallel) | 3-4 hours | 12,000-15,000 | Medium |
| Current config (no changes) | 2-3 hours | 20,000-22,000 | Very High (bad) |

---

## Monitoring During Training

Keep a terminal open watching metrics:

```bash
watch -n 5 "tail -1 experiment/train/YOUR_EXP/logs/metrics.jsonl | python3 -m json.tool | grep -E 'eval/mean_objective|tune/loss_'"
```

**Good signs** (first 5 iterations):
- `eval/mean_objective` decreasing
- `tune/loss_value_mean` decreasing
- `tune/grad_norm_mean` < 10

**Bad signs** (would indicate ongoing issue):
- `eval/mean_objective` increasing
- `tune/loss_value_mean` > 2.0
- `tune/grad_norm_mean` > 50

---

## If You're Still Having Issues

1. **Save this issue**: `/home/bxs/thesis/vrpbtw/rl/FINE_TUNING_EVALUATION.md`
2. **Check memory**: `[[fine-tuning-degradation-root-cause]]`
3. **Review detailed analysis**:
   - Part 3: Root cause breakdown
   - Part 5: Solutions ranked by effort/impact
   - Part 7: Component-by-component issues

---

## Key Insight

> **Fine-tuning from random initialization fails because the network cannot learn accurate value estimates in time. The value function is the signal that guides PPO updates. Without good value estimates (which require either pre-training or meta-learning), PPO diverges.**

The fix is not to tune hyperparameters (rollout_length, ppo_epochs, learning_rate) but to **initialize from a pre-trained network**.
