# CRITICAL BUG: MetaAgent Optimizer Configuration

## Issue Summary

🔴 **CRITICAL BUG FOUND**

MetaAgent is configured with `optimizer: unspecified`, which creates a **None optimizer**. This prevents MetaAgent from updating its policy.

---

## Evidence

### Configuration (meta.yaml Line 24)
```yaml
meta_agent:
  name: meta
  optimizer: unspecified  # ← Maps to None in registry
```

### Registry (registry.py)
```python
_OPTIMIZER_REGISTRY: Dict[str, type | None] = {
    "adam": optim.Adam,
    "sgd": optim.SGD,
    "adamw": optim.AdamW,
    "unspecified": None,  # ← This is the problem
}
```

### Builder (registry.py)
```python
opt_class = _OPTIMIZER_REGISTRY[opt_type]
optimizer = (
    None if opt_class is None else opt_class(network.parameters(), lr=opt_lr)
)
```

When `opt_type = "unspecified"`, optimizer becomes `None`.

### MetaAgent.update() (agent.py Lines 509-534)

```python
def update(self, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
    task_losses = batch["task_losses"]
    meta_loss = task_losses.mean()

    grad_norm = torch.tensor(0.0, device=meta_loss.device)
    if self.optimizer is not None:  # ← BUG: optimizer is None!
        self.optimizer.zero_grad()
        meta_loss.backward()
        grad_norm_val = torch.nn.utils.clip_grad_norm_(...)
        grad_norm = torch.as_tensor(grad_norm_val, device=meta_loss.device)
        self.optimizer.step()  # ← NEVER EXECUTED

    return {
        "loss": meta_loss,
        "grad_norm": grad_norm,
    }
```

**The problem**: When optimizer is None, the entire optimization block is skipped!

---

## What Actually Happens

### Current Flow (BROKEN)
```
trainer.meta_agent.update({"task_losses": task_losses})
    ↓
meta_loss = task_losses.mean()  # ✓ Loss computed
    ↓
if self.optimizer is not None:  # ✗ False (optimizer is None)
    # This block NEVER EXECUTES
    optimizer.zero_grad()
    meta_loss.backward()        # NEVER HAPPENS
    clip_grad_norm_(...)        # NEVER HAPPENS
    optimizer.step()            # NEVER HAPPENS
    ↓
return {"loss": meta_loss, "grad_norm": 0.0}  # ✗ No gradients computed!
```

### Consequence
- MetaAgent computes task losses ✓
- MetaAgent returns loss values ✓
- MetaAgent **does NOT update network weights** ✗
- MetaAgent **does NOT compute gradients** ✗
- Meta-learning **effectively disabled** ✗

---

## Why This Happened

Possible design intent:
```
# Maybe the intent was:
# "Let MetaTrainer handle the optimization, not MetaAgent"
```

But that's not what happens - **no one else calls optimizer.step()**!

Looking at the code:
- `MetaTrainer.meta_train()` calls `self.meta_agent.update()` (line 259)
- MetaAgent.update() should handle optimization
- Nothing else updates meta_agent

---

## Impact

### On Meta-Learning
- **BROKEN**: Meta-agent network is never updated
- **CONSEQUENCE**: No gradient flow to meta_agent
- **RESULT**: Meta-learning is effectively disabled

### On Training
- Task losses are computed correctly
- Sub-agent adapts correctly  
- Query losses computed correctly
- **But meta_agent stays at random initialization**

### What Training Actually Does
```
Iteration 1:
  Collect support set from random meta_agent weights
  Adapt sub_agent
  Evaluate query set with adapted sub_agent
  Compute query loss
  Return loss to trainer
  
MetaAgent.update():
  Compute mean loss
  Check if optimizer is not None
  Optimizer is None, so skip .backward() and .step()
  Return loss (no weights updated!)

Iteration 2:
  Collect support set from SAME random meta_agent weights (unchanged!)
  ...repeat...
```

**Result**: Meta-agent never learns. Training is a no-op.

---

## The Fix

### Option 1: Add Optimizer (RECOMMENDED)

Change config to use adam optimizer:

```yaml
meta_agent:
  name: meta
  learning_rate: 0.001
  optimizer: adam  # ← Changed from "unspecified"
```

**Why this is correct**:
- Meta-agent needs to update weights
- Adam is standard for meta-learning
- Matches sub_agent optimizer

### Option 2: Remove Optimizer Condition (WRONG)

Don't do this:
```python
# WRONG - will crash if optimizer is None
meta_loss.backward()
clip_grad_norm_(...)
optimizer.step()  # ← AttributeError if optimizer is None
```

---

## Correct Implementation

```python
def update(self, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
    """Update network on meta-loss (average of task losses)."""
    task_losses = batch["task_losses"]
    meta_loss = task_losses.mean()

    grad_norm = torch.tensor(0.0, device=meta_loss.device)
    
    # Meta-agent MUST have an optimizer
    if self.optimizer is None:
        raise ValueError(
            "MetaAgent requires an optimizer. "
            "Set optimizer: adam (not 'unspecified') in config."
        )
    
    self.optimizer.zero_grad()
    meta_loss.backward()
    grad_norm_val = torch.nn.utils.clip_grad_norm_(
        self.network.parameters(), self.max_grad_norm
    )
    grad_norm = torch.as_tensor(grad_norm_val, device=meta_loss.device)
    self.optimizer.step()

    return {
        "loss": meta_loss,
        "grad_norm": grad_norm,
    }
```

---

## Testing the Fix

Before and after:

### Before (BROKEN)
```python
# Line 24 in meta.yaml
optimizer: unspecified

# Result:
# - meta_agent weights never updated
# - Training does nothing
# - Loss will be random/constant
```

### After (FIXED)
```python
# Line 24 in meta.yaml
optimizer: adam

# Result:
# - meta_agent weights updated each iteration
# - Gradients properly computed and clipped
# - Meta-learning works correctly
```

---

## Why This Bug Existed

Possible reasons:

1. **Incomplete initial design**: "unspecified" was added as a placeholder
2. **Confusion with design intent**: Maybe thought trainer would handle optimization
3. **Testing artifact**: Created for testing without optimization
4. **Incomplete refactoring**: Changed update() but forgot to update config

Whatever the reason, **this must be fixed before any training**.

---

## Verification

Run this to confirm the bug:

```python
import torch
from core.registry import build_agents
from config import load_config

cfg = load_config('configs/trainer/meta.yaml')
agents = build_agents(cfg)
meta_agent = agents['meta_agent']

# Check optimizer
print(f"Meta-agent optimizer: {meta_agent.optimizer}")
# Output: None ← THIS IS THE BUG

# Try to update
batch = {"task_losses": torch.tensor([1.0, 2.0, 3.0])}
result = meta_agent.update(batch)
print(f"Grad norm: {result['grad_norm']}")
# Output: tensor(0.) ← No gradients because no .backward()
```

---

## Final Verdict

🔴 **CRITICAL BUG - MUST FIX**

**Impact**: Meta-learning completely broken
**Fix**: Change `optimizer: unspecified` to `optimizer: adam`
**Priority**: HIGHEST - Blocks all meta-learning

Without this fix, MetaAgent never updates and meta-learning is non-functional.

