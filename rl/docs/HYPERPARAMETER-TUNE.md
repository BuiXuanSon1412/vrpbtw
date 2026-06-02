# MetaTrainer Hyperparameter Tuning Guide

**Configuration: Fine-Tuning Only (Meta-Learning Disabled)**

This guide provides comprehensive instructions for tuning MetaTrainer hyperparameters when training with fine-tuning enabled and meta-learning disabled. In this configuration, the trainer acts as a **task-specific PPO learner** without FOMAML meta-adaptation.

---

## Table of Contents

1. [Configuration Overview](#configuration-overview)
2. [Fine-Tuning Control Parameters](#fine-tuning-control-parameters)
3. [Tune Agent Parameters](#tune-agent-parameters)
4. [Hyperparameter Tuning Strategy](#hyperparameter-tuning-strategy)
5. [Common Issues & Solutions](#common-issues--solutions)
6. [Example Configurations](#example-configurations)
7. [Monitoring & Evaluation](#monitoring--evaluation)

---

## Configuration Overview

### YAML Structure

```yaml
trainer:
  phases:
    meta_learning:
      enabled: false            # Disable FOMAML meta-learning

    fine_tuning:
      enabled: true             # Enable PPO fine-tuning
      control:
        # Rollout collection parameters
        batch_size: 1            # Parallel environments
        n_iteration: 100         # Iterations per task
        rollout_length: 256      # Timesteps per iteration

        # PPO optimization parameters
        ppo_epochs: 1            # Gradient passes per rollout
        minibatch_size: 32       # Mini-batch size

        # Evaluation & checkpointing
        eval_interval: 1         # Evaluate every N iterations
        checkpoint_interval: 10

      early_stopping:
        patience: 1000
        min_delta: 0.0001

agent:
  tune_agent:
    name: ppo
    learning_rate: 0.0001
    optimizer: adam
    clip_eps: 0.2
    value_coef: 0.5
    entropy_coef: 0.0
    max_grad_norm: 0.5
```

---

## Fine-Tuning Control Parameters

### Rollout Collection Parameters

#### `batch_size` (Parallel Environments)

**What it does:** Number of environments running in parallel during rollout collection.

**Range:** `1` (serial) to `16+` (parallel)

**Details:**
- `batch_size=1`: Single environment, sequential rollouts (memory efficient, slow)
  - Collects `rollout_length` timesteps per iteration
  - Simple but limited parallelism
  - Good for memory-constrained systems

- `batch_size>1`: Multiple environments in parallel (uses SubprocVecEnv)
  - Collects `rollout_length × batch_size` timesteps total
  - Fast rollout collection (~1.2-1.5× speedup per extra worker)
  - Higher memory usage
  - Better gradient estimates from diverse trajectories

**Recommended:**
```yaml
batch_size: 1   # For single-GPU, standard setups
batch_size: 4   # For multi-GPU or high-memory systems
batch_size: 8   # For large parallel clusters
```

**Impact on other parameters:**
- Larger batch_size → can use larger minibatch_size
- Larger batch_size → better gradient estimates, can reduce ppo_epochs

**Tuning:**
```bash
# Monitor timesteps per iteration
timesteps_per_iteration = rollout_length × batch_size
# Example: 256 × 4 = 1024 timesteps per iteration
```

---

#### `n_iteration` (Iterations Per Task)

**What it does:** Number of rollout collections and PPO optimization cycles per task.

**Range:** `10` to `500+`

**Details:**
- Total timesteps per task = `n_iteration × rollout_length × batch_size`
- Total gradient updates per task = `n_iteration × ppo_epochs × ceil(timesteps_per_iteration / minibatch_size)`

**Recommended:**
```yaml
n_iteration: 50    # Fast experimentation (13K timesteps with default params)
n_iteration: 100   # Standard training (25.6K timesteps)
n_iteration: 200   # Long training for hard tasks (51.2K timesteps)
```

**Impact:**
- Too low (< 20): Insufficient training, policy doesn't converge
- Too high (> 300): Diminishing returns, overfitting risk

**Tuning Tips:**
```bash
# Check entropy convergence
# Good: entropy decreases from 0.8 → 0.2 within first 50 iterations
# Bad: entropy flat or increasing → increase entropy_coef

# Check objective improvement
# Good: objective improves by 30%+ in first 50 iterations
# Bad: objective flat → increase learning_rate or ppo_epochs
```

---

#### `rollout_length` (Timesteps Per Rollout)

**What it does:** Number of environment steps to collect before PPO optimization.

**Range:** `32` to `1024`

**Details:**
- Larger rollout → better advantage estimation, more stable learning
- Smaller rollout → faster iteration, more frequent updates
- Interacts with GAE lambda/gamma for advantage computation

**Recommended:**
```yaml
rollout_length: 64    # Quick updates, less stable
rollout_length: 256   # Standard (good balance)
rollout_length: 512   # Longer horizons, more stable but slower
```

**Relationship to other parameters:**
```
Total timesteps per iteration = rollout_length × batch_size
Number of minibatches per epoch = ceil(total_timesteps / minibatch_size)
```

**Tuning:**
```bash
# For VRP problems (routing, scheduling):
rollout_length: 256   # Captures most of episode (typical episode ~200-500 steps)

# For large/complex instances:
rollout_length: 512   # Full episodes, better trajectory data

# For fast iteration:
rollout_length: 64    # Quick feedback, but risk of high variance
```

---

### PPO Optimization Parameters

#### `ppo_epochs` (Gradient Passes Per Rollout)

**What it does:** Number of times to pass over collected data with mini-batch SGD.

**Range:** `1` to `10`

**Details:**
- Each epoch: shuffle data, iterate through mini-batches
- Increases sample reuse but risk of overfitting
- Critical tuning parameter

**Recommended:**
```yaml
ppo_epochs: 1   # Single pass (default, safest)
ppo_epochs: 2   # Two passes (more optimization)
ppo_epochs: 3   # Three passes (higher variance, overfitting risk)
```

**⚠️ Known Issue:** From memory, `ppo_epochs: 3` causes **fine-tuning degradation** (objective worsens from 3068→4556). Recommend **ppo_epochs: 1** for this codebase.

**Tuning Decision Tree:**

```
Does objective improve steadily?
├─ YES: Keep ppo_epochs=1
├─ NO: Check entropy
│  ├─ Entropy high: Increase entropy_coef first, not ppo_epochs
│  └─ Entropy converged: Check policy_loss
│     ├─ Policy loss increasing: ppo_epochs too high, reduce to 1
│     └─ Policy loss stable: Increase learning_rate instead
```

---

#### `minibatch_size` (SGD Batch Size)

**What it does:** Number of timesteps in each mini-batch during SGD optimization.

**Range:** `16` to `256`

**Details:**
- Larger minibatch → smoother gradients, less noisy updates
- Smaller minibatch → faster updates, higher variance, better exploration
- Number of mini-batches per epoch = `ceil(timesteps_per_iteration / minibatch_size)`

**Recommended:**
```yaml
minibatch_size: 32    # Standard (8 minibatches for 256 timesteps)
minibatch_size: 64    # Smoother gradients, needs higher LR
minibatch_size: 128   # Very smooth, risk of stagnation
```

**Relationship:**
```
timesteps_per_iteration = rollout_length × batch_size = 256 × 1 = 256
minibatches_per_epoch = ceil(256 / 32) = 8
ppo_updates_per_task = n_iteration × ppo_epochs × minibatches_per_epoch
                     = 100 × 1 × 8 = 800
```

**Tuning:**
```bash
# Rule of thumb: minibatch_size should be 1/8 of total timesteps
# For 256 timesteps: minibatch_size = 32
# For 512 timesteps: minibatch_size = 64

# If training unstable: increase minibatch_size
# If training slow: decrease minibatch_size (but not below 16)
```

---

### Evaluation & Checkpointing

#### `eval_interval` (Evaluation Frequency)

**What it does:** Evaluate policy every N iterations (determines when to save checkpoints and track early stopping).

**Range:** `1` to `10`

**Recommended:**
```yaml
eval_interval: 1    # Evaluate every iteration (standard, best monitoring)
eval_interval: 5    # Evaluate every 5 iterations (for large n_iteration)
eval_interval: 10   # Sparse evaluation (only for very long training)
```

**Impact:**
- `eval_interval: 1` → More checkpoints, better tracking, slightly slower
- `eval_interval: 5` → Fewer evaluations, faster training, less monitoring

**Typical usage:**
```yaml
n_iteration: 100
eval_interval: 1    # Evaluates 100 times
# vs
n_iteration: 500
eval_interval: 5    # Evaluates 100 times (same number, less frequent)
```

---

#### `checkpoint_interval` (Checkpoint Frequency)

**What it does:** Save model checkpoint every N iterations (independent of eval_interval).

**Recommended:**
```yaml
checkpoint_interval: 10     # Save every 10 iterations
checkpoint_interval: 50     # Save every 50 (sparse checkpoints)
```

**Impact:**
- Checkpoints use disk space: ~7-8MB per checkpoint
- For `n_iteration: 100, checkpoint_interval: 10` → 10 checkpoints per task

---

### Early Stopping Parameters

#### `patience` (Early Stopping Patience)

**What it does:** Stop training if objective doesn't improve for N consecutive evaluations.

**Recommended:**
```yaml
patience: 10        # Stop after 10 evals without improvement (strict)
patience: 50        # Stop after 50 evals (lenient)
patience: 1000      # Effectively disabled (train full n_iteration)
```

**Typical settings:**
```yaml
# For quick experimentation
eval_interval: 1
patience: 10        # Stop after 10 iterations without improvement

# For careful training
eval_interval: 5
patience: 20        # Stop after 100 iterations (5 evals × 20 patience)
```

---

#### `min_delta` (Improvement Threshold)

**What it does:** Minimum improvement threshold for considering an evaluation a success.

**Range:** `0.0001` to `1.0`

**Details:**
- If `new_objective < best_objective - min_delta`, counts as improvement
- Prevents patience counter from incrementing on tiny improvements

**Recommended:**
```yaml
min_delta: 0.0001   # Any improvement counts (loose)
min_delta: 1.0      # Need >1 point improvement (strict)
```

**Typical:**
```yaml
min_delta: 0.0001   # Default, works well for most problems
```

---

## Tune Agent Parameters

### Learning Rate

**What it does:** Step size for gradient updates (Adam optimizer).

**Range:** `0.00001` to `0.01`

**Recommended:**
```yaml
learning_rate: 0.0001   # Standard PPO learning rate
learning_rate: 0.0003   # More aggressive (if converging slowly)
learning_rate: 0.00003  # More conservative (if unstable)
```

**Tuning Decision:**

```
Is training unstable? (loss oscillating, entropy jumping)
├─ YES, reduce learning_rate: 0.0001 → 0.00005
├─ NO, check convergence speed
│  ├─ Slow (objective flat after 20 iterations): increase learning_rate
│  └─ Fast (objective improves steadily): keep it
```

**Common values for this codebase:**
```yaml
learning_rate: 0.0001     # Default (works well)
learning_rate: 0.0005     # For aggressive training
learning_rate: 0.00005    # For stable training on hard tasks
```

---

### Optimizer

**What it does:** Algorithm for gradient updates.

**Options:**
```yaml
optimizer: adam        # Adaptive (recommended)
optimizer: sgd         # Stochastic gradient descent (not recommended)
```

**Recommended:** Always use `adam` for PPO.

**Why:**
- Adam adapts per-parameter learning rates
- Better convergence than SGD for PPO
- Built-in momentum and RMSprop

---

### Clipping Epsilon (`clip_eps`)

**What it does:** PPO policy gradient clipping range (prevents large policy shifts).

**Range:** `0.1` to `0.5`

**Recommended:**
```yaml
clip_eps: 0.2       # Standard PPO (safest)
clip_eps: 0.1       # Conservative (smaller steps)
clip_eps: 0.3       # Aggressive (larger steps)
```

**Details:**
```
PPO clipping:
ratio = exp(log_prob_new - log_prob_old)
clipped_ratio = clamp(ratio, 1-clip_eps, 1+clip_eps)
policy_loss = -min(ratio × advantage, clipped_ratio × advantage)
```

**Tuning:**
```yaml
clip_eps: 0.2       # Default, good balance
clip_eps: 0.1       # If policy diverges (large jumps in objective)
clip_eps: 0.3       # If learning is slow (need larger updates)
```

---

### Value Coefficient (`value_coef`)

**What it does:** Weight for value function loss in total loss.

**Range:** `0.1` to `1.0`

**Formula:**
```
total_loss = policy_loss + value_coef × value_loss + entropy_coef × entropy_loss
```

**Recommended:**
```yaml
value_coef: 0.5     # Standard (balance policy and value learning)
value_coef: 0.1     # Emphasis on policy
value_coef: 1.0     # Emphasis on value function
```

**Tuning:**
```yaml
value_coef: 0.5     # Default, generally works well

# If advantage estimates are poor (high variance):
value_coef: 1.0     # Learn value function more carefully

# If value function is already good:
value_coef: 0.1     # Focus on policy improvement
```

---

### Entropy Coefficient (`entropy_coef`)

**What it does:** Regularization strength for policy entropy (exploration).

**Range:** `0.0` to `0.1`

**Formula:**
```
total_loss = policy_loss + value_coef × value_loss + entropy_coef × entropy_loss
entropy_loss = -mean(entropies)  # Negative, so entropy_coef encourages low entropy
```

**Recommended:**
```yaml
entropy_coef: 0.0       # No entropy regularization (pure exploitation)
entropy_coef: 0.01      # Standard (default)
entropy_coef: 0.05      # Strong entropy regularization (more exploration)
entropy_coef: 0.1       # Very strong (forces high entropy)
```

**⚠️ Critical: This is crucial for entropy convergence (see monitoring guide)**

**Tuning Decision Tree:**

```
Check entropy trajectory during training:
├─ Entropy decreases smoothly (0.8 → 0.2): entropy_coef=0.01 is GOOD
├─ Entropy stays high (> 0.7 at end): entropy_coef too LOW
│  └─ SOLUTION: Increase to 0.05 or 0.1
├─ Entropy crashes to 0 early: entropy_coef too HIGH
│  └─ SOLUTION: Decrease to 0.0 or 0.005
└─ Entropy oscillates wildly: Learning rate too high
   └─ SOLUTION: Reduce learning_rate first, then adjust entropy_coef
```

**Example configurations:**

```yaml
# For exploration-heavy tasks:
entropy_coef: 0.05

# For exploitation-heavy tasks (optimal routes well-defined):
entropy_coef: 0.0

# For balanced learning:
entropy_coef: 0.01
```

---

### Max Gradient Norm (`max_grad_norm`)

**What it does:** Gradient clipping to prevent exploding gradients.

**Range:** `0.1` to `1.0`

**Recommended:**
```yaml
max_grad_norm: 0.5      # Standard (safest)
max_grad_norm: 1.0      # Less aggressive clipping
max_grad_norm: 0.1      # Very conservative
```

**Details:**
- Clips gradient norm to this value before optimizer step
- Prevents training instability from large gradients
- Generally set once and rarely needs tuning

**When to change:**
```yaml
max_grad_norm: 1.0      # If gradients are naturally small
max_grad_norm: 0.5      # Default (good for most)
max_grad_norm: 0.25     # If training is unstable/oscillates
```

---

## Hyperparameter Tuning Strategy

### Phase 1: Establish Baseline (1-2 runs)

Start with these safe defaults:

```yaml
trainer:
  phases:
    fine_tuning:
      control:
        batch_size: 1
        n_iteration: 100
        rollout_length: 256
        ppo_epochs: 1
        minibatch_size: 32
        eval_interval: 1
        checkpoint_interval: 10
      early_stopping:
        patience: 1000
        min_delta: 0.0001

agent:
  tune_agent:
    learning_rate: 0.0001
    optimizer: adam
    clip_eps: 0.2
    value_coef: 0.5
    entropy_coef: 0.01
    max_grad_norm: 0.5
```

**Monitor:**
- Does objective improve?
- Does entropy decrease?
- Are there any errors?

---

### Phase 2: Identify Bottleneck (Check logs)

Run `plot_convergence.py` to visualize:

```bash
python scripts/plot_convergence.py --exp-ids 100 --dpi 200
```

**Analyze 5 plots:**

1. **Loss Convergence**
   - Should decrease over time
   - If flat: increase learning_rate or ppo_epochs

2. **Objective Convergence**
   - Should improve (decrease for minimization)
   - If flat: increase learning_rate

3. **Gradient Norm Stability**
   - Should stay < 1.0
   - If oscillating: reduce learning_rate

4. **Service Rate** (if applicable)
   - Should increase toward 1.0
   - Check task feasibility

5. **Entropy Convergence** (NEW)
   - Should decrease from ~0.8 to ~0.2
   - If high: increase entropy_coef
   - If crashes to 0: decrease entropy_coef

---

### Phase 3: Targeted Tuning

**If objective doesn't improve:**
```yaml
# Try increasing learning rate
learning_rate: 0.0001 → 0.0003

# Or increase PPO optimization
ppo_epochs: 1 → 2  (⚠️ But watch for overfitting)

# Or longer rollouts
rollout_length: 256 → 512
```

**If training is unstable:**
```yaml
# Reduce learning rate
learning_rate: 0.0001 → 0.00005

# Or increase clipping
clip_eps: 0.2 → 0.1

# Or increase minibatch size
minibatch_size: 32 → 64
```

**If entropy doesn't converge:**
```yaml
# Entropy too high
entropy_coef: 0.01 → 0.05

# Entropy too low (crashes to 0)
entropy_coef: 0.01 → 0.0
```

---

### Phase 4: Fine-Tuning (Optimize for your task)

Once baseline works, optimize:

```yaml
# For SPEED (get results quickly)
n_iteration: 100 → 50
eval_interval: 1 → 5
checkpoint_interval: 10 → 20

# For QUALITY (best possible objective)
n_iteration: 100 → 200
rollout_length: 256 → 512
ppo_epochs: 1 → 2  (if stable)

# For STABILITY (reliable training)
minibatch_size: 32 → 64
clip_eps: 0.2 → 0.1
learning_rate: 0.0001 → 0.00005
```

---

## Common Issues & Solutions

### Issue 1: Objective Doesn't Improve

**Symptoms:**
- Objective stays constant or increases
- Policy not learning

**Root Causes & Solutions:**

| Cause | Diagnosis | Solution |
|-------|-----------|----------|
| Learning rate too low | Loss decreases but objective flat | Increase `learning_rate: 0.0001 → 0.0003` |
| Insufficient optimization | Single epoch might not be enough | Try `ppo_epochs: 1 → 2` (watch for overfitting) |
| Poor advantage estimates | High advantage variance | Increase `rollout_length: 256 → 512` |
| Bad network initialization | Random policy is bad | Check network architecture in agent config |
| Environment issue | Task infeasible or reward broken | Verify task config and reward function |

---

### Issue 2: Training Unstable (Oscillating Loss/Objective)

**Symptoms:**
- Loss/objective jumps up and down
- Gradient norm spikes
- Performance randomly good/bad

**Root Causes & Solutions:**

| Cause | Diagnosis | Solution |
|-------|-----------|----------|
| Learning rate too high | Large loss jumps | Reduce `learning_rate: 0.0001 → 0.00005` |
| Minibatch too small | Noisy gradients | Increase `minibatch_size: 32 → 64` |
| Clipping too loose | Large policy changes | Tighten `clip_eps: 0.2 → 0.1` |
| Entropy coefficient wrong | Wild policy changes | Adjust `entropy_coef` (see entropy monitoring) |
| Value function unstable | Volatile advantage estimates | Increase `value_coef: 0.5 → 1.0` |

---

### Issue 3: Entropy Doesn't Converge

**Symptoms:**
- Entropy stays high (> 0.5) throughout training
- OR entropy crashes to 0 immediately
- Curriculum doesn't expand (if using meta-learning)

**Solutions:**

**If entropy stays high:**
```yaml
entropy_coef: 0.01 → 0.05  # Increase regularization
ppo_epochs: 1 → 2          # More optimization
n_iteration: 100 → 200     # Longer training
```

**If entropy crashes:**
```yaml
entropy_coef: 0.01 → 0.0   # Disable entropy regularization
# OR reduce stronger than needed
learning_rate: 0.0001 → 0.0003  # Learn faster, exit collapse
```

---

### Issue 4: PPO Epochs Causes Degradation

**Symptoms:**
- With `ppo_epochs: 1`: objective = 3068
- With `ppo_epochs: 3`: objective = 4556 (worse)

**Solution:**
```yaml
ppo_epochs: 1  # Use single pass only (confirmed safe in this codebase)
```

**Why:** Multiple epochs can cause overfitting to limited rollout data. Use longer rollouts instead if needed:
```yaml
ppo_epochs: 1
rollout_length: 256 → 512  # Collect more diverse data instead
```

---

## Example Configurations

### Fast Experimentation

**Goal:** Test ideas quickly, sacrifice some quality

```yaml
trainer:
  phases:
    fine_tuning:
      control:
        batch_size: 1
        n_iteration: 50         # ← Shorter
        rollout_length: 128     # ← Shorter
        ppo_epochs: 1
        minibatch_size: 32
        eval_interval: 5        # ← Less frequent
        checkpoint_interval: 25 # ← Less frequent
      early_stopping:
        patience: 100
        min_delta: 0.0001

agent:
  tune_agent:
    learning_rate: 0.0003       # ← More aggressive
    optimizer: adam
    clip_eps: 0.2
    value_coef: 0.5
    entropy_coef: 0.01
    max_grad_norm: 0.5
```

**Expected:** ~3-5 minutes per task, reasonable performance

---

### Standard Training

**Goal:** Good balance of speed and quality

```yaml
trainer:
  phases:
    fine_tuning:
      control:
        batch_size: 1           # or 2-4 if you have parallelism
        n_iteration: 100        # Standard
        rollout_length: 256     # Standard
        ppo_epochs: 1           # Single pass (safe)
        minibatch_size: 32      # Standard
        eval_interval: 1        # Full monitoring
        checkpoint_interval: 10
      early_stopping:
        patience: 1000
        min_delta: 0.0001

agent:
  tune_agent:
    learning_rate: 0.0001       # Standard
    optimizer: adam
    clip_eps: 0.2
    value_coef: 0.5
    entropy_coef: 0.01          # Monitor entropy!
    max_grad_norm: 0.5
```

**Expected:** ~10-15 minutes per task, good performance

---

### Quality Training (For Final Results)

**Goal:** Best possible objective, even if slow

```yaml
trainer:
  phases:
    fine_tuning:
      control:
        batch_size: 4           # Parallel collection
        n_iteration: 200        # ← Longer
        rollout_length: 512     # ← Longer
        ppo_epochs: 1           # Still single (safe)
        minibatch_size: 64      # ← Larger
        eval_interval: 1
        checkpoint_interval: 20
      early_stopping:
        patience: 2000          # ← Very lenient
        min_delta: 0.0001

agent:
  tune_agent:
    learning_rate: 0.00005      # ← Conservative
    optimizer: adam
    clip_eps: 0.1               # ← Tight
    value_coef: 1.0             # ← Emphasis on value
    entropy_coef: 0.01
    max_grad_norm: 0.5
```

**Expected:** ~30-45 minutes per task, excellent performance

---

### Stability-Focused (For Difficult Tasks)

**Goal:** Reliable convergence, handle hard problems

```yaml
trainer:
  phases:
    fine_tuning:
      control:
        batch_size: 2           # Some parallelism for robustness
        n_iteration: 150
        rollout_length: 512     # Longer trajectories
        ppo_epochs: 1
        minibatch_size: 64      # Smooth gradients
        eval_interval: 2
        checkpoint_interval: 15
      early_stopping:
        patience: 500
        min_delta: 0.001        # ← Require meaningful improvement

agent:
  tune_agent:
    learning_rate: 0.00005      # Conservative
    optimizer: adam
    clip_eps: 0.1               # Tight clipping
    value_coef: 1.0             # Strong value learning
    entropy_coef: 0.05          # Encourage exploration
    max_grad_norm: 0.3          # Tight gradient clipping
```

**Expected:** Stable convergence, handles edge cases

---

## Monitoring & Evaluation

### Key Metrics to Monitor

**1. Training Loss (`tune/loss_total_mean`)**
- Should decrease monotonically or in steps
- If increasing: learning rate too high or data quality issues

**2. Policy Loss (`tune/loss_policy_mean`)**
- Should decrease as policy improves
- If increasing: policy diverging, reduce learning_rate

**3. Value Loss (`tune/loss_value_mean`)**
- Should decrease as value estimates improve
- Can be noisy, watch the trend

**4. Entropy Loss (`tune/loss_entropy_mean`)**
- Should decrease from ~-0.8 to ~-0.2
- See entropy monitoring guide for tuning

**5. Gradient Norm (`tune/grad_norm_mean`)**
- Should stay < 1.0 (clipped)
- If consistently > 0.9: gradients large, may need more clipping

**6. Evaluation Objective (`eval/mean_objective`)**
- Should improve (decrease for minimization)
- Check improvement is significant, not just noise

---

### Logging & Visualization

**Extract key metrics:**
```bash
# View entropy convergence
grep "tune/loss_entropy_mean" logs/metrics.jsonl | tail -20

# View objective improvement
grep "eval/mean_objective" logs/metrics.jsonl

# Plot all metrics
python scripts/plot_convergence.py --exp-ids 100 --dpi 200
```

---

### Checklist Before Final Training

- [ ] Baseline runs successfully without errors
- [ ] Objective improves steadily in first 20% of iterations
- [ ] Entropy decreases from high (~0.8) to low (~0.2)
- [ ] Loss converges (doesn't oscillate wildly)
- [ ] Gradient norms stay below 1.0
- [ ] No out-of-memory errors
- [ ] Early stopping patience is reasonable (won't stop too early)
- [ ] Checkpoints are being saved regularly

---

## Parameter Interaction Summary

| If You Want | Adjust | Range | Caution |
|------------|--------|-------|---------|
| **Faster learning** | `learning_rate` | 0.0001 → 0.001 | Risk instability |
| **Stable learning** | `minibatch_size` | 32 → 64 | Slower updates |
| **Better estimation** | `rollout_length` | 256 → 512 | Slower collection |
| **More optimization** | `ppo_epochs` | 1 → 2 | Risk overfitting |
| **More exploration** | `entropy_coef` | 0.0 → 0.05 | Less exploitation |
| **Fewer oscillations** | `clip_eps` | 0.2 → 0.1 | Slower updates |

---

## References

- PPO paper: Schulman et al., "Proximal Policy Optimization Algorithms"
- GAE paper: Schulman et al., "High-Dimensional Continuous Control Using Generalized Advantage Estimation"
- See `ENTROPY-MONITORING.md` for detailed entropy convergence guide
- See `ENVIRONMENT-CONFIGURATION.md` for task/environment setup
