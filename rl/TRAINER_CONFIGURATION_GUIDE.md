# Trainer Configuration Guide

## Overview

Two training algorithms available:
1. **MetaTrainer** (MAML with curriculum) — meta-learning across multiple tasks
2. **POMOTrainer** — independent task-specific training with multiple optima

Choose based on your goal:
- **Use MetaTrainer** if you want a **generalizable policy** that works across task distributions
- **Use POMOTrainer** if you want **specialized policies** for each task

---

## MetaTrainer Configuration

### Structure Overview

```yaml
trainer:
  name: meta                    # Required: "meta"
  
  collector:                    # How to collect trajectories
    name: gae                   # Generalized Advantage Estimation
    params:
      rollout_length: 256       # Trajectory length per collection
      gamma: 0.99               # Discount factor
      gae_lambda: 0.95          # GAE exponential decay
  
  agents:                       # Policy networks
    meta_agent:                 # Shared meta-policy
    sub_agent:                  # Task-specific inner-loop policy
    tune_agent:                 # Fine-tuning policy (after meta-learning)
    eval_agent:                 # Evaluation policy
  
  evaluators:                   # Evaluation during training
    meta_eval:                  # Meta-learning phase evaluation
    tune_eval:                  # Fine-tuning phase evaluation
  
  phases:
    meta_learning:              # Phase 1: Meta-learning
    fine_tuning:                # Phase 2: Task-specific fine-tuning
```

### Phase 1: Meta-Learning Configuration

```yaml
phases:
  meta_learning:
    enabled: true               # Set to false to skip meta-learning
    
    # Curriculum Learning
    curriculum:
      entropy_threshold: 0.5    # Policy entropy target before expanding tasks
      check_interval: 1         # Check curriculum every N batches
    
    # Training Loop Control
    control:
      epochs: 200               # Number of meta-training epochs
      batches_per_epoch: 50     # FOMAML iterations per epoch
      eval_interval: 1          # Evaluate every N epochs
      checkpoint_interval: 10   # Save checkpoint every N epochs
    
    # Early Stopping
    early_stopping:
      patience: 20              # Stop if no improvement for N eval intervals
      min_delta: 0.0001         # Minimum improvement threshold
```

**What happens**:
1. Start with easiest task only
2. For each batch:
   - Collect support set (K examples per task)
   - Compute inner-loop gradients (FOMAML)
   - Compute adapted parameters
   - Collect query set with adapted params
   - Update meta_agent on query loss
3. When entropy < threshold: add next task to curriculum
4. Every eval_interval epochs: evaluate on middle-difficulty task
5. If no improvement for patience epochs: stop early

**Typical values**:
- Small tasks (N=10-20 customers): epochs=50-100, batches=20-30
- Medium tasks (N=50-100): epochs=100-200, batches=30-50
- Large tasks (N=200+): epochs=200-500, batches=50-100

---

### Phase 2: Fine-Tuning Configuration

```yaml
phases:
  fine_tuning:
    enabled: true               # Set to false for meta-learning only
    
    control:
      epochs: 50                # Fine-tuning epochs per task
      batches_per_epoch: 100    # Training batches per epoch
      ppo_epochs: 3             # Gradient steps per batch (PPO standard)
      eval_interval: 1          # Evaluate every N epochs
      checkpoint_interval: 10   # Save checkpoint every N epochs
    
    early_stopping:
      patience: 10              # Stop if no improvement for N intervals
      min_delta: 0.0001         # Minimum improvement
```

**What happens**:
1. For each task:
   - Start from meta-learned weights
   - Collect trajectory batch
   - Apply ppo_epochs gradient steps to same batch
   - Evaluate every eval_interval epochs
   - Save best checkpoint
   - Early stop if no improvement

**How ppo_epochs works**:
```
Per epoch:
  For each of 100 batches:
    Collect 1 trajectory
    For i = 1 to 3 (ppo_epochs):
      Update policy on same trajectory
      
Total updates per epoch = 100 batches × 3 ppo_epochs = 300 updates
```

**Typical values**:
- ppo_epochs: 3-5 (more = better sample efficiency, slower)
- epochs: 20-100 (shorter than meta-learning since starting from good weights)
- batches_per_epoch: 50-200 (depends on compute budget)

---

### Agent Configuration (MetaTrainer)

```yaml
agents:
  meta_agent:
    name: meta                  # Aggregates losses across tasks
    learning_rate: 0.001        # Meta-learning rate (usually lower)
    optimizer: unspecified      # Will use default (adam)
    max_grad_norm: 0.5          # Gradient clipping
  
  sub_agent:
    name: ppo                   # Task-specific inner-loop policy
    learning_rate: 0.001        # Inner-loop learning rate
    optimizer: adam
    clip_eps: 0.2               # PPO clipping parameter
    value_coef: 0.5             # Value loss weight
    entropy_coef: 0.01          # Exploration bonus weight
    max_grad_norm: 0.5
  
  tune_agent:
    name: ppo                   # Fine-tuning policy (copy of meta_agent)
    learning_rate: 0.001        # Can be higher than meta-learning
    optimizer: adam
    # ... same as sub_agent ...
  
  eval_agent:
    name: ppo                   # Used for evaluation only
    # ... same as tune_agent ...
```

**Learning rate strategy**:
- `meta_agent`: Lower (0.0001-0.001) — aggregating across tasks
- `sub_agent`: Standard (0.001-0.005) — task-specific adaptation
- `tune_agent`: Can match or exceed sub_agent (0.001-0.01) — already pre-trained

---

## POMOTrainer Configuration

### Structure Overview

```yaml
trainer:
  name: pomo                    # Required: "pomo"
  
  collector:
    name: pomo                  # Multiple starting points per instance
    params: {}                  # No parameters
  
  agents:
    train_agent:                # Task-specific training policy
    eval_agent:                 # Evaluation policy
  
  evaluators:
    train_eval:                 # Training evaluation
  
  phases:
    training:                   # Single training phase per task
```

### Phase: Training Configuration

```yaml
phases:
  training:
    control:
      epochs: 50                # Epochs per task
      batches_per_epoch: 10     # Batches per epoch
      instances_per_batch: 1    # How many problem instances per batch
      eval_interval: 1          # Evaluate every N epochs
      checkpoint_interval: 10   # Save every N epochs
    
    early_stopping:             # NEW: Added with recent fixes
      patience: 20              # Stop if no improvement for N intervals
      min_delta: 0.0001         # Minimum improvement threshold
```

**What happens**:
1. For each task (independent policy):
   - For each epoch:
     - For each batch:
       - For each instance:
         - Collect from multiple starting points (POMO)
         - Aggregate into batch
       - Update policy once (no ppo_epochs)
     - Evaluate if (epoch % eval_interval == 0)
     - Early stop if no improvement

**Key difference from MetaTrainer**:
- No ppo_epochs (update once per batch, not multiple times)
- No curriculum (trains single task, not multiple)
- Simpler loop structure

**How data collection works**:
```
POMO Collection:
  Instance 1:
    Starting point A → full episode → return_A, log_probs_A
    Starting point B → full episode → return_B, log_probs_B
    Starting point C → full episode → return_C, log_probs_C
    → Combined: (log_probs=[A,B,C], rewards=[A,B,C])
  
  Baseline per instance = mean(reward_A, B, C)
  Advantage = (return_i - baseline) for each starting point
```

---

### Agent Configuration (POMOTrainer)

```yaml
agents:
  train_agent:
    name: pomo                  # POMO per-instance baseline
    learning_rate: 0.001        # Standard learning rate
    optimizer: adam
    entropy_coef: 0.1           # Encourages exploration
    max_grad_norm: 0.5          # Gradient clipping
  
  eval_agent:
    name: pomo                  # Evaluation only
    # ... same as train_agent ...
```

**POMO-specific parameters**:
- `entropy_coef`: Higher than PPO (0.05-0.2) — POMO benefits from exploration

---

## Configuration Comparison

### MetaTrainer vs POMOTrainer

| Aspect | MetaTrainer | POMOTrainer |
|--------|-------------|------------|
| **Goal** | Generalizable policy | Task-specific policies |
| **Tasks** | Multiple (curriculum) | One at a time |
| **Inner loop** | FOMAML support/query split | Direct trajectory collection |
| **Updates/batch** | 1 (outer-loop only) | 1 (direct gradient) |
| **Curriculum** | Entropy-based expansion | None |
| **Training time** | Longer (meta-learning overhead) | Shorter (direct training) |
| **Sample efficiency** | High (MAML pre-training) | Medium (task-specific) |
| **Compute/task** | Moderate | Low |
| **Parallelization** | Tasks during meta-learning | Difficult (sequential tasks) |

---

## Common Configuration Patterns

### Quick Prototyping (MetaTrainer)

```yaml
meta_learning:
  enabled: true
  control:
    epochs: 10
    batches_per_epoch: 5
    eval_interval: 2
fine_tuning:
  enabled: true
  control:
    epochs: 5
    batches_per_epoch: 10
    ppo_epochs: 1
```
**Total time**: ~5 min on GPU

---

### Realistic Meta-Learning (MetaTrainer)

```yaml
meta_learning:
  enabled: true
  curriculum:
    entropy_threshold: 0.4
    check_interval: 2
  control:
    epochs: 100
    batches_per_epoch: 30
    eval_interval: 5
fine_tuning:
  enabled: true
  control:
    epochs: 50
    batches_per_epoch: 100
    ppo_epochs: 3
    eval_interval: 1
```
**Total time**: ~2-4 hours on GPU

---

### Fine-Tuning Only (MetaTrainer as PPO)

```yaml
meta_learning:
  enabled: false              # Skip meta-learning
fine_tuning:
  enabled: true
  control:
    epochs: 100
    batches_per_epoch: 100
    ppo_epochs: 3
```
**Equivalent to**: Standard PPO with curriculum disabled
**Total time**: ~1-2 hours on GPU

---

### Task-Specific Training (POMOTrainer)

```yaml
training:
  control:
    epochs: 50
    batches_per_epoch: 20
    instances_per_batch: 1
    eval_interval: 2
  early_stopping:
    patience: 15
    min_delta: 0.001
```
**Total time per task**: ~30-60 min on GPU
**For 5 tasks**: ~2.5-5 hours sequential

---

## Hyperparameter Tuning

### Learning Rate Strategy

```yaml
# Conservative (most stable)
meta_agent:
  learning_rate: 0.0001
sub_agent:
  learning_rate: 0.001
tune_agent:
  learning_rate: 0.001

# Aggressive (faster but riskier)
meta_agent:
  learning_rate: 0.001
sub_agent:
  learning_rate: 0.005
tune_agent:
  learning_rate: 0.01
```

### Entropy Coefficient Strategy

```yaml
# Conservative (exploitation)
sub_agent:
  entropy_coef: 0.001         # Low exploration
tune_agent:
  entropy_coef: 0.005

# Exploratory (POMO)
train_agent:
  entropy_coef: 0.1           # High exploration for multiple optima
```

### Early Stopping Strategy

```yaml
# Aggressive (stop quick on plateau)
patience: 5
min_delta: 0.01

# Conservative (wait longer)
patience: 30
min_delta: 0.0001
```

---

## Debugging Configuration Issues

### "Training diverges / loss explodes"
- Reduce learning rate by 2-10x
- Increase max_grad_norm (0.5 → 1.0)
- Reduce entropy_coef

### "Training stuck / no improvement"
- Increase learning rate 2-5x
- Increase entropy_coef (exploration)
- Reduce patience (earlier stopping)

### "Too slow / out of memory"
- Reduce rollout_length (256 → 64)
- Reduce batches_per_epoch
- Reduce instances_per_batch
- Reduce epochs

### "High variance between runs"
- Increase ppo_epochs (better amortization)
- Increase rollout_length (more samples)
- Reduce learning rate (stable updates)

---

## File Locations

- **MetaTrainer config**: `configs/trainer/meta.yaml`
- **POMOTrainer config**: `configs/trainer/pomo.yaml`
- **Default config**: `configs/train.yaml`

