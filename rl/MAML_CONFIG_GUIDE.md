# MetaTrainer (MAML) Configuration Guide

## Overview

MetaTrainer has **two phases**:
1. **Meta-Learning**: Train a meta-policy on task distribution
2. **Fine-Tuning**: Adapt the meta-policy to each task independently

Each phase has different computational characteristics and configuration strategies.

---

## Memory Requirements Calculation

### Meta-Learning Phase
```
GPU Memory ≈ num_active_tasks × trajectory_overhead × batches_per_epoch (accumulated)

Where:
  - num_active_tasks: Number of tasks in curriculum (starts small, grows)
  - trajectory_overhead: ~50-100KB per trajectory
  - batches_per_epoch: Number of gradient updates
```

**Key difference from POMO:**
- No `instances_per_batch` parameter
- Each batch computes loss across **all active tasks simultaneously**
- Memory grows as curriculum expands (more tasks added)

### Fine-Tuning Phase
```
GPU Memory ≈ batches_per_epoch × ppo_epochs × trajectory_overhead

Where:
  - batches_per_epoch: Trajectories per epoch
  - ppo_epochs: Number of PPO optimization steps per batch
  - Done SEQUENTIALLY for each task (not in parallel)
```

**Key difference:**
- Single task at a time (sequential, not parallel)
- PPO optimization multiplier (ppo_epochs × batches)
- Generally less memory intensive than POMO with large instances_per_batch

---

## Phase-Specific Considerations

### Meta-Learning Phase

**Computational Cost Per Epoch:**
```
Cost = num_active_tasks × batches_per_epoch × K_inner × 2
       (inner loop + outer loop gradient computation)
```

**Typical Growth:**
```
Epoch 1:   1-2 active tasks
Epoch 20:  3-4 active tasks (curriculum expansion)
Epoch 100: 5-10 active tasks (full curriculum)
```

**Memory Scaling:**
```
Early epochs: Lower memory (few tasks)
Late epochs:  Higher memory (many tasks)
```

### Fine-Tuning Phase

**Computational Cost Per Task Per Epoch:**
```
Cost = batches_per_epoch × ppo_epochs
```

**Total Cost Per Phase:**
```
Total = num_tasks × epochs × batches_per_epoch × ppo_epochs
```

**Example:**
- 10 tasks × 50 epochs × 100 batches × 10 ppo_epochs = 500,000 gradient updates

---

## Common GPU Configurations

### High-End GPUs (Research)

#### **NVIDIA A100 (40GB HBM2e)**

```yaml
# Maximum Configuration (Paper-grade)
phases:
  meta_learning:
    enabled: true
    control:
      epochs: 200
      batches_per_epoch: 50
      eval_interval: 1
  fine_tuning:
    enabled: true
    control:
      epochs: 100
      batches_per_epoch: 100
      ppo_epochs: 10
      eval_interval: 1

# Breakdown:
#   Meta-Learning:
#     - Epochs: 200 with growing task pool (1→10 tasks)
#     - Batches/epoch: 50
#     - Avg GPU Memory: 4-8GB (grows over time)
#     - Time per epoch: 2-5 minutes (faster early, slower late)
#     - Total time: 8-17 hours
#     - Total updates: ~200 * 50 * 2 = 20,000
#
#   Fine-Tuning (per task):
#     - Epochs: 100
#     - Batches/epoch: 100
#     - PPO epochs: 10
#     - GPU Memory: ~2-3GB (constant)
#     - Time per task: 1-2 hours
#     - Time all tasks (10): 10-20 hours
#     - Total updates: 10 * 100 * 100 * 10 = 1,000,000
#
#   Total Training Time: ~25-40 hours
#   Best for: Full MAML research, curriculum learning studies
```

```yaml
# Balanced Configuration (Recommended)
phases:
  meta_learning:
    enabled: true
    control:
      epochs: 100
      batches_per_epoch: 30
      eval_interval: 2
  fine_tuning:
    enabled: true
    control:
      epochs: 50
      batches_per_epoch: 50
      ppo_epochs: 5
      eval_interval: 2

# Breakdown:
#   Meta-Learning:
#     - Total time: ~4-7 hours
#     - Total updates: ~6,000
#     - Avg GPU Memory: 2-4GB
#
#   Fine-Tuning:
#     - Time all tasks: ~5-8 hours
#     - Total updates: 250,000
#
#   Total Training Time: ~10-15 hours
#   Best for: Good meta-learning + fast iteration
```

---

#### **NVIDIA V100 (32GB HBM2)**

```yaml
# Recommended Configuration
phases:
  meta_learning:
    enabled: true
    control:
      epochs: 80
      batches_per_epoch: 30
      eval_interval: 2
  fine_tuning:
    enabled: true
    control:
      epochs: 40
      batches_per_epoch: 40
      ppo_epochs: 4
      eval_interval: 2

# Breakdown:
#   Meta-Learning: ~4-6 hours
#   Fine-Tuning: ~3-5 hours
#   Total: ~8-12 hours
```

```yaml
# Conservative Configuration
phases:
  meta_learning:
    enabled: true
    control:
      epochs: 50
      batches_per_epoch: 20
      eval_interval: 2
  fine_tuning:
    enabled: true
    control:
      epochs: 30
      batches_per_epoch: 30
      ppo_epochs: 3
      eval_interval: 2

# Breakdown:
#   Meta-Learning: ~2-3 hours
#   Fine-Tuning: ~1.5-2.5 hours
#   Total: ~4-6 hours
```

---

#### **NVIDIA RTX 4090 (24GB GDDR6X)**

```yaml
# Maximum Configuration
phases:
  meta_learning:
    enabled: true
    control:
      epochs: 80
      batches_per_epoch: 25
      eval_interval: 2
  fine_tuning:
    enabled: true
    control:
      epochs: 50
      batches_per_epoch: 50
      ppo_epochs: 5
      eval_interval: 2

# Breakdown:
#   Meta-Learning: ~5-8 hours
#   Fine-Tuning: ~4-6 hours
#   Total: ~10-14 hours
#   Best for: Single-user research workstation
```

```yaml
# Quick Iteration
phases:
  meta_learning:
    enabled: true
    control:
      epochs: 30
      batches_per_epoch: 15
      eval_interval: 5
  fine_tuning:
    enabled: true
    control:
      epochs: 20
      batches_per_epoch: 20
      ppo_epochs: 3
      eval_interval: 5

# Breakdown:
#   Meta-Learning: ~1-2 hours
#   Fine-Tuning: ~1-1.5 hours
#   Total: ~2-3.5 hours
#   Best for: Hyperparameter tuning, quick experiments
```

---

#### **NVIDIA RTX 3090 (24GB GDDR6X)**

```yaml
# Recommended Configuration
phases:
  meta_learning:
    enabled: true
    control:
      epochs: 50
      batches_per_epoch: 20
      eval_interval: 2
  fine_tuning:
    enabled: true
    control:
      epochs: 30
      batches_per_epoch: 30
      ppo_epochs: 3
      eval_interval: 2

# Breakdown:
#   Meta-Learning: ~3-4 hours
#   Fine-Tuning: ~2-3 hours
#   Total: ~5-7 hours
```

---

### Mid-Range GPUs

#### **NVIDIA RTX 3080 Ti (12GB GDDR6X)**

```yaml
# Recommended Configuration
phases:
  meta_learning:
    enabled: true
    control:
      epochs: 40
      batches_per_epoch: 15
      eval_interval: 2
  fine_tuning:
    enabled: true
    control:
      epochs: 25
      batches_per_epoch: 20
      ppo_epochs: 2
      eval_interval: 2

# Breakdown:
#   Meta-Learning: ~2-3 hours
#   Fine-Tuning: ~1-1.5 hours
#   Total: ~3-4.5 hours
```

#### **NVIDIA RTX 3070 Ti (8GB GDDR6)**

```yaml
# Conservative Configuration
phases:
  meta_learning:
    enabled: true
    control:
      epochs: 30
      batches_per_epoch: 10
      eval_interval: 3
  fine_tuning:
    enabled: true
    control:
      epochs: 20
      batches_per_epoch: 15
      ppo_epochs: 2
      eval_interval: 3

# Breakdown:
#   Meta-Learning: ~1.5-2 hours
#   Fine-Tuning: ~1-1.5 hours
#   Total: ~2.5-3.5 hours
```

#### **NVIDIA RTX 3060 (12GB GDDR6)**

```yaml
# Conservative Configuration
phases:
  meta_learning:
    enabled: true
    control:
      epochs: 25
      batches_per_epoch: 10
      eval_interval: 3
  fine_tuning:
    enabled: true
    control:
      epochs: 15
      batches_per_epoch: 15
      ppo_epochs: 2
      eval_interval: 3

# Breakdown:
#   Meta-Learning: ~1-1.5 hours
#   Fine-Tuning: ~0.5-1 hour
#   Total: ~1.5-2.5 hours
```

---

### Cloud/Datacenter GPUs

#### **NVIDIA Tesla T4 (16GB GDDR6)**

```yaml
# Recommended Configuration
phases:
  meta_learning:
    enabled: true
    control:
      epochs: 30
      batches_per_epoch: 10
      eval_interval: 3
  fine_tuning:
    enabled: true
    control:
      epochs: 20
      batches_per_epoch: 15
      ppo_epochs: 2
      eval_interval: 3

# Breakdown:
#   Meta-Learning: ~2-3 hours (slower GPU)
#   Fine-Tuning: ~1.5-2 hours
#   Total: ~3.5-5 hours
#   Best for: Google Colab, AWS, cost-efficient cloud
```

---

## Quick Comparison: Meta vs POMO

### Same Training Budget (10 hours on RTX 4090)

#### **MetaTrainer Configuration**
```yaml
meta_learning:
  epochs: 50
  batches_per_epoch: 25
  # Time: ~5-6 hours
  # Learns task distribution
fine_tuning:
  epochs: 40
  batches_per_epoch: 30
  ppo_epochs: 3
  # Time: ~4-5 hours
  # Adapts to each task
# Total: 10 hours
```

#### **POMOTrainer Configuration**
```yaml
training:
  epochs: 50
  batches_per_epoch: 2000
  instances_per_batch: 8
# Total: 10 hours
# Direct training on single task
```

**Key Differences:**
- Meta: Learns transferable policy + task-specific adaptation
- POMO: Single-task optimization only
- Meta is better for multiple tasks with shared structure
- POMO is better for single task, maximum performance

---

## Recommended Starter Configurations for MVRPBTW

### **If you have < 6GB VRAM**
```yaml
phases:
  meta_learning:
    enabled: true
    control:
      epochs: 20
      batches_per_epoch: 10
      eval_interval: 2
      checkpoint_interval: 5
  fine_tuning:
    enabled: true
    control:
      epochs: 15
      batches_per_epoch: 10
      ppo_epochs: 2
      eval_interval: 2
      checkpoint_interval: 5

# Total Time: ~1-1.5 hours
# Use case: Laptop, limited GPU memory
```

---

### **If you have 6-12GB VRAM**
```yaml
phases:
  meta_learning:
    enabled: true
    control:
      epochs: 40
      batches_per_epoch: 20
      eval_interval: 2
      checkpoint_interval: 5
  fine_tuning:
    enabled: true
    control:
      epochs: 30
      batches_per_epoch: 25
      ppo_epochs: 3
      eval_interval: 2
      checkpoint_interval: 5

# Total Time: ~3-4 hours
# Use case: Mid-range GPU, balanced learning
```

---

### **If you have 12-24GB VRAM**
```yaml
phases:
  meta_learning:
    enabled: true
    control:
      epochs: 80
      batches_per_epoch: 30
      eval_interval: 2
      checkpoint_interval: 10
  fine_tuning:
    enabled: true
    control:
      epochs: 50
      batches_per_epoch: 50
      ppo_epochs: 5
      eval_interval: 2
      checkpoint_interval: 10

# Total Time: ~8-12 hours
# Use case: High-end consumer GPU, research workstation
```

---

### **If you have 24GB+ VRAM (Paper-matching)**
```yaml
phases:
  meta_learning:
    enabled: true
    control:
      epochs: 150
      batches_per_epoch: 50
      eval_interval: 5
      checkpoint_interval: 10
  fine_tuning:
    enabled: true
    control:
      epochs: 100
      batches_per_epoch: 100
      ppo_epochs: 10
      eval_interval: 5
      checkpoint_interval: 10

# Total Time: ~25-40 hours
# Use case: Research clusters, reproducing papers
```

---

## Expected Convergence Behavior

### Meta-Learning Phase
```
Epoch 1:   Task entropy ≈ 0.95 (high uncertainty)
Epoch 20:  Task entropy ≈ 0.70 (curriculum expansion starts)
Epoch 50:  Task entropy ≈ 0.40 (more tasks added)
Epoch 100: Task entropy ≈ 0.15 (curriculum complete)
```

**Curriculum Learning:**
- Early epochs: 1-2 easy tasks
- Mid epochs: Gradually add harder tasks
- Late epochs: Full task distribution

**Performance:**
- Meta-learning objective improves slowly (task distribution learning)
- Fine-tuning objective improves rapidly per-task (adaptation)

---

### Fine-Tuning Phase (Per-Task)

```
Task 1:
  Epoch 1:  objective ≈ 10000 (meta-learned starting point)
  Epoch 10: objective ≈ 5000  (rapid adaptation)
  Epoch 30: objective ≈ 3000  (convergence)

Task 2:
  Epoch 1:  objective ≈ 9500  (better meta starting point)
  Epoch 10: objective ≈ 4800
  Epoch 30: objective ≈ 2800  (slightly better due to meta-learning)
```

**Key insight:**
- Meta-learning reduces required fine-tuning epochs
- Better starting policy = faster convergence per task
- Diminishing returns after 30-40 epochs per task

---

## Optimization Tips

### **Memory Optimization**
```yaml
# If OOM during meta-learning:

# Option 1: Reduce batches_per_epoch (less data per epoch)
batches_per_epoch: 15  # ← From 25

# Option 2: Reduce epochs (faster curriculum)
epochs: 50  # ← From 100

# Option 3: Disable meta-learning (skip to fine-tuning)
meta_learning:
  enabled: false  # ← Use pre-trained or scratch
```

```yaml
# If OOM during fine-tuning:

# Option 1: Reduce ppo_epochs (fewer optimization steps)
ppo_epochs: 2  # ← From 5

# Option 2: Reduce batches_per_epoch (less data per epoch)
batches_per_epoch: 30  # ← From 50

# Option 3: Reduce epochs (shorter per-task training)
epochs: 30  # ← From 50
```

---

### **Speed Optimization**
```yaml
# Increase eval_interval (less frequent evaluation)
eval_interval: 5  # ← From 2
# Saves ~40% time by reducing evaluation overhead

# Increase checkpoint_interval
checkpoint_interval: 20  # ← From 10
# Saves ~10% time by reducing checkpoint writes
```

---

### **Convergence Optimization**

```yaml
# For better meta-learning:
meta_learning:
  epochs: 150  # ← Increase for better task distribution
  batches_per_epoch: 50  # ← Increase for more updates

# For better per-task performance:
fine_tuning:
  epochs: 80  # ← More epochs
  batches_per_epoch: 100  # ← More data per epoch
  ppo_epochs: 8  # ← More optimization steps
```

---

## Decision Matrix

Choose based on your GPU and priority:

| GPU | Speed Priority | Quality Priority |
|-----|---|---|
| A100 | 100/25/50/25/3 (4h) | 200/50/100/100/10 (40h) |
| V100 | 50/20/30/30/2 (4h) | 80/30/40/40/4 (12h) |
| RTX 4090 | 40/15/25/25/2 (3h) | 80/25/50/50/5 (14h) |
| RTX 3090 | 40/15/25/25/2 (5h) | 50/20/30/30/3 (7h) |
| RTX 3080 Ti | 30/12/20/20/2 (3h) | 40/15/25/25/2 (4.5h) |
| RTX 3070 Ti | 25/10/15/15/2 (2.5h) | 30/10/20/20/2 (3.5h) |
| Tesla T4 | 25/10/15/15/2 (4h) | 30/10/20/20/2 (5h) |

Format: `meta_epochs/meta_batches/fine_epochs/fine_batches/ppo_epochs (time)`

---

## Monitoring During Training

```bash
# Watch GPU usage during meta-learning
watch -n 1 nvidia-smi

# Expected memory growth:
# Epoch 1-10:   2-3GB (few tasks)
# Epoch 50:     4-5GB (more tasks)
# Epoch 100:    5-7GB (full curriculum)

# During fine-tuning:
# Constant: 2-4GB (single task at a time)
```

---

## Meta-Learning vs Fine-Tuning Trade-offs

### More Meta-Learning (increase epochs/batches)
**Pros:**
- Better task distribution understanding
- Faster fine-tuning convergence
- Better generalization to new tasks

**Cons:**
- Longer total training time
- More meta-updates needed for convergence
- Memory peaks late in training

### More Fine-Tuning (increase epochs/batches/ppo_epochs)
**Pros:**
- Better per-task performance
- Higher final objectives
- Simpler curriculum

**Cons:**
- Longer total training time
- Wasted effort on very difficult tasks
- Less transferability

### Balanced Approach (Recommended)
```yaml
# Typical ratio:
meta_learning_time: 40% of total
fine_tuning_time: 60% of total

# Example (10 hour budget):
meta_learning: 4 hours
fine_tuning: 6 hours
```

---

## References

- **MAML Paper**: Finn et al., "Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks" (ICML 2017)
- **POMO Paper**: Kwon et al., "POMO: Policy Optimization with Multiple Optima for Reinforcement Learning" (ICLR 2021)
- **Curriculum Learning**: Bengio et al., "Curriculum Learning" (ICML 2009)
