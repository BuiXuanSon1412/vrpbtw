# POMO Training Configuration Guide

## Memory Requirements Calculation

### Formula
```
GPU Memory per Batch ≈ instances_per_batch × N × trajectory_memory_overhead

Where:
  - instances_per_batch = number of problem instances per batch
  - N = problem size (customers/nodes)
  - trajectory_memory_overhead ≈ 50-100KB per trajectory (depends on network size)
```

### For MVRPBTW (N=10)
```
GPU Memory ≈ instances_per_batch × 10 × 75KB
```

Examples:
- `instances_per_batch=1`: ~750KB
- `instances_per_batch=8`: ~6MB
- `instances_per_batch=32`: ~24MB
- `instances_per_batch=64`: ~48MB

*Note: Actual usage is higher due to gradients, optimizer states, and intermediate activations.*

---

## Common GPU Configurations

### High-End GPUs (Research/Production)

#### **NVIDIA A100 (40GB HBM2e)**
```yaml
# Maximum Configuration (Paper-matching)
epochs: 100
batches_per_epoch: 10000
instances_per_batch: 64

# Breakdown:
#   - GPU Memory: ~6.4GB per batch
#   - Batches/sec: ~2-3
#   - Time per epoch: ~1-2 hours
#   - Total training time: ~100-200 hours
#   - Trajectories/epoch: 6.4M
#   - Best for: Large-scale research, multi-node training
```

```yaml
# Balanced Configuration (Recommended)
epochs: 50
batches_per_epoch: 5000
instances_per_batch: 32

# Breakdown:
#   - GPU Memory: ~3.2GB per batch
#   - Batches/sec: ~3-4
#   - Time per epoch: ~20-30 minutes
#   - Total training time: ~20-30 hours
#   - Trajectories/epoch: 1.6M
#   - Best for: Good convergence + reasonable time
```

---

#### **NVIDIA V100 (32GB HBM2)**
```yaml
# Maximum Configuration
epochs: 80
batches_per_epoch: 5000
instances_per_batch: 32

# Breakdown:
#   - GPU Memory: ~3.2GB per batch
#   - Batches/sec: ~2-3
#   - Time per epoch: ~30-40 minutes
#   - Total training time: ~40-55 hours
#   - Trajectories/epoch: 1.6M
#   - Best for: Research labs, long-running experiments
```

```yaml
# Conservative Configuration
epochs: 50
batches_per_epoch: 2000
instances_per_batch: 16

# Breakdown:
#   - GPU Memory: ~1.6GB per batch
#   - Batches/sec: ~4-5
#   - Time per epoch: ~7-10 minutes
#   - Total training time: ~6-9 hours
#   - Trajectories/epoch: 320K
```

---

#### **NVIDIA RTX 4090 (24GB GDDR6X)**
```yaml
# Maximum Configuration
epochs: 50
batches_per_epoch: 3000
instances_per_batch: 16

# Breakdown:
#   - GPU Memory: ~1.6GB per batch
#   - Batches/sec: ~3-4
#   - Time per epoch: ~15-20 minutes
#   - Total training time: ~12-17 hours
#   - Trajectories/epoch: 480K
#   - Best for: High-end consumer GPU, single-user research
```

```yaml
# Balanced Configuration
epochs: 40
batches_per_epoch: 1500
instances_per_batch: 8

# Breakdown:
#   - GPU Memory: ~800MB per batch
#   - Batches/sec: ~5-6
#   - Time per epoch: ~5 minutes
#   - Total training time: ~3-4 hours
#   - Trajectories/epoch: 120K
#   - Best for: Fast iteration, hyperparameter tuning
```

---

#### **NVIDIA RTX 3090 (24GB GDDR6X)**
```yaml
# Maximum Configuration
epochs: 40
batches_per_epoch: 2000
instances_per_batch: 8

# Breakdown:
#   - GPU Memory: ~800MB per batch
#   - Batches/sec: ~2-3 (slower than 4090)
#   - Time per epoch: ~15-20 minutes
#   - Total training time: ~10-13 hours
#   - Trajectories/epoch: 160K
```

```yaml
# Fast Iteration
epochs: 20
batches_per_epoch: 500
instances_per_batch: 4

# Breakdown:
#   - GPU Memory: ~400MB per batch
#   - Batches/sec: ~5-6
#   - Time per epoch: ~2 minutes
#   - Total training time: ~40 minutes
#   - Trajectories/epoch: 40K
#   - Best for: Quick experiments, debugging
```

---

### Mid-Range GPUs (Workstations)

#### **NVIDIA RTX 3080 Ti (12GB GDDR6X)**
```yaml
# Recommended Configuration
epochs: 30
batches_per_epoch: 1000
instances_per_batch: 4

# Breakdown:
#   - GPU Memory: ~400MB per batch
#   - Batches/sec: ~3-4
#   - Time per epoch: ~5-7 minutes
#   - Total training time: ~2.5-3.5 hours
#   - Trajectories/epoch: 120K
#   - Best for: Single-task fine-tuning, quick prototyping
```

#### **NVIDIA RTX 3070 Ti (8GB GDDR6)**
```yaml
# Recommended Configuration
epochs: 25
batches_per_epoch: 800
instances_per_batch: 2

# Breakdown:
#   - GPU Memory: ~200MB per batch
#   - Batches/sec: ~4-5
#   - Time per epoch: ~3-4 minutes
#   - Total training time: ~1.5-2 hours
#   - Trajectories/epoch: 40K
#   - Best for: Laptops, budget workstations
```

#### **NVIDIA RTX 3060 (12GB GDDR6)**
```yaml
# Conservative Configuration
epochs: 20
batches_per_epoch: 500
instances_per_batch: 2

# Breakdown:
#   - GPU Memory: ~200MB per batch
#   - Batches/sec: ~2-3
#   - Time per epoch: ~3-5 minutes
#   - Total training time: ~1-1.5 hours
#   - Trajectories/epoch: 40K
#   - Best for: Budget-friendly, light workloads
```

---

### Cloud/Datacenter GPUs

#### **NVIDIA Tesla T4 (16GB GDDR6)**
```yaml
# Recommended Configuration
epochs: 25
batches_per_epoch: 500
instances_per_batch: 4

# Breakdown:
#   - GPU Memory: ~400MB per batch
#   - Batches/sec: ~1-2 (slower)
#   - Time per epoch: ~5-10 minutes
#   - Total training time: ~2-4 hours
#   - Trajectories/epoch: 200K
#   - Best for: Google Colab, AWS, cost-efficient cloud training
```

#### **NVIDIA L4 (24GB GDDR6)**
```yaml
# Balanced Configuration
epochs: 40
batches_per_epoch: 1500
instances_per_batch: 8

# Breakdown:
#   - GPU Memory: ~800MB per batch
#   - Batches/sec: ~1.5-2
#   - Time per epoch: ~20-30 minutes
#   - Total training time: ~15-20 hours
#   - Trajectories/epoch: 480K
#   - Best for: Cloud inference-optimized, newer datacenter
```

---

### CPU-Only (Not Recommended for POMO)

#### **High-End CPU (64+ cores, e.g., AMD Threadripper, Intel Xeon)**
```yaml
# Minimal Configuration
epochs: 10
batches_per_epoch: 10
instances_per_batch: 1

# Breakdown:
#   - Memory: ~500MB-1GB per batch
#   - Batches/sec: ~0.01-0.05
#   - Time per epoch: ~5-20 minutes
#   - Total training time: ~50-200 minutes
#   - Trajectories/epoch: 100
#   - Best for: NOT RECOMMENDED - extremely slow
```

**Note:** CPU training for POMO is impractical due to:
- No tensor parallelization
- Slow trajectory collection
- Poor gradient computation performance
- ~100-1000× slower than GPU

---

## Quick Decision Table

**Choose based on your GPU and desired training time:**

| GPU | Recommended | Config | Training Time | Trajectories |
|-----|-------------|--------|--------------|--------------|
| A100 (40GB) | Paper-matching | 100/10000/64 | 100-200h | 64M |
| A100 | Practical | 50/5000/32 | 20-30h | 1.6M |
| V100 (32GB) | Balanced | 80/5000/32 | 40-55h | 1.6M |
| V100 | Fast | 50/2000/16 | 6-9h | 320K |
| RTX 4090 | Fast | 50/3000/16 | 12-17h | 480K |
| RTX 4090 | Quick | 40/1500/8 | 3-4h | 120K |
| RTX 3090 | Fast | 40/2000/8 | 10-13h | 160K |
| RTX 3090 | Quick | 20/500/4 | 40 min | 40K |
| RTX 3080 Ti | Practical | 30/1000/4 | 2.5-3.5h | 120K |
| RTX 3070 Ti | Practical | 25/800/2 | 1.5-2h | 40K |
| Tesla T4 | Cloud | 25/500/4 | 2-4h | 200K |
| CPU-only | **NOT REC.** | 10/10/1 | 50-200 min | 100 |

---

## Optimization Tips

### 1. **Memory Optimization**
```yaml
# If OOM errors occur:
# Step 1: Reduce instances_per_batch
instances_per_batch: 32  # ← Decrease by 50%

# Step 2: Increase batches to compensate
batches_per_epoch: 8000  # ← Increase by 2×

# Step 3: Still OOM? Reduce network complexity or use gradient checkpointing
```

### 2. **Speed Optimization**
```yaml
# If training too slow:
# Step 1: Increase instances_per_batch (if memory allows)
instances_per_batch: 64

# Step 2: Reduce batches_per_epoch
batches_per_epoch: 1000

# Step 3: Reduce eval_interval (less frequent evaluation)
eval_interval: 5  # Evaluate every 5 epochs instead of 1
```

### 3. **Convergence Optimization**
```yaml
# If convergence is slow:
# → Increase batches_per_epoch (more data/epoch)
batches_per_epoch: 10000

# → Keep instances_per_batch large (better gradient estimates)
instances_per_batch: 64

# → This gives better convergence even if it takes longer
```

---

## Expected Convergence Behavior

### Typical Performance Curve (MVRPBTW, N=10)
```
Epoch 1:   objective ≈ 15000-20000 (random policy)
Epoch 5:   objective ≈ 8000-10000   (initial learning)
Epoch 20:  objective ≈ 5000-6000    (good convergence)
Epoch 50:  objective ≈ 3000-4000    (fine-tuning)
Epoch 100: objective ≈ 2000-3000    (saturation)
```

**Rule of thumb:**
- Significant improvement: First 20-30 epochs
- Marginal improvement: Epochs 30-50
- Saturation: After epoch 50

---

## For Your MVRPBTW Problem

### Recommended Starter Configurations

#### **If you have < 6GB VRAM:**
```yaml
# pomo.yaml
trainer:
  phases:
    training:
      control:
        epochs: 20
        batches_per_epoch: 500
        instances_per_batch: 2
        eval_interval: 2
        checkpoint_interval: 5
```
**Training time:** ~30-60 minutes
**Trajectories:** 40K

---

#### **If you have 6-12GB VRAM:**
```yaml
# pomo.yaml
trainer:
  phases:
    training:
      control:
        epochs: 30
        batches_per_epoch: 1000
        instances_per_batch: 4
        eval_interval: 2
        checkpoint_interval: 5
```
**Training time:** ~2-3 hours
**Trajectories:** 120K

---

#### **If you have 12-24GB VRAM:**
```yaml
# pomo.yaml
trainer:
  phases:
    training:
      control:
        epochs: 50
        batches_per_epoch: 2000
        instances_per_batch: 8
        eval_interval: 2
        checkpoint_interval: 10
```
**Training time:** ~5-8 hours
**Trajectories:** 400K

---

#### **If you have 24GB+ VRAM (Paper-matching):**
```yaml
# pomo.yaml
trainer:
  phases:
    training:
      control:
        epochs: 100
        batches_per_epoch: 5000
        instances_per_batch: 32
        eval_interval: 5
        checkpoint_interval: 10
```
**Training time:** ~20-40 hours
**Trajectories:** 16M

---

## Monitoring During Training

Check if your configuration is working:

```bash
# Monitor GPU memory usage
watch -n 1 nvidia-smi

# Look for:
#   - Sustained 70-90% GPU utilization
#   - Consistent memory usage (not spiking)
#   - Batches/sec rate (should be stable)
```

If you see:
- **Memory spikes → OOM risk**: Reduce `instances_per_batch`
- **GPU util < 50%**: Increase `instances_per_batch` or `batches_per_epoch`
- **Very slow batches**: Network may be bottleneck, check data loading

---

## References

- **POMO Paper**: Kwon et al., "POMO: Policy Optimization with Multiple Optima for Reinforcement Learning" (ICLR 2021)
  - Uses: epochs=100, batches=10000, instances=64 on TPU/GPU clusters
  
- **AttentionModel**: Vinyals et al., "Attention, Learn to Solve Routing Problems!" (ICLR 2019)
  - Uses: 2M training steps with batch_size=128 on single GPU
