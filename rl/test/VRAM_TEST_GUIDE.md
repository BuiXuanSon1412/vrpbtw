# GPU VRAM Usage Test Guide

## Overview

`test/vram_usage.py` measures GPU memory (VRAM) required for trainer updates on different tasks.

## What It Measures

### For Each (Task, Trainer) Pair:

1. **Peak VRAM** - Maximum GPU memory used during one update step
2. **Allocated Memory Δ** - Change in GPU memory allocation before/after update
3. **Reserved Memory Δ** - Change in GPU memory reserved by PyTorch
4. **Update Time** - Wallclock time for one update step
5. **Batch Details** - Episodes per batch used for update

## Trainers Tested

| Trainer | Type | Description |
|---------|------|-------------|
| **meta** | MetaTrainer | Meta-learning trainer (MAML-style) |
| **pomo** | POMOTrainer | POMO (Attention Model) trainer |

## Tasks Tested

All tasks from `configs/environment/mvrpbtw.yaml`:

**Small (N=10):**
- 001_N10_F2_RC - Random-Clustered distribution
- 003_N10_F2_C - Clustered distribution

**Medium (N=20):**
- 006_N20_F3_C - Clustered distribution

**Large (N=50):**
- 009_N50_F5_C - Clustered distribution

## Usage Examples

### Run All Measurements
```bash
python test/vram_usage.py
```

### Test Specific Trainer
```bash
# Test only MetaTrainer
python test/vram_usage.py --trainer meta

# Test only POMOTrainer
python test/vram_usage.py --trainer pomo
```

### Test Specific Task
```bash
# Test only small tasks
python test/vram_usage.py --task 001_N10_F2_RC

# Test only large tasks
python test/vram_usage.py --task 009_N50_F5_C
```

### Vary Batch Size
```bash
# Smaller batch (fewer episodes)
python test/vram_usage.py --batch-size 2

# Larger batch (more episodes per update)
python test/vram_usage.py --batch-size 16
```

### Combinations
```bash
# MetaTrainer on 001_N10_F2_RC with batch_size=8
python test/vram_usage.py --trainer meta --task 001_N10_F2_RC --batch-size 8

# All POMO measurements with batch_size=4
python test/vram_usage.py --trainer pomo
```

## Output Format

### Table 1: Peak VRAM Usage
```
PEAK VRAM USAGE:
Task                 Trainer         Peak VRAM (MB)       Alloc Δ (MB)         Time (s)       
001_N10_F2_RC        meta                    1234.5              245.3           0.123
001_N10_F2_RC        pomo                     987.3              182.1           0.089
003_N10_F2_C         meta                    1245.1              256.8           0.145
003_N10_F2_C         pomo                     998.7              192.5           0.098
...
```

### Table 2: Memory Allocation Details
```
VRAM ALLOCATION COMPARISON:
Task                 Trainer         Before (MB)          After (MB)           Delta (MB)    
001_N10_F2_RC        meta                 512.3                757.6              245.3
001_N10_F2_RC        pomo                 501.2                683.3              182.1
...
```

### Table 3: Detailed Breakdown
```
DETAILS:

001_N10_F2_RC + meta:
  Batch size:           4
  Peak VRAM:            1234.5MB
  Allocated (before):   512.3MB
  Allocated (after):    757.6MB
  Allocated Δ:          245.3MB
  Reserved (before):    1024.0MB
  Reserved (after):     1280.0MB
  Reserved Δ:           256.0MB
  Update time:          0.123s
```

### Summary Statistics
```
SUMMARY STATISTICS:
meta            | Mean peak VRAM:  1239.8MB | Max peak VRAM: 1512.3MB
pomo            | Mean peak VRAM:   993.0MB | Max peak VRAM: 1245.6MB
```

## Interpretation

### Key Metrics

**Peak VRAM:**
- Maximum memory needed to run one update step
- Important for determining GPU requirement (e.g., RTX 3090 has 24GB)

**Allocated Δ:**
- Shows how much GPU memory is consumed by the update
- Larger values = more memory-intensive computation
- Can grow with batch size and problem size

**Update Time:**
- Wallclock time for one update step
- Useful for estimating training time
- May correlate with VRAM (more computation = more memory)

### Common Patterns

**MetaTrainer vs POMOTrainer:**
- MetaTrainer typically uses more VRAM (meta-learning has gradient computation overhead)
- POMOTrainer usually faster (single forward pass scoring)

**Small vs Large Tasks:**
- Larger tasks (N=50) use significantly more VRAM
- Memory grows non-linearly with problem size

**Batch Size Effect:**
- Larger batches consume more VRAM
- Peak memory increases roughly linearly with batch_size

## Example Workflow

### 1. Baseline Measurement
```bash
# Measure all configurations with default batch_size=4
python test/vram_usage.py > vram_baseline.txt
```

### 2. Identify Memory Bottlenecks
```bash
# Test on largest task
python test/vram_usage.py --task 009_N50_F5_C
```

### 3. Find Safe Batch Size
```bash
# Try increasing batch sizes to find maximum
python test/vram_usage.py --batch-size 2
python test/vram_usage.py --batch-size 4
python test/vram_usage.py --batch-size 8
python test/vram_usage.py --batch-size 16
```

### 4. Compare Trainers
```bash
# Which trainer is more memory efficient?
python test/vram_usage.py --trainer meta
python test/vram_usage.py --trainer pomo
```

## Implementation Notes

### Measurement Methodology

1. **Clear GPU cache** - Remove stale allocations
2. **Reset peak memory stats** - Track only this update's memory
3. **Collect batch** - Generate episode data using trainer's collector
4. **Perform update** - Execute one trainer.update() step
5. **Record peak VRAM** - torch.cuda.max_memory_allocated()
6. **Cleanup** - Delete objects and empty cache

### Why These Metrics?

- **Peak VRAM** = what GPU needs to have free (hard requirement)
- **Allocated Δ** = how much memory update consumes
- **Reserved Δ** = GPU memory reserved but not used (fragmentation)
- **Update Time** = practical performance metric

### Limitations

- CPU and GPU transfers included in measurements
- Batch collection time not separated from update time
- Not accounting for concurrent GPU operations
- Measurements on fresh GPU state (no other processes)

## Troubleshooting

### "CUDA not available"
```bash
# Ensure GPU is available
nvidia-smi

# Check PyTorch CUDA support
python -c "import torch; print(torch.cuda.is_available())"
```

### Peak VRAM seems too high
- Check GPU is not running other tasks
- Verify batch_size parameter
- Run on smaller task (e.g., 001_N10_F2_RC) to baseline

### Test hangs or crashes
- Reduce batch_size to 1
- Test single trainer first
- Check GPU memory with nvidia-smi during run
- Look for out-of-memory errors in GPU logs

## Files

- `test/vram_usage.py` - Main test script
- `test/README.md` - Quick reference
- `test/VRAM_TEST_GUIDE.md` - This file

## Related Metrics

Also see:
- `test/usage.py` - CPU usage and episode duration
- `configs/train.yaml` - Default configuration
- `configs/environment/mvrpbtw.yaml` - Task definitions
