# Test Suite

Tests for VRPBTW RL framework.

## Test Files

### 1. usage.py - Measure CPU and Duration Metrics

Measures CPU usage and episode rollout duration for each task in `mvrpbtw.yaml`.

**Basic usage:**
```bash
python test/usage.py
```

**Run multiple episodes per task:**
```bash
python test/usage.py --episodes 10
```

**Measure specific task:**
```bash
python test/usage.py --task 001_N10_F2_RC
```

**Use GPU:**
```bash
python test/usage.py --device cuda
```

**Custom config:**
```bash
python test/usage.py --config configs/train.yaml --episodes 5
```

**Output metrics:**

| Metric | Description |
|--------|-------------|
| **Objective** | Episode solution quality (lower is better) |
| **Duration** | Time to complete one episode rollout (seconds) |
| **CPU** | Average CPU usage during rollout (%) |
| **Memory** | Peak memory delta during rollout (MB) |
| **Steps** | Number of decision steps in episode |

**Example output:**
```
DURATION & OBJECTIVE:
Task                 Objective            Duration (s)         Steps          
001_N10_F2_RC        7399.53 ± 2931.15    0.139 ± 0.020        9.0 ± 2.0      

CPU USAGE & MEMORY:
Task                 CPU (%)              Memory (MB)         
001_N10_F2_RC        71.6 ± 71.6          6.6 ± 6.3           
```

## Available Tasks

From `configs/environment/mvrpbtw.yaml`:

| Task ID | Customers | Fleets | Distribution |
|---------|-----------|--------|--------------|
| 001_N10_F2_RC | 10 | 2 | Random-Clustered |
| 002_N10_F2_R | 10 | 2 | Random |
| 003_N10_F2_C | 10 | 2 | Clustered |
| 004_N20_F3_RC | 20 | 3 | Random-Clustered |
| 005_N20_F3_R | 20 | 3 | Random |
| 006_N20_F3_C | 20 | 3 | Clustered |
| 007_N50_F5_RC | 50 | 5 | Random-Clustered |
| 008_N50_F5_R | 50 | 5 | Random |
| 009_N50_F5_C | 50 | 5 | Clustered |
| 010_N100_F10_RC | 100 | 10 | Random-Clustered |
| 011_N100_F10_R | 100 | 10 | Random |
| 012_N100_F10_C | 100 | 10 | Clustered |

### 2. vram_usage.py - Measure GPU VRAM for Trainer Updates

Measures GPU VRAM usage during trainer update steps for each task and trainer type.

**Requirements:**
- GPU with CUDA support

**Basic usage:**
```bash
python test/vram_usage.py
```

**Specific trainer:**
```bash
python test/vram_usage.py --trainer meta
python test/vram_usage.py --trainer pomo
```

**Specific task:**
```bash
python test/vram_usage.py --task 001_N10_F2_RC
```

**Custom batch size:**
```bash
python test/vram_usage.py --batch-size 8
```

**Output metrics:**

| Metric | Description |
|--------|-------------|
| **Peak VRAM** | Maximum GPU memory used during update (MB) |
| **Allocated Δ** | GPU memory allocated before/after update (MB) |
| **Reserved Δ** | GPU memory reserved by PyTorch (MB) |
| **Update time** | Time to perform one update step (seconds) |

**Example output:**
```
PEAK VRAM USAGE:
Task                 Trainer         Peak VRAM (MB)       Alloc Δ (MB)         Time (s)       
001_N10_F2_RC        meta                    1234.5              245.3           0.123
001_N10_F2_RC        pomo                     987.3              182.1           0.089
```

---

## Implementation Details

### Metrics Collected

1. **Objective**: Route cost (distance + time penalties)
2. **Duration**: Wall-clock time per episode
3. **CPU Usage**: Process CPU percentage during execution
4. **Memory**: Peak resident set size increase
5. **Steps**: Number of decision steps

### Sampling Strategy

- CPU usage sampled every 1ms during episode
- Measurements averaged across N episodes
- Both mean and standard deviation reported

### Environment Setup

- Fresh environment instance per task
- Deterministic agent (argmax action selection)
- Seeded RNG (numpy/torch) for reproducibility
