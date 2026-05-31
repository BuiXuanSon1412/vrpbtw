# Evaluate.py Usage Guide

## Overview

`evaluate.py` is a standalone evaluation script that loads trained checkpoints and evaluates them on fresh problem instances without retraining.

**Key Features:**
- ✅ Evaluate single or multiple checkpoints
- ✅ Support greedy, beam search, and best-of-N sampling decoding
- ✅ Test on different problem sizes (generalization)
- ✅ Generate CSV results for analysis
- ✅ Works with MetaTrainer and POMOTrainer checkpoints

## Configuration

### Main Config: `configs/evaluate.yaml`

```yaml
device: cpu                      # 'cpu' or 'cuda'

reproducibility:
  seed:
    random_seed: 42
    numpy_seed: 42
    torch_seed: 42

experiment:
  name: 001                       # Name for this evaluation run

directories:
  source: 001_pomo               # Training experiment to evaluate from
  base_dir: experiment/evaluate   # Where to save evaluation results
  artifacts: artifacts            # Subdirectory for outputs

evaluation:
  deterministic: true             # Greedy decoding (no sampling)
  n_eval_episodes: 100            # Number of test instances
  decoding:
    beam_width: 1                 # 1=greedy, >1=beam search
```

### Key Settings

| Parameter | Meaning |
|-----------|---------|
| `source` | Training experiment folder (e.g., `100_PPO_N10_C`) |
| `n_eval_episodes` | Number of fresh instances to test on |
| `beam_width` | 1 = greedy, 5 = beam search width 5 |
| `deterministic` | true = greedy, false = stochastic sampling |

## Basic Usage

### 1. Evaluate All Checkpoints (Default)

Evaluates all `.pt` files from a training experiment:

```bash
python evaluate.py
```

This uses the default config (`configs/evaluate.yaml`), which evaluates checkpoints from `source: 001_pomo`.

**Output:**
```
experiment/evaluate/
└── 001_pomo/
    └── 001/
        ├── artifacts/
        │   ├── 001_pomo_tune_best_003_N10_F2_C.csv
        │   ├── 001_pomo_tune_best_006_N20_F3_C.csv
        │   └── summary.csv
        └── evaluate.yaml
```

### 2. Evaluate Specific Training Experiment

Create a custom config file:

```bash
# configs/eval_custom.yaml
device: cpu
experiment:
  name: eval_100
directories:
  source: 100_PPO_N10_C      # ← Your training experiment
  base_dir: experiment/evaluate
  artifacts: artifacts
evaluation:
  n_eval_episodes: 100
  decoding:
    beam_width: 1
```

Then run:

```bash
python evaluate.py --config configs/eval_custom.yaml
```

### 3. Evaluate Specific Checkpoint

```bash
# By filename
python evaluate.py --checkpoint tune_best_003_N10_F2_C.pt

# By full path
python evaluate.py --checkpoint experiment/train/100_PPO_N10_C/checkpoints/tune_best_003_N10_F2_C.pt
```

## Advanced Usage

### Greedy Decoding (Default - Fastest)

```bash
python evaluate.py --config configs/eval_custom.yaml
```

Takes the greedy action at each step. **Fast, deterministic.**

### Beam Search (Better Quality)

```bash
python evaluate.py --config configs/eval_custom.yaml --beam 5
```

Explores top-5 partial solutions at each step. **Slower, better quality.**

**Typical beam widths:**
- `--beam 1`: Greedy (baseline)
- `--beam 3`: Good quality, 3x slower
- `--beam 5`: High quality, 5x slower
- `--beam 10`: Very high quality, 10x slower

### Best-of-N Sampling (Diverse Rollouts)

```bash
python evaluate.py --config configs/eval_custom.yaml --samples 8
```

Runs the stochastic policy 8 times per instance, keeps best solution. **Slower, higher variance.**

### Test Generalization (Different Problem Size)

Evaluate a checkpoint trained on N10 problems but tested on N20:

```bash
python evaluate.py \
  --checkpoint tune_best_003_N10_F2_C.pt \
  --customers 20
```

This overrides the problem size at evaluation time.

### Override Evaluation Episodes

```bash
# Quick evaluation (10 instances)
python evaluate.py --episodes 10

# Thorough evaluation (500 instances)
python evaluate.py --episodes 500
```

### Override Device

```bash
# CPU evaluation
python evaluate.py --device cpu

# GPU evaluation
python evaluate.py --device cuda
```

### Combine Multiple Overrides

```bash
python evaluate.py \
  --config configs/eval_custom.yaml \
  --checkpoint tune_best_003_N10_F2_C.pt \
  --episodes 200 \
  --beam 5 \
  --device cuda
```

## Output Format

### Per-Checkpoint CSV

**File:** `artifacts/{checkpoint_name}.csv`

```csv
metric,value
mean_objective,4617.26
std_objective,2781.36
best_objective,104.01
worst_objective,12254.95
median_objective,4237.71
mean_reward,7.70
mean_time_s,0.144
n_episodes,100.0
mean_cost,257.26
std_cost,78.43
best_cost,104.01
mean_service_rate,0.782
std_service_rate,0.139
best_service_rate,1.0
```

### Summary CSV (Multiple Checkpoints)

**File:** `artifacts/summary.csv`

```csv
checkpoint,mean_objective,std_objective,best_objective,mean_service_rate,...
tune_best_003_N10_F2_C.pt,4617.26,2781.36,104.01,0.782,...
tune_best_006_N20_F3_C.pt,49715.75,12345.67,2000.00,0.649,...
```

## Common Workflows

### Workflow 1: Quick Quality Check

```bash
# After training, evaluate best checkpoint
python evaluate.py \
  --config configs/eval_custom.yaml \
  --checkpoint tune_best_003_N10_F2_C.pt \
  --episodes 50
```

**Output:** Quick CSV with 50 instances (1-2 minutes)

### Workflow 2: Detailed Evaluation for Publication

```bash
# Thorough evaluation with beam search
python evaluate.py \
  --config configs/eval_custom.yaml \
  --checkpoint tune_best_003_N10_F2_C.pt \
  --episodes 200 \
  --beam 5 \
  --device cuda
```

**Output:** Comprehensive results with 200 instances + beam search

### Workflow 3: Compare Multiple Checkpoints

```bash
# Evaluate all checkpoints from training experiment
python evaluate.py \
  --config configs/eval_custom.yaml \
  --episodes 100 \
  --device cuda
```

**Output:** `summary.csv` with results for all checkpoints

**Then analyze:**
```python
import pandas as pd

df = pd.read_csv("experiment/evaluate/100_PPO_N10_C/eval_100/artifacts/summary.csv")
print(df[["checkpoint", "mean_objective", "mean_service_rate"]])
```

### Workflow 4: Test Generalization

```bash
# Checkpoint trained on N10, test on N20
python evaluate.py \
  --checkpoint tune_best_003_N10_F2_C.pt \
  --customers 20 \
  --episodes 100
```

**Expected results:**
- If trained well: Performance degrades but stays reasonable
- If overfitted: Performance drops significantly
- If generalizable: Performance similar to trained size

### Workflow 5: Batch Evaluation (All Tasks)

```bash
# Create custom config
cat > configs/eval_batch.yaml << 'EOF'
device: cuda
experiment:
  name: batch_eval
directories:
  source: 100_PPO_N10_C
  base_dir: experiment/evaluate
  artifacts: artifacts
evaluation:
  n_eval_episodes: 100
  decoding:
    beam_width: 1
EOF

# Run evaluation
python evaluate.py --config configs/eval_batch.yaml

# Analyze results
python << 'PYTHON'
import pandas as pd
df = pd.read_csv("experiment/evaluate/100_PPO_N10_C/batch_eval/artifacts/summary.csv")
print("\nBest Checkpoints:")
print(df.sort_values("mean_objective")[["checkpoint", "mean_objective", "mean_service_rate"]].head(3))
PYTHON
```

## Understanding Results

### Metrics Explained

| Metric | Meaning | Good Value |
|--------|---------|-----------|
| `mean_objective` | Average cost | **Lower is better** |
| `std_objective` | Cost variance | **Lower is better** |
| `best_objective` | Minimum cost | **Lower is better** |
| `worst_objective` | Maximum cost | **Lower is better** |
| `mean_service_rate` | % customers served | **Higher is better** (close to 1.0) |
| `mean_time_s` | Average solution time | **Lower is faster** |
| `mean_cost` | Average travel cost | **Lower is better** |
| `best_cost` | Minimum travel cost | **Lower is better** |

### Example Results Interpretation

```
Checkpoint: tune_best_003_N10_F2_C.pt

mean_objective:   4617.26  ← Average solution quality
std_objective:    2781.36  ← High variance (instance difficulty varies)
best_objective:   104.01   ← Best solution found
worst_objective:  12254.95 ← Worst solution found
mean_service_rate: 0.782   ← Serves ~78% of customers on average
best_service_rate: 1.0     ← Can serve all customers on some instances
mean_time_s:      0.144    ← Solves in ~0.14 seconds per instance
```

**Interpretation:** Good checkpoint - serves most customers, moderate cost variance

## Troubleshooting

### Error: "Config not found"
```bash
# Make sure evaluate config exists
ls configs/evaluate.yaml

# Or specify correct path
python evaluate.py --config configs/eval_custom.yaml
```

### Error: "Training config not found"
```bash
# Check training experiment directory exists
ls experiment/train/100_PPO_N10_C/config/config.yaml

# Or update directories.source in evaluate config
```

### Error: "Checkpoint not found"
```bash
# List available checkpoints
ls experiment/train/100_PPO_N10_C/checkpoints/

# Use exact filename
python evaluate.py --checkpoint tune_best_003_N10_F2_C.pt
```

### Slow evaluation
```bash
# Use fewer episodes
python evaluate.py --episodes 20

# Use CPU instead of GPU (if GPU is slow)
python evaluate.py --device cpu

# Use greedy instead of beam search
python evaluate.py --beam 1
```

### OOM (Out of Memory)
```bash
# Reduce episodes
python evaluate.py --episodes 50

# Use smaller beam width
python evaluate.py --beam 1

# Use CPU
python evaluate.py --device cpu
```

## Performance Comparison

### Decoding Strategy Comparison

| Strategy | Quality | Speed | Use Case |
|----------|---------|-------|----------|
| **Greedy** | Good | 1x | Baseline, quick check |
| **Beam (width=3)** | Better | 3x | Publications |
| **Beam (width=5)** | Best | 5x | High-quality results |
| **Best-of-8** | Better | 8x | Diversity analysis |

### Typical Results

For a trained model on 100 instances:

```
Greedy (1s):        mean_objective = 4617
Beam-5 (4s):        mean_objective = 4200  (9% better)
Best-of-8 (8s):     mean_objective = 4100  (11% better)
```

## Integration with Analysis

### Quick comparison of two checkpoints

```bash
# Evaluate checkpoint A
python evaluate.py --checkpoint tune_best_003_N10_F2_C.pt --episodes 100

# Evaluate checkpoint B
python evaluate.py --checkpoint tune_best_006_N20_F3_C.pt --episodes 100

# Compare
python << 'EOF'
import pandas as pd

a = pd.read_csv("experiment/evaluate/100_PPO_N10_C/artifacts/tune_best_003_N10_F2_C.csv")
b = pd.read_csv("experiment/evaluate/100_PPO_N10_C/artifacts/tune_best_006_N20_F3_C.csv")

print("Comparison:")
print(f"A objective: {float(a[a['metric'] == 'mean_objective']['value']):.2f}")
print(f"B objective: {float(b[b['metric'] == 'mean_objective']['value']):.2f}")

a_sr = float(a[a['metric'] == 'mean_service_rate']['value'])
b_sr = float(b[b['metric'] == 'mean_service_rate']['value'])
print(f"A service rate: {a_sr:.3f}")
print(f"B service rate: {b_sr:.3f}")
EOF
```

## Tips

1. **Always use `--episodes 100+`** for reliable statistics (lower variance)
2. **Use beam search for papers** - significant quality improvements with modest slowdown
3. **Save results to Git** - CSVs are small and useful for tracking progress
4. **Test generalization** - evaluate on different problem sizes to check robustness
5. **Use deterministic evaluation** - set seed consistently for reproducible results
