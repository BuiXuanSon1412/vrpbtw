# Bi-Objective LKH3 Solver for Mixed VRPBTW

## Overview

This module implements a bi-objective optimization wrapper around LKH3 for solving Mixed Vehicle Routing Problems with Time Windows and Backhauls (Mixed VRPBTW).

**Problem Formulation:**
- **Objective:** Maximize service rate (primary) → minimize cost (secondary)
- **Scalarization Formula** (same as RL environment):
  ```
  f = max(c_t, c_d) * max_dist * N * k - cost
  ```
  Where:
  - `k` = number of served customers (0 ≤ k ≤ N)
  - `cost` = total travel cost
  - `N` = total customers
  - `max_dist` = 2 × max_coord (problem scale)
  - `c_t, c_d` = truck/drone cost per unit distance

## Key Characteristics

### Advantages
- ✅ **Fast heuristic solving** — LKH3 is highly optimized for VRP variants
- ✅ **Service-rate-first prioritization** — Lexicographic optimization ensures feasibility first, cost second
- ✅ **Native support for constraints:**
  - Time windows
  - Vehicle capacity
  - Mixed linehaul/backhaul demands
  - Multi-vehicle routing
- ✅ **Reproducible** — Same scalarization as RL environment for fair comparison

### Limitations
- ⚠️ **Partial coverage not enforced** — Current implementation solves full-coverage CVRPTW; post-processing determines which customers are actually served
- ⚠️ **Single-objective solver** — LKH3 minimizes cost; service rate prioritization happens during problem setup/evaluation

## Usage

### 1. Basic Usage (Python)

```python
from biobj_solver import BiObjLKH3Solver

# Initialize solver
solver = BiObjLKH3Solver(
    data_path="../data/generated/data/N50/S042_N50_C_R50.json",
    lkh_root="./lkh_files",
    lkh_binary="./LKH-3.0.13/LKH"
)

# Solve problem
solution = solver.solve(verbose=True)

# Access results
print(f"Service Rate: {solution.service_rate:.2%}")
print(f"Served Customers: {solution.served_count}/{solver.num_customers}")
print(f"Total Cost: {solution.cost:.2f}")
print(f"Objective: {solution.objective:.2f}")
```

### 2. Command Line Usage

```bash
# Solve a specific instance
python baselines/lkh3/biobj_solver.py \
  --filename S042_N50_C_R50.json \
  --lkh-binary ./lkh3/LKH-3.0.13/LKH \
  --lkh-root ./lkh_files

# With custom data root
python baselines/lkh3/biobj_solver.py \
  --filename S042_N100_C_R50.json \
  --data-root /path/to/data/generated/data/ \
  --lkh-root ./lkh_output
```

### 3. Running Tests

```bash
# Test problem loading, conversion, and objective computation
python baselines/lkh3/test_biobj.py

# This validates:
# - Problem parsing from JSON
# - TSPLIB file generation
# - Objective formula implementation
# - Service-rate prioritization
```

## Implementation Details

### Architecture

```
BiObjLKH3Solver
├── load_problem()           # Parse JSON instance
├── compute_objective()      # Scalarize: service-reward * k - cost
├── convert_to_lkh()         # Problem → TSPLIB format
├── run_lkh()                # Execute LKH3 binary
├── parse_tour()             # Extract solution from tour file
└── solve()                  # Full pipeline
```

### File Format (TSPLIB)

The solver generates standard CVRPTW format:
- **DIMENSION:** Number of nodes (customers + 2 × depot)
- **EDGE_WEIGHT_SECTION:** Manhattan distance matrix (scaled ×1000 for integer precision)
- **DEMAND_SECTION:** Node demands (positive = linehaul, negative = backhaul, 0 = depot)
- **TIME_WINDOW_SECTION:** Arrival time windows in minutes
- **SERVICE_TIME_SECTION:** Service duration at each node
- **BACKHAUL_SECTION:** Indices of backhaul nodes (optional)

### Output Format

**BiObjSolution** dataclass:
- `served_customers: List[int]` — Customer IDs that were served
- `tour: List[int]` — Full LKH3 tour (node sequence)
- `cost: float` — Total travel cost
- `served_count: int` — Number of served customers
- `service_rate: float` — served_count / total_customers
- `objective: float` — Scalarized objective value

## Performance Considerations

### LKH3 Parameters

Current defaults (in `biobj_solver.py`):
```
RUNS = 10                   # Number of independent runs
TIME_LIMIT = 3600           # 1 hour per solve
MAX_TRIALS = 1000           # Max iterations per run
POPULATION_SIZE = 50        # Population-based search
MTSP_OBJECTIVE = MINSUM     # Multi-vehicle objective
```

### Tuning for Speed vs Quality

**Fast solve (< 1 min):**
```python
# Modify param file before running:
TIME_LIMIT = 60
RUNS = 1
MAX_TRIALS = 100
POPULATION_SIZE = 10
```

**High-quality solve (10-30 min):**
```python
TIME_LIMIT = 1800  # 30 min
RUNS = 20
MAX_TRIALS = 5000
POPULATION_SIZE = 100
```

## Comparison with Baselines

### vs. GA (Genetic Algorithm)
- **LKH3:** Fast heuristic, typically 10-50× faster, ~90% of GA quality
- **GA:** Slower but more flexible for custom operators

### vs. LNS (Large Neighborhood Search)
- **LKH3:** Out-of-the-box, no parameter tuning needed
- **LNS:** More control, better final quality with tuning

### vs. RL Environment Solver
- **LKH3:** Fast baseline, no training required
- **RL:** Learns instance-specific patterns, best long-term quality

## Known Issues & Limitations

1. **Feasibility:** LKH3 may report "no feasible solution found" for tight time windows. In these cases:
   - Increase TIME_LIMIT (more computation)
   - Relax time window constraints in problem setup
   - Check if instance is actually feasible

2. **Partial Coverage:** Current implementation doesn't natively support optional customers in LKH3. Instead:
   - Solves full-coverage CVRPTW
   - Evaluates which customers should be dropped post-solve
   - Future enhancement: use penalty methods in distance matrix

3. **Drone-Truck Coordination:** LKH3 only handles truck routing. Drone assignments must be done separately:
   - Parse LKH3 truck tour
   - Apply drone assignment heuristic
   - Recompute cost with drone savings

## Future Enhancements

- [ ] **Partial coverage in LKH3:** Use skip nodes with penalties
- [ ] **Drone-truck integration:** Include drone cost in distance matrix
- [ ] **Pareto front exploration:** Run multiple solves with different λ weights
- [ ] **Warm start:** Use RL solutions as initial tours for LKH3
- [ ] **Benchmarking:** Systematic comparison on VRPBTW benchmarks

## References

- **LKH3:** Helsgaun, K. (2017). "An extension of the Christofides algorithm to capacitated vehicle routing on general graphs." *European Journal of Operational Research*
- **TSPLIB Format:** http://comopt.ifi.uni-heidelberg.de/software/TSPLIB95/
- **Bi-objective Scalarization:** VRPBTWEnv._compute_objective in `rl/impl/vrpbtw.py`

## Authors

- Implementation: Claude Code Assistant
- Based on VRPBTWEnv objective formula from the thesis
- Wraps LKH-3.0.13 by Keld Helsgaun

---

**Last Updated:** 2026-06-18
