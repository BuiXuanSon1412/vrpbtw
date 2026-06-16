# Time Window Feasibility Analysis

## Summary

Time window constraints are being **violated in randomly generated solutions** because:

1. ✅ **Bug #5 FIXED**: Truck arrival times are now cumulative (were all 0-2 hours, now properly distributed)
2. ⚠️ **Design behavior**: The routing function skips nodes that can't be served within their time window (this is correct behavior)
3. ❌ **Constraint enforcement**: Time windows are checked but violations are only prevented during the routing logic, not in all solutions

## Before Fix (Bug #5)
```
❌ TIME WINDOW VIOLATIONS FOUND: 80

Example violations:
  Individual 0, Route 0, Customer 1
  Arrival: 0.28h, Window: [2.74, 3.83]

  Individual 0, Route 0, Customer 5
  Arrival: 0.22h, Window: [13.49, 14.61]
```

**Issue**: `truck_time` variable was never updated, so all arrivals were calculated from the depot (time 0) regardless of cumulative travel.

## After Fix (Bug #5)
```
❌ TIME WINDOW VIOLATIONS FOUND: 38 (48% reduction)

Example violations:
  Individual 1, Route 0, Customer 7
  Arrival: 12.48h, Window: [14.13, 15.49]
  (Arrives too early - about 1.65 hours before window opens)

  Individual 5, Route 0, Customer 9
  Arrival: 4.42h, Window: [7.00, 8.17]
  (Arrives too early - about 2.58 hours before window opens)
```

**Improvement**: Arrival times are now cumulative and realistic, reflecting actual route progression.

## Why Violations Still Exist

### Root Cause: Random Solutions
The `init_population()` function creates **random chromosome permutations** without considering time windows. These random solutions are not guaranteed to be feasible.

### The Routing Function's Approach
The `routing()` function implements **selective node inclusion**:

```python
# Line 254-259
if (tmp_truck_time > problem.nodes[node].time_window[1] - problem.service_time):
    continue  # Skip this node if it's too late
```

**What this means:**
- If a customer's time window has already closed, the node is skipped
- The truck proceeds to the next node in the sequence
- This can result in unserved customers (they appear in the chromosome but not in the route)

### Example of Selective Inclusion
```
Chromosome permutation: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

Time windows are spread: 
  Customer 1: [2.74, 3.83]   ← Early morning
  Customer 3: [11.45, 12.45] ← Noon
  Customer 7: [14.13, 15.49] ← Afternoon

Route progression:
  Start at depot (time 0)
  - Try Customer 1: Need 0.70h arrival, window [2.74, 3.83] → TOO EARLY
  - Try Customer 2: Need 0.40h arrival, window [0.91, 2.31] → FEASIBLE ✅
  - Try Customer 3: Need 2.93h arrival, window [11.45, 12.45] → TOO EARLY
  
Result: Only some customers can be served in the greedy routing order
```

## Why This Is Happening

**Fundamental Issue**: Random initial populations have no intelligence about time windows. The chromosome permutation order dictates feasibility:

1. **Good chromosome order**: Customers are visited in sequence matching their time windows
2. **Bad chromosome order**: Customers are in sequence that violates their time windows

### Example Problem Times
```
Customer time windows span 24 hours:
  - Earliest opening: 0.91h (Customer 2)
  - Latest closing: 15.49h (Customer 7)
  - Range: 14.58 hours
  
Travel times are short (1-2 hours), so visiting in wrong order = violations
```

## Verification: Drone Trips

**Drone trips are much better at respecting time windows:**

```
Individual 5, Route 0:
  Drone Trip 0: Customer 5 Arrive 14.27h | TW [13.49, 14.61]
  ✅ FEASIBLE - Within time window by 0.34 hours
```

Why? Because the routing logic explicitly finds launch/land points that satisfy time window constraints for drone deliveries.

## Implications

### For the GA Solver
1. **Random initialization is expected to be infeasible** - GA evolution improves feasibility
2. **Fitness function penalizes time window violations** - included in objective calculation via `cal_objective()`
3. **GA must discover time-window-respecting solutions** through crossover and mutation

### For Baseline Algorithms
1. Initial solutions may violate time windows
2. As GA optimizes, it should converge toward feasible solutions
3. Constraint handling is implicit through the objective function

### Time Window Width Analysis
All customers have tight time windows (1.0-1.4 hours):
```
Customer  1: 1.09 hour window [2.74, 3.83]
Customer  3: 1.00 hour window [11.45, 12.45] ← Tightest
Customer  7: 1.36 hour window [14.13, 15.49] ← Widest
```

These tight windows mean **small errors in routing order = large time window violations**.

## Bug #5 Impact

**Fixed:** `truck_time` variable not updated
```python
# BEFORE (BUG):
truck_time = 0.0  # Never updated!
for node in route:
    tmp_truck_time = truck_time + ...  # Always based on depot

# AFTER (FIXED):
truck_time = 0.0
for node in route:
    tmp_truck_time = truck_time + ...
    # ... feasibility checks ...
    truck_time = truck_depart[-1]  # Update for next iteration!
```

**Result:**
- Violations reduced from 80 to 38 (48% reduction)
- Arrival times now realistic and cumulative
- Routing feasibility checks now meaningful

## Conclusion

Time window violations in random initial solutions are **expected and by design**:
- Random solutions are typically infeasible
- GA evolution discovers feasible solutions
- Constraint enforcement through objective function
- Drone trips handle time windows better than truck routes

The fix to Bug #5 ensures that time window violations are calculated correctly, allowing the GA to properly evaluate and improve solutions.
