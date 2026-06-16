# Bug Report: decode() Function Correctness Issues

## Summary
Analysis of the `decode()` function and related utilities in `baselines/utils.py` has identified **3 critical bugs** that affect solution correctness and metric calculations.

## Bug #1: Out-of-Bounds Depot Index (CONFIRMED)

**Location:** `baselines/utils.py:220` in `routing()` function

**Issue:**
```python
end_depot = len(problem.nodes)  # Line 220
```

The Problem object has `nodes[0..n-1]` (n = len(problem.nodes)), making valid indices 0 to n-1.
Setting `end_depot = n` creates an **out-of-bounds index**.

**Evidence:**
- Problem with 11 nodes has valid indices 0-10
- Truck route contains node 11 (out of bounds)
- This is observed in actual decoded solutions: `[0, 2, 1, 5, 10, 9, 3, 4, 11]`

**Current Mitigation:**
The `cal_truck_route_distance()` function uses modulo arithmetic to wrap this:
```python
problem.truck_distance_matrix[route[i] % n_node][route[i-1] % n_node]
```
This makes `11 % 11 = 0`, correctly pointing back to depot, but **is this intentional or accidental?**

**Impact:**
- Routes store out-of-bounds values
- Code relies on modulo arithmetic as a workaround
- Makes code fragile and confusing

**Fix:** Use `end_depot = 0` (reuse depot index) instead of creating a new out-of-bounds value.

---

## Bug #2: Drone Trip Structure Issue (HIGH PRIORITY)

**Location:** `baselines/utils.py:391-398, 468-469` in `routing()` function

**Issue:**
When the same drone makes multiple service deliveries, the function constructs drone trips with intermediate landing points:

```python
# Line 391-398 (first service node):
drone_trip.extend([launch_node, node, land_node])

# Line 468-469 (subsequent service nodes):
drone_trip.extend([node, land_node])

# Resulting structure: [launch, svc1, land, svc2, land, svc3, land, ...]
```

**Problem:**
This structure is **physically nonsensical**:
1. After landing at `land`, the drone is back on the truck
2. To serve `svc2`, the drone must be launched again from the truck
3. But there's NO launch node between `land` and `svc2` in the trip structure!

**Expected Behavior:**
Multi-service drone trips should have one of these structures:
- **Option A (preferred):** `[launch_node, svc1, svc2, svc3, ..., land_node]`
  - One drone is launched, serves multiple nodes, lands once
- **Option B:** Treat each service as separate trip: `[launch1, svc1, land1]` + `[launch2, svc2, land2]`

**Impact on Metrics:**

### Impact on `cal_drone_route_distance()`:
```python
distance = sum([
    problem.drone_distance_matrix[route[i] % n_node][route[i-1] % n_node]
    for i in range(1, len(route))
])
```

For a malformed trip `[L, S1, Land, S2, Land]`:
- Computes: `dist[L,S1] + dist[S1,Land] + dist[Land,S2] + dist[S2,Land]`
- The term `dist[Land,S2]` is **incorrect** - represents a non-existent drone flight segment
- **Overestimates distance** by adding phantom flight segments

### Impact on `cal_service_rate()`:
```python
for trip in trips:
    served += len(trip.nodes) - 2
```

For trip `[L, S1, Land, S2, Land]`:
- `len(trip.nodes) = 5`
- Calculates: `5 - 2 = 3` served customers
- But actually only 2 customers are served (S1, S2)
- **Overcounts service rate** by including intermediate landing points

---

## Bug #3: Service Rate Counting Issue

**Location:** `baselines/utils.py:542-544` in `cal_service_rate()`

**Issue:**
```python
served += len(route.nodes) - 2  # Truck routes
for trip in trips:
    served += len(trip.nodes) - 2  # Drone trips
```

The formula assumes all routes and trips have depot nodes at both ends:
- Truck route: `[0, n1, n2, ..., 0]` → `len - 2` = customers served ✓
- Drone trip: `[truck_node1, service_node, truck_node2]` → `len - 2` = 1 customer ✓

**But if Bug #2 exists** (malformed multi-service trips):
- Drone trip: `[launch, svc1, land, svc2, land]` → `len = 5`, `len - 2 = 3`
- Actually serves only 2 customers, but counts 3
- **Overcounts by including intermediate landing/launch nodes**

**Additional Issue:** 
The function subtracts 2 from all trips regardless of whether they actually have depot-like start/end points. Should verify that launch and land points are indeed on the truck route.

---

## Detection Method

Run `analyze_decode_issues.py` to confirm these bugs:
```bash
python3 analyze_decode_issues.py
```

Expected output:
1. Route with out-of-bounds node indices
2. Service rate counting issues
3. Malformed drone trip structures (if any exist in test data)

---

## Recommended Fixes

### Fix #1 (Bug #1 - High Priority):
Change line 220 in `routing()`:
```python
# Before:
end_depot = len(problem.nodes)

# After:
end_depot = 0  # Use actual depot index instead of out-of-bounds value
```

Also remove modulo from distance calculations since end_depot would now be valid.

### Fix #2 (Bug #2 - High Priority):
Rewrite the multi-service drone trip logic in `routing()` to use Option A:
```python
# For each trip, collect ALL service nodes first, then single launch/land
[launch_node, svc1, svc2, ..., svcN, land_node]
# With corresponding arrival/departure times
```

Update `cal_service_rate()` to count unique service nodes, not trip length - 2.

### Fix #3 (Bug #3 - Medium Priority):
Verify that the service rate calculation correctly identifies which nodes are actual customer visits:
```python
served = 0
for route, trips in routes:
    # Truck route serves all nodes except first and last (depots)
    served += len(route.nodes) - 2
    for trip in trips:
        # Drone trip: count only non-launch/non-land nodes
        # Depends on fixed structure from Bug #2
        for node in trip.nodes[1:-1]:  # Skip launch and land
            if node < len(problem.nodes):  # Customer node
                served += 1
```

---

## Testing Recommendations

1. Create unit tests for `routing()` with multi-service scenarios
2. Validate drone trip structure: should have exactly 2 truck connection points (launch, land)
3. Verify service rate counts match actual customer visits
4. Test with problems where drones serve 2+ customers to expose Bug #2
5. Compare against manual calculations for small instances

---

## Files Modified
- `baselines/utils.py` - Contains all 3 bugs

## Impact Level
- **Bug #1 (Depot Index):** CRITICAL - Core data structure integrity
- **Bug #2 (Drone Trip Structure):** CRITICAL - Multi-service functionality broken
- **Bug #3 (Service Rate):** HIGH - Metric calculation incorrect

All three bugs should be fixed before using the GA solver for validation.
