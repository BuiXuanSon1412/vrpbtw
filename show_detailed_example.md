# Individual and Solution Example - Detailed Breakdown

## Overview

This document shows a concrete example of:
1. An **Individual** (chromosome-based solution representation used by GA)
2. Its decoded **Solution** (actual routes with constraints applied)
3. How the decode function transforms one into the other

---

## Example Problem Instance

**Problem:** VRPBTW with 10 customers + 1 depot, 2 vehicles, drone delivery option

```
Nodes:
  - Node 0: DEPOT (location: 50.0, 50.0)
  - Nodes 1-10: CUSTOMERS with demands, time windows, and coordinates

Fleet:
  - 2 trucks with 200 unit capacity
  - Drones with 20 unit capacity (limited by weight)
  - System duration: 24 hours
  - Drone trip max duration: varies by instance
```

---

## Step 1: Individual (Chromosome)

### Chromosome Structure
```
Chromosome = [Permutation, Mask]

Permutation:  [7, 11, 4, 3, 2, 10, 6, 1, 5, 8, 9]
Mask:         [0, 0,  1, 0, 0, 0,  0, 0, 0, 0, 0]
```

### What Each Part Means

**Permutation:** Ordering of nodes including delimiters
- Positions 0-10 contain customer IDs (1-10) and delimiters (11, 12)
- Delimiter 11 marks boundary between vehicle 1 and vehicle 2
- Delimiter 12 (if exists) marks end of vehicle 2

| Pos | Node | Meaning |
|-----|------|---------|
| 0 | 7 | Customer 7 |
| 1 | 11 | **Fleet Delimiter** (vehicle boundary) |
| 2 | 4 | Customer 4 |
| 3 | 3 | Customer 3 |
| ... | ... | ... |

**Mask:** Delivery method for each node
- **0** = Truck delivery
- **1** = Drone delivery (linehaul - delivery)
- **-1** = Drone pickup (backhaul - pickup)

In this example:
- Position 0 (Node 7): Mask=0 → Deliver by truck
- Position 2 (Node 4): Mask=1 → Deliver by drone
- All others: Mask=0 → Deliver by truck

---

## Step 2: Decoding Process

The `decode(Individual, Problem)` function:

```
1. REPAIR CHROMOSOME
   - Remove impossible drone operations
   - Check feasibility
   
2. PARTITION BY DELIMITERS
   - Split by delimiter node (11)
   - Create separate sequences for each vehicle
   
   Sequence 1: [7] (before delimiter 11)
   Sequence 2: [4, 3, 2, 10, 6, 1, 5, 8, 9] (after delimiter 11)

3. FOR EACH SEQUENCE → BUILD ROUTES
   - Separate truck nodes (mask=0) from drone nodes (mask=1/-1)
   - Create truck route with truck nodes
   - For each drone node, find feasible launch/land points
   - Create drone trips: [launch_point, service_node, land_point]
   
4. APPLY CONSTRAINTS
   - Check time windows
   - Check capacity constraints
   - Check system duration
   
5. RETURN SOLUTION
   - Vehicle routes with timing information
   - Feasibility verified
```

---

## Step 3: Solution (Decoded Routes)

### Result

```
Solution contains 2 routes (one per vehicle):

ROUTE 0:
  Truck: [DEPOT(0) → Customer 7 → DEPOT(0)]
  Drone trips: 0
  
ROUTE 1:
  Truck: [DEPOT(0) → Customer 3 → Customer 2 → Customer 10 → ... → DEPOT(0)]
  Drone trips: 0 (in this example, node 4 couldn't be served by drone)
```

### Route Details with Timing

**ROUTE 0:**

| Stop | Node | Type | Arrival Time | Service Time | Departure Time |
|------|------|------|--------------|--------------|----------------|
| 1 | DEPOT | Start | 0:00 | - | 0:00 |
| 2 | Customer 7 | Delivery | 0:45 | 11:53 | 11:53 |
| 3 | DEPOT | End | 1:14 | - | 1:22 |

Status:
- ✅ Customer 7 served by truck
- Customer demand: -13 units (return/pickup)
- Service time: Arrived 0:45, service 11:53-11:53, departed 11:53

**ROUTE 1:**

| Stop | Node | Type | Arrival | Service | Departure |
|------|------|------|---------|---------|-----------|
| 1 | DEPOT | Start | 0:00 | - | 0:00 |
| 2 | Cust 3 | Truck | 0:75 | 11:53 | 11:53 |
| 3 | Cust 2 | Truck | ... | ... | ... |
| ... | ... | ... | ... | ... | ... |
| 9 | DEPOT | End | ... | - | ... |

Status:
- ✅ Customers 3, 2, 10, 6, 1, 5, 8, 9 served by truck
- ✅ Customer 4 attempted but couldn't find feasible drone trip
- ✓ Route respects all time windows
- ✓ Truck capacity never exceeded

---

## Solution Metrics

```
Fitness: 19488.49
  = objective considering both cost and service rate

Service Rate: 0.8182 (9 out of 11 nodes served)
  = (customers served) / (total customers)
  = 9 / 11
  
Cost: 588.49 units
  = (truck_distance × truck_cost) 
    + (drone_distance × drone_cost)
    + (num_routes × basis_cost)
```

---

## Key Implementation Details (Fixed)

### Bug Fixes Applied

#### ✅ Bug #1: Depot Index
```python
# WRONG (out of bounds):
end_depot = len(problem.nodes)  # = 11 for 11 nodes, but valid: 0-10

# CORRECT (valid index):
end_depot = 0  # Reuse depot at index 0
```

#### ✅ Bug #2: Drone Trip Structure
```python
# WRONG (malformed):
drone_trip = [launch, service1, land, service2, land, ...]
# Problem: Can't fly from land to service2 without relaunching!

# CORRECT (proper structure):
drone_trip = [launch, service, land]
# Each trip serves exactly one customer
# Multiple services = multiple independent trips
```

#### ✅ Bug #3: Service Rate Counting
```python
# WRONG (with malformed trips):
served += len(trip.nodes) - 2  # Overcounts intermediate nodes

# CORRECT (with proper 3-node trips):
served += len(trip.nodes) - 2  # = 3 - 2 = 1 (exactly one customer)
```

---

## The Complete Data Structure

```python
# INDIVIDUAL (Genetic Algorithm Level)
Individual(
    chromosome = [
        permutation=[7, 11, 4, 3, 2, 10, 6, 1, 5, 8, 9],
        mask=[0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]
    ]
)

# SOLUTION (Problem-Level)
Solution(
    routes=[
        (
            # Route 0
            truck_route=Route(
                nodes=[0, 7, 0],
                arrival=[0.00, 0.75, 1.14],
                departure=[0.00, 11.53, 1.22]
            ),
            drone_trips=[]  # No drones for this route
        ),
        (
            # Route 1
            truck_route=Route(
                nodes=[0, 3, 2, 10, 6, 1, 5, 8, 9, 0],
                arrival=[0.00, 0.75, ..., ...],
                departure=[0.00, 11.53, ..., ...]
            ),
            drone_trips=[]  # Customer 4 couldn't find feasible drone trip
        )
    ]
)
```

---

## Why This Matters

1. **Genetic Algorithm** works with Individuals (chromosomes)
   - Crossover and mutation operate on the chromosome representation
   - Simple, compact encoding

2. **Feasibility Evaluation** works with Solutions
   - Actual routes with all constraints applied
   - Timing feasibility verified
   - Capacity constraints checked

3. **The Bridge** is the `decode()` function
   - Transforms chromosome → actual routes
   - Applies problem constraints
   - Calculates feasibility and cost

4. **Bug Fixes** ensure correctness
   - Valid node indices (Bug #1)
   - Realistic drone operations (Bug #2)
   - Accurate metric calculations (Bug #3)

---

## Summary

| Aspect | Individual | Solution |
|--------|------------|----------|
| **Level** | Genetic Algorithm | Problem Domain |
| **Structure** | Permutation + Mask | Routes with timing |
| **Usage** | GA operations (crossover, mutation) | Cost calculation, feasibility check |
| **Constraints** | None (raw chromosome) | All constraints applied |
| **Example** | `[7, 11, 4, ...], [0, 0, 1, ...]` | 2 routes with times and drones |

The transformation from Individual to Solution is where the **magic happens** - where GA's abstract representation meets the real-world problem constraints!
