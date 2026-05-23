# Time-Related Feasibility Verifications

All time constraints in VRPBTW are checked before any action is taken. Here's how they work:

## 1. **Truck Service Feasibility** (`_truck_feasible`)

**Lines 690-715**

Checks if truck k can serve node j:

```
arrive_j = current_time[k] + distance[truck_node, j] / v_truck
service_end = max(arrive_j, tw_open[j]) + service_time[j]

Feasible if:
  ✓ service_end <= tw_close[j]          (respects time window at j)
  ✓ service_end + dist[j, DEPOT] / v_truck <= T_max  (can return to depot by horizon)
```

**Key Variables:**
- `_truck_current_time(state, k)` — derived from `truck_depart[k][-1]` (last service completion)
- `tw_open[j]`, `tw_close[j]` — time window [open, close] at node j
- `T_max` — episode time horizon (seconds)
- `v_truck` — truck speed (km/h)

---

## 2. **Truck Return Feasibility** (`_truck_return_feasible`)

**Lines 717-725**

Checks if truck k can return to depot:

```
arrive_depot = current_time[k] + distance[truck_node, DEPOT] / v_truck

Feasible if:
  ✓ arrive_depot <= T_max
```

---

## 3. **Drone Launch Feasibility** (`_drone_launch_feasible` + `_compute_drone_launch_time`)

**Lines 700-729 and 876-964**

Multi-stage feasibility with three launch-time cases:

### **Case 1: No-Delay Launch**
```
launch_time = truck_depart[launch_idx]
depart_t = launch_time + launch_time_offset
arrive_j = depart_t + distance[launch_node, j] / v_drone

Feasible if:
  ✓ arrive_j <= tw_close[j]
  ✓ service_end = max(arrive_j, tw_open[j]) + service_time[j]
  ✓ service_end + distance[j, land_node] / v_drone + land_time <= truck_depart[land_idx]
  ✓ trip_duration = service_end + distance[j, land_node] / v_drone <= t_max (trip duration)
  ✓ truck can still serve subsequent nodes after late landing
```

### **Case 2: Earliest Launch**
```
launch_time = truck_arrive[launch_idx]
(same time window and trip duration checks as Case 1)
```

### **Case 3: Delayed Launch**
```
max_truck_delay = compute_max_truck_delay(state, k, land_idx)
delayed_truck_depart = truck_depart[land_idx] + max_truck_delay

(same time window and trip duration checks, but with delayed_truck_depart)
```

**Key Variables:**
- `truck_arrive[k][i]`, `truck_depart[k][i]` — arrival/departure times at route position i
- `drone_trips_node[k][t]` — nodes in trip t of drone k
- `t_max` — maximum trip duration (seconds)
- `launch_time_offset` — time to prepare drone for flight
- `land_time` — time to land drone

---

## 4. **Drone Extend Feasibility** (`_drone_extend_feasible`)

**Lines 820-844**

Checks if active drone k can extend to serve node j:

```
arrive_j = current_time[k] + distance[drone_node, j] / v_drone
service_end = max(arrive_j, tw_open[j]) + service_time[j]

Feasible if:
  ✓ service_end <= tw_close[j]
  ✓ elapsed_time = current_time[k] - launch_time[k]
  ✓ elapsed_time + distance[j, landing_node] / v_drone <= t_max (trip duration)
  ✓ drone has feasible landing after extending to j
```

**Key Variables:**
- `_drone_current_time(state, k)` — derived from `drone_depart[k][-1][-1]`
- `_elapsed_trip_time(state, k)` — current_time - launch_time
- `t_max` — maximum trip duration (seconds)

---

## 5. **Drone Land Feasibility** (`_drone_land_feasible`)

**Lines 804-824**

Checks if drone k can land at truck_routes[k][land_idx]:

```
elapsed = current_time[k] - launch_time[k]
t_back = distance[drone_node, land_node] / v_drone
drone_land_time = current_time[k] + t_back + land_time

Feasible if:
  ✓ elapsed + t_back <= t_max (can complete trip in time)
  ✓ drone_land_time <= T_max (lands before episode horizon)
  ✓ truck can service subsequent nodes after late landing
```

---

## 6. **Late Landing Check** (`_late_land_feasible`)

**Lines 640-668**

Simulates remaining truck route from landing node to verify no time window violations:

```
for each subsequent node in truck route:
  arrive = time_at_landing + distance[landing_node, next_node] / v_truck
  if arrive > tw_close[next_node]:
    return False  (can't reach in time)
  
  service_end = max(arrive, tw_open[next_node]) + service_time[next_node]
  update time to service_end

Check return to depot:
  time_to_depot = service_end + distance[last_node, DEPOT] / v_truck
  return time_to_depot <= T_max
```

---

## 7. **Max Truck Delay Computation** (`_compute_max_truck_delay`)

**Lines 826-876**

Computes how long truck can wait at landing node before violating time windows:

```
max_delay = infinity

for each subsequent node in truck route starting from landing_idx:
  if truck delays by delta, service_end becomes: service_end + delta
  delta must satisfy: service_end + delta <= tw_close[next_node]
  delay_slack = tw_close[next_node] - service_end
  max_delay = min(max_delay, delay_slack)

Also check return to depot:
  arrive_depot + delta <= T_max
  delay_slack = T_max - arrive_depot
  max_delay = min(max_delay, delay_slack)

Return max_delay (can be negative if impossible to delay)
```

---

## Summary Table

| Constraint | Truck | Drone | Type |
|-----------|-------|-------|------|
| Time window at node | ✓ `tw_open ≤ arrive ≤ tw_close` | ✓ `tw_open ≤ arrive ≤ tw_close` | Hard |
| Episode horizon | ✓ `service_end + return ≤ T_max` | ✓ `arrive ≤ T_max` | Hard |
| Trip duration | — | ✓ `elapsed + return_flight ≤ t_max` | Hard |
| Truck delay accommodation | ✓ via `_compute_max_truck_delay` | — | Hard |
| Late landing recovery | ✓ via `_late_land_feasible` | ✓ via `_late_land_feasible` | Hard |

---

## Time Tracking Data Structures

**Explicit Time Arrays** (initialized at each action):
- `truck_arrive[k][i]` — arrival at truck_routes[k][i]
- `truck_depart[k][i]` — departure (service end) at truck_routes[k][i]
- `drone_arrive[k][t][i]` — arrival at drone_trips_node[k][t][i]
- `drone_depart[k][t][i]` — departure at drone_trips_node[k][t][i]

**Derived Accessors:**
- `_truck_current_time(state, k)` → `truck_depart[k][-1]`
- `_drone_current_time(state, k)` → `drone_depart[k][-1][-1]`
- `_drone_launch_time(state, k)` → `drone_depart[k][-1][0]` (first node in current trip)
