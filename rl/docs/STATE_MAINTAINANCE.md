# VRPBTWState Property Maintenance

Verification that all properties in VRPBTWState are properly initialized, copied, and updated.

## Property Maintenance Matrix

| Property | initial_state | _copy_state | _apply_truck | _apply_drone_launch | _apply_drone_extend | _apply_drone_land | _update_current_loads |
|----------|---|---|---|---|---|---|---|
| truck_node | ✓ | ✓ | ✓ | - | - | - | - |
| truck_prev_node | ✓ | ✓ | ✓ | - | - | - | - |
| truck_phase | ✓ | ✓ | ✓ (_update_truck_phase) | - | - | - | - |
| drone_node | ✓ | ✓ | - | ✓ | ✓ | ✓ | - |
| drone_active | ✓ | ✓ | - | ✓ | - | ✓ | - |
| drone_phase | ✓ | ✓ | - | ✓ | - | ✓ (_update_drone_phase) | - |
| drone_land_node | ✓ | ✓ | - | ✓ | ✓ | ✓ | - |
| served | ✓ | ✓ | ✓ | ✓ | ✓ | - | - |
| current_cost | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | - |
| current_truck_load | ✓ | ✓ | - | - | - | - | ✓ |
| current_drone_load | ✓ | ✓ | - | - | - | - | ✓ |
| truck_load | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | - |
| drone_load | ✓ | ✓ | - | ✓ | ✓ | ✓ | - |
| truck_routes | ✓ | ✓ | ✓ | - | - | - | - |
| drone_trips | ✓ | ✓ | - | ✓ | ✓ | ✓ | - |
| truck_arrive | ✓ | ✓ | ✓ | ✓ | - | ✓ | - |
| truck_depart | ✓ | ✓ | ✓ | ✓ | - | ✓ | - |
| drone_arrive | ✓ | ✓ | - | ✓ | ✓ | ✓ | - |
| drone_depart | ✓ | ✓ | - | ✓ | ✓ | ✓ | - |

## Summary

**Total Properties:** 19  
**Status:** ✅ All properties properly maintained

Each property is:
1. **Initialized** in `initial_state()`
2. **Copied** in `_copy_state()`
3. **Updated** in appropriate apply methods based on action type

## Key Invariants

### Drone Node Maintenance
- **When active:** `drone_node[k]` = current position in active trip
- **When inactive:** `drone_node[k]` = `truck_prev_node[k]` (ready for next launch)

### Drone Landing Node Maintenance
- **When active:** `drone_land_node[k]` = first feasible landing node (updated after each extend)
- **When inactive:** `drone_land_node[k]` = `truck_node[k]` (current truck position)

### Truck Load Maintenance (Overload Prevention)
- **When inactive:** `current_truck_load[k][0] = Q_t - max(truck_load[k])`
- **When active:** `current_truck_load[k][0] = Q_t - max(pre_landing_load, post_landing_load + drone_backhaul)`
  - `pre_landing_load = max(truck_load[k][0:land_idx])`
  - `post_landing_load = max(truck_load[k][land_idx:truck_node_idx+1])`
  - `drone_backhaul = drone_load[k][-1][-1]` (backhaul at current drone position)

This ensures truck never overloads when drone lands with its backhaul.

### Drone Load Maintenance
- `current_drone_load[k][0]` = min(drone remaining linehaul, truck pre-launch remaining)
- `current_drone_load[k][1]` = max(drone remaining backhaul, truck post-landing remaining)
- Computed in `_update_current_loads()` which is called in all apply methods

### Served Tracking
- Only customers (not depot) mark served status
- Marked in `_apply_truck()`, `_apply_drone_launch()`, `_apply_drone_extend()`
- Never unmarked (monotonic)

### Cost Accumulation
- Truck moves: `current_cost += c_t * manhattan_distance`
- Drone moves: `current_cost += c_d * euclidean_distance`
- Accumulated in all apply methods
