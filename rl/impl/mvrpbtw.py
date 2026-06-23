"""
impl/mvrpbtw.py
---------------
MVRPBTW: Multi-phase Vehicle Routing Problem with Time Windows and Drone Delivery
(Without phase constraint enforcement)

Identical to VRPBTWEnv but without linehaul→backhaul phase ordering constraint.
Vehicles can serve backhauls and linehauls in any order.

This addresses instance generation issues where time windows become infeasible
when phase constraints are strictly enforced.
"""

from impl.vrpbtw import (
    VRPBTWEnv,
    VRPBTWState,
    ActionMask,
    DEPOT,
    TRUCK,
    DRONE,
)
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np


class MVRPBTWEnv(VRPBTWEnv):
    """
    Multi-phase VRPBTW without phase constraint enforcement.

    Inherits all methods from VRPBTWEnv but:
    - Sets phased=False (no phase switching)
    - Overrides _truck_phase_ok: Always returns True (no phase check)
    - Overrides _drone_phase_ok: Always returns True (no phase check)

    Phase state is still tracked in state (truck_phase, drone_phase) for
    reference and logging, but does not restrict feasibility or trigger updates.
    """

    def __init__(self, cfg: dict):
        super().__init__(cfg)
        self.phased = False

    def _truck_phase_ok(self, state: VRPBTWState, k: int, j: int) -> bool:
        """
        Override: trucks can serve any customer regardless of phase.

        Original (VRPBTWEnv):
            phase == 0 → can only serve linehaul (demand > 0)
            phase == 1 → can only serve backhaul (demand < 0)

        New (MVRPBTWEnv):
            No phase constraint. Any customer can be served at any time.
        """
        return True

    def _drone_phase_ok(self, state: VRPBTWState, k: int, j: int) -> bool:
        """
        Override: drones can serve any customer regardless of locked phase.

        Original (VRPBTWEnv):
            phase == 0 → can only serve linehaul (demand > 0)
            phase == 1 → can only serve backhaul (demand < 0)

        New (MVRPBTWEnv):
            No phase constraint. Any customer can be served at any time.
        """
        return True

    @classmethod
    def from_config(cls, cfg: Dict) -> "MVRPBTWEnv":
        """Factory method: instantiate MVRPBTWEnv from config dict."""
        props = cfg.get("properties", cfg)
        return cls(props)


# ─────────────────────────────────────────────────────────────────────────────
# ParallelMVRPBTW: Parallel vehicle routing (all vehicles available)
# ─────────────────────────────────────────────────────────────────────────────


class ParallelMVRPBTW(MVRPBTWEnv):
    """
    Parallel VRPBTW: All vehicles (K trucks + K drones) available simultaneously.
    Action space: 2K × (N+1) bilevel (choose any vehicle and node).
    """

# ─────────────────────────────────────────────────────────────────────────────
# MonoMVRPBTW: Mono-vehicle Sequential Routing
# ─────────────────────────────────────────────────────────────────────────────


class MonoMVRPBTW(MVRPBTWEnv):
    """
    Mono-vehicle VRPBTW: Sequential vehicle routing with single-node action space.

    Routes one vehicle at a time in interleaved order: truck 0, drone 0, truck 1, drone 1, ..., truck K-1, drone K-1

    Vehicle indexing for routing:
    - current_vehicle_idx = 0 → truck 0 (fleet 0)
    - current_vehicle_idx = 1 → drone 0 (fleet 0)
    - current_vehicle_idx = 2 → truck 1 (fleet 1)
    - current_vehicle_idx = 3 → drone 1 (fleet 1)
    - ...
    - current_vehicle_idx = 2K-1 → drone K-1 (fleet K-1)

    Parent MVRPBTW vehicle indexing (for action encoding):
    - 0 to K-1: trucks (truck 0, truck 1, ..., truck K-1)
    - K to 2K-1: drones (drone 0, drone 1, ..., drone K-1)

    Key features:
    - Action space: (N+1) node selection only (unilevel)
    - current_vehicle_idx: Tracks routing sequence (0 to 2K-1)
    - Flexible drone launch: Drones can launch from ANY node in truck's route
    - Auto-advance: Moves to next vehicle when current vehicle returns to depot

    Key overrides:
    - step(): Converts unilevel action (node) to bilevel action (node, vehicle)
    - _drone_launch_feasible: Allows launching from ANY node in truck route (with feasibility checks)
    """

    def __init__(self, cfg: dict):
        super().__init__(cfg)
        self.current_vehicle_idx = 0

    def reset(self):
        """Reset environment and vehicle index.

        Resets to idx=0 which corresponds to truck 0 (first fleet).

        Returns:
            Initial observation
        """
        self.current_vehicle_idx = 0
        return super().reset()

    def get_action_mask(self, state: VRPBTWState):
        """Compute unilevel action mask for current vehicle only.

        Independently computes feasibility for the current vehicle, not extracted from parent.
        This avoids deadlock where parent skips depot because other vehicles can serve.
        """
        N1 = self.n_customers + 1
        mask = np.zeros(N1, dtype=bool)

        k, vehicle_type = self._get_current_vehicle_info()

        if vehicle_type == TRUCK:
            # Check feasible serving nodes for this truck
            for j in range(1, N1):
                if self._truck_feasible(state, k, j):
                    mask[j] = True

            # If no serving nodes, allow return to depot
            if not mask.any() and self._truck_return_feasible(state, k):
                mask[DEPOT] = True
        else:  # DRONE
            if state.drone_active[k]:
                # Drone is active: check extending to unserved customers
                for j in range(1, N1):
                    if self._drone_extend_feasible(state, k, j):
                        mask[j] = True
                # Check landing at nodes on truck route
                for land_idx in self._landing_nodes(state, k):
                    if self._drone_land_feasible(state, k, land_idx):
                        land_node = int(state.truck_routes[k][land_idx])
                        mask[land_node] = True
            else:
                # Drone is inactive: check launching
                for j in range(1, N1):
                    if self._drone_launch_feasible(state, k, j):
                        mask[j] = True
                # Inactive drone can always wait at depot
                mask[DEPOT] = True

        return ActionMask.from_bool_array(mask)

    def _get_current_vehicle_info(self) -> Tuple[int, int]:
        """Get current vehicle index and type (truck or drone).

        Routing sequence is interleaved by fleet:
        - idx 0 → truck 0 (fleet 0)
        - idx 1 → drone 0 (fleet 0)
        - idx 2 → truck 1 (fleet 1)
        - idx 3 → drone 1 (fleet 1)
        - ...

        Returns:
            (k, vehicle_type) where k is fleet index, vehicle_type is TRUCK or DRONE constant
        """
        k = self.current_vehicle_idx // 2
        is_drone = self.current_vehicle_idx % 2 == 1
        vehicle_type = DRONE if is_drone else TRUCK
        return k, vehicle_type

    def _update_vehicle_index(self, state: VRPBTWState) -> None:
        """Update vehicle index when current vehicle returns to depot.

        A vehicle is considered done when it's at the depot.
        """
        k, vehicle_type = self._get_current_vehicle_info()
        current_node = (
            int(state.drone_node[k]) if vehicle_type == DRONE else int(state.truck_node[k])
        )

        if current_node == DEPOT:
            n_vehicles = 2 * self.n_fleets
            self.current_vehicle_idx = (self.current_vehicle_idx + 1) % n_vehicles

    def step(self, action: int) -> Tuple[Union[np.ndarray, Dict], Optional[float], bool, bool, Dict]:
        """Execute one step: route current vehicle to selected node.

        For AM (unilevel action space), action is a node index only.
        Converts it to bilevel (node, global_vehicle_idx) before passing to parent.

        Routing sequence is interleaved (T0→D0→T1→D1→...),
        but parent MVRPBTW uses grouped indexing (T0→T1→...→D0→D1→...),
        so we convert current_vehicle_idx to the parent's indexing scheme.

        Args:
            action: Node index to visit (unilevel for AM)

        Returns:
            (obs, reward, terminated, truncated, info)
        """
        node = action
        k, vehicle_type = self._get_current_vehicle_info()

        # Convert interleaved indexing to parent's grouped indexing:
        # Parent MVRPBTW: 0 to K-1 are trucks, K to 2K-1 are drones
        if vehicle_type == TRUCK:
            parent_v_idx = k
        else:
            parent_v_idx = self.n_fleets + k

        bilevel_action = node * (2 * self.n_fleets) + parent_v_idx

        obs, reward, terminated, truncated, info = super().step(bilevel_action)

        if not terminated:
            self._update_vehicle_index(self._current_state)

        return obs, reward, terminated, truncated, info

    def _drone_launch_feasible(self, state: VRPBTWState, k: int, j: int) -> bool:
        """
        Override: Drone can launch from ANY node in its truck's route (not just prev_node).

        Original MVRPBTW/VRPBTWEnv:
            launch_node = state.truck_prev_node[k]  (fixed to one node)

        New (MonoMVRPBTW):
            launch_node can be ANY node in state.truck_routes[k]
            Check all possible launch nodes for feasibility.
        """
        if state.drone_active[k] or state.served[j]:
            return False
        if not self._drone_phase_ok(state, k, j):
            return False
        if state.drone_node[k] == state.truck_node[k]:
            return False

        demand = self.demands[j]
        if demand > 0:
            if demand > state.current_drone_load[k][0]:
                return False
        else:
            if demand < state.current_drone_load[k][1]:
                return False

        land_node = int(state.truck_node[k])
        truck_route = state.truck_routes[k]

        for launch_idx, launch_node_id in enumerate(truck_route):
            launch_node_id = int(launch_node_id)

            if launch_idx >= len(state.truck_depart[k]):
                continue

            launch_time = state.truck_depart[k][launch_idx]

            has_duplicate = False
            for existing_trip in state.drone_trips[k]:
                if (
                    existing_trip
                    and int(existing_trip[0]) == launch_node_id
                    and int(existing_trip[-1]) == land_node
                ):
                    has_duplicate = True
                    break
            if has_duplicate:
                continue

            t_launch_to_j = self.euclidean_dist[launch_node_id, j] / self.v_d
            earliest_at_j = launch_time + t_launch_to_j

            if earliest_at_j > self.tw_close[j]:
                continue

            feasible_lands = self._get_feasible_land_nodes(state, k, j, launch_time)
            if len(feasible_lands) > 0:
                return True

        return False


# ─────────────────────────────────────────────────────────────────────────────
# SequentialMVRPBTW: Fleet-Sequential Routing
# ─────────────────────────────────────────────────────────────────────────────


class SequentialMVRPBTW(MVRPBTWEnv):
    """
    Sequential VRPBTW: Fleet-sequential routing with bilevel action space for current fleet.

    Routes truck-drone pairs sequentially: (truck 0, drone 0), then (truck 1, drone 1), etc.
    Designed for testing parallel environment efficiency.

    Vehicle indexing (from parent MVRPBTW):
    - 0 to K-1: trucks (truck 0, truck 1, ..., truck K-1)
    - K to 2K-1: drones (drone 0, drone 1, ..., drone K-1)

    Key features:
    - Action space: 2 × (N+1) bilevel (vehicle_in_pair, node) for current fleet
    - current_fleet_idx: Tracks which fleet pair is being routed (0 to K-1)
    - Drone launch: From previous node only (inherits parent behavior)
    - Auto-advance: Moves to next fleet when both vehicles return to depot

    Action encoding:
    - action // (N+1) → vehicle in pair (0=truck, 1=drone)
    - action % (N+1) → node to visit
    """

    def __init__(self, cfg: dict):
        super().__init__(cfg)
        self.current_fleet_idx = 0

    def reset(self):
        """Reset environment and fleet index.

        Returns:
            Initial observation
        """
        self.current_fleet_idx = 0
        return super().reset()

    def get_action_mask(self, state: VRPBTWState):
        """Compute bilevel action mask for current fleet pair only.

        Returns mask of shape (2×(N+1),) for (vehicle_in_pair, node) encoding:
        - vehicle_in_pair ∈ {0=truck, 1=drone}
        - node ∈ {0, ..., N}
        """
        N1 = self.n_customers + 1
        k = self.current_fleet_idx
        mask = np.zeros(2 * N1, dtype=bool)

        # Check if there are any feasible serving nodes for this fleet pair
        has_feasible_serving = False

        # Truck (vehicle_in_pair = 0)
        feasible_truck_nodes = []
        for j in range(1, N1):
            if self._truck_feasible(state, k, j):
                feasible_truck_nodes.append(j)
                # Encode as (vehicle=0, node=j)
                mask[0 * N1 + j] = True
                has_feasible_serving = True

        # Drone (vehicle_in_pair = 1)
        if state.drone_active[k]:
            # Check extending to unserved customers
            for j in range(1, N1):
                if self._drone_extend_feasible(state, k, j):
                    mask[1 * N1 + j] = True
                    has_feasible_serving = True
            # Check landing at nodes on truck route
            for land_idx in self._landing_nodes(state, k):
                if self._drone_land_feasible(state, k, land_idx):
                    land_node = int(state.truck_routes[k][land_idx])
                    mask[1 * N1 + land_node] = True
                    has_feasible_serving = True
        else:
            # Inactive drone: check launching
            for j in range(1, N1):
                if self._drone_launch_feasible(state, k, j):
                    mask[1 * N1 + j] = True
                    has_feasible_serving = True

        # Only allow return-to-depot if there are NO feasible serving nodes
        if not has_feasible_serving:
            if self._truck_return_feasible(state, k):
                mask[0 * N1 + DEPOT] = True

        return ActionMask.from_bool_array(mask)

    def _update_fleet_index(self, state: VRPBTWState) -> None:
        """Update fleet index when both vehicles in current fleet return to depot.

        A fleet is considered done when both truck k and drone k are at the depot.
        """
        k = self.current_fleet_idx
        truck_at_depot = int(state.truck_node[k]) == DEPOT
        drone_at_depot = int(state.drone_node[k]) == DEPOT

        if truck_at_depot and drone_at_depot:
            self.current_fleet_idx = (self.current_fleet_idx + 1) % self.n_fleets

    def step(self, action: int) -> Tuple[Union[np.ndarray, Dict], Optional[float], bool, bool, Dict]:
        """Execute one step: route current fleet vehicles.

        For SequentialMVRPBTW (bilevel actions for current fleet pair):
        action encodes (vehicle_in_pair, node) where:
        - vehicle_in_pair ∈ {0=truck, 1=drone}
        - node ∈ {0, ..., N}

        Vehicle indexing (from parent MVRPBTW):
        - 0 to K-1: trucks
        - K to 2K-1: drones

        Args:
            action: Encoded bilevel action for current fleet (0 to 2×(N+1)-1)

        Returns:
            (obs, reward, terminated, truncated, info)
        """
        N1 = self.n_customers + 1
        vehicle_in_pair = action // N1
        node = action % N1

        k = self.current_fleet_idx
        if vehicle_in_pair == 0:
            v_idx = k
        else:
            v_idx = self.n_fleets + k

        bilevel_action = node * (2 * self.n_fleets) + v_idx

        obs, reward, terminated, truncated, info = super().step(bilevel_action)

        if not terminated:
            self._update_fleet_index(self._current_state)

        return obs, reward, terminated, truncated, info
