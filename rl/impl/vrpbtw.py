"""
problems/vrpbtw.py
------------------
VRPBTW with heterogeneous truck-drone fleets.

Reward design
-------------
Per-step reward uses potential-based shaping:
    r_t = -(f(s_{t+1}) - f(s_t))

where f(s) = total_cost + c_t * max_dist * N * (N - k)
    total_cost: total travel cost
    N: total customers
    k: served customers

This objective prioritizes serving customers (lexicographic) while minimizing cost.
Penalties are maintained incrementally in VRPBTWState.

At termination:
    feasible   -> r_T = -(f(s_T) - f(s_{T-1}))  (potential-based shaping)
    infeasible -> r_T = -1e6                     (hard penalty for infeasible end)

Constants
---------
NODE_FEAT_DIM  = 6   [x, y, linehaul_demand, backhaul_demand, tw_open, tw_close]
VEH_FEAT_DIM   = 6   [x, y, linehaul_bound, backhaul_bound, time, deadline]
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from core.environment import (
    ActionMask,
    Environment,
    Solution,
    StepResult,
)

# ---------------------------------------------------------------------------
# Feature-dimension constants
# ---------------------------------------------------------------------------

NODE_FEAT_DIM = 6
VEH_FEAT_DIM = 6

TRUCK = 0
DRONE = 1
DEPOT = 0


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------


@dataclass
class VRPBTWState:
    # Per fleet (K,)
    truck_node: np.ndarray
    truck_prev_node: np.ndarray  # (K,) previous truck node (second-most-recent)
    truck_phase: np.ndarray

    drone_node: np.ndarray
    drone_active: np.ndarray
    drone_launch_node: np.ndarray  # (K,) truck node at which drone k last launched
    drone_phase: (
        np.ndarray
    )  # (K,) phase when drone k launched (locked for trip duration)

    # Global
    served: np.ndarray  # (N+1,) bool

    # Incremental objective accumulators
    current_cost: float  # total travel cost so far

    # Policy inputs: per-vehicle load bounds (K, 2)
    current_truck_load: np.ndarray  # [max_linehaul_can_serve, min_backhaul_can_serve]
    current_drone_load: np.ndarray  # same structure

    # Global load tracking: per-position/per-trip loads
    truck_load: List[List[float]]  # [k][pos] load carried from pos to pos+1
    drone_load: List[List[List[float]]]  # [k][trip][pos] same for drone trips

    # Routes (for logging + evaluate())
    truck_routes: List[List[int]]
    drone_trips_node: List[List[List[int]]]  # [k][trip][pos] node IDs
    drone_trips_mask: List[List[List[int]]]  # [k][trip][pos] 0=non-customer 1=customer

    # Explicit time tracking per node/trip
    truck_arrive: List[List[float]]  # [k][i] arrival at truck_routes[k][i]
    truck_depart: List[
        List[float]
    ]  # [k][i] departure (service_end) at truck_routes[k][i]
    drone_arrive: List[
        List[List[float]]
    ]  # [k][t][i] arrival at drone_trips_node[k][t][i]
    drone_depart: List[
        List[List[float]]
    ]  # [k][t][i] departure at drone_trips_node[k][t][i]


def _copy_state(s: VRPBTWState) -> VRPBTWState:
    return VRPBTWState(
        truck_node=s.truck_node.copy(),
        truck_prev_node=s.truck_prev_node.copy(),
        truck_phase=s.truck_phase.copy(),
        drone_node=s.drone_node.copy(),
        drone_active=s.drone_active.copy(),
        drone_launch_node=s.drone_launch_node.copy(),
        drone_phase=s.drone_phase.copy(),
        served=s.served.copy(),
        current_cost=s.current_cost,
        current_truck_load=s.current_truck_load.copy(),
        current_drone_load=s.current_drone_load.copy(),
        truck_load=[list(r) for r in s.truck_load],
        drone_load=[[list(t) for t in trips] for trips in s.drone_load],
        truck_routes=[list(r) for r in s.truck_routes],
        drone_trips_node=[[list(t) for t in trips] for trips in s.drone_trips_node],
        drone_trips_mask=[[list(m) for m in trips] for trips in s.drone_trips_mask],
        truck_arrive=[[t for t in times] for times in s.truck_arrive],
        truck_depart=[[t for t in times] for times in s.truck_depart],
        drone_arrive=[
            [[t for t in times] for times in trips] for trips in s.drone_arrive
        ],
        drone_depart=[
            [[t for t in times] for times in trips] for trips in s.drone_depart
        ],
    )


# ---------------------------------------------------------------------------
# VRPBTWProblem
# ---------------------------------------------------------------------------


class VRPBTWEnv(Environment):
    def __init__(self, cfg: Dict[str, Any]):
        super().__init__(name="VRPBTW")
        # Task ID (set when reset() is called with task_id)
        self.task_id: Optional[str] = None

        # Available tasks from config (task_id format: "{difficulty}_N{customers}_F{fleets}_{distribution}")
        self.tasks: List[str] = cfg.get("tasks", [])

        # Instance parameters (set dynamically in reset via encode_instance)
        self.n_customers: int = 10
        self.n_fleets: int = 2
        self.K: int = 2

        # Static node feature arrays (placeholder, resized in encode_instance)
        self.coords: np.ndarray = np.zeros((1, 2), dtype=np.float32)
        self.demands: np.ndarray = np.zeros(1, dtype=np.float32)
        self.tw_open: np.ndarray = np.zeros(1, dtype=np.float32)
        self.tw_close: np.ndarray = np.zeros(1, dtype=np.float32)
        self.service_times: np.ndarray = np.zeros(1, dtype=np.float32)
        self.manhattan_dist: np.ndarray = np.zeros((1, 1), dtype=np.float32)
        self.euclidean_dist: np.ndarray = np.zeros((1, 1), dtype=np.float32)

        # Static environment parameters from config
        self.max_coord: float = float(cfg.get("max_coord", 100.0))
        self.Q_t: float = float(cfg.get("capacity_truck", 200.0))
        self.Q_d: float = float(cfg.get("capacity_drone", 20.0))
        self.T_max: float = float(cfg.get("t_max_system_h", 24.0))
        self.t_max: float = float(cfg.get("drone_duration_h", 1.0))
        self.v_t: float = float(cfg.get("v_truck_km_h", 40.0))
        self.v_d: float = float(cfg.get("v_drone_km_h", 60.0))
        self.c_t: float = float(cfg.get("truck_cost_unit", 1.0))
        self.c_d: float = float(cfg.get("drone_cost_unit", 0.5))
        self.launch_time: float = float(cfg.get("drone_takeoff_min", 1.0)) / 60.0
        self.land_time: float = float(cfg.get("drone_landing_min", 1.0)) / 60.0
        self.service_time: float = float(cfg.get("service_time_min", 5.0)) / 60.0

        # Instance generation parameters (for _generate_instance)
        self.demand_range_linehaul: Tuple[int, int] = (
            int(cfg.get("demand_range_linehaul_min", 5)),
            int(cfg.get("demand_range_linehaul_max", 10)),
        )
        self.demand_range_backhaul: Tuple[int, int] = (
            int(cfg.get("demand_range_backhaul_min", 5)),
            int(cfg.get("demand_range_backhaul_max", 10)),
        )
        self.time_window_scaling_factor: float = float(
            cfg.get("time_window_scaling_factor", 1.0)
        )

        # Phase constraint enforcement: True for VRPBTW, False for MVRPBTW
        self.phased: bool = True

        self._linehaul_idx: np.ndarray = np.array([], dtype=np.int32)
        self._backhaul_idx: np.ndarray = np.array([], dtype=np.int32)

    @classmethod
    def from_config(cls, cfg: Dict) -> "VRPBTWEnv":
        """Factory method: instantiate VRPBTWEnv from config dict.

        Instance-specific parameters (n_customers, n_fleets) are set dynamically
        in reset() via _generate_instance() when task_id is provided.
        """
        # Support both old (flat) and new (properties) structure
        props = cfg.get("properties", cfg)
        return cls(props)

    # ------------------------------------------------------------------
    # Instance generation
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_task_id(task_id: str) -> Tuple[int, int, str]:
        """Parse task ID format: {difficulty}_N{customers}_F{fleets}_{distribution}.

        Returns: (n_customers, n_fleets, distribution)
        Example: "001_N10_F2_RC" -> (10, 2, "RC")
        """
        parts = task_id.split("_")
        if len(parts) < 4:
            raise ValueError(f"Invalid task_id format: {task_id!r}")

        try:
            n_customers = int(parts[1][1:])  # Skip 'N'
            n_fleets = int(parts[2][1:])  # Skip 'F'
            dist = parts[3]
        except (ValueError, IndexError) as e:
            raise ValueError(f"Failed to parse task_id {task_id!r}: {e}")

        return n_customers, n_fleets, dist

    def _generate_coords(self, num_customers: int, dist_type: str) -> List[List[float]]:
        """Generate customer coordinates based on distribution type (R, C, RC).
        Uses global numpy RNG state set by seed_everything().
        """
        if dist_type == "R":
            return np.random.uniform(
                0, self.max_coord, size=(num_customers, 2)
            ).tolist()

        elif dist_type == "C":
            coords = []
            remaining_nodes = num_customers

            if num_customers <= 200:
                std_dev = self.max_coord / 25
            elif num_customers <= 400:
                std_dev = self.max_coord / 32
            else:
                std_dev = self.max_coord / 40
            min_dist = std_dev * 4

            centers = []
            while remaining_nodes > 0:
                current_cluster_size = min(
                    remaining_nodes, int(np.random.uniform(10, 16))
                )

                proposal = np.random.uniform(5, self.max_coord - 5, size=(2,))
                valid_center = False
                attempts = 0
                while not valid_center and attempts < 1000:
                    proposal = np.random.uniform(5, self.max_coord - 5, size=(2,))
                    if not centers:
                        valid_center = True
                    else:
                        dists = [
                            np.linalg.norm(proposal - np.array(c)) for c in centers
                        ]
                        if min(dists) >= min_dist:
                            valid_center = True
                    attempts += 1

                centers.append(proposal.tolist())

                for _ in range(current_cluster_size):
                    point = np.random.normal(proposal, std_dev)
                    coords.append(np.clip(point, 0, self.max_coord).tolist())

                remaining_nodes -= current_cluster_size

            return coords

        elif dist_type == "RC":
            n_c = num_customers // 2
            return self._generate_coords(n_c, "C") + self._generate_coords(
                num_customers - n_c, "R"
            )

        raise ValueError(f"Unknown distribution type: {dist_type}")

    def _generate_instance(self, task_id: str) -> Dict[str, Any]:
        """Generate modifiable VRPBTW instance attributes from task_id.

        Args:
            task_id: Task ID in format "{difficulty}_N{customers}_F{fleets}_{distribution}"

        Returns:
            Dict with modifiable attributes (depot, customers, n_fleets).
            Static attributes (capacities, speeds, etc.) are already on self.

        Uses global numpy RNG state set by seed_everything().
        """
        n_customers, n_fleets, dist_type = self._parse_task_id(task_id)

        # Generate coordinates using global numpy RNG (seeded by seed_everything())
        coords = self._generate_coords(n_customers, dist_type)
        depot_coord = [self.max_coord / 2.0, self.max_coord / 2.0]

        # Generate customers with time windows and demands
        customers = []
        linehaul_count = int(n_customers * 0.5)
        types = ["LINEHAUL"] * linehaul_count + ["BACKHAUL"] * (
            n_customers - linehaul_count
        )
        np.random.shuffle(types)

        for i in range(n_customers):
            node_type = types[i]
            demand_range = (
                self.demand_range_linehaul
                if node_type == "LINEHAUL"
                else self.demand_range_backhaul
            )

            dist_km = np.linalg.norm(np.array(coords[i]) - np.array(depot_coord))
            min_reach_time = dist_km / self.v_t
            ready_h = float(np.random.uniform(min_reach_time * 1.1, self.T_max * 0.7))
            width_h = float(
                np.random.uniform(
                    1.0,
                    1.0
                    + (self.time_window_scaling_factor * (dist_km / self.max_coord)),
                )
            )

            demand = float(int(np.random.randint(demand_range[0], demand_range[1] + 1)))
            if node_type == "BACKHAUL":
                demand = -demand
            customers.append(
                [
                    float(coords[i][0]),
                    float(coords[i][1]),
                    float(round(ready_h, 4)),
                    float(round(min(ready_h + width_h, self.T_max), 4)),
                    demand,
                ]
            )

        return {
            "depot": depot_coord,
            "customers": customers,
            "n_fleets": n_fleets,
        }

    # ------------------------------------------------------------------
    # encode_instance
    # ------------------------------------------------------------------

    def encode_instance(self, raw_instance: Dict) -> None:
        depot = np.array(raw_instance["depot"], dtype=np.float32)
        customers = np.array(raw_instance["customers"], dtype=np.float32)

        self.n_customers = len(customers)
        self.K = int(raw_instance["n_fleets"])
        self.n_fleets = self.K

        coords_all = np.vstack([depot, customers[:, :2]])
        tw_open_all = np.concatenate([[0.0], customers[:, 2]])
        tw_close_all = np.concatenate([[self.T_max], customers[:, 3]])
        demands_all = np.concatenate([[0.0], customers[:, 4]])
        svc_all = np.full(self.n_customers + 1, self.service_time, dtype=np.float32)
        svc_all[DEPOT] = 0.0

        self.coords = coords_all.astype(np.float32)
        self.tw_open = tw_open_all.astype(np.float32)
        self.tw_close = tw_close_all.astype(np.float32)
        self.demands = demands_all.astype(np.float32)
        self.service_times = svc_all

        dx = self.coords[:, None, 0] - self.coords[None, :, 0]
        dy = self.coords[:, None, 1] - self.coords[None, :, 1]
        self.euclidean_dist = np.sqrt(dx**2 + dy**2).astype(np.float32)
        self.manhattan_dist = (np.abs(dx) + np.abs(dy)).astype(np.float32)

        cust_d = self.demands[1:]
        self._linehaul_idx = (np.where(cust_d > 0)[0] + 1).astype(np.int32)
        self._backhaul_idx = (np.where(cust_d < 0)[0] + 1).astype(np.int32)

    # ------------------------------------------------------------------
    # reset override — computes nadir internally before returning state
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Objective functions
    # ------------------------------------------------------------------

    def _compute_objective(self, cost: float, served_count: int) -> float:
        """
        New lexicographic objective: minimize cost, prioritize serving customers.
        f = total_cost + c_t * max_dist * N * (N - k)

        Where:
        - total_cost: total travel cost
        - c_t: truck cost unit
        - max_dist: maximum distance (2 * max_coord)
        - N: total customers
        - k: served customers
        - (N-k): unserved customers

        This ensures k+1 served customers always beats k served customers,
        even with worst-case additional distance.
        """
        N = self.n_customers
        k = served_count
        max_dist = 2.0 * self.max_coord
        unserved_penalty = self.c_t * max_dist * N * (N - k)
        return cost + unserved_penalty

    def compute_return(self) -> float:
        """Compute return from current solution: -objective / (max_dist * max_cost_unit).

        Gets solution from current state, extracts objective, negates it to convert
        from cost minimization to return maximization, and normalizes by both spatial
        and cost scales for scale-independence across instances.

        Returns
        -------
        float
            Normalized return in range [-(N), ~0]:
            - Worst (no customers served): ~-(N) where N = n_customers
            - Best (all served, minimal cost): ~-ε (negative but small)
        """
        solution = self.current_solution()
        normalized_factor = (
            2.0 * self.max_coord * max(self.c_t, self.c_d) * self.n_customers
        )
        return -solution.objective / normalized_factor

    # ------------------------------------------------------------------
    # initial_state
    # ------------------------------------------------------------------

    def initial_state(self) -> VRPBTWState:
        K = self.K
        served = np.zeros(self.n_customers + 1, dtype=bool)
        served[DEPOT] = True
        return VRPBTWState(
            truck_node=np.zeros(K, dtype=np.int32),
            truck_prev_node=np.zeros(K, dtype=np.int32),
            truck_phase=np.zeros(K, dtype=np.int32),
            drone_node=np.zeros(K, dtype=np.int32),
            drone_active=np.zeros(K, dtype=bool),
            drone_launch_node=np.zeros(K, dtype=np.int32),
            drone_phase=np.zeros(K, dtype=np.int32),
            served=served,
            current_cost=0.0,
            current_truck_load=np.stack(
                [
                    np.full(K, self.Q_t, dtype=np.float32),
                    np.full(K, -self.Q_t, dtype=np.float32),
                ],
                axis=1,
            ),
            current_drone_load=np.stack(
                [
                    np.full(K, self.Q_d, dtype=np.float32),
                    np.full(K, -self.Q_d, dtype=np.float32),
                ],
                axis=1,
            ),
            truck_load=[[0.0] for _ in range(K)],
            drone_load=[[] for _ in range(K)],
            truck_routes=[[DEPOT] for _ in range(K)],
            drone_trips_node=[[] for _ in range(K)],
            drone_trips_mask=[[] for _ in range(K)],
            truck_arrive=[[0.0] for _ in range(K)],
            truck_depart=[[0.0] for _ in range(K)],
            drone_arrive=[[] for _ in range(K)],
            drone_depart=[[] for _ in range(K)],
        )

    # ------------------------------------------------------------------
    # Time accessors (derived from time arrays)
    # ------------------------------------------------------------------

    def _truck_current_time(self, state: VRPBTWState, k: int) -> float:
        """Get truck k's current time (departure from last node)."""
        return state.truck_depart[k][-1] if state.truck_depart[k] else 0.0

    def _drone_current_time(self, state: VRPBTWState, k: int) -> float:
        """Get drone k's current time (last recorded time in current trip)."""
        if (
            state.drone_active[k]
            and state.drone_depart[k]
            and state.drone_depart[k][-1]
        ):
            return state.drone_depart[k][-1][-1]
        return 0.0

    def _drone_launch_time(self, state: VRPBTWState, k: int) -> float:
        """Get drone k's launch time (departure from launch node in current trip)."""
        if (
            state.drone_active[k]
            and state.drone_depart[k]
            and state.drone_depart[k][-1]
        ):
            return state.drone_depart[k][-1][0]
        return 0.0

    # ------------------------------------------------------------------
    # Update current load bounds (policy inputs)
    # ------------------------------------------------------------------

    def _update_current_loads(self, state: VRPBTWState, k: int) -> None:
        """Recompute current_truck_load and current_drone_load from truck_load / drone_load."""
        # Truck
        loads = state.truck_load[k]
        if loads:
            state.current_truck_load[k][0] = self.Q_t - max(loads)  # remaining linehaul
            state.current_truck_load[k][1] = -(
                self.Q_t - loads[-1]
            )  # remaining backhaul (negative)

        # Drone (inactive = full capacity)
        if state.drone_active[k] and state.drone_load[k] and state.drone_load[k][-1]:
            dloads = state.drone_load[k][-1]
            state.current_drone_load[k][0] = self.Q_d - max(dloads)
            state.current_drone_load[k][1] = -(self.Q_d - dloads[-1])
        else:
            state.current_drone_load[k][0] = self.Q_d
            state.current_drone_load[k][1] = -self.Q_d

    # ------------------------------------------------------------------
    # Action encoding / decoding
    # ------------------------------------------------------------------

    def encode_action(self, node: int, vehicle_idx: int) -> int:
        return node * (2 * self.K) + vehicle_idx

    def decode_action(self, action: int) -> Tuple[int, int]:
        return action // (2 * self.K), action % (2 * self.K)

    def vehicle_fleet_type(self, vehicle_idx: int) -> Tuple[int, int]:
        if vehicle_idx < self.K:
            return vehicle_idx, TRUCK
        return vehicle_idx - self.K, DRONE

    # ------------------------------------------------------------------
    # Action mask
    # ------------------------------------------------------------------

    def get_action_mask(self, state: VRPBTWState) -> ActionMask:
        N1 = self.n_customers + 1
        V = 2 * self.K
        mask = np.zeros(N1 * V, dtype=bool)

        # Check if there are any feasible serving nodes across all vehicles
        has_feasible_serving = False

        for v_idx in range(V):
            k, vtype = self.vehicle_fleet_type(v_idx)
            if vtype == TRUCK:
                for j in range(1, N1):
                    if self._truck_feasible(state, k, j):
                        mask[self.encode_action(j, v_idx)] = True
                        has_feasible_serving = True
            else:
                if state.drone_active[k]:
                    # Check extending to unserved customers
                    for j in range(1, N1):
                        if self._drone_extend_feasible(state, k, j):
                            mask[self.encode_action(j, v_idx)] = True
                            has_feasible_serving = True
                    # Check landing at nodes on truck route
                    for land_idx in self._landing_nodes(state, k):
                        if self._drone_land_feasible(state, k, land_idx):
                            land_node = state.truck_routes[k][land_idx]
                            mask[self.encode_action(land_node, v_idx)] = True
                            has_feasible_serving = True
                else:
                    for j in range(1, N1):
                        if self._drone_launch_feasible(state, k, j):
                            mask[self.encode_action(j, v_idx)] = True
                            has_feasible_serving = True

        # Only allow return-to-depot if there are NO feasible serving nodes
        if not has_feasible_serving:
            for v_idx in range(V):
                k, vtype = self.vehicle_fleet_type(v_idx)
                if vtype == TRUCK and self._truck_return_feasible(state, k):
                    mask[self.encode_action(DEPOT, v_idx)] = True

        return ActionMask.from_bool_array(mask)

    # ------------------------------------------------------------------
    # Feasibility helpers
    # ------------------------------------------------------------------

    def _landing_nodes(self, state: VRPBTWState, k: int) -> List[int]:
        """
        Return indices into truck_routes[k] where drone can land.
        Drone can land from launch_node (exclusive) to current_truck_node (inclusive).
        """
        launch_node = int(state.drone_launch_node[k])
        route = state.truck_routes[k]

        # Find the first position of the launch node in the truck route (where drone was launched)
        launch_idx = -1
        for i in range(len(route)):
            if route[i] == launch_node:
                launch_idx = i
                break

        # Return indices of all nodes from launch_idx+1 to end (inclusive)
        return list(range(launch_idx + 1, len(route)))

    def _has_feasible_landing_after_extend(
        self, state: VRPBTWState, k: int, j: int
    ) -> bool:
        """
        Check if drone can land somewhere after extending to node j.
        Returns True if at least one feasible landing node exists.
        """
        from_node = int(state.drone_node[k])
        landing_indices = self._landing_nodes(state, k)

        for land_idx in landing_indices:
            land_node = state.truck_routes[k][land_idx]
            t_to_j = self.euclidean_dist[from_node, j] / self.v_d
            t_j_to_land = self.euclidean_dist[j, land_node] / self.v_d

            # Calculate drone landing time
            arrive_j = (
                self._drone_current_time(state, k)
                + self.launch_time
                + t_to_j
                + self.land_time
            )
            service_end = max(arrive_j, self.tw_open[j]) + self.service_times[j]
            trip_start = state.drone_depart[k][-1][0]
            trip_end = service_end + self.launch_time + t_j_to_land + self.land_time

            if trip_end - trip_start > self.t_max:
                continue

            # Check if truck can still service subsequent nodes
            if self._late_land_feasible(state, k, land_idx, trip_end):
                return True

        return False

    def _late_land_feasible(
        self, state: VRPBTWState, k: int, land_idx: int, drone_land_time: float
    ) -> bool:
        """
        If drone lands at truck_routes[k][land_idx] at drone_land_time (possibly late),
        check if truck can still service all subsequent nodes without missing customers.

        Simulates truck's remaining route from landing node with new ready time.
        """
        route = state.truck_routes[k]
        land_node = route[land_idx]
        t = drone_land_time
        node = land_node

        # Simulate remaining route from landing node onwards
        for i in range(land_idx + 1, len(route)):
            next_node = route[i]
            dist = self.manhattan_dist[node, next_node]
            arrive = t + dist / self.v_t

            # Check time window feasibility for next_node
            if arrive > self.tw_close[next_node]:
                return False  # Can't reach next_node on time

            service_start = max(arrive, self.tw_open[next_node])
            t = service_start + self.service_times[next_node]
            node = next_node

        # Check return to depot is feasible
        dist_to_depot = self.manhattan_dist[node, DEPOT]
        return t + dist_to_depot / self.v_t <= self.T_max

    def _truck_phase_ok(self, state: VRPBTWState, k: int, j: int) -> bool:
        """Check if truck can serve node j based on truck's current phase."""
        phase = int(state.truck_phase[k])
        demand = self.demands[j]
        if phase == 0 and demand < 0:
            return False
        if phase == 1 and demand > 0:
            return False
        return True

    def _drone_phase_ok(self, state: VRPBTWState, k: int, j: int) -> bool:
        """Check if drone can serve node j based on drone's locked launch phase."""
        phase = int(state.drone_phase[k])
        demand = self.demands[j]
        if phase == 0 and demand < 0:
            return False
        if phase == 1 and demand > 0:
            return False
        return True

    def _truck_feasible(self, state: VRPBTWState, k: int, j: int) -> bool:
        if j == DEPOT or state.served[j]:
            return False
        # Prevent serving after returning to depot: if truck is at DEPOT and route has more than initial DEPOT,
        # the truck has already returned and cannot serve more
        if int(state.truck_node[k]) == DEPOT and len(state.truck_routes[k]) > 1:
            return False
        if not self._truck_phase_ok(state, k, j):
            return False
        # Check capacity using current load bounds
        demand = self.demands[j]
        if demand > 0:  # linehaul
            if demand > state.current_truck_load[k][0]:
                return False
        else:  # backhaul
            if demand < state.current_truck_load[k][1]:
                return False
        from_node = int(state.truck_node[k])
        arrive = (
            self._truck_current_time(state, k)
            + self.manhattan_dist[from_node, j] / self.v_t
        )
        service_end = max(arrive, self.tw_open[j]) + self.service_times[j]
        if service_end > self.tw_close[j]:
            return False
        return service_end + self.manhattan_dist[j, DEPOT] / self.v_t <= self.T_max

    def _truck_return_feasible(self, state: VRPBTWState, k: int) -> bool:
        from_node = int(state.truck_node[k])
        if from_node == DEPOT:
            return False
        arrive = (
            self._truck_current_time(state, k)
            + self.manhattan_dist[from_node, DEPOT] / self.v_t
        )
        return arrive <= self.T_max

    def _elapsed_trip_time(self, state: VRPBTWState, k: int) -> float:
        return float(
            self._drone_current_time(state, k) - self._drone_launch_time(state, k)
        )

    def _min_return_time(self, state: VRPBTWState, k: int, from_node: int) -> float:
        """Minimum flight time from from_node to any unserved node.

        Returns the minimum time for drone to reach a landing node (where truck can be).
        If no feasible landing node exists, returns infinity.
        """
        unserved_nodes = ~state.served
        reachable_nodes = np.where(unserved_nodes)[0]
        if len(reachable_nodes) == 0:
            return float("inf")
        distances = self.euclidean_dist[from_node, reachable_nodes]
        return float(np.min(distances)) / self.v_d

    def _drone_launch_feasible(self, state: VRPBTWState, k: int, j: int) -> bool:
        """Check if drone k can launch to serve node j.

        Performs early checks, then uses _compute_drone_launch_time for constraint checking.
        """
        # Early feasibility checks
        if state.drone_active[k] or state.served[j]:
            return False
        if not self._truck_phase_ok(state, k, j):
            return False

        # Check capacity using current load bounds
        demand = self.demands[j]
        if demand > 0:  # linehaul
            if demand > state.current_drone_load[k][0]:
                return False
        else:  # backhaul
            if demand < state.current_drone_load[k][1]:
                return False

        # Drone must be at current truck location or depot
        drone_at = int(state.drone_node[k])
        land_node = int(state.truck_node[k])
        if drone_at != DEPOT and drone_at != land_node:
            return False

        # Prevent duplicate trips: don't launch if a trip with same (launch, land) already exists
        launch_node = int(state.truck_prev_node[k])
        for existing_trip in state.drone_trips_node[k]:
            if (
                existing_trip
                and int(existing_trip[0]) == launch_node
                and int(existing_trip[-1]) == land_node
            ):
                return False

        # Detailed constraint checking using launch time computation
        launch_time = self._compute_trip_start(state, k, j)
        return launch_time > float("-inf")

    def _drone_extend_feasible(self, state: VRPBTWState, k: int, j: int) -> bool:
        if not state.drone_active[k] or state.served[j]:
            return False
        if not self._drone_phase_ok(state, k, j):
            return False
        # Check capacity using current load bounds
        demand = self.demands[j]
        if demand > 0:  # linehaul
            if demand > state.current_drone_load[k][0]:
                return False
        else:  # backhaul
            if demand < state.current_drone_load[k][1]:
                return False
        from_node = int(state.drone_node[k])
        t_to_j = self.euclidean_dist[from_node, j] / self.v_d
        arrive_j = (
            self._drone_current_time(state, k)
            + self.launch_time
            + t_to_j
            + self.land_time
        )
        service_end = max(arrive_j, self.tw_open[j]) + self.service_times[j]
        if service_end > self.tw_close[j]:
            return False
        t_back = self._min_return_time(state, k, j)

        trip_start = state.drone_depart[k][-1][0]
        trip_end = service_end + self.launch_time + t_back + self.land_time
        if trip_end - trip_start > self.t_max:
            return False

        # Check if drone can still land somewhere after extending to j
        return self._has_feasible_landing_after_extend(state, k, j)

    def _drone_land_feasible(self, state: VRPBTWState, k: int, land_idx: int) -> bool:
        """Check if drone can land at truck_routes[k][land_idx] without losing customers.

        Args:
            land_idx: index into truck_routes[k]
        """
        if not state.drone_active[k]:
            return False

        land_node = state.truck_routes[k][land_idx]
        from_node = int(state.drone_node[k])
        t_back = self.euclidean_dist[from_node, land_node] / self.v_d

        trip_start = state.drone_depart[k][-1][0]
        trip_end = (
            self._drone_current_time(state, k)
            + self.launch_time
            + t_back
            + self.land_time
        )

        if trip_end - trip_start > self.t_max:
            return False
        if trip_end > self.T_max:
            return False

        # Check if late landing would cause truck to miss future customers
        return self._late_land_feasible(state, k, land_idx, trip_end)

    def _compute_max_truck_delay(
        self, state: VRPBTWState, k: int, land_idx: int
    ) -> float:
        """
        Compute maximum time truck can wait at landing node without pushing any subsequent service end above tw_close.

        Truck waits at land_idx (its current position) for drone to land. This delay propagates
        to all subsequent nodes, so we must find the tightest constraint.

        Args:
            state: current state
            k: fleet index
            land_idx: index of landing node in truck_routes[k] (usually len(routes)-1)

        Returns:
            Maximum delay in seconds; inf if no constraint, negative if impossible.
        """
        route = state.truck_routes[k]
        max_delay = float("inf")

        # Simulate truck's remaining route from land_idx onwards
        truck_time = state.truck_depart[k][land_idx]
        current_node = route[land_idx]

        for i in range(land_idx + 1, len(route)):
            next_node = route[i]
            dist = self.manhattan_dist[current_node, next_node]
            travel_time = dist / self.v_t
            arrive_next = truck_time + travel_time
            service_end_next = (
                max(arrive_next, self.tw_open[next_node])
                + self.service_times[next_node]
            )

            # If truck delays by delta at land_idx, service_end_next becomes:
            # service_end_next + delta must not exceed tw_close[next_node]
            delay_slack = self.tw_close[next_node] - service_end_next
            max_delay = min(max_delay, delay_slack)

            truck_time = service_end_next
            current_node = next_node

        # Also check return to depot
        dist_to_depot = self.manhattan_dist[current_node, DEPOT]
        travel_time = dist_to_depot / self.v_t
        arrive_depot = truck_time + travel_time
        delay_slack = self.T_max - arrive_depot
        max_delay = min(max_delay, delay_slack)

        return max_delay

    def _compute_trip_start(self, state: VRPBTWState, k: int, j: int) -> float:
        """
        Compute optimal drone launch time for serving node j.

        Tries three cases in preference order:
        1. Launch at truck_depart[launch_idx] (no truck delay, preferred)
        2. Launch at truck_arrive[launch_idx] (earliest, no delay, alternative)
        3. Delayed launch where truck waits at launch_node (last resort)

        For each case, checks:
        - Drone can reach j within time window
        - Drone can return and land before truck departs
        - Trip duration fits within t_max
        - Truck can service subsequent nodes after late landing

        Returns:
            Launch time (seconds from episode start) if feasible, -inf otherwise.
        """
        launch_node = int(state.truck_prev_node[k])
        land_node = int(state.truck_node[k])
        launch_idx = state.truck_routes[k].index(launch_node)
        land_idx = len(state.truck_routes[k]) - 1

        earliest_launch = state.truck_arrive[k][launch_idx]
        original_truck_depart = state.truck_depart[k][land_idx]

        # Flight and service times
        t_to_j = self.euclidean_dist[launch_node, j] / self.v_d
        t_back = self.euclidean_dist[j, land_node] / self.v_d
        service_time_j = self.service_times[j]

        # Helper: check if drone can serve j and land given a depart_t and truck_land_depart
        def is_feasible_at(depart_t: float, truck_land_depart_val: float) -> bool:
            arrive_j = depart_t + self.launch_time + t_to_j + self.land_time

            # Time window at j
            if arrive_j > self.tw_close[j]:
                return False

            serve_start = max(arrive_j, self.tw_open[j])
            service_end = serve_start + service_time_j

            # Trip duration and end time
            trip_start = depart_t
            trip_end = service_end + self.launch_time + t_back + self.land_time
            trip_duration = trip_end - trip_start

            if trip_duration > self.t_max or trip_end > self.T_max:
                return False

            # Must land before truck departs
            if trip_end > truck_land_depart_val:
                return False

            # Check if truck can service subsequent nodes after late landing
            return self._late_land_feasible(state, k, land_idx, trip_end)

        # Case 1: Launch at truck_depart[launch_idx] (no truck delay)
        launch_time_case1 = state.truck_depart[k][launch_idx]
        if is_feasible_at(launch_time_case1, original_truck_depart):
            return launch_time_case1

        # Case 2: Launch at truck_arrive[launch_idx] (earliest, still no delay)
        launch_time_case2 = earliest_launch
        if launch_time_case2 < launch_time_case1:
            if is_feasible_at(launch_time_case2, original_truck_depart):
                return launch_time_case2

        # Case 3: Delayed launch (truck waits at landing_node for drone to land)
        max_truck_delay = self._compute_max_truck_delay(state, k, land_idx)
        if max_truck_delay > 0:
            # Truck can delay by up to max_truck_delay
            delayed_truck_depart = original_truck_depart + max_truck_delay

            # Find latest feasible depart_t within delayed window
            # depart_t + launch_time + t_to_j + land_time + service + launch_time + t_back + land_time <= delayed_truck_depart
            # depart_t <= delayed_truck_depart - 2*launch_time - t_to_j - t_back - 2*land_time - service
            latest_launch_for_delay = (
                delayed_truck_depart
                - 2 * self.launch_time
                - t_to_j
                - service_time_j
                - t_back
                - 2 * self.land_time
            )

            # Constrain to be >= earliest_launch
            candidate_launch_case3 = max(earliest_launch, latest_launch_for_delay)
            if candidate_launch_case3 <= delayed_truck_depart:
                if is_feasible_at(candidate_launch_case3, delayed_truck_depart):
                    return candidate_launch_case3

        return float("-inf")

    # ------------------------------------------------------------------
    # apply_action
    # ------------------------------------------------------------------

    def apply_action(self, state: VRPBTWState, action: int) -> StepResult:
        node, v_idx = self.decode_action(action)
        k, vtype = self.vehicle_fleet_type(v_idx)

        # Compute objective before action for potential-based shaping
        prev_served = int(state.served[1:].sum())
        prev_obj = self._compute_objective(state.current_cost, prev_served)

        state = _copy_state(state)

        # Apply action (assumed feasible — policy ensures this)
        if vtype == TRUCK:
            self._apply_truck(state, k, node)
        else:  # DRONE
            if state.drone_active[k]:
                # Landing if node is on truck's route, else extend
                if node in state.truck_routes[k]:
                    self._apply_drone_land(state, k, node)
                else:
                    self._apply_drone_extend(state, k, node)
            else:
                self._apply_drone_launch(state, k, node)

        terminated = self._is_terminated(state)

        # Compute reward: potential-based shaping = -(f_next - f_prev)
        curr_served = int(state.served[1:].sum())
        curr_obj = self._compute_objective(state.current_cost, curr_served)
        normalized_factor = (
            2.0 * self.max_coord * max(self.c_t, self.c_d) * self.n_customers
        )

        reward = -(curr_obj - prev_obj) / normalized_factor

        next_mask = (
            self.get_action_mask(state)
            if not terminated
            else ActionMask.all_valid(self.action_space_size)
        )

        return StepResult(
            next_state=state,
            terminated=terminated,
            truncated=False,
            action_mask=next_mask,
            reward=reward,
            info={
                "node": node,
                "fleet": k,
                "vehicle": "truck" if vtype == TRUCK else "drone",
                "served_count": curr_served,
                "current_cost": state.current_cost,
            },
        )

    # ------------------------------------------------------------------
    # Transition helpers  (update state in-place, no return value)
    # ------------------------------------------------------------------

    def _apply_truck(self, state: VRPBTWState, k: int, j: int) -> None:
        from_node = int(state.truck_node[k])
        dist = self.manhattan_dist[from_node, j]
        arrive = self._truck_current_time(state, k) + dist / self.v_t
        serve_start = max(arrive, self.tw_open[j]) if j != DEPOT else arrive
        tardiness = max(arrive - self.tw_close[j], 0.0) if j != DEPOT else 0.0

        service_end = serve_start + self.service_times[j]
        state.truck_prev_node[k] = from_node
        state.truck_node[k] = j
        state.truck_routes[k].append(j)

        # Track arrival and departure times at this node
        state.truck_arrive[k].append(arrive)
        state.truck_depart[k].append(service_end)

        if j != DEPOT:
            state.served[j] = True
            self._update_truck_phase(state, k)

        state.current_cost += self.c_t * dist

        # Update load arrays
        if j != DEPOT:
            demand = self.demands[j]
            state.truck_load[k].append(state.truck_load[k][-1])  # inherit previous load
            if demand > 0:  # linehaul: all previous positions carry j's goods
                for i in range(len(state.truck_load[k]) - 1):
                    state.truck_load[k][i] += demand
            else:  # backhaul: only the new position accumulates pickup
                state.truck_load[k][-1] -= demand  # -= negative = +=
        self._update_current_loads(state, k)

    def _apply_drone_launch(self, state: VRPBTWState, k: int, j: int) -> None:
        launch_node = int(state.truck_prev_node[k])
        land_node = int(state.truck_node[k])
        state.drone_launch_node[k] = launch_node

        # Compute optimal launch time using preference-order logic
        depart_t = self._compute_trip_start(state, k, j)
        assert depart_t > float("-inf"), (
            f"Launch should be feasible (policy guarantees)"
        )

        launch_idx = state.truck_routes[k].index(launch_node)
        land_idx = len(state.truck_routes[k]) - 1

        # Compute arrival and service times based on depart_t
        t_to_j = (
            self.launch_time
            + (self.euclidean_dist[launch_node, j] / self.v_d)
            + self.land_time
        )
        arrive_j = depart_t + t_to_j
        serve_start = max(arrive_j, self.tw_open[j])
        service_end = serve_start + self.service_times[j]
        dist = self.euclidean_dist[launch_node, j]

        state.drone_node[k] = j
        state.drone_active[k] = True
        state.drone_phase[k] = int(state.truck_phase[k])
        state.served[j] = True

        state.current_cost += self.c_d * dist

        # Start new trip with launch_node and j
        state.drone_trips_node[k].append([launch_node, j])
        state.drone_trips_mask[k].append([0, 1])
        state.drone_load[k].append([0.0])  # launch_node starts at 0
        state.drone_load[k][-1].append(
            state.drone_load[k][-1][-1]
        )  # j inherits launch_node's load (0)

        # Track drone times aligned with drone_trips_node[k][-1]
        # [launch_node, j] -> arrivals=[0.0, arrive_j], departures=[depart_t, service_end]
        state.drone_arrive[k].append(
            [0.0, arrive_j]
        )  # 0.0 for launch_node (no arrival), arrive_j for j
        state.drone_depart[k].append(
            [depart_t, service_end]
        )  # depart_t from launch_node, service_end from j

        # Update loads based on demand type
        demand = self.demands[j]
        if demand > 0:  # linehaul
            for i in range(launch_idx + 1):  # truck positions up to launch_node
                state.truck_load[k][i] += demand
            for i in range(
                len(state.drone_load[k][-1]) - 1
            ):  # all trip positions except j
                state.drone_load[k][-1][i] += demand
        else:  # backhaul
            state.drone_load[k][-1][-1] -= demand  # j accumulates pickup

        self._update_current_loads(state, k)

    def _apply_drone_extend(self, state: VRPBTWState, k: int, j: int) -> None:
        from_node = int(state.drone_node[k])
        dist = self.euclidean_dist[from_node, j]
        arrive_j = (
            self._drone_current_time(state, k)
            + self.launch_time
            + (dist / self.v_d)
            + self.land_time
        )
        serve_start = max(arrive_j, self.tw_open[j])
        tardiness = max(arrive_j - self.tw_close[j], 0.0)
        service_end = serve_start + self.service_times[j]

        state.drone_node[k] = j
        state.served[j] = True

        # Extend trip with new customer node - keep arrays aligned with drone_trips_node[k][-1]
        state.drone_arrive[k][-1].append(arrive_j)
        state.drone_depart[k][-1].append(service_end)

        state.current_cost += self.c_d * dist

        # Extend current trip
        state.drone_trips_node[k][-1].append(j)
        state.drone_trips_mask[k][-1].append(1)
        state.drone_load[k][-1].append(
            state.drone_load[k][-1][-1]
        )  # j inherits previous load

        # Update loads based on demand type
        demand = self.demands[j]
        if demand > 0:  # linehaul
            launch_node_val = int(state.drone_launch_node[k])
            launch_idx = state.truck_routes[k].index(launch_node_val)
            for i in range(launch_idx + 1):  # truck up to launch_node
                state.truck_load[k][i] += demand
            for i in range(
                len(state.drone_load[k][-1]) - 1
            ):  # all trip positions except j
                state.drone_load[k][-1][i] += demand
        else:  # backhaul
            state.drone_load[k][-1][-1] -= demand  # j accumulates pickup

        self._update_current_loads(state, k)

    def _apply_drone_land(self, state: VRPBTWState, k: int, land: int) -> None:
        from_node = int(state.drone_node[k])
        dist = self.euclidean_dist[from_node, land]
        arrive = (
            self._drone_current_time(state, k)
            + self.launch_time
            + (dist / self.v_d)
            + self.land_time
        )

        state.drone_node[k] = land
        state.drone_active[k] = False
        self._update_drone_phase(state, k)

        # Landing has no tardiness — cost only
        state.current_cost += self.c_d * dist

        # Close current trip
        state.drone_trips_node[k][-1].append(land)
        state.drone_trips_mask[k][-1].append(0)

        # Track arrival/departure at landing node - keep arrays aligned with drone_trips_node[k][-1]
        state.drone_arrive[k][-1].append(arrive)  # arrival at landing node
        state.drone_depart[k][-1].append(
            0.0
        )  # 0.0 for departure (unset - only set if landing node becomes launch node of next trip)

        # Update truck departure times based on drone trip completion
        # At launch node: truck depart time should be >= drone depart time at launch node
        launch_node = int(state.drone_trips_node[k][-1][0])
        launch_idx = state.truck_routes[k].index(launch_node)
        drone_depart_at_launch = state.drone_depart[k][-1][0]
        state.truck_depart[k][launch_idx] = max(state.truck_depart[k][launch_idx], drone_depart_at_launch)

        # At land node: truck depart time should be >= drone arrive time at land node
        land_idx = state.truck_routes[k].index(land)
        state.truck_depart[k][land_idx] = max(state.truck_depart[k][land_idx], arrive)

        # Transfer drone backhaul load to truck positions from landing onwards
        landing_load = state.drone_load[k][-1][-1]  # last drone node's load
        for i in range(land_idx, len(state.truck_load[k])):
            state.truck_load[k][i] += landing_load
        state.drone_load[k][-1].append(0.0)  # close trip

        self._update_current_loads(state, k)

    # ------------------------------------------------------------------
    # Phase update
    # ------------------------------------------------------------------

    def _update_truck_phase(self, state: VRPBTWState, k: int) -> None:
        """Update truck phase: switches to 1 when all feasible nodes are backhaul (if phased=True)."""
        if not self.phased or state.truck_phase[k] == 1:
            return
        # Check if any unserved linehaul customer is still feasible
        for j in self._linehaul_idx:
            if not state.served[j] and self._truck_feasible(state, k, j):
                return  # Still have feasible linehaul, stay in phase 0
        # All feasible customers are backhaul, switch to phase 1
        state.truck_phase[k] = 1

    def _update_drone_phase(self, state: VRPBTWState, k: int) -> None:
        """Update drone phase on landing: sync to truck's current phase (if phased=True)."""
        if self.phased:
            state.drone_phase[k] = int(state.truck_phase[k])

    # ------------------------------------------------------------------
    # Termination
    # ------------------------------------------------------------------

    def _is_terminated(self, state: VRPBTWState) -> bool:
        drones_idle = not state.drone_active.any()
        trucks_at_depot = bool((state.truck_node == DEPOT).all())
        return drones_idle and trucks_at_depot

    # ------------------------------------------------------------------
    # state_to_obs
    # ------------------------------------------------------------------

    """
    All normalisation and static graph construction happens here, once per
    step, before anything enters the network.  The policy network receives
    only plain normalised numpy arrays — no scalar parameters, no raw coords.

    New obs dict keys:
    node_features     (N+1, 5)   normalised  [x, y, demand, tw_open, tw_close]
    vehicle_features  (2K,  5)   normalised  [x, y, rem_load, tw_open, tw_close]
    truck_edge_index  (2, E)     int32        E = (N+1)*N  all ordered pairs i≠j
    truck_edge_attr   (E,  2)    float32      [cost, time]  normalised
    drone_edge_index  (2, E)     int32
    drone_edge_attr   (E,  2)    float32      [cost, time]  normalised

    Normalisation rules
    -------------------
    Universal spatial normalizer: max_dist = 2 * max_coord (longest Manhattan distance)
    Universal time normalizer: T_scale = max_dist / min(truck_speed, drone_speed) (slowest vehicle)
    Universal capacity normalizer: max_capacity = max(Q_t, Q_d)

    x, y          / max_dist
    demand        / max_capacity
    tw_open/close / T_scale
    rem_load      / max_capacity  (both trucks and drones)
    vehicle tw_open, tw_close  / T_scale   (see details below)

    edge cost   = dist * cost_unit / (max_dist * max(c_t, c_d))
                    manhattan dist * c_t  for truck edges
                    euclidean dist * c_d  for drone edges
    edge time   = dist / max_dist / (speed * T_scale)
                    manhattan / max_dist / (v_t * T_scale)  for truck edges
                    euclidean / max_dist / (v_d * T_scale)  for drone edges

    Vehicle feature semantics
    -------------------------
    Truck k:
        tw_open  = truck_time[k] / T_scale   (earliest next departure = now)
        tw_close = T_max / T_scale            (system deadline)

    Drone k  (NOT on trip, drone_active[k] == False):
        tw_open  = truck_time[k] / T_max   (drone boards when truck arrives)
        tw_close = (drone_launch_time[k] + t_max) / T_max

    Drone k  (ON trip, drone_active[k] == True):
        tw_open  = drone_time[k] / T_max   (earliest next departure from current node)
        tw_close = (drone_launch_time[k] + t_max) / T_max

    The static graph is rebuilt from scratch each call using the instance's
    distance matrices, which are already stored on self after encode_instance.
    This is O(N^2) numpy work — fast enough for N<=50.
    """

    def state_to_obs(self, state) -> dict:
        N1 = self.n_customers + 1
        # Universal spatial normalizer: longest possible distance (Manhattan, corner to corner)
        max_dist = (
            2.0 * self.max_coord
        )  # longest distance in grid: (0,0) to (max_coord, max_coord)

        # Normalize time by the longest possible distance / slowest vehicle speed
        # This puts the entire temporal range in [0, 1] and is consistent across instances
        T = max_dist / min(
            self.v_t, self.v_d
        )  # time to traverse longest path at slowest speed

        mc = max(self.c_t, self.c_d) + 1e-8  # max cost unit
        norm_cost_denom = max_dist * mc  # cost normalisation denominator
        max_capacity = max(self.Q_t, self.Q_d) + 1e-8  # universal load normalizer

        # ── Node features ────────────────────────────────────────────────
        # Demand is zeroed for served nodes so the network can distinguish
        # visited (demand=0) from unvisited (demand≠0) without an extra feature.
        # Served linehaul/backhaul nodes also leave the l_idx/b_idx index sets
        # in NodeEncoder, stopping them from influencing routing cross-attention.
        effective_demand = np.where(state.served, 0.0, self.demands).astype(np.float32)
        linehaul_demand = np.where(self.demands > 0, effective_demand, 0.0)
        backhaul_demand = np.where(self.demands < 0, effective_demand, 0.0)
        node_features = np.stack(
            [
                self.coords[:, 0] / max_dist,
                self.coords[:, 1] / max_dist,
                linehaul_demand / max_capacity,
                backhaul_demand / max_capacity,
                self.tw_open / T,
                self.tw_close / T,
            ],
            axis=1,
        ).astype(np.float32)  # (N+1, 6)

        # ── Vehicle features ─────────────────────────────────────────────
        truck_rows = []
        drone_rows = []
        for k in range(self.K):
            # Truck k: [x, y, linehaul_bound, backhaul_bound, time, deadline]
            tx, ty = self.coords[state.truck_node[k]]
            t_lh = float(state.current_truck_load[k][0]) / self.Q_t
            t_bh = float(state.current_truck_load[k][1]) / self.Q_t
            t_open = float(self._truck_current_time(state, k)) / T
            t_close = self.T_max / T  # system deadline normalized by time scale
            truck_rows.append(
                np.array(
                    [tx / max_dist, ty / max_dist, t_lh, t_bh, t_open, t_close],
                    dtype=np.float32,
                )
            )

            # Drone k: [x, y, linehaul_bound, backhaul_bound, time, deadline]
            dx, dy = self.coords[state.drone_node[k]]
            d_lh = float(state.current_drone_load[k][0]) / self.Q_d
            d_bh = float(state.current_drone_load[k][1]) / self.Q_d
            d_close = float(self._drone_launch_time(state, k) + self.t_max) / T
            if state.drone_active[k]:
                d_open = float(self._drone_current_time(state, k)) / T  # already flying
            else:
                d_open = (
                    float(self._truck_current_time(state, k)) / T
                )  # boards when truck arrives
            drone_rows.append(
                np.array(
                    [dx / max_dist, dy / max_dist, d_lh, d_bh, d_open, d_close],
                    dtype=np.float32,
                )
            )

        # Vehicle index order matches encode_action()/vehicle_fleet_type():
        # [truck_0..truck_{K-1}, drone_0..drone_{K-1}]
        vehicle_features = np.stack(truck_rows + drone_rows, axis=0)  # (2K, 6)

        # ── Candidate edges: all ordered (i, j), i != j ─────────────────────
        src, dst = np.meshgrid(np.arange(N1), np.arange(N1), indexing="ij")
        src, dst = src.ravel(), dst.ravel()
        valid = src != dst
        src, dst = src[valid], dst[valid]

        man = self.manhattan_dist[src, dst]
        euc = self.euclidean_dist[src, dst]

        truck_cost = (man * self.c_t) / norm_cost_denom
        truck_time = (man / max_dist) / (self.v_t * T + 1e-8)
        drone_cost = (euc * self.c_d) / norm_cost_denom
        drone_time = (euc / max_dist) / (self.v_d * T + 1e-8)

        truck_edge_attr_all = np.stack([truck_cost, truck_time], axis=1).astype(
            np.float32
        )
        drone_edge_attr_all = np.stack([drone_cost, drone_time], axis=1).astype(
            np.float32
        )

        # ── Sparse edge sets ──────────────────────────────────────────────
        #
        # Truck subgraph — keep endpoints that are routing-relevant for the truck:
        #   • unserved customers        (future truck targets)
        #   • depot                     (always kept — trucks return, drones land)
        #   • current truck node(s)     (planning origin)
        #   • landing candidates        (post-launch truck nodes the active drone
        #                                can rendezvous with; kept so the GNN
        #                                propagates structure around those sites)
        #
        # Drone subgraph — same keep set as truck, plus the current drone node:
        #   • current drone node(s)     (active: extend-trip or land decisions;
        #                                inactive: next-launch origin)
        #
        # Edges where EITHER endpoint falls outside the keep set are dropped.
        # Nodes left with no edges in either subgraph still participate in the
        # attention encoders (NodeEncoder / VehicleEncoder) but receive no GNN
        # message aggregation — their Z_graph embedding is feature-only.

        keep_truck = ~state.served  # unserved customers
        keep_truck[DEPOT] = True
        for k in range(self.K):
            keep_truck[int(state.truck_node[k])] = True  # current truck pos
            if state.drone_active[k]:
                for lc_idx in self._landing_nodes(
                    state, k
                ):  # landing candidates (route indices)
                    lc_node = state.truck_routes[k][lc_idx]
                    keep_truck[lc_node] = True

        keep_drone = keep_truck.copy()
        for k in range(self.K):
            keep_drone[int(state.drone_node[k])] = True  # current drone pos

        t_mask = keep_truck[src] & keep_truck[dst]
        d_mask = keep_drone[src] & keep_drone[dst]

        truck_edge_index = np.stack([src[t_mask], dst[t_mask]], axis=0).astype(
            np.int32
        )  # (2, E_t)
        drone_edge_index = np.stack([src[d_mask], dst[d_mask]], axis=0).astype(
            np.int32
        )  # (2, E_d)

        return {
            "node_features": node_features,  # (N+1, 5)
            "vehicle_features": vehicle_features,  # (2K,  5)
            "truck_edge_index": truck_edge_index,  # (2,   E_t)
            "truck_edge_attr": truck_edge_attr_all[t_mask],  # (E_t, 2)
            "drone_edge_index": drone_edge_index,  # (2,   E_d)
            "drone_edge_attr": drone_edge_attr_all[d_mask],  # (E_d, 2)
        }

    # ------------------------------------------------------------------
    # evaluate
    # ------------------------------------------------------------------

    def evaluate(self, state: VRPBTWState) -> Tuple[float, int]:
        """
        Evaluate solution metrics by replaying stored routes.

        Returns:
            (total_cost, served_count) - raw solution properties
        """
        total_cost = 0.0

        for k in range(self.K):
            t, prev = 0.0, DEPOT
            for j in state.truck_routes[k]:
                dist = self.manhattan_dist[prev, j]
                t += dist / self.v_t
                if j != DEPOT:
                    t = max(t, self.tw_open[j]) + self.service_times[j]
                total_cost += self.c_t * dist
                prev = j

            t_d, prev_d = 0.0, DEPOT
            for trip_nodes, trip_mask in zip(
                state.drone_trips_node[k], state.drone_trips_mask[k]
            ):
                in_trip = False
                for node, is_cust in zip(trip_nodes, trip_mask):
                    dist = self.euclidean_dist[prev_d, node]
                    total_cost += self.c_d * dist
                    if not in_trip and not is_cust:
                        t_d = max(t_d + dist / self.v_d, 0.0)
                    elif not in_trip and is_cust:
                        t_d += self.launch_time + dist / self.v_d
                        t_d = max(t_d, self.tw_open[node]) + self.service_times[node]
                        in_trip = True
                    elif in_trip and is_cust:
                        t_d += dist / self.v_d
                        t_d = max(t_d, self.tw_open[node]) + self.service_times[node]
                    else:
                        t_d += dist / self.v_d + self.land_time
                        in_trip = False
                    prev_d = node

        served_count = int(state.served[1:].sum())
        return total_cost, served_count

    def is_complete(self, state: VRPBTWState) -> bool:
        return self._is_terminated(state)

    def decode_solution(self, state: VRPBTWState) -> Solution:
        cost, served_count = self.evaluate(state)
        obj = self._compute_objective(cost, served_count)
        return Solution(
            problem_name=self.name,
            raw_state=state,
            objective=obj,
            metadata={
                "total_cost": cost,
                "served_count": served_count,
                "n_customers": self.n_customers,
                "truck_routes": [list(r) for r in state.truck_routes],
                "drone_trips_node": [
                    [list(t) for t in trips] for trips in state.drone_trips_node
                ],
                "drone_trips_mask": [
                    [list(m) for m in trips] for trips in state.drone_trips_mask
                ],
                "unserved": int((~state.served[1:]).sum()),
            },
        )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def action_space_size(self) -> int:
        return (self.n_customers + 1) * 2 * self.K

    @property
    def observation_shape(self) -> Tuple[int, ...]:
        return (self.n_customers + 1, NODE_FEAT_DIM)

    @property
    def n_vehicles(self) -> int:
        return 2 * self.K

    @property
    def linehaul_indices(self) -> np.ndarray:
        return self._linehaul_idx

    @property
    def backhaul_indices(self) -> np.ndarray:
        return self._backhaul_idx

    def get_starting_actions(self) -> np.ndarray:
        """
        Return candidate first actions for POMO: linehaul customers with first fleet (truck and drone).

        For backhaul VRP, linehaul customers must be served first, and only the first
        fleet can make the initial dispatch. This includes both truck and drone options
        for each linehaul customer, reducing the candidate action space.

        Vehicle indices for first fleet (fleet 0):
        - Truck: vehicle_idx = 0
        - Drone: vehicle_idx = K

        Returns
        -------
        np.ndarray: Indices of candidate actions (linehaul customers with first fleet vehicles).
        """
        starting_actions = []
        for j in self._linehaul_idx:
            # Truck for first fleet (0)
            starting_actions.append(self.encode_action(int(j), 0))
            # Drone for first fleet (K)
            starting_actions.append(self.encode_action(int(j), self.K))
        return np.array(starting_actions, dtype=np.int64)
