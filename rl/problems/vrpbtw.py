"""
problems/vrpbtw.py
------------------
VRPBTW with heterogeneous truck-drone fleets.

Reward design
-------------
Per-step reward uses potential-based shaping:
    r_t = F(s_{t+1}) - F(s_t)

where F(s) = -(0.5 * f_cost(s) / z_cost + 0.5 * f_maxtard(s) / z_tard)

f_cost and f_maxtard are maintained incrementally in VRPBTWState so no
full evaluate() replay is needed at each step.

z_cost, z_tard are nadir approximations from the heuristic solution,
computed once at episode reset via set_nadir().

At termination:
    feasible   -> r_T = F(s_T) - F(s_{T-1})  (normal shaped step)
    infeasible -> r_T = -2.0                  (Option 1: hard penalty,
                                               always worse than feasible F
                                               which lies in roughly [-1, 0])

Constants
---------
NODE_FEAT_DIM  = 5   [x, y, demand, tw_open, tw_close]
VEH_FEAT_DIM   = 5   [x, y, load, time, deadline]
EDGE_FEAT_DIM  = 6   [vtype, travel_time, dist, depart, arrive, tardiness]
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from core.problem import (
    ActionMask,
    Problem,
    Solution,
    StepResult,
)

# ---------------------------------------------------------------------------
# Feature-dimension constants  (imported by networks/hacn.py)
# ---------------------------------------------------------------------------

NODE_FEAT_DIM = 5
VEH_FEAT_DIM = 5
EDGE_FEAT_DIM = 6

TRUCK = 0
DRONE = 1
DEPOT = 0

_TARD_EPS = 1e-6  # clamp z_tard to avoid division by zero


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------


@dataclass
class VRPBTWState:
    # Per fleet (K,)
    truck_node: np.ndarray
    truck_time: np.ndarray
    truck_load: np.ndarray
    truck_phase: np.ndarray

    drone_node: np.ndarray
    drone_time: np.ndarray
    drone_load: np.ndarray
    drone_launch_time: np.ndarray
    drone_active: np.ndarray

    # Global
    served: np.ndarray  # (N+1,) bool

    # Incremental objective accumulators
    current_cost: float  # total travel cost so far
    current_max_tard: float  # max tardiness across served customers so far

    # Routes (for logging + evaluate())
    truck_routes: List[List[int]]
    drone_route_nodes: List[List[int]]
    drone_route_mask: List[List[int]]

    # Incremental graph (input to vehicle GNN)
    edge_index: np.ndarray  # (2, E)
    edge_attr: np.ndarray  # (E, EDGE_FEAT_DIM)
    edge_fleet: np.ndarray  # (E,)


# ---------------------------------------------------------------------------
# State copy helper
# ---------------------------------------------------------------------------


def _copy_state(s: VRPBTWState) -> VRPBTWState:
    return VRPBTWState(
        truck_node=s.truck_node.copy(),
        truck_time=s.truck_time.copy(),
        truck_load=s.truck_load.copy(),
        truck_phase=s.truck_phase.copy(),
        drone_node=s.drone_node.copy(),
        drone_time=s.drone_time.copy(),
        drone_load=s.drone_load.copy(),
        drone_launch_time=s.drone_launch_time.copy(),
        drone_active=s.drone_active.copy(),
        served=s.served.copy(),
        current_cost=s.current_cost,
        current_max_tard=s.current_max_tard,
        truck_routes=[list(r) for r in s.truck_routes],
        drone_route_nodes=[list(r) for r in s.drone_route_nodes],
        drone_route_mask=[list(m) for m in s.drone_route_mask],
        edge_index=s.edge_index.copy(),
        edge_attr=s.edge_attr.copy(),
        edge_fleet=s.edge_fleet.copy(),
    )


# ---------------------------------------------------------------------------
# VRPBTWProblem
# ---------------------------------------------------------------------------


class VRPBTWProblem(Problem):
    def __init__(self, n_customers: int = 10, n_fleets: int = 2):
        super().__init__(name="VRPBTW")
        self.n_customers = n_customers
        self.n_fleets = n_fleets
        self.K = n_fleets

        self.coords: np.ndarray = np.zeros((1, 2), dtype=np.float32)
        self.demands: np.ndarray = np.zeros(1, dtype=np.float32)
        self.tw_open: np.ndarray = np.zeros(1, dtype=np.float32)
        self.tw_close: np.ndarray = np.zeros(1, dtype=np.float32)
        self.service_times: np.ndarray = np.zeros(1, dtype=np.float32)
        self.manhattan_dist: np.ndarray = np.zeros((1, 1), dtype=np.float32)
        self.euclidean_dist: np.ndarray = np.zeros((1, 1), dtype=np.float32)

        self.Q_t: float = 1.0
        self.Q_d: float = 1.0
        self.T_max: float = 1.0
        self.t_max: float = 1.0
        self.v_t: float = 1.0
        self.v_d: float = 2.0
        self.c_t: float = 1.0
        self.c_d: float = 0.5
        self.launch_time: float = 0.0
        self.land_time: float = 0.0
        self.lambda_weight: float = 0.5

        self._linehaul_idx: np.ndarray = np.array([], dtype=np.int32)
        self._backhaul_idx: np.ndarray = np.array([], dtype=np.int32)

        # Nadir reference values — set at episode reset via set_nadir()
        self._z_cost: float = 1.0
        self._z_tard: float = 1.0

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
        tw_close_all = np.concatenate(
            [[float(raw_instance["system_duration"])], customers[:, 3]]
        )
        demands_all = np.concatenate([[0.0], customers[:, 4]])
        svc = float(raw_instance.get("service_time", 0.0))
        svc_all = np.full(self.n_customers + 1, svc, dtype=np.float32)
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

        self.Q_t = float(raw_instance["truck_capacity"])
        self.Q_d = float(raw_instance["drone_capacity"])
        self.T_max = float(raw_instance["system_duration"])
        self.t_max = float(raw_instance["trip_duration"])
        self.v_t = float(raw_instance["truck_speed"])
        self.v_d = float(raw_instance["drone_speed"])
        self.c_t = float(raw_instance["truck_cost"])
        self.c_d = float(raw_instance["drone_cost"])
        self.launch_time = float(raw_instance.get("launch_time", 0.0))
        self.land_time = float(raw_instance.get("land_time", 0.0))
        self.lambda_weight = float(raw_instance.get("lambda_weight", 0.5))

        cust_d = self.demands[1:]
        self._linehaul_idx = (np.where(cust_d > 0)[0] + 1).astype(np.int32)
        self._backhaul_idx = (np.where(cust_d < 0)[0] + 1).astype(np.int32)

    # ------------------------------------------------------------------
    # reset override — computes nadir internally before returning state
    # ------------------------------------------------------------------

    def reset(self, raw_instance: Any) -> VRPBTWState:
        """
        Encode the instance, compute per-instance nadir approximations
        from the heuristic, then return the initial state.

        Overrides Problem.reset() so that Environment never needs to
        know about objectives, heuristics, or normalisation — those are
        internal concerns of VRPBTWProblem.
        """
        self.encode_instance(raw_instance)
        self._n_steps = 0
        self._set_nadir_from_heuristic()
        return self.initial_state()

    def _set_nadir_from_heuristic(self) -> None:
        """
        Run the nearest-neighbour heuristic and register (z_cost, z_tard)
        as the per-instance nadir approximations for reward normalisation.
        z_tard is clamped to _TARD_EPS to avoid division by zero when the
        heuristic produces zero tardiness.
        """
        z_cost, z_tard = self._heuristic_nadir()
        self._z_cost = max(z_cost, _TARD_EPS)
        self._z_tard = max(z_tard, _TARD_EPS)

    # ------------------------------------------------------------------
    # Potential function
    # ------------------------------------------------------------------

    def _F(self, cost: float, max_tard: float) -> float:
        """
        Scalarised normalised objective.  Always <= 0 for typical solutions.
        F = -(0.5 * cost / z_cost + 0.5 * max_tard / z_tard)
        """
        return -(0.5 * cost / self._z_cost + 0.5 * max_tard / self._z_tard)

    # ------------------------------------------------------------------
    # initial_state
    # ------------------------------------------------------------------

    def initial_state(self) -> VRPBTWState:
        K = self.K
        served = np.zeros(self.n_customers + 1, dtype=bool)
        served[DEPOT] = True
        return VRPBTWState(
            truck_node=np.zeros(K, dtype=np.int32),
            truck_time=np.zeros(K, dtype=np.float32),
            truck_load=np.full(K, self.Q_t, dtype=np.float32),
            truck_phase=np.zeros(K, dtype=np.int32),
            drone_node=np.zeros(K, dtype=np.int32),
            drone_time=np.zeros(K, dtype=np.float32),
            drone_load=np.full(K, self.Q_d, dtype=np.float32),
            drone_launch_time=np.zeros(K, dtype=np.float32),
            drone_active=np.zeros(K, dtype=bool),
            served=served,
            current_cost=0.0,
            current_max_tard=0.0,
            truck_routes=[[] for _ in range(K)],
            drone_route_nodes=[[] for _ in range(K)],
            drone_route_mask=[[] for _ in range(K)],
            edge_index=np.zeros((2, 0), dtype=np.int32),
            edge_attr=np.zeros((0, EDGE_FEAT_DIM), dtype=np.float32),
            edge_fleet=np.zeros(0, dtype=np.int32),
        )

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

        for v_idx in range(V):
            k, vtype = self.vehicle_fleet_type(v_idx)
            if vtype == TRUCK:
                for j in range(1, N1):
                    if self._truck_feasible(state, k, j):
                        mask[self.encode_action(j, v_idx)] = True
                if self._truck_return_feasible(state, k):
                    mask[self.encode_action(DEPOT, v_idx)] = True
            else:
                if state.drone_active[k]:
                    for j in range(1, N1):
                        if self._drone_extend_feasible(state, k, j):
                            mask[self.encode_action(j, v_idx)] = True
                    for land in self._landing_nodes(state, k):
                        if self._drone_land_feasible(state, k, land):
                            mask[self.encode_action(land, v_idx)] = True
                else:
                    for j in range(1, N1):
                        if self._drone_launch_feasible(state, k, j):
                            mask[self.encode_action(j, v_idx)] = True

        return ActionMask.from_bool_array(mask)

    # ------------------------------------------------------------------
    # Feasibility helpers
    # ------------------------------------------------------------------

    def _landing_nodes(self, state: VRPBTWState, k: int) -> List[int]:
        nodes = [DEPOT]
        t_node = int(state.truck_node[k])
        if t_node != DEPOT:
            nodes.append(t_node)
        return nodes

    def _phase_ok(self, state: VRPBTWState, k: int, j: int) -> bool:
        phase = int(state.truck_phase[k])
        demand = self.demands[j]
        if phase == 0 and demand < 0:
            return False
        if phase == 1 and demand > 0:
            return False
        return True

    def _truck_feasible(self, state: VRPBTWState, k: int, j: int) -> bool:
        if j == DEPOT or state.served[j]:
            return False
        if not self._phase_ok(state, k, j):
            return False
        if abs(self.demands[j]) > state.truck_load[k]:
            return False
        from_node = int(state.truck_node[k])
        arrive = state.truck_time[k] + self.manhattan_dist[from_node, j] / self.v_t
        if arrive > self.tw_close[j]:
            return False
        depart = max(arrive, self.tw_open[j]) + self.service_times[j]
        return depart + self.manhattan_dist[j, DEPOT] / self.v_t <= self.T_max

    def _truck_return_feasible(self, state: VRPBTWState, k: int) -> bool:
        from_node = int(state.truck_node[k])
        if from_node == DEPOT:
            return False
        arrive = state.truck_time[k] + self.manhattan_dist[from_node, DEPOT] / self.v_t
        return arrive <= self.T_max

    def _elapsed_trip_time(self, state: VRPBTWState, k: int) -> float:
        return float(state.drone_time[k] - state.drone_launch_time[k])

    def _min_return_time(self, state: VRPBTWState, k: int, from_node: int) -> float:
        times = [self.euclidean_dist[from_node, DEPOT] / self.v_d]
        t_node = int(state.truck_node[k])
        if t_node != DEPOT:
            times.append(self.euclidean_dist[from_node, t_node] / self.v_d)
        return float(min(times))

    def _drone_launch_feasible(self, state: VRPBTWState, k: int, j: int) -> bool:
        if state.drone_active[k] or state.served[j]:
            return False
        if not self._phase_ok(state, k, j):
            return False
        if abs(self.demands[j]) > state.drone_load[k]:
            return False
        drone_at = int(state.drone_node[k])
        truck_at = int(state.truck_node[k])
        if drone_at != DEPOT and drone_at != truck_at:
            return False
        t_out = self.euclidean_dist[drone_at, j] / self.v_d
        t_back = self._min_return_time(state, k, j)
        if self.launch_time + t_out + t_back > self.t_max:
            return False
        return state.drone_time[k] + self.launch_time + t_out <= self.tw_close[j]

    def _drone_extend_feasible(self, state: VRPBTWState, k: int, j: int) -> bool:
        if not state.drone_active[k] or state.served[j]:
            return False
        if not self._phase_ok(state, k, j):
            return False
        if abs(self.demands[j]) > state.drone_load[k]:
            return False
        from_node = int(state.drone_node[k])
        t_to_j = self.euclidean_dist[from_node, j] / self.v_d
        t_back = self._min_return_time(state, k, j)
        elapsed = self._elapsed_trip_time(state, k)
        if elapsed + t_to_j + t_back > self.t_max:
            return False
        return state.drone_time[k] + t_to_j <= self.tw_close[j]

    def _drone_land_feasible(self, state: VRPBTWState, k: int, land: int) -> bool:
        if not state.drone_active[k]:
            return False
        from_node = int(state.drone_node[k])
        t_back = self.euclidean_dist[from_node, land] / self.v_d
        elapsed = self._elapsed_trip_time(state, k)
        if elapsed + t_back > self.t_max:
            return False
        return state.drone_time[k] + t_back + self.land_time <= self.T_max

    # ------------------------------------------------------------------
    # apply_action
    # ------------------------------------------------------------------

    def apply_action(self, state: VRPBTWState, action: int) -> StepResult:
        node, v_idx = self.decode_action(action)
        k, vtype = self.vehicle_fleet_type(v_idx)

        # Snapshot potential before transition
        F_before = self._F(state.current_cost, state.current_max_tard)

        state = _copy_state(state)

        if vtype == TRUCK:
            self._apply_truck(state, k, node)
        else:
            if state.drone_active[k]:
                landing = self._landing_nodes(state, k)
                if node in landing or state.served[node]:
                    self._apply_drone_land(state, k, node)
                else:
                    self._apply_drone_extend(state, k, node)
            else:
                self._apply_drone_launch(state, k, node)

        terminated = self._is_terminated(state)
        infeasible_end = False

        next_mask = (
            self.get_action_mask(state)
            if not terminated
            else ActionMask.all_valid(self.action_space_size)
        )
        if not terminated and next_mask.is_empty():
            terminated = True
            infeasible_end = True

        # ── Reward (Option 1: infeasibility as hard terminal penalty) ──
        if terminated and infeasible_end:
            reward = -2.0
        else:
            reward = self._F(state.current_cost, state.current_max_tard) - F_before

        return StepResult(
            next_state=state,
            reward=reward,
            terminated=terminated,
            truncated=False,
            action_mask=next_mask,
            info={
                "node": node,
                "fleet": k,
                "vehicle": "truck" if vtype == TRUCK else "drone",
                "served_count": int(state.served[1:].sum()),
                "current_cost": state.current_cost,
                "current_max_tard": state.current_max_tard,
            },
        )

    # ------------------------------------------------------------------
    # Transition helpers  (update state in-place, no return value)
    # ------------------------------------------------------------------

    def _apply_truck(self, state: VRPBTWState, k: int, j: int) -> None:
        from_node = int(state.truck_node[k])
        dist = self.manhattan_dist[from_node, j]
        arrive = state.truck_time[k] + dist / self.v_t
        serve_start = max(arrive, self.tw_open[j]) if j != DEPOT else arrive
        tardiness = max(arrive - self.tw_close[j], 0.0) if j != DEPOT else 0.0

        state.truck_time[k] = serve_start + self.service_times[j]
        state.truck_node[k] = j
        state.truck_load[k] -= abs(self.demands[j])
        state.truck_routes[k].append(j)
        if j != DEPOT:
            state.served[j] = True
            self._update_phase(state, k)

        state.current_cost += self.c_t * dist
        if j != DEPOT:
            state.current_max_tard = max(state.current_max_tard, tardiness)

        self._add_edge(
            state,
            from_node,
            j,
            k,
            TRUCK,
            depart_time=arrive,
            arrive_time=arrive,
            tardiness=tardiness,
        )

    def _apply_drone_launch(self, state: VRPBTWState, k: int, j: int) -> None:
        from_node = int(state.drone_node[k])
        dist = self.euclidean_dist[from_node, j]
        depart_t = state.drone_time[k] + self.launch_time
        arrive_j = depart_t + dist / self.v_d
        serve_start = max(arrive_j, self.tw_open[j])
        tardiness = max(arrive_j - self.tw_close[j], 0.0)
        launch_node = int(state.truck_node[k])

        state.drone_launch_time[k] = state.drone_time[k]
        state.drone_time[k] = serve_start + self.service_times[j]
        state.drone_node[k] = j
        state.drone_load[k] -= abs(self.demands[j])
        state.drone_active[k] = True
        state.served[j] = True
        self._update_phase(state, k)

        state.drone_route_nodes[k].extend([launch_node, j])
        state.drone_route_mask[k].extend([0, 1])

        state.current_cost += self.c_d * dist
        state.current_max_tard = max(state.current_max_tard, tardiness)

        self._add_edge(
            state,
            launch_node,
            j,
            k,
            DRONE,
            depart_time=depart_t,
            arrive_time=arrive_j,
            tardiness=tardiness,
        )

    def _apply_drone_extend(self, state: VRPBTWState, k: int, j: int) -> None:
        from_node = int(state.drone_node[k])
        dist = self.euclidean_dist[from_node, j]
        arrive_j = state.drone_time[k] + dist / self.v_d
        serve_start = max(arrive_j, self.tw_open[j])
        tardiness = max(arrive_j - self.tw_close[j], 0.0)

        state.drone_time[k] = serve_start + self.service_times[j]
        state.drone_node[k] = j
        state.drone_load[k] -= abs(self.demands[j])
        state.served[j] = True
        self._update_phase(state, k)

        state.drone_route_nodes[k].append(j)
        state.drone_route_mask[k].append(1)

        state.current_cost += self.c_d * dist
        state.current_max_tard = max(state.current_max_tard, tardiness)

        self._add_edge(
            state,
            from_node,
            j,
            k,
            DRONE,
            depart_time=state.drone_time[k] - self.service_times[j],
            arrive_time=arrive_j,
            tardiness=tardiness,
        )

    def _apply_drone_land(self, state: VRPBTWState, k: int, land: int) -> None:
        from_node = int(state.drone_node[k])
        dist = self.euclidean_dist[from_node, land]
        arrive = state.drone_time[k] + dist / self.v_d + self.land_time
        if land == int(state.truck_node[k]):
            arrive = max(arrive, state.truck_time[k])

        state.drone_time[k] = arrive
        state.drone_node[k] = land
        state.drone_active[k] = False
        state.drone_load[k] = self.Q_d
        state.drone_launch_time[k] = arrive

        state.drone_route_nodes[k].append(land)
        state.drone_route_mask[k].append(0)

        # Landing has no tardiness — cost only
        state.current_cost += self.c_d * dist

        self._add_edge(
            state,
            from_node,
            land,
            k,
            DRONE,
            depart_time=state.drone_time[k] - dist / self.v_d - self.land_time,
            arrive_time=arrive,
            tardiness=0.0,
        )

    # ------------------------------------------------------------------
    # Graph edge builder
    # ------------------------------------------------------------------

    def _add_edge(
        self,
        state: VRPBTWState,
        src: int,
        dst: int,
        fleet: int,
        vtype: int,
        depart_time: float,
        arrive_time: float,
        tardiness: float,
    ) -> None:
        dist = (self.euclidean_dist if vtype == DRONE else self.manhattan_dist)[
            src, dst
        ]
        speed = self.v_d if vtype == DRONE else self.v_t
        T = self.T_max + 1e-6
        feat = np.array(
            [
                float(vtype),
                (dist / speed) / T,
                dist / (speed * T),
                depart_time / T,
                arrive_time / T,
                tardiness / T,
            ],
            dtype=np.float32,
        )
        state.edge_index = np.concatenate(
            [state.edge_index, np.array([[src], [dst]], dtype=np.int32)], axis=1
        )
        state.edge_attr = np.concatenate([state.edge_attr, feat[None]], axis=0)
        state.edge_fleet = np.concatenate([state.edge_fleet, [fleet]], axis=0)

    # ------------------------------------------------------------------
    # Phase update
    # ------------------------------------------------------------------

    def _update_phase(self, state: VRPBTWState, k: int) -> None:
        if (
            state.truck_phase[k] == 0
            and len(self._linehaul_idx) > 0
            and state.served[self._linehaul_idx].all()
        ):
            state.truck_phase[k] = 1

    # ------------------------------------------------------------------
    # Termination
    # ------------------------------------------------------------------

    def _is_terminated(self, state: VRPBTWState) -> bool:
        return bool(state.served[1:].all()) and not state.drone_active.any()

    # ------------------------------------------------------------------
    # state_to_obs
    # ------------------------------------------------------------------

    def state_to_obs(self, state: VRPBTWState) -> Dict[str, np.ndarray]:
        T = self.T_max + 1e-6
        max_coord = float(self.coords.max()) + 1e-6

        node_features = np.stack(
            [
                self.coords[:, 0] / max_coord,
                self.coords[:, 1] / max_coord,
                self.demands / (self.Q_t + 1e-6),
                self.tw_open / T,
                self.tw_close / T,
            ],
            axis=1,
        ).astype(np.float32)

        veh_rows = []
        for k in range(self.K):
            tx, ty = self.coords[state.truck_node[k]]
            veh_rows.append(
                np.array(
                    [
                        tx / max_coord,
                        ty / max_coord,
                        state.truck_load[k] / (self.Q_t + 1e-6),
                        state.truck_time[k] / T,
                        self.T_max / T,
                    ],
                    dtype=np.float32,
                )
            )
            dx, dy = self.coords[state.drone_node[k]]
            tw_unavail = (state.drone_launch_time[k] + self.t_max) / T
            veh_rows.append(
                np.array(
                    [
                        dx / max_coord,
                        dy / max_coord,
                        state.drone_load[k] / (self.Q_d + 1e-6),
                        state.drone_time[k] / T,
                        tw_unavail,
                    ],
                    dtype=np.float32,
                )
            )

        vehicle_features = np.stack(veh_rows, axis=0)

        return {
            "node_features": node_features,
            "vehicle_features": vehicle_features,
            "truck_travel_times": np.stack(
                [
                    self.manhattan_dist[state.truck_node[k]] / (self.v_t * T)
                    for k in range(self.K)
                ],
                axis=0,
            ),
            "drone_travel_times": np.stack(
                [
                    self.euclidean_dist[state.drone_node[k]] / (self.v_d * T)
                    for k in range(self.K)
                ],
                axis=0,
            ),
            "edge_index": state.edge_index.copy(),
            "edge_attr": state.edge_attr.copy(),
            "edge_fleet": state.edge_fleet.copy(),
        }

    # ------------------------------------------------------------------
    # evaluate / scalar_objective
    # ------------------------------------------------------------------

    def evaluate(self, state: VRPBTWState) -> Tuple[float, float]:
        """
        Returns (total_cost, max_tardiness) by replaying stored routes.
        max_tardiness is the worst single time-window violation across
        all customer visits — not the sum.
        """
        total_cost = 0.0
        max_tard = 0.0

        for k in range(self.K):
            t, prev = 0.0, DEPOT
            for j in state.truck_routes[k]:
                dist = self.manhattan_dist[prev, j]
                t += dist / self.v_t
                if j != DEPOT:
                    max_tard = max(max_tard, max(t - self.tw_close[j], 0.0))
                    t = max(t, self.tw_open[j]) + self.service_times[j]
                total_cost += self.c_t * dist
                prev = j

            t_d, prev_d, in_trip = 0.0, DEPOT, False
            for node, is_cust in zip(
                state.drone_route_nodes[k], state.drone_route_mask[k]
            ):
                dist = self.euclidean_dist[prev_d, node]
                total_cost += self.c_d * dist
                if not in_trip and not is_cust:
                    t_d = max(t_d + dist / self.v_d, 0.0)
                elif not in_trip and is_cust:
                    t_d += self.launch_time + dist / self.v_d
                    max_tard = max(max_tard, max(t_d - self.tw_close[node], 0.0))
                    t_d = max(t_d, self.tw_open[node]) + self.service_times[node]
                    in_trip = True
                elif in_trip and is_cust:
                    t_d += dist / self.v_d
                    max_tard = max(max_tard, max(t_d - self.tw_close[node], 0.0))
                    t_d = max(t_d, self.tw_open[node]) + self.service_times[node]
                else:
                    t_d += dist / self.v_d + self.land_time
                    in_trip = False
                prev_d = node

        unserved = int((~state.served[1:]).sum())
        total_cost += unserved * 1000.0
        return total_cost, max_tard

    def scalar_objective(self, state: VRPBTWState) -> float:
        """Normalised scalarised objective using current nadir references."""
        cost, max_tard = self.evaluate(state)
        return self._F(cost, max_tard)

    def is_complete(self, state: VRPBTWState) -> bool:
        return self._is_terminated(state)

    def decode_solution(self, state: VRPBTWState) -> Solution:
        cost, max_tard = self.evaluate(state)
        obj = self._F(cost, max_tard)
        return Solution(
            problem_name=self.name,
            raw_state=state,
            objective=obj,
            metadata={
                "total_cost": cost,
                "max_tardiness": max_tard,
                "z_cost": self._z_cost,
                "z_tard": self._z_tard,
                "served_count": int(state.served[1:].sum()),
                "n_customers": self.n_customers,
                "truck_routes": [list(r) for r in state.truck_routes],
                "drone_route_nodes": [list(r) for r in state.drone_route_nodes],
                "drone_route_mask": [list(m) for m in state.drone_route_mask],
                "unserved": int((~state.served[1:]).sum()),
            },
        )

    def _heuristic_nadir(self) -> Tuple[float, float]:
        """
        Nearest-neighbour truck-only baseline.
        Returns (z_cost, z_tard) as nadir approximations for normalisation,
        where z_tard is max tardiness across all customers (not total).
        Private — called only by _set_nadir_from_heuristic.
        """
        served = np.zeros(self.n_customers + 1, dtype=bool)
        served[DEPOT] = True
        current, t, cost, load, phase = DEPOT, 0.0, 0.0, self.Q_t, 0
        max_tard = 0.0

        while not served[1:].all():
            best_j, best_d = -1, float("inf")
            for j in range(1, self.n_customers + 1):
                if served[j]:
                    continue
                if phase == 0 and self.demands[j] < 0:
                    continue
                if phase == 1 and self.demands[j] > 0:
                    continue
                if abs(self.demands[j]) > load:
                    continue
                d = self.manhattan_dist[current, j]
                arrive = t + d / self.v_t
                if arrive > self.tw_close[j]:
                    continue
                if d < best_d:
                    best_d, best_j = d, j
            if best_j == -1:
                if phase == 0:
                    phase = 1
                    continue
                break
            d = self.manhattan_dist[current, best_j]
            arrive = t + d / self.v_t
            tard = max(arrive - self.tw_close[best_j], 0.0)
            max_tard = max(max_tard, tard)
            t = max(arrive, self.tw_open[best_j]) + self.service_times[best_j]
            cost += self.c_t * d
            load -= abs(self.demands[best_j])
            served[best_j] = True
            current = best_j
            if phase == 0 and len(self._linehaul_idx) > 0:
                if served[self._linehaul_idx].all():
                    phase = 1

        cost += self.c_t * self.manhattan_dist[current, DEPOT]
        unserved = int((~served[1:]).sum())
        cost += unserved * 1000.0
        return cost, max_tard

    def heuristic_solution(self) -> Optional[float]:
        """
        Override base class contract: return scalar heuristic objective.
        Callers expecting Optional[float] (e.g. Evaluator) get the
        normalised scalarised value.  Nadir computation uses
        _heuristic_nadir() directly so this stays a clean scalar.
        """
        z_cost, z_tard = self._heuristic_nadir()
        return self._F(z_cost, z_tard)

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


# ---------------------------------------------------------------------------
# Instance generator
# ---------------------------------------------------------------------------


def generate_vrpbtw(
    n_customers: int = 10,
    n_fleets: int = 1,
    grid_size: float = 100.0,
    linehaul_ratio: float = 0.5,
    lambda_weight: float = 0.5,
    rng: Optional[np.random.Generator] = None,
    **kwargs,
) -> Dict:
    if rng is None:
        rng = np.random.default_rng()

    depot_xy = (grid_size / 2.0, grid_size / 2.0)
    coords = rng.uniform(0.0, grid_size, (n_customers, 2))
    n_lin = max(1, int(n_customers * linehaul_ratio))
    demands = np.concatenate(
        [
            rng.uniform(1.0, 10.0, n_lin),
            -rng.uniform(1.0, 10.0, n_customers - n_lin),
        ]
    )
    idx = rng.permutation(n_customers)
    coords = coords[idx]
    demands = demands[idx]

    depot_arr = np.array(depot_xy)
    dist_depot = np.linalg.norm(coords - depot_arr, axis=1)
    earliest = dist_depot / 1.0
    tw_open = np.maximum(0.0, earliest - rng.uniform(5.0, 15.0, n_customers))
    tw_close = earliest + rng.uniform(20.0, 50.0, n_customers)
    sys_dur = float(tw_close.max() + 30.0)
    t_max = float(grid_size * np.sqrt(2) / (2.0 * 2.0))
    customers = np.column_stack([coords, tw_open, tw_close, demands]).tolist()

    return {
        "depot": list(depot_xy),
        "customers": customers,
        "n_fleets": n_fleets,
        "truck_capacity": 50.0,
        "drone_capacity": 15.0,
        "system_duration": sys_dur,
        "trip_duration": t_max,
        "truck_speed": 1.0,
        "drone_speed": 2.0,
        "truck_cost": 1.0,
        "drone_cost": 0.5,
        "launch_time": 2.0,
        "land_time": 3.0,
        "service_time": 5.0,
        "lambda_weight": lambda_weight,
    }
