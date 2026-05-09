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
    DEPOT,
    TRUCK,
    DRONE,
)
from typing import Any, Dict


class MVRPBTWEnv(VRPBTWEnv):
    """
    Multi-phase VRPBTW without phase constraint enforcement.

    Inherits all methods from VRPBTWEnv but overrides:
    - _truck_phase_ok: Always returns True (no phase check)
    - _drone_phase_ok: Always returns True (no phase check)

    Phase state is still tracked in state (truck_phase, drone_phase) for
    reference and logging, but does not restrict feasibility.
    """

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
