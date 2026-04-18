"""
core/state.py
-------------
Abstract state representation for MDP/environment abstraction.

All environments track state internally and transition through states.
A State encapsulates the complete problem state at a point in time.

Design principle:
  - Problem holds no mutable state
  - State is immutable or at least semantically isolated
  - Episode management (stateful reset/step) is separate from Problem (stateless)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict


class State(ABC):
    """
    Abstract state representation for any MDP.

    A state encapsulates:
    - Decision variables (which node served, vehicle positions, etc.)
    - Constraints (time windows, capacities, etc.)
    - Derived quantities (cost, feasibility, etc.)

    Implementations should be immutable or at least thread-safe.
    """

    @property
    @abstractmethod
    def is_terminal(self) -> bool:
        """True if this state is terminal (no more actions possible)."""
        ...

    @property
    @abstractmethod
    def is_feasible(self) -> bool:
        """True if solution represented by this state is feasible."""
        ...

    @abstractmethod
    def to_dict(self) -> Dict[str, Any]:
        """Serialize state to dict for checkpointing, logging, etc."""
        ...

    @classmethod
    @abstractmethod
    def from_dict(cls, data: Dict[str, Any]) -> State:
        """Deserialize state from dict."""
        ...


# Example concrete implementation (defined in problems/vrpbtw.py):
# class VRPBTWState(State):
#     truck_node: np.ndarray  # (K,)
#     truck_time: np.ndarray  # (K,)
#     drone_node: np.ndarray  # (K,)
#     ...
#
#     @property
#     def is_terminal(self) -> bool:
#         return all_nodes_served()
#
#     @property
#     def is_feasible(self) -> bool:
#         return all_time_windows_met()
