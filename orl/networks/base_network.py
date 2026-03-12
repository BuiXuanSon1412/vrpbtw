# networks/base_network.py

from abc import ABC, abstractmethod
from typing import Optional, Tuple
import torch


class BaseNetwork(ABC):
    @property
    @abstractmethod
    def obs_shape(self) -> Tuple[int, ...]:
        """Shape of the observation this network expects."""
        ...

    @property
    @abstractmethod
    def action_space_size(self) -> int:
        """Number of actions this network can score."""
        ...

    @abstractmethod
    def forward(
        self,
        obs: torch.Tensor,
        action_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns (logits, value)."""
        ...

    @abstractmethod
    def get_action_and_log_prob(
        self,
        obs: torch.Tensor,
        action_mask: Optional[torch.Tensor] = None,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns (action, log_prob, value)."""
        ...

    @abstractmethod
    def evaluate_actions(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        action_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns (log_probs, values, entropy)."""
        ...
