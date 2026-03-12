from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple


class BaseAgent(ABC):
    @abstractmethod
    def select_action(
        self,
        obs,
        action_mask,
        training: bool = True,
    ) -> Tuple[int, float, float]:
        """Returns (action, log_prob, value)."""
        ...

    @abstractmethod
    def update(self) -> Optional[Dict[str, float]]:
        """Perform one learning update. Returns metrics dict or None."""
        ...

    @abstractmethod
    def save(self, path: str) -> None: ...

    @abstractmethod
    def load(self, path: str) -> None: ...
