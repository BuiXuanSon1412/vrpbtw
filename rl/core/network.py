"""
core/policy.py
--------------
Abstract base class for all policy + value networks, plus the
shared neural network sub-modules reused across architectures.

What lives here
---------------
  BaseNetwork          — abstract interface agents call (act, evaluate)
  _MHA                 — multi-head attention (self + cross)
  _FF                  — position-wise feed-forward block
  _InstanceNormWrapper — InstanceNorm1d with the sequence-dim swap
  _make_norm           — factory: InstanceNorm or LayerNorm by flag

"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional, Tuple

import torch
import torch.nn as nn
import math

# ---------------------------------------------------------------------------
# Shared sub-modules
# ---------------------------------------------------------------------------


def _make_norm(use_instance_norm: bool, dim: int) -> nn.Module:
    """Return InstanceNorm or LayerNorm depending on the flag."""
    return _InstanceNormWrapper(dim) if use_instance_norm else nn.LayerNorm(dim)


class _InstanceNormWrapper(nn.Module):
    """InstanceNorm1d that works on (B, T, D) tensors."""

    def __init__(self, dim: int):
        super().__init__()
        self.norm = nn.InstanceNorm1d(dim, affine=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (B, T, D) → transpose → norm → transpose back
        return self.norm(x.transpose(1, 2)).transpose(1, 2)


class _MHA(nn.Module):
    """
    Multi-head attention supporting both self-attention and cross-attention.

    Parameters
    ----------
    D       : embedding dimension
    H       : number of heads (must divide D)
    dropout : dropout on attention weights
    """

    def __init__(self, D: int, H: int, dropout: float = 0.0):
        super().__init__()
        assert D % H == 0, f"embed_dim {D} must be divisible by n_heads {H}"
        self.H = H
        self.Dh = D // H
        self.D = D
        self.Wq = nn.Linear(D, D, bias=False)
        self.Wk = nn.Linear(D, D, bias=False)
        self.Wv = nn.Linear(D, D, bias=False)
        self.Wo = nn.Linear(D, D, bias=False)
        self.drop = nn.Dropout(dropout)

    def forward(
        self,
        q: torch.Tensor,  # (B, Tq, D)
        k: torch.Tensor,  # (B, Tk, D)
        v: torch.Tensor,  # (B, Tk, D)
        mask: Optional[torch.Tensor] = None,  # (B, Tq, Tk) bool True=ignore
    ) -> torch.Tensor:
        B, Tq, _ = q.shape
        Tk = k.shape[1]
        H, Dh = self.H, self.Dh

        def reshape(t: torch.Tensor, T: int) -> torch.Tensor:
            return t.view(B, T, H, Dh).transpose(1, 2)

        Q = reshape(self.Wq(q), Tq)
        K = reshape(self.Wk(k), Tk)
        V = reshape(self.Wv(v), Tk)
        sc = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(Dh)
        if mask is not None:
            sc = sc.masked_fill(mask.unsqueeze(1), float("-inf"))
        at = self.drop(torch.softmax(sc, dim=-1))
        at = torch.nan_to_num(at, nan=0.0)
        out = torch.matmul(at, V).transpose(1, 2).contiguous().view(B, Tq, self.D)
        return self.Wo(out)


class _FF(nn.Module):
    """Position-wise feed-forward: Linear → ReLU → Dropout → Linear."""

    def __init__(self, D: int, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(D, D * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(D * 4, D),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ---------------------------------------------------------------------------
# Network  (abstract base)
# ---------------------------------------------------------------------------


class BaseNetwork(nn.Module, ABC):
    """
    Abstract base for all neural network policies.

    Defines the minimal interface that all policy networks must implement.

    Abstract methods:
    - evaluate: compute logits/values for sampling, or evaluate given actions
    """

    @abstractmethod
    def evaluate(
        self,
        obs,
        action_mask: Optional[torch.Tensor] = None,
        actions: Optional[torch.Tensor] = None,
        context=None,
    ) -> Tuple[torch.Tensor, ...]:
        """
        Compute logits/values for sampling, or evaluate given actions.

        If actions=None: sample mode
            Returns (logits, values)
        If actions provided: evaluation mode
            Returns (log_probs, values, entropy)

        Args:
            obs: observation
            action_mask: (B, action_space) bool mask for valid actions
            actions: optional (B,) action indices for evaluation
            context: optional context

        Returns:
            If actions=None: (logits, values)
                logits: (B, action_space) float32
                values: (B,) float32
            If actions provided: (log_probs, values, entropy)
                log_probs: (B,) float32
                values: (B,) float32
                entropy: (B,) float32
        """
        ...


# ---------------------------------------------------------------------------
# ActorCritic  (extends Network)
# ---------------------------------------------------------------------------


class ActorCritic(BaseNetwork):
    """
    Abstract base for all policy + value networks.

    Concrete architectures must subclass ActorCritic and implement evaluate().

    Abstract methods
    ----------------
    forward          → (logits, value)
    evaluate         → (logits, values, entropy) or (log_probs, values, entropy)
    """

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @abstractmethod
    def forward(
        self,
        obs,
        action_mask: Optional[torch.Tensor] = None,
        context=None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns
        -------
        logits : (B, action_space_size)   raw masked scores
        value  : (B,)                     critic estimate
        """
        ...

    @abstractmethod
    def evaluate(
        self,
        obs,
        action_mask: Optional[torch.Tensor] = None,
        actions: Optional[torch.Tensor] = None,
        context=None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute logits/values for sampling, or evaluate given actions.

        If actions=None: return (logits, values, entropy)
        If actions provided: return (log_probs, values, entropy)
        """
        ...

    # ------------------------------------------------------------------
    # Shared helpers available to all subclasses
    # ------------------------------------------------------------------

    def to_device(self, device: str) -> "ActorCritic":
        return self.to(torch.device(device))

    @staticmethod
    def _apply_mask(logits: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
        if mask is not None:
            logits = logits.masked_fill(~mask, float("-inf"))
        return logits

    @staticmethod
    def _ortho_init(module: nn.Module, gain: float = 1.414) -> None:
        for m in module.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=gain)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
