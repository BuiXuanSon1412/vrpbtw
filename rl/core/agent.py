"""
core/agents.py
--------------
Algorithm-specific RL agents with different batch structures and loss computations.

Classes
-------
  BaseAgent      — abstract base with update interface
  PPOAgent       — Proximal Policy Optimization with clipping
  ReinforceAgent — REINFORCE (policy gradient) without baseline
  POMOAgent      — POMO-style with advantage = reward - baseline

Design
------
Each agent implements a specific RL algorithm and defines:
  - expected batch structure (which keys, shapes)
  - hyperparameters (clip_eps, value_coef, entropy_coef, etc.)
  - loss computation and update logic

Batch structure varies by agent:
  - PPOAgent: advantages, returns, old_log_probs
  - ReinforceAgent: log_probs, returns
  - POMOAgent: log_probs, rewards (computes baseline internally)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

import torch
import torch.nn.utils
import torch.optim as optim


# ---------------------------------------------------------------------------
# BaseAgent (abstract)
# ---------------------------------------------------------------------------


class BaseAgent(ABC):
    """
    Abstract base for algorithm-specific RL agents.

    Each agent encapsulates a specific RL algorithm with its own:
      - batch structure expectations
      - hyperparameters
      - loss computation and update logic
    """

    def __init__(
        self,
        network: Any,
        optimizer: Optional[optim.Optimizer] = None,
        max_grad_norm: float = 0.5,
    ):
        """
        Args:
            network: network (must have parameters() method)
            optimizer: optimizer instance (Adam, SGD, AdamW, etc.), or None for manual updates
            max_grad_norm: gradient clipping threshold
        """
        self.network = network
        self.optimizer = optimizer
        self.max_grad_norm = max_grad_norm

    def act(
        self,
        obs: Any,
        action_mask: torch.Tensor,
        deterministic: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Select action using network.

        Args:
            obs: observation
            action_mask: (B, action_space) bool tensor, True=valid action
            deterministic: if True, take argmax; else sample

        Returns:
            action: (B,) int64
            log_prob: (B,) float32
            value: (B,) float32
            entropy: (B,) float32 entropy of policy distribution
        """
        logits, value, entropy = self.network.evaluate(obs, action_mask, actions=None)

        # Select action (deterministic or stochastic)
        if deterministic:
            action = logits.argmax(dim=-1)
        else:
            dist = torch.distributions.Categorical(logits=logits)
            action = dist.sample()

        # Compute log_prob from logits
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
        log_prob = log_probs.gather(-1, action.unsqueeze(-1)).squeeze(-1)

        return action, log_prob, value, entropy

    @abstractmethod
    def update(self, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """Perform a single gradient update step.

        Args:
            batch: dictionary of batch data (structure depends on algorithm)

        Returns:
            dict with metrics (loss, grad_norm, etc.)
        """
        ...

    @classmethod
    @abstractmethod
    def from_config(
        cls,
        cfg: Dict[str, Any],
        network: Any,
        opt_network: Optional[optim.Optimizer] = None,
    ) -> "BaseAgent":
        """Factory method: instantiate agent from config, policy, and optimizer."""
        ...

    def clone(self, source: "BaseAgent") -> None:
        """Clone policy from source agent into self.

        Replaces self's policy with a cloned copy of source's policy.
        Preserves all other configurations of self (optimizer, hyperparams, etc).

        Args:
            source: agent to clone policy from
        """
        self.network.load_state_dict(source.network.state_dict())


# ---------------------------------------------------------------------------
# PPOAgent
# ---------------------------------------------------------------------------


class PPOAgent(BaseAgent):
    """
    Proximal Policy Optimization (PPO) agent.

    Uses clipped surrogate objective to constrain policy updates.

    Expected batch structure:
      - "log_probs": (B,) or (B*T,) old policy log probabilities
      - "advantages": (B,) or (B*T,) advantage estimates
      - "returns": (B,) or (B*T,) return estimates
      - "masks": optional (B,) or (B*T,) validity masks

    Hyperparameters:
      - clip_eps: PPO clipping epsilon (typical: 0.2)
      - value_coef: coefficient for value loss
      - entropy_coef: coefficient for entropy regularization
    """

    def __init__(
        self,
        network: Any,
        optimizer: Optional[optim.Optimizer] = None,
        clip_eps: float = 0.2,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
        max_grad_norm: float = 0.5,
    ):
        super().__init__(network, optimizer, max_grad_norm)
        self.clip_eps = clip_eps
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef

    @classmethod
    def from_config(
        cls,
        cfg: Dict[str, Any],
        network: Any,
        opt_network: Optional[optim.Optimizer] = None,
    ) -> "PPOAgent":
        clip_eps = cfg.get("clip_eps", 0.2)
        value_coef = cfg.get("value_coef", 0.5)
        entropy_coef = cfg.get("entropy_coef", 0.01)
        max_grad_norm = cfg.get("max_grad_norm", 0.5)
        return cls(
            network=network,
            optimizer=opt_network,
            clip_eps=clip_eps,
            value_coef=value_coef,
            entropy_coef=entropy_coef,
            max_grad_norm=max_grad_norm,
        )

    def update(self, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """PPO update step with clipped surrogate objective.

        Args:
            batch: dict with keys observations, masks, log_probs, advantages, returns

        Returns:
            dict with loss and metrics
        """
        observations = batch["observations"]
        masks = batch["masks"]
        old_log_probs = batch["log_probs"]
        advantages = batch["advantages"]
        returns = batch["returns"]

        # Forward pass: re-evaluate actions under new policy
        # (batch must have 'actions' key for proper PPO re-evaluation)
        if "actions" in batch:
            actions = batch["actions"]
            new_log_probs, values, _ = self.network.evaluate(
                observations, masks, actions=actions
            )
        else:
            # Fallback: compute values only (improper PPO without action re-evaluation)
            logits, values, _ = self.network.evaluate(observations, masks, actions=None)
            new_log_probs = old_log_probs

        # Compute ratio and clipped surrogate loss
        ratio = torch.exp(new_log_probs - old_log_probs)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()

        # Value loss (MSE between predictions and returns)
        value_loss = torch.nn.functional.mse_loss(values, returns)

        # Entropy regularization
        entropy_loss = -new_log_probs.mean()

        total_loss = (
            policy_loss
            + self.value_coef * value_loss
            + self.entropy_coef * entropy_loss
        )

        # Optimization step
        grad_norm = torch.tensor(0.0, device=total_loss.device)
        if self.optimizer is not None:
            self.optimizer.zero_grad()
            total_loss.backward()
            grad_norm_val = torch.nn.utils.clip_grad_norm_(
                self.network.parameters(), self.max_grad_norm
            )
            grad_norm = torch.as_tensor(grad_norm_val, device=total_loss.device)
            self.optimizer.step()

        return {
            "loss": total_loss,
            "grad_norm": grad_norm,
            "policy_loss": policy_loss,
            "value_loss": value_loss,
        }


# ---------------------------------------------------------------------------
# ReinforceAgent
# ---------------------------------------------------------------------------


class ReinforceAgent(BaseAgent):
    """
    REINFORCE (policy gradient) agent without baseline.

    Simplest policy gradient method: gradient of log policy weighted by returns.

    Expected batch structure:
      - "log_probs": (B,) or (B*T,) policy log probabilities
      - "returns": (B,) or (B*T,) return estimates
      - "masks": optional (B,) or (B*T,) validity masks

    Hyperparameters:
      - entropy_coef: coefficient for entropy regularization (optional)
    """

    def __init__(
        self,
        network: Any,
        optimizer: Optional[optim.Optimizer] = None,
        entropy_coef: float = 0.0,
        max_grad_norm: float = 0.5,
    ):
        super().__init__(network, optimizer, max_grad_norm)
        self.entropy_coef = entropy_coef

    @classmethod
    def from_config(
        cls,
        cfg: Dict[str, Any],
        network: Any,
        opt_network: Optional[optim.Optimizer] = None,
    ) -> "ReinforceAgent":
        entropy_coef = cfg.get("entropy_coef", 0.0)
        max_grad_norm = cfg.get("max_grad_norm", 0.5)
        return cls(
            network=network,
            optimizer=opt_network,
            entropy_coef=entropy_coef,
            max_grad_norm=max_grad_norm,
        )

    def update(self, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """REINFORCE update: -mean(log_prob * return).

        Args:
            batch: dict with keys log_probs, returns

        Returns:
            dict with loss and grad_norm metrics
        """
        log_probs = batch["log_probs"]
        returns = batch["returns"]

        # Policy gradient loss (negative because we do gradient ascent)
        policy_loss = -(log_probs * returns).mean()

        # Optional entropy regularization
        entropy_loss = -(log_probs**2).mean() if self.entropy_coef > 0 else torch.tensor(0.0, device=policy_loss.device)
        total_loss = policy_loss + self.entropy_coef * entropy_loss

        # Optimization step
        grad_norm = torch.tensor(0.0, device=total_loss.device)
        if self.optimizer is not None:
            self.optimizer.zero_grad()
            total_loss.backward()
            grad_norm_val = torch.nn.utils.clip_grad_norm_(
                self.network.parameters(), self.max_grad_norm
            )
            grad_norm = torch.as_tensor(grad_norm_val, device=total_loss.device)
            self.optimizer.step()

        return {
            "loss": total_loss,
            "grad_norm": grad_norm,
            "policy_loss": policy_loss,
        }


# ---------------------------------------------------------------------------
# POMOAgent
# ---------------------------------------------------------------------------


class POMOAgent(BaseAgent):
    """
    POMO (Policy Optimization with Multiple Optima) agent.

    Uses empirical advantage = reward - baseline (mean reward).

    Expected batch structure:
      - "log_probs": (B,) or (B*T,) policy log probabilities (can be summed)
      - "rewards": (B,) or (B*T,) reward values

    Hyperparameters:
      - entropy_coef: coefficient for entropy regularization (optional)
    """

    def __init__(
        self,
        network: Any,
        optimizer: Optional[optim.Optimizer] = None,
        entropy_coef: float = 0.0,
        max_grad_norm: float = 0.5,
    ):
        super().__init__(network, optimizer, max_grad_norm)
        self.entropy_coef = entropy_coef

    @classmethod
    def from_config(
        cls,
        cfg: Dict[str, Any],
        network: Any,
        opt_network: Optional[optim.Optimizer] = None,
    ) -> "POMOAgent":
        entropy_coef = cfg.get("entropy_coef", 0.0)
        max_grad_norm = cfg.get("max_grad_norm", 0.5)
        return cls(
            network=network,
            optimizer=opt_network,
            entropy_coef=entropy_coef,
            max_grad_norm=max_grad_norm,
        )

    def update(self, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """POMO update: per-instance baseline with multiple starting points.

        For each instance, computes baseline as mean reward across starting points,
        then advantage = reward - baseline. Loss per instance is mean of log_prob * advantage.
        Batch loss is mean of all instance losses.

        Args:
            batch: dict with keys:
                - log_probs: list of (num_starting_points_i,) tensors, one per instance
                - rewards: list of (num_starting_points_i,) tensors, one per instance
                - entropies: list of (num_starting_points_i,) tensors, one per instance

        Returns:
            dict with loss and grad_norm metrics
        """
        log_probs = batch["log_probs"]      # List of (num_starting_points_i,) tensors
        rewards = batch["rewards"]           # List of (num_starting_points_i,) tensors
        entropies = batch["entropies"]       # List of (num_starting_points_i,) tensors

        instance_losses = []
        policy_losses = []
        total_baseline = 0.0
        num_instances = len(log_probs)
        entropy_values = []

        # Compute loss per instance with per-instance baseline and entropy
        for log_probs_i, rewards_i, entropies_i in zip(log_probs, rewards, entropies):
            # Per-instance baseline
            baseline_i = rewards_i.mean()
            advantages_i = rewards_i - baseline_i

            # Loss for this instance: mean(log_prob * advantage) - entropy_coef * mean(entropy)
            policy_loss_i = -(log_probs_i * advantages_i).mean()
            entropy_bonus_i = entropies_i.mean()
            loss_i = policy_loss_i - self.entropy_coef * entropy_bonus_i

            instance_losses.append(loss_i)
            policy_losses.append(policy_loss_i)
            total_baseline += baseline_i.item()
            entropy_values.append(entropy_bonus_i)

        # Batch loss: mean of instance losses (already includes entropy regularization)
        total_loss = torch.stack(instance_losses).mean()

        # Compute policy loss for logging
        policy_loss = torch.stack(policy_losses).mean()

        # Compute mean entropy bonus across instances for logging
        entropy_bonus = torch.stack(entropy_values).mean() if entropy_values else torch.tensor(0.0, device=total_loss.device)

        # Optimization step
        grad_norm = torch.tensor(0.0, device=total_loss.device)
        if self.optimizer is not None:
            self.optimizer.zero_grad()
            total_loss.backward()
            grad_norm_val = torch.nn.utils.clip_grad_norm_(
                self.network.parameters(), self.max_grad_norm
            )
            grad_norm = torch.as_tensor(grad_norm_val, device=total_loss.device)
            self.optimizer.step()

        # Debug: loss components and gradient info
        instance_losses_vals = [l.item() for l in instance_losses]
        print(f"  POMOAgent: n_instances={num_instances}, "
              f"total_loss={total_loss.item():.6f}, "
              f"policy_loss={policy_loss.item():.6f}, "
              f"entropy_bonus={entropy_bonus.item():.6f}, "
              f"loss_min={min(instance_losses_vals):.6f}, "
              f"loss_max={max(instance_losses_vals):.6f}, "
              f"grad_norm={grad_norm.item():.4f}, "
              f"baseline_mean={total_baseline / max(num_instances, 1):.4f}")

        return {
            "loss": total_loss,
            "grad_norm": grad_norm,
            "policy_loss": policy_loss,
            "entropy_bonus": entropy_bonus,
            "baseline": torch.tensor(total_baseline / max(num_instances, 1), device=total_loss.device),
        }


# ---------------------------------------------------------------------------
# MetaAgent
# ---------------------------------------------------------------------------


class MetaAgent(BaseAgent):
    """Meta-learning agent that aggregates losses across multiple tasks.

    Computes meta-update by averaging losses from different tasks and updating
    the shared policy on the aggregated loss.

    Expected usage:
      - PPOAgent instances compute losses on different tasks
      - MetaAgent.update() receives list of these losses
      - Performs meta-update on aggregated loss
    """

    def __init__(
        self,
        network: Any,
        optimizer: Optional[optim.Optimizer] = None,
        max_grad_norm: float = 0.5,
    ):
        super().__init__(network, optimizer, max_grad_norm)

    @classmethod
    def from_config(
        cls,
        cfg: Dict[str, Any],
        network: Any,
        opt_network: Optional[optim.Optimizer] = None,
    ) -> "MetaAgent":
        max_grad_norm = cfg.get("max_grad_norm", 0.5)
        return cls(
            network=network,
            optimizer=opt_network,
            max_grad_norm=max_grad_norm,
        )

    def update(self, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """Update network on meta-loss (average of task losses).

        Args:
            batch: dict with key "task_losses" containing stacked loss tensor

        Returns:
            dict with meta_loss metric
        """
        task_losses = batch["task_losses"]
        meta_loss = task_losses.mean()

        if self.optimizer is not None:
            self.optimizer.zero_grad()
            meta_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.network.parameters(), self.max_grad_norm
            )
            self.optimizer.step()

        return {"meta_loss": meta_loss}
