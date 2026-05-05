"""
core/collector.py
-----------------
Data collection strategies for on-policy RL.

Classes
-------
  BaseCollector           — abstract interface for trajectory collection
  BaseSequentialCollector — common logic for sequential rollout collectors
  GAECollector            — Generalized Advantage Estimation
  MCCollector             — Monte Carlo returns
  EPCollector             — Episodic rewards from environment
  POMOCollector           — collect from multiple starting points

Design
------
Each collector implements a strategy for gathering trajectory data and computing
returns/advantages. Collectors are algorithm-specific and return batch dicts ready
for agent.update().
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict

import numpy as np
import torch

import globals
from core.agent import BaseAgent
from core.utils import obs_to_tensor


# ---------------------------------------------------------------------------
# BaseCollector (abstract)
# ---------------------------------------------------------------------------


class BaseCollector(ABC):
    """Abstract base class for trajectory collection strategies.

    Each collector returns a batch dict ready for agent.update(), with
    structure determined by the algorithm.
    """

    @abstractmethod
    def collect(
        self,
        agent: BaseAgent,
        env: Any,
    ) -> dict:
        """Collect trajectories and return batch dict for agent.update().

        Args:
            agent: agent instance (with act() method)
            env: environment instance (with reset/step interface)

        Returns:
            dict with algorithm-specific batch structure
        """
        ...

    @classmethod
    @abstractmethod
    def from_config(cls, cfg: Dict[str, Any]) -> "BaseCollector":
        """Factory method: instantiate collector from config."""
        ...


# ---------------------------------------------------------------------------
# BaseSequentialCollector
# ---------------------------------------------------------------------------


class BaseSequentialCollector(BaseCollector):
    """Common logic for sequential rollout-based collectors.

    Subclasses implement _compute_returns() and _compute_advantages()
    to define their specific advantage estimation method.
    """

    def __init__(self, rollout_length: int = 256, gamma: float = 0.99):
        self.rollout_length = rollout_length
        self.gamma = gamma

    def collect(
        self,
        agent: BaseAgent,
        env: Any,
    ) -> dict:
        """Collect complete episodes and return batch dict.

        Args:
            agent: agent instance
            env: environment (already initialized with a task)

        Returns:
            dict with keys: observations, masks, log_probs, advantages, returns, entropies
        """
        observations = []
        masks = []
        actions = []
        log_probs = []
        values = []
        entropies = []
        rewards = []
        dones = []

        obs, info = env.reset()
        action_mask = info["action_mask"]

        while len(log_probs) < self.rollout_length:
            obs_t = obs_to_tensor(obs, device=globals.DEVICE)
            mask_t = torch.tensor(
                action_mask, dtype=torch.bool, device=globals.DEVICE
            ).unsqueeze(0)
            action_t, lp, val, ent = agent.act(obs_t, mask_t, deterministic=False)
            action = int(action_t.item())

            next_obs, reward, terminated, truncated, info = env.step(action)

            observations.append(obs_t)
            masks.append(mask_t)
            actions.append(action_t)
            log_probs.append(lp)
            values.append(val)
            entropies.append(ent)
            rewards.append(reward)
            dones.append(terminated or truncated)

            obs = next_obs
            action_mask = info["action_mask"]

            if terminated or truncated or not action_mask.any():
                obs, info = env.get_obs_info()
                action_mask = info["action_mask"]
                # Safety: if no feasible actions after reset, break collection
                if not action_mask.any():
                    break

        # Convert to tensors
        observations_tensor = torch.cat(observations, dim=0)
        masks_tensor = torch.cat(masks, dim=0)
        actions_tensor = torch.cat(actions, dim=0)
        log_probs_tensor = torch.stack(log_probs)
        values_tensor = torch.stack(values)
        entropies_tensor = torch.stack(entropies)
        rewards_tensor = torch.tensor(
            rewards, dtype=torch.float32, device=globals.DEVICE
        )
        dones_tensor = torch.tensor(dones, dtype=torch.float32, device=globals.DEVICE)

        # Compute returns and advantages using subclass method
        returns = self._compute_returns(rewards_tensor, dones_tensor)
        advantages = self._compute_advantages(values_tensor, returns, dones_tensor)

        return {
            "observations": observations_tensor,
            "masks": masks_tensor,
            "actions": actions_tensor,
            "log_probs": log_probs_tensor,
            "advantages": advantages,
            "returns": returns,
            "entropies": entropies_tensor,
        }

    @abstractmethod
    def _compute_returns(
        self, rewards: torch.Tensor, dones: torch.Tensor
    ) -> torch.Tensor:
        """Compute returns from rewards and done flags."""
        ...

    @abstractmethod
    def _compute_advantages(
        self,
        values: torch.Tensor,
        returns: torch.Tensor,
        dones: torch.Tensor,
    ) -> torch.Tensor:
        """Compute advantages given values and returns."""
        ...


# ---------------------------------------------------------------------------
# GAECollector
# ---------------------------------------------------------------------------


def _compute_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    dones: torch.Tensor,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute GAE advantages and returns.

    Args:
        rewards: (T,) tensor of rewards
        values: (T+1,) tensor of value estimates (includes bootstrap value)
        dones: (T,) tensor of done flags
        gamma: discount factor
        gae_lambda: GAE lambda parameter

    Returns:
        advantages: (T,) tensor of advantage estimates
        returns: (T,) tensor of return estimates
    """
    T = len(rewards)
    advantages = torch.zeros(T, dtype=torch.float32, device=rewards.device)
    gae = 0.0

    for t in reversed(range(T)):
        next_value = values[t + 1]
        current_value = values[t]
        delta = rewards[t] + gamma * next_value * (1 - dones[t]) - current_value
        gae = delta + gamma * gae_lambda * (1 - dones[t]) * gae
        advantages[t] = gae

    returns = advantages + values[:T]
    return advantages, returns


class GAECollector(BaseSequentialCollector):
    """Generalized Advantage Estimation collector.

    Computes advantages using exponential-weighted moving average over TD residuals.
    """

    def __init__(
        self,
        rollout_length: int = 256,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
    ):
        super().__init__(rollout_length, gamma)
        self.gae_lambda = gae_lambda

    @classmethod
    def from_config(cls, cfg: Dict[str, Any]) -> "GAECollector":
        params = cfg.get("params", {})
        return cls(
            rollout_length=params.get("rollout_length", 256),
            gamma=params.get("gamma", 0.99),
            gae_lambda=params.get("gae_lambda", 0.95),
        )

    def collect(
        self,
        agent: BaseAgent,
        env: Any,
    ) -> dict:
        """Collect trajectories with GAE advantage estimation."""
        observations = []
        masks = []
        actions = []
        log_probs = []
        values = []
        entropies = []
        rewards = []
        dones = []

        obs, info = env.reset()
        action_mask = info["action_mask"]

        while len(log_probs) < self.rollout_length:
            obs_t = obs_to_tensor(obs, device=globals.DEVICE)
            mask_t = torch.tensor(
                action_mask, dtype=torch.bool, device=globals.DEVICE
            ).unsqueeze(0)
            action_t, lp, val, ent = agent.act(obs_t, mask_t, deterministic=False)
            action = int(action_t.item())

            next_obs, reward, terminated, truncated, info = env.step(action)

            observations.append(obs_t)
            masks.append(mask_t)
            actions.append(action_t)
            log_probs.append(lp)
            values.append(val)
            entropies.append(ent)
            rewards.append(reward)
            dones.append(terminated or truncated)

            obs = next_obs
            action_mask = info["action_mask"]

            if terminated or truncated or not action_mask.any():
                obs, info = env.get_obs_info()
                action_mask = info["action_mask"]
                # Safety: if no feasible actions after reset, break collection
                if not action_mask.any():
                    break

        # Compute GAE
        observations_tensor = torch.cat(observations, dim=0)
        masks_tensor = torch.cat(masks, dim=0)
        actions_tensor = torch.cat(actions, dim=0)
        log_probs_tensor = torch.stack(log_probs)
        values_tensor = torch.stack(values)
        entropies_tensor = torch.stack(entropies)
        rewards_tensor = torch.tensor(
            rewards, dtype=torch.float32, device=globals.DEVICE
        )
        dones_tensor = torch.tensor(dones, dtype=torch.float32, device=globals.DEVICE)

        # Get bootstrap value
        obs_end = obs_to_tensor(obs, device=globals.DEVICE)
        mask_end = torch.tensor(
            action_mask, dtype=torch.bool, device=globals.DEVICE
        ).unsqueeze(0)
        _, _, bootstrap_val, _ = agent.act(obs_end, mask_end, deterministic=False)
        values_with_bootstrap = torch.cat(
            [values_tensor, bootstrap_val.unsqueeze(0)], dim=0
        )

        advantages, returns = _compute_gae(
            rewards_tensor,
            values_with_bootstrap,
            dones_tensor,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
        )

        return {
            "observations": observations_tensor,
            "masks": masks_tensor,
            "actions": actions_tensor,
            "log_probs": log_probs_tensor,
            "advantages": advantages,
            "returns": returns,
            "entropies": entropies_tensor,
        }

    def _compute_returns(
        self, rewards: torch.Tensor, dones: torch.Tensor
    ) -> torch.Tensor:
        """Not used in overridden collect() but required by interface."""
        return torch.zeros_like(rewards)

    def _compute_advantages(
        self,
        values: torch.Tensor,
        returns: torch.Tensor,
        dones: torch.Tensor,
    ) -> torch.Tensor:
        """Not used in overridden collect() but required by interface."""
        return torch.zeros_like(returns)


# ---------------------------------------------------------------------------
# MCCollector
# ---------------------------------------------------------------------------


class MCCollector(BaseSequentialCollector):
    """Monte Carlo collector.

    Computes returns as discounted sum of rewards. Advantages are return - value.
    """

    @classmethod
    def from_config(cls, cfg: Dict[str, Any]) -> "MCCollector":
        params = cfg.get("params", {})
        return cls(
            rollout_length=params.get("rollout_length", 256),
            gamma=params.get("gamma", 0.99),
        )

    def _compute_returns(
        self, rewards: torch.Tensor, dones: torch.Tensor
    ) -> torch.Tensor:
        """Compute Monte Carlo returns (discounted cumulative sum)."""
        returns = []
        cumsum = 0.0
        for r, d in zip(reversed(rewards.tolist()), reversed(dones.tolist())):
            cumsum = r + self.gamma * cumsum * (1.0 - d)
            returns.insert(0, cumsum)
        return torch.tensor(returns, dtype=torch.float32, device=rewards.device)

    def _compute_advantages(
        self,
        values: torch.Tensor,
        returns: torch.Tensor,
        dones: torch.Tensor,
    ) -> torch.Tensor:
        """Compute advantages as return - value."""
        return returns - values[:-1].detach()


# ---------------------------------------------------------------------------
# EPCollector (Episodic)
# ---------------------------------------------------------------------------


class EPCollector(BaseSequentialCollector):
    """Episodic reward collector.

    Uses environment-computed returns and value-based advantages.
    Suitable for combinatorial optimization problems.
    """

    @classmethod
    def from_config(cls, cfg: Dict[str, Any]) -> "EPCollector":
        params = cfg.get("params", {})
        return cls(
            rollout_length=params.get("rollout_length", 256),
            gamma=params.get("gamma", 0.99),
        )

    def collect(
        self,
        agent: BaseAgent,
        env: Any,
    ) -> dict:
        """Collect trajectories with environment-computed returns."""
        observations = []
        masks = []
        actions = []
        log_probs = []
        values = []
        entropies = []
        rewards = []
        dones = []

        obs, info = env.reset()
        action_mask = info["action_mask"]

        while len(log_probs) < self.rollout_length:
            obs_t = obs_to_tensor(obs, device=globals.DEVICE)
            mask_t = torch.tensor(
                action_mask, dtype=torch.bool, device=globals.DEVICE
            ).unsqueeze(0)
            action_t, lp, val, ent = agent.act(obs_t, mask_t, deterministic=False)
            action = int(action_t.item())

            next_obs, reward, terminated, truncated, info = env.step(action)

            observations.append(obs_t)
            masks.append(mask_t)
            actions.append(action_t)
            log_probs.append(lp)
            values.append(val)
            entropies.append(ent)
            rewards.append(reward)
            dones.append(terminated or truncated)

            obs = next_obs
            action_mask = info["action_mask"]

            if terminated or truncated or not action_mask.any():
                obs, info = env.get_obs_info()
                action_mask = info["action_mask"]
                # Safety: if no feasible actions after reset, break collection
                if not action_mask.any():
                    break

        # Get episode return from environment
        episode_return = env.compute_return()
        observations_tensor = torch.cat(observations, dim=0)
        masks_tensor = torch.cat(masks, dim=0)
        actions_tensor = torch.cat(actions, dim=0)
        log_probs_tensor = torch.stack(log_probs)
        values_tensor = torch.stack(values)
        entropies_tensor = torch.stack(entropies)

        # Use episode return for all steps
        returns = torch.full_like(
            values_tensor[:, 0], episode_return, dtype=torch.float32
        )
        advantages = returns - values_tensor[:, 0].detach()

        return {
            "observations": observations_tensor,
            "masks": masks_tensor,
            "actions": actions_tensor,
            "log_probs": log_probs_tensor,
            "advantages": advantages,
            "returns": returns,
            "entropies": entropies_tensor,
        }

    def _compute_returns(
        self, rewards: torch.Tensor, dones: torch.Tensor
    ) -> torch.Tensor:
        """Not used in overridden collect() but required by interface."""
        return torch.zeros_like(rewards)

    def _compute_advantages(
        self,
        values: torch.Tensor,
        returns: torch.Tensor,
        dones: torch.Tensor,
    ) -> torch.Tensor:
        """Not used in overridden collect() but required by interface."""
        return torch.zeros_like(returns)


# ---------------------------------------------------------------------------
# POMOCollector (POMO-style multiple optima)
# ---------------------------------------------------------------------------


class POMOSampler(BaseCollector):
    """
    Collect from multiple starting points per instance (POMO-style).

    For each feasible starting action: collect complete episode starting from
    that action, accumulating log probabilities across the trajectory.
    Returns episode-level batch (log_probs, rewards) for POMO loss computation.
    """

    def __init__(self):
        """POMO sampler: no rollout_length needed (collects full episodes)."""
        pass

    @classmethod
    def from_config(cls, cfg: Dict[str, Any]) -> "POMOSampler":
        return cls()

    def collect(
        self,
        agent: BaseAgent,
        env: Any,
    ) -> dict:
        """Collect episodes from multiple starting points (POMO-style).

        For each feasible starting action:
          - Compute log_prob for that action
          - Take that action
          - Roll out complete episode, accumulating log probabilities
          - Get final episode return
          - Store (episode_log_prob_sum, episode_return)

        Args:
            agent: agent instance
            env: environment (already initialized with a task, has compute_return())

        Returns:
            dict with keys:
              - "log_probs": list of summed log_probs (one per episode)
              - "rewards": list of episode returns (one per episode)
        """
        obs, info = env.reset()
        action_mask = info["action_mask"]

        # Compute log probs for all feasible starting actions
        obs_t = obs_to_tensor(obs, device=globals.DEVICE)
        mask_t = torch.tensor(
            action_mask, dtype=torch.bool, device=globals.DEVICE
        ).unsqueeze(0)

        feasible_actions = np.where(action_mask)[0].tolist()
        if not feasible_actions:
            return {"log_probs": [], "rewards": []}

        action_log_probs = {}
        for action in feasible_actions:
            act_t = torch.tensor([action], dtype=torch.long, device=globals.DEVICE)
            log_prob, _, _ = agent.network.evaluate(obs_t, mask_t, actions=act_t)
            action_log_probs[int(action)] = log_prob

        episode_log_probs = []
        episode_returns = []

        # print(len(feasible_actions))
        for starting_action, starting_log_prob in action_log_probs.items():
            obs, info = env.reset()

            # First step with starting action
            episode_log_prob = starting_log_prob
            next_obs, _, terminated, truncated, next_info = env.step(
                int(starting_action)
            )

            if not terminated and not truncated and next_info["action_mask"].any():
                obs = next_obs
                mask = next_info["action_mask"]

                # Collect rest of trajectory, accumulating log probabilities
                while True:
                    obs_t = obs_to_tensor(obs, device=globals.DEVICE)
                    mask_t = torch.tensor(
                        mask, dtype=torch.bool, device=globals.DEVICE
                    ).unsqueeze(0)
                    action_t, lp, _, _ = agent.act(obs_t, mask_t, deterministic=False)
                    action = int(action_t.item())
                    episode_log_prob = episode_log_prob + lp
                    next_obs, _, terminated, truncated, info = env.step(action)

                    if terminated or truncated or not info["action_mask"].any():
                        break

                    obs = next_obs
                    mask = info["action_mask"]

            # Get final episode return
            episode_return = env.compute_return()
            episode_log_probs.append(episode_log_prob)
            episode_returns.append(episode_return)

        return {
            "log_probs": episode_log_probs,
            "rewards": episode_returns,
        }
