"""
core/collector.py
-----------------
Data collection strategies for on-policy RL.

Classes
-------
  BaseCollector           — abstract interface for trajectory collection
  GAECollector            — Generalized Advantage Estimation
  MCCollector             — Monte Carlo returns
  POMOSampler             — collect from multiple starting points

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

from core.vec_env import stack_obs, batch_obs_to_tensor

# ---------------------------------------------------------------------------
# BaseCollector (abstract)
# ---------------------------------------------------------------------------


class BaseCollector(ABC):
    """Abstract base class for trajectory collection strategies.

    Each collector returns a batch dict ready for agent.update(), with
    structure determined by the algorithm.
    """

    @abstractmethod
    def collect(self, agent: BaseAgent, env: Any, rollout_length: int) -> dict:
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

    returns = (advantages + values[:T]).detach()
    return advantages, returns


# ---------------------------------------------------------------------------
# GAECollector (Generalized Advantage Estimation)
# ---------------------------------------------------------------------------


class GAECollector(BaseCollector):
    """Generalized Advantage Estimation collector for parallel/serial environments.

    Supports both serial (batch_size=1) and parallel (batch_size>1) collection:
    - Serial: SubprocVecEnv(n_envs=1) collects from single environment
    - Parallel: SubprocVecEnv(n_envs=B) collects from B parallel environments

    For each environment, collects rollout_length timesteps and computes
    per-env advantages using exponential-weighted moving average over TD residuals.

    Returns observations as list of T stacked batches (B, ...) per timestep,
    enabling unified mini-batch logic in trainer for both serial and parallel cases.
    """

    def __init__(
        self,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
    ):
        self.gamma = gamma
        self.gae_lambda = gae_lambda

    @classmethod
    def from_config(cls, cfg: Dict[str, Any]) -> "GAECollector":
        params = cfg.get("params", {})
        return cls(
            gamma=params.get("gamma", 0.99),
            gae_lambda=params.get("gae_lambda", 0.95),
        )

    def collect(
        self,
        agent: BaseAgent,
        env: Any,
        rollout_length: int,
    ) -> dict:
        """Collect trajectory data from vectorized environment.

        For SubprocVecEnv with n_envs=B (serial or parallel):
        - Collects rollout_length timesteps from each of B environments
        - Computes per-env advantages using GAE
        - Returns stacked observations (list of T batched dicts)

        This unified method works for both:
        - Serial: SubprocVecEnv(n_envs=1) with single environment
        - Parallel: SubprocVecEnv(n_envs=B) with B parallel environments

        Args:
            agent: Policy agent with act(obs, mask) returning (actions, log_probs, values, entropies)
            env: SubprocVecEnv instance
            rollout_length: Target timesteps to collect per environment (required)

        Returns:
            dict with observations (list of T batched obs dicts), masks, actions, log_probs,
            advantages, returns, entropies (flattened to B*rollout_length timesteps)
        """

        n_envs = env.n_envs

        obs_list, info_list = env.reset()
        action_masks = np.stack([info["action_mask"] for info in info_list])

        obs_buffer = []
        mask_buffer = []
        action_buffer = []
        logprob_buffer = []
        value_buffer = []
        entropy_buffer = []
        reward_buffer = []
        done_buffer = []

        # Initialize for tracking final episode states (for bootstrap value masking)
        terminateds: np.ndarray = np.zeros(n_envs, dtype=bool)
        truncateds: np.ndarray = np.zeros(n_envs, dtype=bool)

        with torch.no_grad():
            for t in range(rollout_length):
                stacked_obs = stack_obs(obs_list)
                obs_t = batch_obs_to_tensor(stacked_obs, device=globals.DEVICE)
                mask_t = torch.tensor(
                    action_masks, dtype=torch.bool, device=globals.DEVICE
                )

                actions_t, log_probs_t, values_t, entropies_t = agent.act(
                    obs_t, mask_t, deterministic=False
                )

                obs_list, rewards, terminateds, truncateds, info_list = env.step(
                    actions_t.cpu().numpy()
                )
                action_masks = np.stack([info["action_mask"] for info in info_list])

                obs_buffer.append(stacked_obs)
                mask_buffer.append(mask_t)
                action_buffer.append(actions_t)
                logprob_buffer.append(log_probs_t.squeeze(-1))
                value_buffer.append(values_t.squeeze(-1))
                entropy_buffer.append(entropies_t.squeeze(-1))
                reward_buffer.append(
                    torch.tensor(rewards, dtype=torch.float32, device=globals.DEVICE)
                )
                done_buffer.append(
                    torch.tensor(
                        np.logical_or(terminateds, truncateds),
                        dtype=torch.float32,
                        device=globals.DEVICE,
                    )
                )

            stacked_final = stack_obs(obs_list)
            obs_final = batch_obs_to_tensor(stacked_final, device=globals.DEVICE)
            mask_final = torch.tensor(
                action_masks, dtype=torch.bool, device=globals.DEVICE
            )
            _, _, bootstrap_vals, _ = agent.act(
                obs_final, mask_final, deterministic=False
            )
            bootstrap_vals = bootstrap_vals.squeeze(-1)
            bootstrap_vals[np.logical_or(terminateds, truncateds)] = 0.0

        # Compute per-env advantages and returns
        advantages_list = []
        returns_list = []

        for b in range(n_envs):
            rewards_b = torch.stack(
                [reward_buffer[t][b] for t in range(rollout_length)]
            )
            values_b = torch.stack([value_buffer[t][b] for t in range(rollout_length)])
            dones_b = torch.stack([done_buffer[t][b] for t in range(rollout_length)])
            v_with_bootstrap = torch.cat([values_b, bootstrap_vals[b : b + 1]], dim=0)

            adv_b, ret_b = _compute_gae(
                rewards_b,
                v_with_bootstrap,
                dones_b,
                gamma=self.gamma,
                gae_lambda=self.gae_lambda,
            )
            advantages_list.append(adv_b)
            returns_list.append(ret_b)

        advantages = torch.stack(advantages_list, dim=0)
        returns = torch.stack(returns_list, dim=0)

        advantages = advantages.view(-1)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        returns = returns.view(-1)

        masks_stacked = torch.stack(mask_buffer, dim=0)
        masks_flat = masks_stacked.view(n_envs * rollout_length, -1)

        actions_flat = torch.cat(action_buffer, dim=0)
        logprobs_flat = torch.cat(logprob_buffer, dim=0)
        entropies_flat = torch.cat(entropy_buffer, dim=0)

        return {
            "observations": obs_buffer,
            "masks": masks_flat,
            "actions": actions_flat,
            "log_probs": logprobs_flat,
            "advantages": advantages,
            "returns": returns,
            "entropies": entropies_flat,
        }


# ---------------------------------------------------------------------------
# MCCollector
# ---------------------------------------------------------------------------


class MCCollector(BaseCollector):
    """Monte Carlo collector.

    Collects one complete episode per call.
    Computes returns as discounted sum of rewards. Advantages are return - value.

    Note: Each batch is exactly one complete episode (no partial episodes).
    """

    def __init__(self, gamma: float = 0.99):
        self.gamma = gamma

    @classmethod
    def from_config(cls, cfg: Dict[str, Any]) -> "MCCollector":
        params = cfg.get("params", {})
        return cls(
            gamma=params.get("gamma", 0.99),
        )

    def collect(self, agent: BaseAgent, env: Any, rollout_length: int) -> dict:
        """Collect one complete episode and return batch dict."""
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

        with torch.no_grad():
            while True:  # Collect until episode terminates
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
                    break

        # Convert to tensors, squeezing batch dimension from agent.act() returns
        # Keep observations as-is (may be list of dicts for graph-based envs with dynamic structures)
        observations_tensor = observations
        masks_tensor = torch.cat(masks, dim=0)
        actions_tensor = torch.cat(actions, dim=0)
        log_probs_tensor = torch.stack([lp.squeeze(0) for lp in log_probs])
        values_tensor = torch.stack([v.squeeze(0) for v in values])
        entropies_tensor = torch.stack([e.squeeze(0) for e in entropies])
        rewards_tensor = torch.tensor(
            rewards, dtype=torch.float32, device=globals.DEVICE
        )
        dones_tensor = torch.tensor(dones, dtype=torch.float32, device=globals.DEVICE)

        # Compute returns and advantages
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
        return returns - values.detach()


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

    def collect(self, agent: BaseAgent, env: Any, rollout_length: int) -> dict:
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
        episode_entropies = []

        # print(len(feasible_actions))
        for starting_action, starting_log_prob in action_log_probs.items():
            obs, info = env.reset()

            # First step with starting action
            episode_log_prob = starting_log_prob
            episode_reward = 0.0
            step_entropies = []
            next_obs, reward, terminated, truncated, next_info = env.step(
                int(starting_action)
            )
            episode_reward += reward if reward is not None else 0.0

            if not terminated and not truncated and next_info["action_mask"].any():
                obs = next_obs
                mask = next_info["action_mask"]

                # Collect rest of trajectory, accumulating log probabilities, rewards, and entropies
                while True:
                    obs_t = obs_to_tensor(obs, device=globals.DEVICE)
                    mask_t = torch.tensor(
                        mask, dtype=torch.bool, device=globals.DEVICE
                    ).unsqueeze(0)
                    action_t, lp, _, entropy = agent.act(
                        obs_t, mask_t, deterministic=False
                    )
                    action = int(action_t.item())
                    episode_log_prob = episode_log_prob + lp
                    step_entropies.append(entropy)
                    next_obs, reward, terminated, truncated, info = env.step(action)
                    episode_reward += reward if reward is not None else 0.0

                    if terminated or truncated or not info["action_mask"].any():
                        break

                    obs = next_obs
                    mask = info["action_mask"]

            # Use accumulated per-step rewards (store as tensor scalar for consistency)
            episode_return = torch.tensor(
                episode_reward, dtype=torch.float32, device=globals.DEVICE
            )
            episode_log_probs.append(episode_log_prob)
            episode_returns.append(episode_return)

            # Compute episode-level entropy as mean of step-level entropies
            if step_entropies:
                episode_entropy = torch.stack(step_entropies).mean()
            else:
                episode_entropy = torch.tensor(0.0, device=globals.DEVICE)
            episode_entropies.append(episode_entropy)

        return {
            "log_probs": episode_log_probs,
            "rewards": episode_returns,
            "entropies": episode_entropies,
        }
