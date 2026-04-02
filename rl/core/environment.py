"""
core/environment.py
-------------------
Generic Gym-style MDP wrapper for any Problem subclass.

Reward shaping is now handled entirely inside the Problem (vrpbtw.py).
Reward shaping is handled entirely inside the Problem subclass.
Environment is problem-agnostic and has no knowledge of objectives,
heuristics, or normalisation.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from core.problem import Problem, ActionMask


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------


class Environment:
    """
    Wraps any Problem in a step / reset interface.

    Parameters
    ----------
    problem : Any concrete Problem subclass.
    cfg     : EnvConfig dataclass from config.py.
    """

    def __init__(self, problem: Problem, cfg: Any):
        self.problem = problem
        self.cfg = cfg

        self._state: Any = None
        self._current_mask: Optional[ActionMask] = None
        self._current_obs: Any = None
        self._step_count: int = 0
        self._episode_reward: float = 0.0

        # Aggregate stats
        self._total_episodes: int = 0
        self._total_steps: int = 0
        self._episode_rewards: List[float] = []
        self._episode_objectives: List[float] = []

    # ------------------------------------------------------------------
    # Core interface
    # ------------------------------------------------------------------

    def reset(self, raw_instance: Any) -> Tuple[Any, Dict]:
        self._state = self.problem.reset(raw_instance)
        self._current_mask = self.problem.get_action_mask(self._state)
        self._current_obs = self.problem.state_to_obs(self._state)
        self._step_count = 0
        self._episode_reward = 0.0

        return self._current_obs, self._make_info()

    def step(self, action: int) -> Tuple[Any, float, bool, bool, Dict]:
        if self._state is None or self._current_mask is None:
            raise RuntimeError("Call reset() before step().")
        if not self._current_mask.mask[action]:
            raise ValueError(
                f"Action {action} is infeasible. "
                f"Feasible: {self._current_mask.action_indices.tolist()}"
            )

        result = self.problem.apply_action(self._state, action)
        self._state = result.next_state
        self._current_mask = result.action_mask
        self._current_obs = self.problem.state_to_obs(self._state)
        self._step_count += 1
        self._total_steps += 1

        reward = result.reward * self.cfg.reward_scale
        self._episode_reward += reward

        truncated = (
            self.cfg.max_steps is not None and self._step_count >= self.cfg.max_steps
        ) or result.truncated
        terminated = result.terminated

        if terminated or truncated:
            self._total_episodes += 1
            self._episode_rewards.append(self._episode_reward)
            if terminated:
                obj = self.problem.scalar_objective(self._state)
                self._episode_objectives.append(obj)
                result.info["episode_objective"] = obj
            result.info["episode_reward"] = self._episode_reward
            result.info["episode_steps"] = self._step_count

        info = {**result.info, **self._make_info()}
        return self._current_obs, reward, terminated, truncated, info

    # ------------------------------------------------------------------
    # Info dict
    # ------------------------------------------------------------------

    def _make_info(self) -> Dict:
        assert self._current_mask is not None
        info: Dict[str, Any] = {
            "action_mask": self._current_mask.mask,
            "feasible_actions": self._current_mask.action_indices,
            "step": self._step_count,
        }
        if isinstance(self._current_obs, dict):
            for key in (
                "node_features",
                "vehicle_features",
                "edge_index",
                "edge_attr",
                "edge_fleet",
            ):
                if key in self._current_obs:
                    info[key] = self._current_obs[key]
        return info

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def decode_current_solution(self):
        return self.problem.decode_solution(self._state)

    def get_stats(self) -> Dict:
        r = self._episode_rewards
        o = self._episode_objectives
        return {
            "total_episodes": self._total_episodes,
            "total_steps": self._total_steps,
            "mean_ep_reward": float(np.mean(r)) if r else 0.0,
            "mean_objective": float(np.mean(o)) if o else 0.0,
            "best_objective": float(np.max(o)) if o else 0.0,
        }

    def __repr__(self) -> str:
        return (
            f"Environment(problem={self.problem.name!r}, "
            f"max_steps={self.cfg.max_steps}, "
            f"episodes={self._total_episodes})"
        )
