"""
core/trainer.py
---------------
Training-loop implementations.

MetaTrainer — multi-task MAML with curriculum expansion
POMOTrainer — Policy Optimization with Multiple Optima (per-task independent training)

Design principle:
  - Agent holds the policy network; each trainer computes loss via its own method
  - MetaTrainer coordinates multi-task learning with inner-loop adaptation and outer meta-updates
  - Curriculum expansion monitored via task entropy, integrated into train() method
  - POMOTrainer trains each sub-policy for each task independently using POMO collection
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from torch.func import functional_call

import globals
from core.agent import BaseAgent
from core.utils import obs_to_tensor


from core.pool import SubprocVecEnv
from core.pool import stack_obs, batch_obs_to_tensor
# ---------------------------------------------------------------------------
# GAE Computation
# ---------------------------------------------------------------------------


def _compute_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    dones: torch.Tensor,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute Generalized Advantage Estimation (GAE).

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
# BaseTrainer (abstract interface)
# ---------------------------------------------------------------------------


class BaseTrainer(ABC):
    """Abstract base class for training strategies."""

    @abstractmethod
    def train(self) -> Dict[str, Any]:
        """Run training loop and return summary."""
        ...

    @classmethod
    @abstractmethod
    def from_config(
        cls,
        trainer_cfg: Dict[str, Any],
        agents: Dict[str, BaseAgent],
        env: Any,
        evaluators: Dict[str, Any],
        logger: Any,
        env_cfg: Dict[str, Any],
    ) -> "BaseTrainer":
        """Factory method: instantiate trainer from config.

        Args:
            trainer_cfg: trainer config dict (cfg.trainer)
            agents: dict of agent instances (keyed by name)
            env: environment instance (has tasks list and reset interface)
            evaluators: dict of evaluator instances (keyed by phase name, or "default" for single-phase)
            logger: logger instance
            env_cfg: environment configuration dict (optional)
        """
        ...


# ---------------------------------------------------------------------------
# MetaTrainer: Full MAML with curriculum
# ---------------------------------------------------------------------------


class MetaTrainer(BaseTrainer):
    """
    MAML trainer (second-order) with curriculum learning.

    Coordinates:
      1. Task sampling from TaskManager (respects curriculum)
      2. Support/query data collection for each task
      3. Inner-loop adaptation per task
      4. Outer-loop meta-gradient accumulation and update
      5. Curriculum expansion monitoring
    """

    def __init__(
        self,
        agents: Dict[str, BaseAgent],
        env: Any,
        trainer_cfg: Dict[str, Any],
        evaluators: Dict[str, Any],
        logger: Any,
        env_cfg: Dict[str, Any],
    ):
        self.agents = agents
        self.meta_agent = agents["meta_agent"]
        self.sub_agent = agents["sub_agent"]
        self.tune_agent = agents["tune_agent"]
        self.env = env
        self.env_cfg = env_cfg or {}
        self.active_tasks = {env.tasks[0]}  # Start with first (easiest) task
        self.meta_evaluator = evaluators["meta_eval"]
        self.fine_evaluator = evaluators["tune_eval"]
        self.logger = logger

        # Extract config from trainer structure
        phases_cfg = trainer_cfg.get("phases", {})
        estimators_cfg = trainer_cfg.get("estimators", {})

        # Get meta_learning phase
        meta_phase = phases_cfg.get("meta_learning", {})
        self.enable_meta_learning = bool(meta_phase.get("enabled", True))
        curriculum_cfg = meta_phase.get("curriculum", {})
        meta_control_cfg = meta_phase.get("control", {})
        meta_early_stop = meta_phase.get("early_stopping", {})

        # Meta-learning config: epochs/batches instead of timesteps
        self.mcfg = {
            "batch_size": int(meta_control_cfg.get("batch_size", 1)),
            "rollout_length": int(meta_control_cfg.get("rollout_length", 256)),
            "entropy_threshold": float(curriculum_cfg.get("entropy_threshold", 0.5)),
            "curriculum_check_interval": int(curriculum_cfg.get("check_interval", 10)),
            "epochs": int(meta_control_cfg.get("epochs", 200)),
            "eval_interval": int(meta_control_cfg.get("eval_interval", 1)),
            "checkpoint_interval": int(meta_control_cfg.get("checkpoint_interval", 10)),
            "patience": int(meta_early_stop.get("patience", 20)),
            "min_delta": float(meta_early_stop.get("min_delta", 0.0001)),
        }

        # Get fine_tuning phase config
        fine_tune_phase = phases_cfg.get("fine_tuning", {})
        self.enable_fine_tuning = bool(fine_tune_phase.get("enabled", True))
        fine_control_cfg = fine_tune_phase.get("control", {})
        fine_early_stop = fine_tune_phase.get("early_stopping", {})

        # Fine-tuning config: per-task PPO training with optional parallelization
        # Structure: for each task → for each iteration → collect rollout → PPO updates
        ppo_estimator = estimators_cfg.get("ppo", {})
        self.fcfg = {
            # Rollout collection (parallel or serial)
            "batch_size": int(fine_control_cfg.get("batch_size", 1)),
            "n_iteration": int(fine_control_cfg.get("n_iteration", 100)),
            "rollout_length": int(fine_control_cfg.get("rollout_length", 256)),
            # PPO optimization on collected data
            "ppo_epochs": int(fine_control_cfg.get("ppo_epochs", 1)),
            "minibatch_size": int(fine_control_cfg.get("minibatch_size", 32)),
            # Advantage estimation hyperparameters
            "gamma": float(ppo_estimator.get("gamma", 0.99)),
            "gae_lambda": float(ppo_estimator.get("gae_lambda", 0.95)),
            # Evaluation and checkpointing
            "eval_interval": int(fine_control_cfg.get("eval_interval", 1)),
            "checkpoint_interval": int(fine_control_cfg.get("checkpoint_interval", 10)),
            # Early stopping
            "patience": int(fine_early_stop.get("patience", 10)),
            "min_delta": float(fine_early_stop.get("min_delta", 0.0001)),
        }

        # Training state
        self._total_updates = 0
        self._best_objective = float(
            "-inf"
        )  # For maximization problems, higher is better
        self._patience_counter = 0
        self._curriculum_check_counter = 0

    @classmethod
    def from_config(
        cls,
        trainer_cfg: Dict[str, Any],
        agents: Dict[str, BaseAgent],
        env: Any,
        evaluators: Dict[str, Any],
        logger: Any,
        env_cfg: Dict[str, Any],
    ) -> "MetaTrainer":
        if not env.tasks:
            raise ValueError("MetaTrainer.from_config requires env.tasks")

        if "meta_agent" not in agents:
            raise ValueError("MetaTrainer requires 'meta_agent' in agents dict")

        return cls(
            agents=agents,
            env=env,
            trainer_cfg=trainer_cfg,
            evaluators=evaluators,
            logger=logger,
            env_cfg=env_cfg,
        )

    def train(self) -> Dict[str, Any]:
        """Run training pipeline: conditionally execute meta-training and/or fine-tuning.

        Modes:
          - Both enabled: Full MAML pipeline (meta-train + fine-tune)
          - Only meta-learning: Pure meta-learning without fine-tuning
          - Only fine-tuning: Equivalent to normal PPO (task-specific training from scratch)
          - Neither enabled: Error (at least one phase must be enabled)
        """
        if not self.enable_meta_learning and not self.enable_fine_tuning:
            raise ValueError(
                "At least one of meta_learning or fine_tuning must be enabled"
            )

        # Print training header (always shown, regardless of which phases are enabled)
        self._print_header()

        start_time = time.time()
        meta_summary = {}
        fine_tune_summary = {}

        # Phase 1: Meta-training (optional)
        if self.enable_meta_learning:
            meta_summary = self.meta_train()
        else:
            self.logger.log_event(
                "meta_learning_skipped",
                self._total_updates,
                message="Meta-learning disabled; starting fine-tuning with untrained network",
            )

        # Phase 2: Fine-tuning (optional)
        if self.enable_fine_tuning:
            fine_tune_summary = self.fine_tune()
        else:
            self.logger.log_event(
                "fine_tuning_skipped",
                self._total_updates,
                message="Fine-tuning disabled; meta-learning complete",
            )

        # Combine summaries
        summary = {
            "stop_reason": meta_summary.get(
                "stop_reason", fine_tune_summary.get("stop_reason", "completed")
            ),
            "total_updates": self._total_updates,
            "total_epochs": meta_summary.get("total_epochs", 0)
            + fine_tune_summary.get("total_epochs", 0),
            "best_objective": min(
                meta_summary.get("best_objective", float("inf")),
                fine_tune_summary.get("best_objective", float("inf")),
            ),
            "training_time_s": round(time.time() - start_time, 1),
            "meta_learning_enabled": self.enable_meta_learning,
            "fine_tuning_enabled": self.enable_fine_tuning,
            "meta_summary": meta_summary if self.enable_meta_learning else None,
            "fine_tune_summary": fine_tune_summary if self.enable_fine_tuning else None,
        }
        self.logger.log_event(
            "training_complete",
            self._total_updates,
            total_updates=self._total_updates,
            total_epochs=summary["total_epochs"],
            best_objective=summary["best_objective"],
            meta_learning_enabled=self.enable_meta_learning,
            fine_tuning_enabled=self.enable_fine_tuning,
        )
        self.logger.save_summary(summary)
        self.logger.close()
        self._print_footer(summary)
        return summary

    def meta_train(self) -> Dict[str, Any]:
        """Run meta-training loop with epoch/batch structure."""
        start_time = time.time()
        stop_reason = "completed"
        epoch = -1

        # Setup vectorized environment for meta-learning (serial or parallel)

        batch_size = self.mcfg["batch_size"]
        vec_env = SubprocVecEnv(
            env_class=type(self.env),
            env_cfg=self.env_cfg,
            n_envs=batch_size,
            base_seed=42,
        )

        try:
            for epoch in range(self.mcfg["epochs"]):
                epoch_start = time.time()

                try:
                    # Compute task losses across active tasks
                    task_losses, task_metrics = self._compute_task_losses(vec_env)

                    # Update meta-policy on aggregated task losses
                    self.meta_agent.update({"task_losses": task_losses})
                    self._total_updates += 1

                    # Curriculum check per epoch
                    max_entropy = None
                    for task_id, metrics_dict in task_metrics.items():
                        entropy = metrics_dict.get("entropy", 0)
                        if max_entropy is None or entropy > max_entropy:
                            max_entropy = entropy

                    self._curriculum_check_counter += 1
                    if (
                        self._curriculum_check_counter
                        >= self.mcfg["curriculum_check_interval"]
                    ):
                        self._curriculum_check_counter = 0
                        if (
                            max_entropy is not None
                            and max_entropy < self.mcfg["entropy_threshold"]
                        ):
                            if len(self.active_tasks) < len(self.env.tasks):
                                next_task = self.env.tasks[len(self.active_tasks)]
                                self.active_tasks.add(next_task)
                                self.logger.log_event(
                                    "curriculum_expansion",
                                    self._total_updates,
                                    num_tasks=len(self.active_tasks),
                                    task_id=str(next_task),
                                )

                except Exception as e:
                    self.logger.log_exception(
                        e,
                        message=f"Error during meta-training in epoch {epoch}",
                        step=self._total_updates,
                        epoch=epoch,
                    )
                    raise

                # Per-epoch metrics
                epoch_time = time.time() - epoch_start
                epoch_losses_tensor = task_losses.detach()
                train_metrics = {
                    "meta/loss_mean": float(epoch_losses_tensor.mean()),
                    "meta/loss_std": float(epoch_losses_tensor.std()),
                    "meta/num_active_tasks": float(len(self.active_tasks)),
                    "meta/total_updates": float(self._total_updates),
                    "meta/epoch_time_s": epoch_time,
                }

                # Evaluation every eval_interval epochs
                eval_metrics = {}
                if (epoch + 1) % self.mcfg["eval_interval"] == 0:
                    try:
                        median_idx = len(self.env.tasks) // 2
                        eval_task_id = self.env.tasks[median_idx]
                        eval_stats = self.meta_evaluator.evaluate(eval_task_id)
                        eval_metrics = {f"eval/{k}": v for k, v in eval_stats.items()}

                        mean_obj = eval_stats.get("mean_objective", float("-inf"))
                        if mean_obj > self._best_objective + self.mcfg["min_delta"]:
                            self._best_objective = mean_obj
                            self._patience_counter = 0
                            self.logger.save_checkpoint(
                                "meta_best",
                                {
                                    "network_state": self.meta_agent.network.state_dict(),
                                    "epoch": epoch + 1,
                                },
                            )
                            self.logger.log_event(
                                "best_checkpoint",
                                self._total_updates,
                                objective=f"{mean_obj:.4f}",
                            )
                        else:
                            self._patience_counter += 1

                        if self._patience_counter >= self.mcfg["patience"]:
                            stop_reason = "early_stopping"
                            self.logger.log_event(
                                "early_stop",
                                self._total_updates,
                                patience=self.mcfg["patience"],
                            )
                            break
                    except Exception as e:
                        self.logger.log_exception(
                            e,
                            message=f"Error during evaluation in epoch {epoch}",
                            step=self._total_updates,
                            epoch=epoch,
                        )
                        raise

                # Log all metrics
                all_metrics = {**train_metrics, **eval_metrics}
                print_keys = [
                    "meta/loss_mean",
                    "meta/num_active_tasks",
                    "meta/total_updates",
                ]
                if eval_metrics:
                    print_keys.extend(["eval/mean_objective"])
                    if "eval/mean_service_rate" in eval_metrics:
                        print_keys.append("eval/mean_service_rate")
                    if "eval/mean_cost" in eval_metrics:
                        print_keys.append("eval/mean_cost")

                self.logger.log_metrics(
                    all_metrics,
                    step=self._total_updates,
                    print_keys=print_keys,
                )

        except Exception as e:
            stop_reason = "error"
            self.logger.log_exception(
                e,
                message="Fatal error during meta-training",
                step=self._total_updates,
                epoch=epoch,
            )
            raise

        summary = {
            "stop_reason": stop_reason,
            "total_epochs": epoch + 1,
            "total_updates": self._total_updates,
            "best_objective": self._best_objective,
            "training_time_s": round(time.time() - start_time, 1),
            "final_num_active_tasks": len(self.active_tasks),
        }
        self.logger.save_checkpoint(
            "meta_final",
            {
                "network_state": self.meta_agent.network.state_dict(),
                "epoch": epoch + 1,
            },
        )
        self.logger.log_event("meta_training_complete", self._total_updates, **summary)
        return summary

    def collect(
        self,
        agent: BaseAgent,
        env: Any,
        rollout_length: int,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
    ) -> Dict[str, Any]:
        """Collect trajectory data with GAE advantage estimation (meta-learning or fine-tuning).

        Collects rollout_length timesteps from vectorized environment (SubprocVecEnv)
        and computes per-env advantages using GAE.

        Args:
            agent: Policy agent
            env: SubprocVecEnv instance
            rollout_length: Target timesteps to collect per environment
            gamma: Discount factor (default: from fine-tuning config if available, else 0.99)
            gae_lambda: GAE lambda parameter (default: from fine-tuning config if available, else 0.95)

        Returns:
            dict with:
              - observations: list of rollout_length dicts (NOT flattened)
                Each dict contains tensors with batch dimension (batch_size, ...)
                Layout: observations[t] = all batch_size environments' data at timestep t
              - masks, actions, log_probs, advantages, returns, entropies: flattened tensors
                Layout: flat indices map to (timestep, env) via: flat_idx = t*batch_size + e
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
                # Keep batch dimension: values_t shape is (n_envs,) from agent.act()
                logprob_buffer.append(
                    log_probs_t if log_probs_t.dim() == 1 else log_probs_t.squeeze(-1)
                )
                value_buffer.append(
                    values_t if values_t.dim() == 1 else values_t.squeeze(-1)
                )
                entropy_buffer.append(
                    entropies_t if entropies_t.dim() == 1 else entropies_t.squeeze(-1)
                )
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
            # Compute done_mask for bootstrap masking (per-env)
            done_mask = torch.tensor(
                np.logical_or(terminateds, truncateds),
                dtype=torch.bool,
                device=globals.DEVICE,
            )

        # Compute per-env advantages and returns
        advantages_list = []
        returns_list = []

        for b in range(n_envs):
            rewards_b = torch.stack(
                [reward_buffer[t][b] for t in range(rollout_length)]
            )
            values_b = torch.stack([value_buffer[t][b] for t in range(rollout_length)])
            dones_b = torch.stack([done_buffer[t][b] for t in range(rollout_length)])
            # Only use bootstrap if env is not done at final step
            bootstrap_masked = bootstrap_vals[b] * (~done_mask[b]).float()
            v_with_bootstrap = torch.cat(
                [values_b, bootstrap_masked.unsqueeze(0)], dim=0
            )

            adv_b, ret_b = _compute_gae(
                rewards_b,
                v_with_bootstrap,
                dones_b,
                gamma=gamma,
                gae_lambda=gae_lambda,
            )
            # Normalize advantages PER-environment, not globally
            adv_b = (adv_b - adv_b.mean()) / (adv_b.std() + 1e-8)
            advantages_list.append(adv_b)
            returns_list.append(ret_b)

        advantages = torch.stack(advantages_list, dim=0)
        returns = torch.stack(returns_list, dim=0)

        advantages = advantages.view(-1)
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

    def fine_tune(self) -> Dict[str, Any]:
        """Fine-tune policy on each task independently using PPO with optional parallelization.

        For each task:
          1. For each iteration (0 to n_iteration):
             - Collect rollout_length timesteps from batch_size parallel environments (or 1 serial env)
             - Compute advantages/returns with GAE per-environment
          2. PPO optimization:
             - For each ppo_epoch (0 to ppo_epochs):
               - Shuffle collected data and do mini-batch SGD with minibatch_size
               - Compute PPO loss: policy + value + entropy terms
             - Evaluate and checkpoint

        Parallel collection (batch_size > 1):
          - Uses SubprocVecEnv: N worker processes, genuine parallelism
          - Uses GAECollector: per-env GAE computation
          - Mini-batch loop handles both graph (list of dicts) and tensor observations

        Serial collection (batch_size = 1):
          - Uses single environment, standard GAECollector
        """
        self._print_header_tune()
        start_time = time.time()
        agent = self.tune_agent

        # Extract fine-tuning config
        batch_size = int(self.fcfg.get("batch_size", 1))
        n_iteration = int(self.fcfg.get("n_iteration", 100))
        rollout_length = int(self.fcfg.get("rollout_length", 256))
        ppo_epochs = int(self.fcfg.get("ppo_epochs", 1))
        minibatch_size = int(self.fcfg.get("minibatch_size", 32))
        eval_interval = int(self.fcfg.get("eval_interval", 1))
        checkpoint_interval = int(self.fcfg.get("checkpoint_interval", 10))

        # Validate batch_size
        if batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {batch_size}")

        # Setup vectorized environment (serial or parallel)

        vec_env = SubprocVecEnv(
            env_class=type(self.env),
            env_cfg=self.env_cfg,
            n_envs=batch_size,
            base_seed=42,
        )

        best_objective = float("-inf")
        task_summaries = {}

        try:
            for task_id in self.env.tasks:
                self._print_header_task(task_id)
                task_best_objective = float("-inf")
                task_patience_counter = 0
                iteration = -1

                # Reset agent to meta-trained weights for per-task specialization
                if self.enable_meta_learning and hasattr(self, "meta_agent"):
                    agent.network.load_state_dict(self.meta_agent.network.state_dict())
                    self.logger.log_event(
                        "tune_agent_reset",
                        self._total_updates,
                        task=task_id,
                        source="meta_agent",
                    )

                try:
                    for iteration in range(n_iteration):
                        iteration_start = time.time()
                        loss_buffer = []  # List of dicts: {policy, value, entropy, total}
                        grad_norm_buffer = []

                        try:
                            # Step 1: Collect rollout_length timesteps from parallel/serial environment
                            # - Parallel: collects from batch_size envs, returns (batch_size * rollout_length) timesteps
                            # - Serial: collects from 1 env, returns rollout_length timesteps
                            vec_env.retask(task_id)
                            batch = self.collect(agent, vec_env, rollout_length)

                            # Prepare collected data for PPO updates
                            observations = batch["observations"]
                            masks = batch["masks"]
                            actions = batch["actions"]
                            old_log_probs = batch["log_probs"]
                            advantages = batch["advantages"]
                            returns = batch["returns"]
                            entropies = batch["entropies"]

                            num_timesteps = len(actions)

                            # Step 2: PPO optimization - ppo_epochs passes over collected data
                            # For batch_size > 1: process each environment's timesteps separately
                            # For batch_size = 1: same logic, just 1 environment
                            for ppo_epoch in range(ppo_epochs):
                                # Process each environment's timesteps independently
                                for env_id in range(batch_size):
                                    # Get all timesteps for this environment
                                    # With batch_size=4, env_id=0 gets [0, 4, 8, 12, ...]
                                    # With batch_size=1, env_id=0 gets [0, 1, 2, 3, ...]
                                    env_timestep_indices = torch.arange(
                                        env_id,
                                        num_timesteps,
                                        batch_size,
                                        device=old_log_probs.device,
                                        dtype=torch.long,
                                    )

                                    # Shuffle within this environment only
                                    shuffled_indices = env_timestep_indices[
                                        torch.randperm(
                                            len(env_timestep_indices),
                                            device=old_log_probs.device,
                                        )
                                    ]

                                    # Mini-batch SGD on this environment's data
                                    for start_idx in range(
                                        0, len(shuffled_indices), minibatch_size
                                    ):
                                        end_idx = min(
                                            start_idx + minibatch_size,
                                            len(shuffled_indices),
                                        )
                                        batch_indices = shuffled_indices[
                                            start_idx:end_idx
                                        ]

                                        # Extract mini-batch
                                        if isinstance(observations, list):
                                            # observations[t] is a dict with tensors stacked across batch_size
                                            # batch_indices are flat indices (0 to batch_size*rollout_length)
                                            # Convert to (timestep, env) coordinates: flat_idx = t*batch_size + e
                                            # Note: env_timestep_indices constructed as arange(env_id, ..., batch_size)
                                            # guarantees all indices belong to this env_id, so no assertion needed

                                            # Vectorized extraction: convert flat indices to timestep indices
                                            timestep_indices = (
                                                batch_indices // batch_size
                                            )

                                            # Extract observations for these timesteps
                                            mb_observations = []
                                            for timestep_idx in timestep_indices:
                                                stacked_obs = observations[
                                                    int(timestep_idx.item())
                                                ]
                                                single_obs = {}
                                                for k, v in stacked_obs.items():
                                                    if "edge_index" in k:
                                                        # Shared across batch (no env dimension)
                                                        single_obs[k] = v
                                                    else:
                                                        # Extract this env's data: v.shape = (batch_size, ...)
                                                        single_obs[k] = v[
                                                            env_id : env_id + 1
                                                        ]
                                                mb_observations.append(single_obs)
                                        else:
                                            mb_observations = observations[
                                                batch_indices
                                            ]

                                        # Prepare mini-batch dict for agent.update()
                                        minibatch = {
                                            "observations": mb_observations,
                                            "masks": masks[batch_indices],
                                            "actions": actions[batch_indices],
                                            "log_probs": old_log_probs[batch_indices],
                                            "advantages": advantages[batch_indices],
                                            "returns": returns[batch_indices],
                                            "entropies": entropies[batch_indices],
                                        }

                                        # Delegate all PPO logic to agent
                                        update_metrics = agent.update(minibatch)

                                        # Collect metrics
                                        loss_buffer.append(
                                            {
                                                "policy": float(
                                                    update_metrics[
                                                        "policy_loss"
                                                    ].detach()
                                                ),
                                                "value": float(
                                                    update_metrics[
                                                        "value_loss"
                                                    ].detach()
                                                ),
                                                "entropy": float(
                                                    update_metrics["entropy"].detach()
                                                ),
                                                "total": float(
                                                    update_metrics["loss"].detach()
                                                ),
                                            }
                                        )
                                        grad_norm_buffer.append(
                                            float(update_metrics["grad_norm"].detach())
                                        )

                                        self._total_updates += 1

                        except Exception as e:
                            self.logger.log_exception(
                                e,
                                message=f"Error during PPO update for task {task_id} in iteration {iteration}",
                                step=self._total_updates,
                                task=str(task_id),
                                iteration=iteration,
                            )
                            raise

                        # Per-iteration aggregation
                        iteration_time = time.time() - iteration_start
                        train_metrics = {
                            "tune/total_updates": float(self._total_updates),
                            "tune/iteration_time_s": iteration_time,
                        }

                        # Log aggregated loss components and gradient norm at eval_interval
                        if (iteration + 1) % eval_interval == 0 and loss_buffer:
                            # Extract loss components
                            policy_losses = torch.tensor(
                                [l["policy"] for l in loss_buffer]
                            )
                            value_losses = torch.tensor(
                                [l["value"] for l in loss_buffer]
                            )
                            entropy_losses = torch.tensor(
                                [l["entropy"] for l in loss_buffer]
                            )
                            total_losses = torch.tensor(
                                [l["total"] for l in loss_buffer]
                            )
                            grad_norm_array = torch.tensor(grad_norm_buffer)

                            train_metrics.update(
                                {
                                    # Policy loss
                                    "tune/loss_policy_mean": float(
                                        policy_losses.mean()
                                    ),
                                    "tune/loss_policy_std": float(policy_losses.std()),
                                    # Value loss (raw, not scaled)
                                    "tune/loss_value_mean": float(value_losses.mean()),
                                    "tune/loss_value_std": float(value_losses.std()),
                                    # Entropy loss (raw, not scaled)
                                    "tune/loss_entropy_mean": float(
                                        entropy_losses.mean()
                                    ),
                                    "tune/loss_entropy_std": float(
                                        entropy_losses.std()
                                    ),
                                    # Total (combined)
                                    "tune/loss_total_mean": float(total_losses.mean()),
                                    "tune/loss_total_std": float(total_losses.std()),
                                    # Gradient norm
                                    "tune/grad_norm_mean": float(
                                        grad_norm_array.mean()
                                    ),
                                    "tune/grad_norm_std": float(grad_norm_array.std()),
                                }
                            )

                        # Evaluation
                        eval_metrics = {}
                        if (iteration + 1) % eval_interval == 0:
                            try:
                                eval_stats = self.fine_evaluator.evaluate(task_id)
                                eval_metrics = {
                                    f"eval/{k}": v for k, v in eval_stats.items()
                                }

                                mean_obj = eval_stats.get(
                                    "mean_objective", float("-inf")
                                )
                                if mean_obj > task_best_objective + self.fcfg.get(
                                    "min_delta", 0.0001
                                ):
                                    task_best_objective = mean_obj
                                    task_patience_counter = 0
                                    self.logger.save_checkpoint(
                                        f"tune_best_{task_id}",
                                        {
                                            "network_state": agent.network.state_dict(),
                                            "iteration": iteration + 1,
                                        },
                                    )
                                    self.logger.log_event(
                                        "tune_best_checkpoint",
                                        self._total_updates,
                                        task=task_id,
                                        objective=f"{mean_obj:.4f}",
                                    )
                                else:
                                    task_patience_counter += 1

                            except Exception as e:
                                self.logger.log_exception(
                                    e,
                                    message=f"Error during evaluation for task {task_id} in iteration {iteration}",
                                    step=self._total_updates,
                                    task=str(task_id),
                                    iteration=iteration,
                                )
                                raise

                            # Early stopping: check patience threshold
                            if task_patience_counter >= self.fcfg.get("patience", 1000):
                                self.logger.log_event(
                                    "fine_tune_early_stop",
                                    self._total_updates,
                                    task=task_id,
                                    iteration=iteration,
                                    patience=self.fcfg.get("patience", 1000),
                                )
                                break

                        # Checkpointing
                        if (iteration + 1) % checkpoint_interval == 0:
                            self.logger.save_checkpoint(
                                f"tune_{task_id}_iter_{iteration + 1}",
                                {
                                    "network_state": agent.network.state_dict(),
                                    "iteration": iteration + 1,
                                },
                            )

                        # Log metrics
                        all_metrics = {**train_metrics, **eval_metrics}
                        print_keys = ["tune/total_updates"]

                        # Add loss components to print if available
                        if "tune/loss_policy_mean" in train_metrics:
                            print_keys.extend(
                                [
                                    "tune/loss_policy_mean",
                                    "tune/loss_value_mean",
                                    "tune/loss_entropy_mean",
                                    "tune/loss_total_mean",
                                ]
                            )

                        if eval_metrics:
                            print_keys.append("eval/mean_objective")
                            if "eval/mean_service_rate" in eval_metrics:
                                print_keys.append("eval/mean_service_rate")
                            if "eval/mean_cost" in eval_metrics:
                                print_keys.append("eval/mean_cost")

                        self.logger.log_metrics(
                            all_metrics,
                            step=self._total_updates,
                            print_keys=print_keys,
                        )

                except Exception as e:
                    self.logger.log_exception(
                        e,
                        message=f"Error during fine-tuning iteration loop for task {task_id}",
                        step=self._total_updates,
                        task=str(task_id),
                    )
                    raise

                best_objective = max(best_objective, task_best_objective)
                task_summaries[task_id] = {
                    "best_objective": float(task_best_objective),
                    "iterations_completed": iteration + 1,
                }

                # Final checkpoint
                self.logger.save_checkpoint(
                    f"tune_final_{task_id}",
                    {
                        "network_state": agent.network.state_dict(),
                        "iteration": iteration + 1,
                    },
                )

        except Exception as e:
            self.logger.log_exception(
                e,
                message="Fatal error during fine-tuning",
                step=self._total_updates,
            )
            raise
        finally:
            vec_env.close()

        summary = {
            "stop_reason": "completed",
            "total_iterations": n_iteration,
            "total_updates": self._total_updates,
            "best_objective": best_objective,
            "training_time_s": round(time.time() - start_time, 1),
            "task_summaries": task_summaries,
        }
        self.logger.log_event("fine_tuning_complete", self._total_updates, **summary)
        return summary

    def _print_header_tune(self) -> None:
        from datetime import datetime

        # Get tune agent hyperparameters
        tune_lr = getattr(self.tune_agent, "learning_rate", 0.001)
        tune_optimizer = (
            self.tune_agent.optimizer.__class__.__name__
            if self.tune_agent.optimizer
            else "None"
        )
        entropy_coef = getattr(self.tune_agent, "entropy_coef", 0.01)
        max_grad_norm = getattr(self.tune_agent, "max_grad_norm", 0.5)

        # Calculate statistics
        # Per-task statistics
        timesteps_per_iteration = self.fcfg["rollout_length"] * self.fcfg["batch_size"]
        timesteps_per_task = self.fcfg["n_iteration"] * timesteps_per_iteration
        total_timesteps = len(self.env.tasks) * timesteps_per_task

        # PPO updates per task = number of gradient steps on mini-batches per task
        minibatches_per_epoch = -(
            -timesteps_per_iteration // self.fcfg["minibatch_size"]
        )  # ceiling division
        ppo_updates_per_task = (
            self.fcfg["n_iteration"] * self.fcfg["ppo_epochs"] * minibatches_per_epoch
        )

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        lines = [
            f"\n{'=' * 80}",
            f"                       FINE-TUNING PHASE STARTED",
            f"{'=' * 80}",
            f"Timestamp               : {timestamp}",
            f"",
            f"Training Schedule",
            f"  Tasks                   : {len(self.env.tasks)}",
            f"  Iterations/Task         : {self.fcfg['n_iteration']}",
            f"  Rollout Length          : {self.fcfg['rollout_length']} timesteps per iteration",
            f"  Batch Size              : {self.fcfg['batch_size']} (parallel environments)",
            f"  PPO Epochs              : {self.fcfg['ppo_epochs']} (gradient passes per rollout)",
            f"  Mini-batch Size         : {self.fcfg['minibatch_size']}",
            f"  Total Timesteps         : {total_timesteps:,}",
            f"  Timesteps Per Task      : {timesteps_per_task:,}",
            f"  PPO Updates Per Task    : {ppo_updates_per_task:,}",
            f"",
            f"Evaluation & Checkpointing",
            f"  Eval Interval           : {self.fcfg['eval_interval']} iteration(s)",
            f"  Checkpoint Interval     : {self.fcfg['checkpoint_interval']} iteration(s)",
            f"  Early Stopping Patience : {self.fcfg['patience']} evals",
            f"  Min Delta (improvement) : {self.fcfg['min_delta']}",
            f"",
            f"Configuration",
            f"  Fine-Tune LR            : {tune_lr}",
            f"  Fine-Tune Optimizer     : {tune_optimizer}",
            f"  Entropy Coefficient     : {entropy_coef}",
            f"  Max Grad Norm           : {max_grad_norm}",
            f"  GAE Lambda              : {self.fcfg['gae_lambda']}",
            f"  Gamma (discount)        : {self.fcfg['gamma']}",
            f"  Device                  : {globals.DEVICE}",
            f"",
            f"Output Directories",
            f"  Experiment              : {self.logger.log_dir.parent}",
            f"  Checkpoints             : {self.logger.checkpoint_dir}",
            f"  Logs                    : {self.logger.log_dir}",
            f"",
            f"{'=' * 80}",
        ]

        print("\n".join(lines))

    def _compute_task_losses(
        self, vec_env: Any
    ) -> Tuple[torch.Tensor, Dict[Any, Dict[str, float]]]:
        """Compute task losses across all active tasks using FOMAML (first-order MAML).

        For each active task:
          1. Clone sub_agent from meta_agent (fresh copy)
          2. Inner loop: collect support set, compute gradients without stepping
          3. Compute adapted parameters using gradient information
          4. Outer loop: collect query set, evaluate with adapted parameters
          5. Return query loss (connected to meta_agent for outer-loop update)

        Args:
            vec_env: Vectorized environment (SubprocVecEnv with n_envs batch_size)

        Returns:
            (task_losses_tensor, task_metrics)
        """
        task_losses: List[torch.Tensor] = []
        task_metrics: Dict[Any, Dict[str, float]] = {}

        # Extract agent hyperparameters (safe access for type checker)
        clip_eps = getattr(self.sub_agent, "clip_eps", 0.2)
        value_coef = getattr(self.sub_agent, "value_coef", 0.5)
        rollout_length = self.mcfg["rollout_length"]

        # Process each active task
        for task_id in self.active_tasks:
            # Clone meta_agent to sub_agent (fresh copy for this task)
            sub_agent = self.sub_agent
            sub_agent.clone(self.meta_agent)

            # Inner loop: collect support set and compute gradients (FOMAML)
            vec_env.retask(task_id)
            support_batch = self.collect(
                sub_agent, vec_env, rollout_length, gamma=0.99, gae_lambda=0.95
            )

            # Compute support loss manually to retain graph for adapted parameters
            observations = support_batch["observations"]
            masks = support_batch["masks"]
            old_log_probs = support_batch["log_probs"]
            advantages = support_batch["advantages"]
            returns = support_batch["returns"]

            # Convert observations from batch format (list of stacked dicts/tensors)
            # to per-timestep format (list of single obs per timestep)
            # collect() returns obs with batch dimension; extract [0] for batch_size
            if observations and isinstance(observations[0], dict):
                observations = [
                    {
                        k: v[0] if isinstance(v, torch.Tensor) else v[0]
                        for k, v in obs.items()
                    }
                    for obs in observations
                ]
            elif observations:
                observations = [
                    obs[0] if isinstance(obs, torch.Tensor) else obs[0]
                    for obs in observations
                ]

            # Flatten batch dimension from masks (collect returns (batch_size*T, n_actions))
            masks = masks.view(-1, masks.shape[-1])

            # Handle observations as list (graph-based with dynamic sizes)
            if isinstance(observations, list):
                # Process each observation through network and collect outputs
                # masks is (T, n_actions) after concatenation, so need to iterate properly
                logits_list = []
                values_list = []
                for i, obs_t in enumerate(observations):
                    mask_t = masks[i : i + 1]  # Keep batch dimension (1, n_actions)
                    logits_t, values_t, _ = sub_agent.network.evaluate(
                        obs_t, mask_t, actions=None
                    )
                    logits_list.append(logits_t)
                    values_list.append(values_t)
                logits = torch.cat(logits_list, dim=0)
                values = torch.cat(values_list, dim=0)
            else:
                logits, values, _ = sub_agent.network.evaluate(
                    observations, masks, actions=None
                )

            # Verify shape compatibility for gather operation
            assert logits.shape[:-1] == support_batch["actions"].shape, (
                f"Support logits batch shape {logits.shape[:-1]} != actions shape {support_batch['actions'].shape}"
            )

            ratio = torch.exp(
                torch.nn.functional.log_softmax(logits, dim=-1)
                .gather(-1, support_batch["actions"].unsqueeze(-1))
                .squeeze(-1)
                - old_log_probs
            )
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * advantages
            support_policy_loss = -torch.min(surr1, surr2).mean()
            support_value_loss = torch.nn.functional.mse_loss(values, returns)

            # Entropy regularization (match fine-tuning phase)
            entropy_coef = getattr(self.sub_agent, "entropy_coef", 0.01)
            support_entropies = support_batch.get("entropies", torch.tensor(0.0))
            if (
                isinstance(support_entropies, torch.Tensor)
                and support_entropies.numel() > 0
            ):
                support_entropy_loss = -support_entropies.mean()
            else:
                support_entropy_loss = torch.tensor(
                    0.0, device=support_policy_loss.device
                )

            support_loss = (
                support_policy_loss
                + value_coef * support_value_loss
                + entropy_coef * support_entropy_loss
            )
            support_loss_tensor = support_loss.detach()

            # Compute adapted parameters via gradient (FOMAML: no optimizer.step())
            # Inner learning rate (use same as outer for simplicity)
            inner_lr = (
                0.001
                if sub_agent.optimizer is None
                else sub_agent.optimizer.defaults.get("lr", 0.001)
            )
            grads = torch.autograd.grad(
                support_loss,
                sub_agent.network.parameters(),
                create_graph=True,
                allow_unused=True,  # Some params may not be connected to loss
            )
            # Handle None gradients (from unused parameters)
            grads_safe = [
                g if g is not None else torch.zeros_like(p)
                for p, g in zip(sub_agent.network.parameters(), grads)
            ]
            adapted_params = {
                n: p - inner_lr * g
                for (n, p), g in zip(sub_agent.network.named_parameters(), grads_safe)
            }

            # Outer loop: collect query set and evaluate with adapted parameters
            vec_env.retask(task_id)
            query_batch = self.collect(
                sub_agent, vec_env, rollout_length, gamma=0.99, gae_lambda=0.95
            )

            # Evaluate query loss with adapted parameters using functional_call
            # This maintains gradient flow through the adaptation for proper meta-gradients
            query_observations = query_batch["observations"]
            query_masks = query_batch["masks"]

            # Convert observations from batch format to per-timestep format
            if query_observations and isinstance(query_observations[0], dict):
                query_observations = [
                    {
                        k: v[0] if isinstance(v, torch.Tensor) else v[0]
                        for k, v in obs.items()
                    }
                    for obs in query_observations
                ]
            elif query_observations:
                query_observations = [
                    obs[0] if isinstance(obs, torch.Tensor) else obs[0]
                    for obs in query_observations
                ]

            # Flatten batch dimension from masks
            query_masks = query_masks.view(-1, query_masks.shape[-1])

            # Create params dict for functional_call (adapted parameters only)
            # Filter by actual parameters, not state_dict (which includes buffers)
            adapted_params_dict = {
                name: adapted_params[name]
                for name in dict(sub_agent.network.named_parameters()).keys()
                if name in adapted_params
            }

            # Helper function to evaluate network with functional_call
            def evaluate_with_adapted_params():
                if isinstance(query_observations, list):
                    # Process each observation with adapted parameters
                    query_logits_list = []
                    query_values_list = []
                    for i, obs_t in enumerate(query_observations):
                        mask_t = query_masks[i : i + 1]
                        # Use functional_call to evaluate with adapted params
                        outputs = functional_call(
                            sub_agent.network,
                            adapted_params_dict,
                            (obs_t, mask_t, None),
                        )
                        logits_t, values_t, _ = outputs
                        query_logits_list.append(logits_t)
                        query_values_list.append(values_t)
                    query_logits = torch.cat(query_logits_list, dim=0)
                    query_values = torch.cat(query_values_list, dim=0)
                else:
                    # Non-list observations
                    outputs = functional_call(
                        sub_agent.network,
                        adapted_params_dict,
                        (query_observations, query_masks, None),
                    )
                    query_logits, query_values, _ = outputs

                return query_logits, query_values

            query_logits, query_values = evaluate_with_adapted_params()

            # Compute query loss with adapted parameters
            query_old_log_probs = query_batch["log_probs"]
            query_advantages = query_batch["advantages"]
            query_returns = query_batch["returns"]
            query_entropies = query_batch.get("entropies", torch.tensor(0.0))

            # Verify shape compatibility for gather operation
            assert query_logits.shape[:-1] == query_batch["actions"].shape, (
                f"Logits batch shape {query_logits.shape[:-1]} != actions shape {query_batch['actions'].shape}"
            )

            query_ratio = torch.exp(
                torch.nn.functional.log_softmax(query_logits, dim=-1)
                .gather(-1, query_batch["actions"].unsqueeze(-1))
                .squeeze(-1)
                - query_old_log_probs
            )
            query_surr1 = query_ratio * query_advantages
            query_surr2 = (
                torch.clamp(query_ratio, 1 - clip_eps, 1 + clip_eps) * query_advantages
            )
            query_policy_loss = -torch.min(query_surr1, query_surr2).mean()
            query_value_loss = torch.nn.functional.mse_loss(query_values, query_returns)

            # Entropy regularization (match fine-tuning phase)
            entropy_coef = getattr(self.sub_agent, "entropy_coef", 0.01)
            if (
                isinstance(query_entropies, torch.Tensor)
                and query_entropies.numel() > 0
            ):
                query_entropy_loss = -query_entropies.mean()
            else:
                query_entropy_loss = torch.tensor(0.0, device=query_policy_loss.device)

            query_loss = (
                query_policy_loss
                + value_coef * query_value_loss
                + entropy_coef * query_entropy_loss
            )

            task_losses.append(query_loss)

            # Compute mean entropy from query batch for curriculum learning
            entropy = 0.0
            if "entropies" in query_batch:
                entropy = float(query_batch["entropies"].mean().item())

            task_metrics[task_id] = {
                "support_loss": support_loss_tensor.item(),
                "query_loss": query_loss.detach().item(),
                "improvement": (support_loss_tensor - query_loss.detach()).item(),
                "entropy": entropy,
            }

        return torch.stack(task_losses), task_metrics

    def _print_header(self) -> None:
        from datetime import datetime

        active_task_ids = sorted(self.active_tasks)
        total_tasks = len(self.env.tasks)

        # Determine algorithm name based on enabled phases
        if self.enable_meta_learning and self.enable_fine_tuning:
            algo = "FOMAML + PPO Fine-Tuning"
        elif self.enable_meta_learning:
            algo = "FOMAML (Meta-Learning Only)"
        elif self.enable_fine_tuning:
            algo = "PPO (Fine-Tuning Only)"
        else:
            algo = "INVALID"

        # Calculate training statistics
        # Meta-learning: one update per epoch
        meta_batches = self.mcfg["epochs"] if self.enable_meta_learning else 0

        # Fine-tuning: per-task statistics
        if self.enable_fine_tuning:
            timesteps_per_iteration = (
                self.fcfg["rollout_length"] * self.fcfg["batch_size"]
            )
            timesteps_per_task = self.fcfg["n_iteration"] * timesteps_per_iteration
            fine_timesteps = len(self.env.tasks) * timesteps_per_task

            # PPO updates per task = gradient steps on mini-batches per task
            minibatches_per_epoch = -(
                -timesteps_per_iteration // self.fcfg["minibatch_size"]
            )  # ceiling division
            fine_ppo_steps_per_task = (
                self.fcfg["n_iteration"]
                * self.fcfg["ppo_epochs"]
                * minibatches_per_epoch
            )
            fine_ppo_steps = len(self.env.tasks) * fine_ppo_steps_per_task
        else:
            fine_timesteps = 0
            fine_ppo_steps = 0
            fine_ppo_steps_per_task = 0

        total_updates = meta_batches + fine_ppo_steps

        # Get meta-agent hyperparameters
        meta_lr = getattr(self.meta_agent, "learning_rate", 0.001)
        meta_optimizer = (
            self.meta_agent.optimizer.__class__.__name__
            if self.meta_agent.optimizer
            else "None"
        )

        # Get tune-agent hyperparameters (if fine-tuning enabled)
        tune_lr = None
        tune_optimizer = None
        entropy_coef = None
        if self.enable_fine_tuning:
            tune_lr = getattr(self.tune_agent, "learning_rate", 0.001)
            tune_optimizer = (
                self.tune_agent.optimizer.__class__.__name__
                if self.tune_agent.optimizer
                else "None"
            )
            entropy_coef = getattr(self.tune_agent, "entropy_coef", 0.01)

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        lines = [
            f"\n{'=' * 80}",
            f"                          TRAINING STARTED",
            f"{'=' * 80}",
            f"Experiment              : {self.logger.experiment_name}",
            f"Timestamp               : {timestamp}",
            f"Algorithm               : {algo}",
            f"",
            f"Training Schedule",
        ]

        if self.enable_meta_learning:
            lines.extend(
                [
                    f"  Meta-Learning Phase",
                    f"    Epochs              : {self.mcfg['epochs']}",
                    f"    Batch Size          : {self.mcfg['batch_size']} (parallel environments)",
                    f"    Rollout Length      : {self.mcfg['rollout_length']} timesteps per collection",
                    f"    Updates/Epoch       : 1 (support + query per epoch)",
                    f"    Total Updates       : {meta_batches:,}",
                    f"    Task Curriculum     : Start with {len(self.active_tasks)} task(s), expand every {self.mcfg['curriculum_check_interval']} epochs if entropy < {self.mcfg['entropy_threshold']}",
                ]
            )

        if self.enable_fine_tuning:
            lines.extend(
                [
                    f"  Fine-Tuning Phase",
                    f"    Iterations/Task     : {self.fcfg['n_iteration']}",
                    f"    Rollout Length      : {self.fcfg['rollout_length']} timesteps per iteration",
                    f"    PPO Epochs          : {self.fcfg['ppo_epochs']} (gradient passes per rollout)",
                    f"    Mini-batch Size     : {self.fcfg['minibatch_size']}",
                    f"    Total Timesteps     : {fine_timesteps:,}",
                    f"    PPO Updates/Task    : {fine_ppo_steps_per_task:,}",
                    f"    Total PPO Updates   : {fine_ppo_steps:,}",
                    f"    Tasks               : {len(self.env.tasks)} task(s)",
                ]
            )

        lines.extend(
            [
                f"",
                f"Training Summary",
                f"  Total Timesteps (Meta+Fine) : {meta_batches + fine_timesteps:,}",
                f"  Total Gradient Updates      : {total_updates:,}",
                f"  Environment                 : {self.env.__class__.__name__}",
                f"  Device                      : {globals.DEVICE}",
                f"",
                f"Configuration",
                f"  Meta LR                     : {meta_lr}",
                f"  Meta Optimizer              : {meta_optimizer}",
                f"  Meta Max Grad Norm          : {self.meta_agent.max_grad_norm if hasattr(self.meta_agent, 'max_grad_norm') else 0.5}",
            ]
        )

        if self.enable_fine_tuning:
            lines.extend(
                [
                    f"  Tune LR                     : {tune_lr}",
                    f"  Tune Optimizer              : {tune_optimizer}",
                    f"  Tune Max Grad Norm          : {self.tune_agent.max_grad_norm if hasattr(self.tune_agent, 'max_grad_norm') else 0.5}",
                    f"  Entropy Coefficient         : {entropy_coef}",
                    f"  GAE Lambda                  : {self.fcfg['gae_lambda']}",
                    f"  Gamma (discount)            : {self.fcfg['gamma']}",
                ]
            )

        lines.extend(
            [
                f"  Rollout Strategy            : Serial (1 environment), variable-length batches",
                f"",
                f"Output Directories",
                f"  Experiment                  : {self.logger.log_dir.parent}",
                f"  Checkpoints                 : {self.logger.checkpoint_dir}",
                f"  Logs                        : {self.logger.log_dir}",
                f"",
                f"{'=' * 80}",
            ]
        )

        print("\n".join(lines))

    def _print_footer(self, summary: Dict) -> None:
        phases = []
        if summary.get("meta_learning_enabled"):
            phases.append("meta-learning")
        if summary.get("fine_tuning_enabled"):
            phases.append("fine-tuning")
        phases_str = " + ".join(phases) if phases else "none"

        print(
            f"\n{'=' * 64}\n"
            f"  Done ({summary['stop_reason']})\n"
            f"  Phases     : {phases_str}\n"
            f"  Updates    : {summary['total_updates']:,}\n"
            f"  Epochs     : {summary['total_epochs']:,}\n"
            f"  Best Obj   : {summary['best_objective']:.4f}\n"
            f"  Time       : {summary['training_time_s']:.1f}s\n"
            f"{'=' * 64}\n"
        )

    def _print_header_task(self, task_id: str) -> None:
        print(f"\n{'-' * 64}\n  Task: {task_id}\n{'-' * 64}")


# ---------------------------------------------------------------------------
# POMOTrainer: Policy Optimization with Multiple Optima
# ---------------------------------------------------------------------------


class POMOTrainer(BaseTrainer):
    """
    POMOTrainer: Policy Optimization with Multiple Optima.

    Trains a separate sub-policy for each task independently using POMO collection.
    For each task:
      - Collect episodes from multiple starting points per instance
      - Compute policy gradients via POMOAgent
      - Train for N epochs with early stopping and checkpointing

    Collection: POMOSampler (multiple starting points per task instance)
    Agent: POMOAgent (empirical advantage = reward - baseline)
    """

    def __init__(
        self,
        agents: Dict[str, BaseAgent],
        env: Any,
        trainer_cfg: Dict[str, Any],
        evaluators: Dict[str, Any],
        logger: Any,
    ):
        self.agents = agents
        self.train_agent = agents["train_agent"]
        self.env = env
        self.evaluator = evaluators["train_eval"]
        self.logger = logger

        # Setup task iteration from env
        if not env.tasks:
            raise ValueError("POMOTrainer requires env.tasks")

        # Extract config from training phase
        phases_cfg = trainer_cfg.get("phases", {})
        training_phase = phases_cfg.get("training", {})
        control_cfg = training_phase.get("control", {})
        early_stop_cfg = training_phase.get("early_stopping", {})

        self.tcfg = {
            "epochs": int(control_cfg["epochs"]),
            "batches_per_epoch": int(control_cfg["batches_per_epoch"]),
            "instances_per_batch": int(control_cfg["instances_per_batch"]),
            "eval_interval": int(control_cfg.get("eval_interval", 1)),
            "checkpoint_interval": int(control_cfg["checkpoint_interval"]),
            "patience": int(early_stop_cfg.get("patience", 20)),
            "min_delta": float(early_stop_cfg.get("min_delta", 0.0001)),
        }

        self._total_updates = 0
        self._total_instances = 0

    @classmethod
    def from_config(
        cls,
        trainer_cfg: Dict[str, Any],
        agents: Dict[str, BaseAgent],
        env: Any,
        evaluators: Dict[str, Any],
        logger: Any,
        env_cfg: Dict[str, Any],
    ) -> "POMOTrainer":
        return cls(
            agents=agents,
            env=env,
            trainer_cfg=trainer_cfg,
            evaluators=evaluators,
            logger=logger,
        )

    def collect(
        self,
        agent: BaseAgent,
        env: Any,
    ) -> Dict[str, Any]:
        """Collect episodes from multiple starting points (POMO-style).

        For each feasible starting action:
          - Compute log_prob for that action
          - Take that action and roll out complete episode
          - Accumulate log probabilities and rewards
          - Store (episode_log_prob_sum, episode_return)

        Args:
            agent: agent instance
            env: environment (already initialized with a task)

        Returns:
            dict with keys:
              - "log_probs": list of summed log_probs (one per episode)
              - "rewards": list of episode returns (one per episode)
              - "entropies": list of episode entropies
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
            return {"log_probs": [], "rewards": [], "entropies": []}

        action_log_probs = {}
        for action in feasible_actions:
            act_t = torch.tensor([action], dtype=torch.long, device=globals.DEVICE)
            log_prob, _, _ = agent.network.evaluate(obs_t, mask_t, actions=act_t)
            action_log_probs[int(action)] = log_prob

        episode_log_probs = []
        episode_returns = []
        episode_entropies = []

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

                # Collect rest of trajectory, accumulating log probabilities and rewards
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

            # Store episode results
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

    def train(self) -> Dict[str, Any]:
        """Run POMO training loop with epoch-based batch training per task.

        For each task:
          - Train for N epochs, accumulating returns from all POMO starting points
          - Log per-epoch aggregated statistics (mean/std/percentiles)
          - Evaluate using self.evaluator every eval_interval epochs
          - Track best objective and save checkpoints
        """
        start_time = time.time()
        self._print_header()
        agent = self.train_agent

        epochs = self.tcfg["epochs"]
        batches_per_epoch = self.tcfg["batches_per_epoch"]
        instances_per_batch = self.tcfg["instances_per_batch"]
        eval_interval = self.tcfg["eval_interval"]
        checkpoint_interval = self.tcfg["checkpoint_interval"]

        all_task_summaries = {}

        for task_id in self.env.tasks:
            self._print_header_task(task_id)
            self.env.retask(task_id)

            best_objective = float(
                "-inf"
            )  # For maximization problems, higher is better
            best_epoch = -1
            patience_counter = 0

            for epoch in range(epochs):
                epoch_start = time.time()
                epoch_losses = []
                epoch_returns = []
                epoch_grad_norms = []

                # Training phase: accumulate statistics across all batches
                for batch_idx in range(batches_per_epoch):
                    batch_log_probs = []  # List of tensors, one per instance
                    batch_rewards = []  # List of tensors, one per instance
                    batch_entropies = []  # List of tensors, one per instance

                    for _ in range(instances_per_batch):
                        self.env.retask(task_id)
                        batch_data = self.collect(agent, self.env)
                        # Stack episodes for this instance
                        if batch_data["log_probs"]:
                            instance_log_probs = torch.stack(batch_data["log_probs"])
                            instance_rewards = torch.stack(batch_data["rewards"])
                            instance_entropies = torch.stack(
                                batch_data.get("entropies", [])
                            )
                            batch_log_probs.append(instance_log_probs)
                            batch_rewards.append(instance_rewards)
                            batch_entropies.append(instance_entropies)

                    # Update after batch
                    if batch_log_probs:
                        batch = {
                            "log_probs": batch_log_probs,  # List of (num_starting_points_i,) tensors
                            "rewards": batch_rewards,  # List of (num_starting_points_i,) tensors
                            "entropies": batch_entropies,  # List of (num_starting_points_i,) tensors
                        }
                        metrics = agent.update(batch)
                        loss_val = metrics.get("loss", 0.0)
                        loss_val = (
                            loss_val.detach()
                            if isinstance(loss_val, torch.Tensor)
                            else loss_val
                        )
                        epoch_losses.append(float(loss_val))
                        grad_norm_val = metrics.get("grad_norm", -1.0)
                        grad_norm_val = (
                            grad_norm_val.detach()
                            if isinstance(grad_norm_val, torch.Tensor)
                            else grad_norm_val
                        )
                        epoch_grad_norms.append(float(grad_norm_val))
                        self._total_updates += 1
                    else:
                        epoch_losses.append(0.0)

                    # Flatten instance rewards (list of tensors) into epoch_returns
                    # Batch GPU→CPU transfer instead of per-instance
                    gpu_rewards = [
                        r for r in batch_rewards if isinstance(r, torch.Tensor)
                    ]
                    if gpu_rewards:
                        batch_rewards_gpu = torch.cat(gpu_rewards)
                        epoch_returns.extend(batch_rewards_gpu.detach().cpu().tolist())
                    # Handle non-tensor rewards
                    for r in batch_rewards:
                        if not isinstance(r, torch.Tensor):
                            epoch_returns.extend(r)
                    self._total_instances += instances_per_batch

                # Compute per-epoch training statistics
                epoch_time = time.time() - epoch_start
                train_returns = np.array(epoch_returns)

                epoch_losses_array = (
                    np.array(epoch_losses) if epoch_losses else np.array([])
                )
                epoch_grad_norms_array = (
                    np.array(epoch_grad_norms) if epoch_grad_norms else np.array([])
                )

                train_metrics = {
                    "train/loss_mean": float(np.mean(epoch_losses))
                    if epoch_losses
                    else 0.0,
                    "train/loss_std": float(np.std(epoch_losses))
                    if epoch_losses
                    else 0.0,
                    "train/return_mean": float(np.mean(train_returns))
                    if len(train_returns) > 0
                    else 0.0,
                    "train/return_std": float(np.std(train_returns))
                    if len(train_returns) > 0
                    else 0.0,
                    "train/return_p10": float(np.percentile(train_returns, 10))
                    if len(train_returns) > 0
                    else 0.0,
                    "train/return_p50": float(np.percentile(train_returns, 50))
                    if len(train_returns) > 0
                    else 0.0,
                    "train/return_p90": float(np.percentile(train_returns, 90))
                    if len(train_returns) > 0
                    else 0.0,
                    "train/grad_norm_mean": float(np.mean(epoch_grad_norms))
                    if epoch_grad_norms
                    else 0.0,
                    "train/grad_norm_max": float(np.max(epoch_grad_norms))
                    if epoch_grad_norms
                    else 0.0,
                    "train/total_updates": float(self._total_updates),
                    "train/total_instances": float(self._total_instances),
                    "train/epoch_time_s": epoch_time,
                }

                # Evaluation
                eval_metrics = {}
                if (epoch + 1) % eval_interval == 0:
                    eval_stats = self.evaluator.evaluate(task_id)
                    eval_metrics = {f"eval/{k}": v for k, v in eval_stats.items()}

                    # Track best objective with early stopping
                    mean_obj = eval_stats.get("mean_objective", float("-inf"))
                    if mean_obj > best_objective + self.tcfg["min_delta"]:
                        best_objective = mean_obj
                        best_epoch = epoch + 1
                        patience_counter = 0
                        self.logger.save_checkpoint(
                            f"{task_id}_best",
                            {
                                "network_state": agent.network.state_dict(),
                                "epoch": epoch + 1,
                                "mean_objective": mean_obj,
                            },
                        )
                        self.logger.log_event(
                            "best_checkpoint",
                            self._total_updates,
                            task=task_id,
                            epoch=epoch + 1,
                            objective=f"{mean_obj:.4f}",
                        )
                    else:
                        patience_counter += 1

                    # Early stopping check
                    if patience_counter >= self.tcfg["patience"]:
                        self.logger.log_event(
                            "early_stop",
                            self._total_updates,
                            task=task_id,
                            patience=self.tcfg["patience"],
                        )
                        break

                # Log all metrics
                all_metrics = {**train_metrics, **eval_metrics}
                print_keys = [
                    "train/loss_mean",
                    "train/return_mean",
                    "train/return_std",
                    "train/grad_norm_mean",
                ]
                if eval_metrics:
                    print_keys.extend(
                        [
                            "eval/mean_objective",
                            "eval/std_objective",
                        ]
                    )
                    if "eval/mean_cost" in eval_metrics:
                        print_keys.append("eval/mean_cost")
                    if "eval/mean_service_rate" in eval_metrics:
                        print_keys.append("eval/mean_service_rate")

                self.logger.log_metrics(
                    all_metrics,
                    step=self._total_updates,
                    total_steps=self.tcfg["epochs"] * self.tcfg["batches_per_epoch"],
                    print_keys=print_keys,
                )

                # Periodic checkpoint
                if (epoch + 1) % checkpoint_interval == 0:
                    self.logger.save_checkpoint(
                        f"{task_id}_epoch_{epoch + 1}",
                        {
                            "network_state": agent.network.state_dict(),
                            "epoch": epoch + 1,
                        },
                    )

            # Final checkpoint for task
            self.logger.save_checkpoint(
                f"{task_id}_final",
                {
                    "network_state": agent.network.state_dict(),
                    "epoch": epochs,
                },
            )

            task_summary = {
                "best_objective": float(best_objective),
                "best_epoch": best_epoch,
                "final_epoch": epochs,
            }
            all_task_summaries[task_id] = task_summary

            # Task completion summary
            print(f"\n{'-' * 64}")
            print(f"Task {task_id} Complete")
            print(f"  Best objective: {best_objective:.4f} (epoch {best_epoch})")
            print(f"  Total epochs: {epochs}")
            print(f"  Total updates: {self._total_updates}")
            print(f"{'-' * 64}\n")

            self.logger.log_event(
                "task_complete",
                self._total_updates,
                task=task_id,
                **task_summary,
            )

        # Experiment summary
        summary = {
            "stop_reason": "completed",
            "total_updates": self._total_updates,
            "total_instances": self._total_instances,
            "training_time_s": round(time.time() - start_time, 1),
            "task_summaries": all_task_summaries,
        }

        self.logger.log_event(
            "training_complete",
            self._total_updates,
            total_updates=self._total_updates,
            total_instances=self._total_instances,
            training_time_s=summary["training_time_s"],
        )
        self.logger.save_summary(summary)
        self.logger.close()
        self._print_footer(summary)
        return summary

    def _print_header(self) -> None:
        total_tasks = len(self.env.tasks)
        total_batches = self.tcfg["epochs"] * self.tcfg["batches_per_epoch"]
        total_instances = (
            total_batches * self.tcfg["instances_per_batch"]
            if self.tcfg["instances_per_batch"] > 0
            else total_batches
        )

        train_lr = getattr(self.train_agent, "learning_rate", 0.001)
        train_optimizer = (
            self.train_agent.optimizer.__class__.__name__
            if self.train_agent.optimizer
            else "None"
        )
        entropy_coef = getattr(self.train_agent, "entropy_coef", 0.01)
        max_grad_norm = getattr(self.train_agent, "max_grad_norm", 0.5)

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        lines = [
            f"\n{'=' * 80}",
            f"                          TRAINING STARTED",
            f"{'=' * 80}",
            f"Experiment              : {self.logger.experiment_name}",
            f"Timestamp               : {timestamp}",
            f"Algorithm               : POMO (Policy Optimization with Multiple Optima)",
            f"",
            f"Training Schedule",
            f"  Epochs                  : {self.tcfg['epochs']}",
            f"  Batches/Epoch           : {self.tcfg['batches_per_epoch']}",
            f"  Instances/Batch         : {self.tcfg['instances_per_batch']}",
            f"  Eval Interval           : {self.tcfg['eval_interval']}",
            f"  Checkpoint Interval     : {self.tcfg['checkpoint_interval']}",
            f"  Total Batches           : {total_batches:,}",
            f"  Total Instances         : {total_instances:,}",
            f"",
            f"Training Summary",
            f"  Tasks                   : {total_tasks}",
            f"  Environment             : {self.env.__class__.__name__}",
            f"  Device                  : {globals.DEVICE}",
            f"",
            f"Configuration",
            f"  Learning Rate           : {train_lr}",
            f"  Optimizer               : {train_optimizer}",
            f"  Entropy Coefficient     : {entropy_coef}",
            f"  Max Grad Norm           : {max_grad_norm}",
            f"  Rollout Strategy        : Serial (1 environment), per-task training",
            f"",
            f"Output Directories",
            f"  Experiment              : {self.logger.log_dir.parent}",
            f"  Checkpoints             : {self.logger.checkpoint_dir}",
            f"  Logs                    : {self.logger.log_dir}",
            f"",
            f"{'=' * 80}",
        ]

        print("\n".join(lines))

    def _print_header_task(self, task_id: str) -> None:
        print(f"\n{'-' * 64}\n  Task: {task_id}\n{'-' * 64}")

    def _print_footer(self, summary: Dict) -> None:
        print(
            f"\n{'=' * 64}\n"
            f"  Done ({summary['stop_reason']})\n"
            f"  Updates    : {summary['total_updates']:,}\n"
            f"  Instances  : {summary['total_instances']:,}\n"
            f"  Time       : {summary['training_time_s']:.1f}s\n"
            f"{'=' * 64}\n"
        )
