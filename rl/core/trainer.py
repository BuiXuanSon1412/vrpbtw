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
from typing import Any, Dict, List, Tuple

import numpy as np
import torch

import globals
from core.agent import BaseAgent
from core.collector import BaseCollector


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
        collector: BaseCollector,
    ) -> "BaseTrainer":
        """Factory method: instantiate trainer from config.

        Args:
            trainer_cfg: trainer config dict (cfg.trainer)
            agents: dict of agent instances (keyed by name)
            env: environment instance (has tasks list and reset interface)
            evaluators: dict of evaluator instances (keyed by phase name, or "default" for single-phase)
            logger: logger instance
            collector: collector instance for trajectory collection
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
        collector: BaseCollector,
    ):
        self.agents = agents
        self.meta_agent = agents["meta_agent"]
        self.sub_agent = agents["sub_agent"]
        self.tune_agent = agents["tune_agent"]
        self.env = env
        self.collector = collector
        self.active_tasks = {env.tasks[0]}  # Start with first (easiest) task
        self.meta_evaluator = evaluators["meta_eval"]
        self.fine_evaluator = evaluators["tune_eval"]
        self.logger = logger

        # Extract config from trainer structure
        phases_cfg = trainer_cfg.get("phases", {})

        # Get meta_learning phase
        meta_phase = phases_cfg.get("meta_learning", {})
        self.enable_meta_learning = bool(meta_phase.get("enabled", True))
        curriculum_cfg = meta_phase.get("curriculum", {})
        meta_control_cfg = meta_phase.get("control", {})
        meta_early_stop = meta_control_cfg.get("early_stopping", {})

        # Meta-learning config: epochs/batches instead of timesteps
        self.mcfg = {
            "entropy_threshold": float(curriculum_cfg.get("entropy_threshold", 0.5)),
            "curriculum_check_interval": int(curriculum_cfg.get("check_interval", 1)),
            "epochs": int(meta_control_cfg.get("epochs", 200)),
            "batches_per_epoch": int(meta_control_cfg.get("batches_per_epoch", 50)),
            "eval_interval": int(meta_control_cfg.get("eval_interval", 1)),
            "checkpoint_interval": int(meta_control_cfg.get("checkpoint_interval", 10)),
            "patience": int(meta_early_stop.get("patience", 20)),
            "min_delta": float(meta_early_stop.get("min_delta", 0.0001)),
        }

        # Get fine_tuning phase config
        fine_tune_phase = phases_cfg.get("fine_tuning", {})
        self.enable_fine_tuning = bool(fine_tune_phase.get("enabled", True))
        fine_control_cfg = fine_tune_phase.get("control", {})
        fine_early_stop = fine_control_cfg.get("early_stopping", {})

        # Fine-tuning config: epochs/batches instead of timesteps
        self.fcfg = {
            "epochs": int(fine_control_cfg.get("epochs", 50)),
            "batches_per_epoch": int(fine_control_cfg.get("batches_per_epoch", 100)),
            "ppo_epochs": int(fine_control_cfg.get("ppo_epochs", 3)),
            "eval_interval": int(fine_control_cfg.get("eval_interval", 1)),
            "checkpoint_interval": int(fine_control_cfg.get("checkpoint_interval", 10)),
            "patience": int(fine_early_stop.get("patience", 10)),
            "min_delta": float(fine_early_stop.get("min_delta", 0.0001)),
        }

        # Training state
        self._total_updates = 0
        self._best_objective = float("-inf")
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
        collector: BaseCollector,
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
            collector=collector,
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
            raise ValueError("At least one of meta_learning or fine_tuning must be enabled")

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
            "stop_reason": meta_summary.get("stop_reason", fine_tune_summary.get("stop_reason", "completed")),
            "total_updates": self._total_updates,
            "total_epochs": meta_summary.get("total_epochs", 0)
            + fine_tune_summary.get("total_epochs", 0),
            "best_objective": max(
                meta_summary.get("best_objective", float("-inf")),
                fine_tune_summary.get("best_objective", float("-inf")),
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
        self._print_header()
        start_time = time.time()
        stop_reason = "completed"
        epoch = -1

        try:
            for epoch in range(self.mcfg["epochs"]):
                epoch_start = time.time()
                epoch_losses = []
                epoch_task_metrics = {}

                try:
                    for _ in range(self.mcfg["batches_per_epoch"]):
                        # Compute task losses across active tasks
                        task_losses, task_metrics = self._compute_task_losses()

                        # Update meta-policy on aggregated task losses
                        self.meta_agent.update({"task_losses": task_losses})
                        self._total_updates += 1

                        epoch_losses.extend(task_losses)
                        for task_id, metrics_dict in task_metrics.items():
                            if task_id not in epoch_task_metrics:
                                epoch_task_metrics[task_id] = {}
                            for key, val in metrics_dict.items():
                                if key not in epoch_task_metrics[task_id]:
                                    epoch_task_metrics[task_id][key] = []
                                epoch_task_metrics[task_id][key].append(val)

                        # Curriculum check per batch
                        max_entropy = None
                        for task_id, metrics_dict in task_metrics.items():
                            entropy = metrics_dict.get("entropy", 0)
                            if max_entropy is None or entropy > max_entropy:
                                max_entropy = entropy

                        if max_entropy is not None:
                            self._curriculum_check_counter += 1
                            if (
                                self._curriculum_check_counter
                                >= self.mcfg["curriculum_check_interval"]
                            ):
                                self._curriculum_check_counter = 0
                                if max_entropy < self.mcfg["entropy_threshold"]:
                                    if len(self.active_tasks) < len(self.env.tasks):
                                        next_task = self.env.tasks[
                                            len(self.active_tasks)
                                        ]
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
                        message=f"Error during meta-training batch computation in epoch {epoch}",
                        step=self._total_updates,
                        epoch=epoch,
                    )
                    raise

                # Per-epoch aggregation
                epoch_time = time.time() - epoch_start
                epoch_losses_array = (
                    torch.stack(epoch_losses).detach().cpu().numpy()
                    if epoch_losses
                    else np.array([])
                )
                train_metrics = {
                    "meta/loss_mean": float(np.mean(epoch_losses_array))
                    if len(epoch_losses_array) > 0
                    else 0.0,
                    "meta/loss_std": float(np.std(epoch_losses_array))
                    if len(epoch_losses_array) > 0
                    else 0.0,
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

    def fine_tune(self) -> Dict[str, Any]:
        """Fine-tune policy on each task independently after meta-training.

        Uses the tune_agent to adapt the meta-learned policy
        to each task independently using PPO-style optimization.
        """
        self._print_header_tune()
        start_time = time.time()
        agent = self.tune_agent

        total_epochs = 0
        best_objective = float("-inf")
        task_summaries = {}

        try:
            for task_id in self.env.tasks:
                self._print_header_task(task_id)
                task_best_objective = float("-inf")
                task_patience_counter = 0
                epoch = -1

                try:
                    for epoch in range(self.fcfg["epochs"]):
                        epoch_start = time.time()
                        epoch_losses = []
                        epoch_grad_norms = []

                        try:
                            for batch_idx in range(self.fcfg["batches_per_epoch"]):
                                # Collect trajectory
                                self.env.retask(task_id)
                                batch = self.collector.collect(agent, self.env)

                                # Reuse each rollout for ppo_epochs gradient updates.
                                # old_log_probs stay fixed (from collection); the clipped
                                # surrogate ratio naturally limits drift across epochs.
                                for _ in range(self.fcfg["ppo_epochs"]):
                                    metrics = agent.update(batch)
                                    self._total_updates += 1
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
                        except Exception as e:
                            self.logger.log_exception(
                                e,
                                message=f"Error during batch collection/update for task {task_id} in epoch {epoch}",
                                step=self._total_updates,
                                task=str(task_id),
                                epoch=epoch,
                            )
                            raise

                        # Per-epoch aggregation
                        epoch_time = time.time() - epoch_start
                        train_metrics = {
                            "tune/loss_mean": float(np.mean(epoch_losses))
                            if epoch_losses
                            else 0.0,
                            "tune/loss_std": float(np.std(epoch_losses))
                            if epoch_losses
                            else 0.0,
                            "tune/grad_norm_mean": float(np.mean(epoch_grad_norms))
                            if epoch_grad_norms
                            else 0.0,
                            "tune/grad_norm_max": float(np.max(epoch_grad_norms))
                            if epoch_grad_norms
                            else 0.0,
                            "tune/total_updates": float(self._total_updates),
                            "tune/epoch_time_s": epoch_time,
                        }

                        # Evaluation
                        eval_metrics = {}
                        if (epoch + 1) % self.fcfg["eval_interval"] == 0:
                            try:
                                eval_stats = self.fine_evaluator.evaluate(task_id)
                                eval_metrics = {
                                    f"eval/{k}": v for k, v in eval_stats.items()
                                }

                                mean_obj = eval_stats.get(
                                    "mean_objective", float("-inf")
                                )
                                if (
                                    mean_obj
                                    > task_best_objective + self.fcfg["min_delta"]
                                ):
                                    task_best_objective = mean_obj
                                    task_patience_counter = 0
                                    self.logger.save_checkpoint(
                                        f"tune_best_{task_id}",
                                        {
                                            "network_state": agent.network.state_dict(),
                                            "epoch": epoch + 1,
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

                                if task_patience_counter >= self.fcfg["patience"]:
                                    self.logger.log_event(
                                        "tune_early_stop",
                                        self._total_updates,
                                        task=task_id,
                                        patience=self.fcfg["patience"],
                                    )
                                    break
                            except Exception as e:
                                self.logger.log_exception(
                                    e,
                                    message=f"Error during evaluation for task {task_id} in epoch {epoch}",
                                    step=self._total_updates,
                                    task=str(task_id),
                                    epoch=epoch,
                                )
                                raise

                        # Log all metrics
                        all_metrics = {**train_metrics, **eval_metrics}
                        print_keys = [
                            "tune/loss_mean",
                            "tune/grad_norm_mean",
                            "tune/total_updates",
                        ]
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
                        message=f"Error during fine-tuning epoch loop for task {task_id}",
                        step=self._total_updates,
                        task=str(task_id),
                    )
                    raise

                best_objective = max(best_objective, task_best_objective)
                total_epochs += epoch + 1
                task_summaries[task_id] = {
                    "best_objective": float(task_best_objective),
                    "epochs_completed": epoch + 1,
                }

                self.logger.save_checkpoint(
                    f"tune_final_{task_id}",
                    {
                        "network_state": agent.network.state_dict(),
                        "epoch": epoch + 1,
                    },
                )
        except Exception as e:
            self.logger.log_exception(
                e,
                message="Fatal error during fine-tuning",
                step=self._total_updates,
            )
            raise

        summary = {
            "stop_reason": "completed",
            "total_epochs": total_epochs,
            "total_updates": self._total_updates,
            "best_objective": best_objective,
            "training_time_s": round(time.time() - start_time, 1),
            "task_summaries": task_summaries,
        }
        self.logger.log_event("fine_tuning_complete", self._total_updates, **summary)
        return summary

    def _print_header_tune(self) -> None:
        print(
            f"\n{'=' * 64}\n"
            f"  Fine-Tuning Phase\n"
            f"  Tasks: {len(self.env.tasks)}\n"
            f"  Epochs/task: {self.fcfg['epochs']} × {self.fcfg['batches_per_epoch']} batches\n"
            f"{'=' * 64}"
        )

    def _compute_task_losses(
        self,
    ) -> Tuple[torch.Tensor, Dict[Any, Dict[str, float]]]:
        """Compute task losses across all active tasks using FOMAML (first-order MAML).

        For each active task:
          1. Clone sub_agent from meta_agent (fresh copy)
          2. Inner loop: collect support set, compute gradients without stepping
          3. Compute adapted parameters using gradient information
          4. Outer loop: collect query set, evaluate with adapted parameters
          5. Return query loss (connected to meta_agent for outer-loop update)

        Returns:
            (task_losses_tensor, task_metrics)
        """
        task_losses: List[torch.Tensor] = []
        task_metrics: Dict[Any, Dict[str, float]] = {}

        # Process each active task
        for task_id in self.active_tasks:
            # Clone meta_agent to sub_agent (fresh copy for this task)
            sub_agent = self.sub_agent
            sub_agent.clone(self.meta_agent)

            # Inner loop: collect support set and compute gradients (FOMAML)
            self.env.retask(task_id)
            support_batch = self.collector.collect(sub_agent, self.env)

            # Compute support loss manually to retain graph for adapted parameters
            observations = support_batch["observations"]
            masks = support_batch["masks"]
            old_log_probs = support_batch["log_probs"]
            advantages = support_batch["advantages"]
            returns = support_batch["returns"]

            # Handle observations as list (graph-based with dynamic sizes)
            if isinstance(observations, list):
                # Process each observation through network and collect outputs
                # masks is (T, n_actions) after concatenation, so need to iterate properly
                logits_list = []
                values_list = []
                for i, obs_t in enumerate(observations):
                    mask_t = masks[i:i+1]  # Keep batch dimension (1, n_actions)
                    logits_t, values_t, _ = sub_agent.network.evaluate(obs_t, mask_t, actions=None)
                    logits_list.append(logits_t)
                    values_list.append(values_t)
                logits = torch.cat(logits_list, dim=0)
                values = torch.cat(values_list, dim=0)
            else:
                logits, values, _ = sub_agent.network.evaluate(observations, masks, actions=None)
            ratio = torch.exp(torch.nn.functional.log_softmax(logits, dim=-1).gather(-1, support_batch["actions"].unsqueeze(-1)).squeeze(-1) - old_log_probs)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - sub_agent.clip_eps, 1 + sub_agent.clip_eps) * advantages
            support_loss = -torch.min(surr1, surr2).mean() + sub_agent.value_coef * torch.nn.functional.mse_loss(values, returns)
            support_loss_tensor = support_loss.detach()

            # Compute adapted parameters via gradient (FOMAML: no optimizer.step())
            # Inner learning rate (use same as outer for simplicity)
            inner_lr = 0.001 if sub_agent.optimizer is None else sub_agent.optimizer.defaults.get('lr', 0.001)
            grads = torch.autograd.grad(support_loss, sub_agent.network.parameters(), create_graph=True)
            adapted_params = {n: p - inner_lr * g for (n, p), g in zip(sub_agent.network.named_parameters(), grads)}

            # Outer loop: collect query set and evaluate with adapted parameters
            self.env.retask(task_id)
            query_batch = self.collector.collect(sub_agent, self.env)

            # Temporarily apply adapted parameters to network (maintains gradient connection)
            # Save original params
            original_params = {n: p.detach().clone() for n, p in sub_agent.network.named_parameters()}

            # Assign adapted parameters (these retain gradients through autograd.grad call)
            for name, param in sub_agent.network.named_parameters():
                if name in adapted_params:
                    param.data = adapted_params[name].data

            # Evaluate query loss with adapted network parameters
            query_observations = query_batch["observations"]
            query_masks = query_batch["masks"]

            # Handle observations as list (graph-based with dynamic sizes)
            if isinstance(query_observations, list):
                # Process each observation through network and collect outputs
                query_logits_list = []
                query_values_list = []
                for i, obs_t in enumerate(query_observations):
                    mask_t = query_masks[i:i+1]  # Keep batch dimension (1, n_actions)
                    logits_t, values_t, _ = sub_agent.network.evaluate(obs_t, mask_t, actions=None)
                    query_logits_list.append(logits_t)
                    query_values_list.append(values_t)
                query_logits = torch.cat(query_logits_list, dim=0)
                query_values = torch.cat(query_values_list, dim=0)
            else:
                query_logits, query_values, _ = sub_agent.network.evaluate(query_observations, query_masks, actions=None)

            # Compute query loss
            query_old_log_probs = query_batch["log_probs"]
            query_advantages = query_batch["advantages"]
            query_returns = query_batch["returns"]

            query_ratio = torch.exp(torch.nn.functional.log_softmax(query_logits, dim=-1).gather(-1, query_batch["actions"].unsqueeze(-1)).squeeze(-1) - query_old_log_probs)
            query_surr1 = query_ratio * query_advantages
            query_surr2 = torch.clamp(query_ratio, 1 - sub_agent.clip_eps, 1 + sub_agent.clip_eps) * query_advantages
            query_loss = -torch.min(query_surr1, query_surr2).mean() + sub_agent.value_coef * torch.nn.functional.mse_loss(query_values, query_returns)

            # Restore original parameters
            for name, param in sub_agent.network.named_parameters():
                param.data = original_params[name]

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
        active_task_ids = sorted(self.active_tasks)
        total_tasks = len(self.env.tasks)

        # Determine algorithm name based on enabled phases
        if self.enable_meta_learning and self.enable_fine_tuning:
            algo = "MAML (Meta-Learning + Fine-Tuning)"
        elif self.enable_meta_learning:
            algo = "MAML (Meta-Learning Only)"
        elif self.enable_fine_tuning:
            algo = "PPO (Fine-Tuning Only)"
        else:
            algo = "INVALID"

        lines = [
            f"\n{'=' * 64}",
            f"  Algorithm  : {algo}",
            f"  Tasks      : {active_task_ids} (of {total_tasks} total)",
        ]

        if self.enable_meta_learning:
            lines.append(f"  Meta       : {self.mcfg['epochs']} epochs × {self.mcfg['batches_per_epoch']} batches")
        if self.enable_fine_tuning:
            lines.append(f"  Fine-tune  : {self.fcfg['epochs']} epochs × {self.fcfg['batches_per_epoch']} batches")

        lines.extend([
            f"  Device     : {globals.DEVICE}",
            f"{'=' * 64}"
        ])

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
        collector: BaseCollector,
    ):
        self.agents = agents
        self.train_agent = agents["train_agent"]
        self.env = env
        self.evaluator = evaluators["train_eval"]
        self.logger = logger
        self.collector = collector

        # Setup task iteration from env
        if not env.tasks:
            raise ValueError("POMOTrainer requires env.tasks")

        # Extract config from training phase
        phases_cfg = trainer_cfg.get("phases", {})
        training_phase = phases_cfg.get("training", {})
        control_cfg = training_phase.get("control", {})

        self.tcfg = {
            "epochs": int(control_cfg["epochs"]),
            "batches_per_epoch": int(control_cfg["batches_per_epoch"]),
            "instances_per_batch": int(control_cfg["instances_per_batch"]),
            "eval_interval": int(control_cfg.get("eval_interval", 1)),
            "checkpoint_interval": int(control_cfg["checkpoint_interval"]),
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
        collector: BaseCollector,
    ) -> "POMOTrainer":
        return cls(
            agents=agents,
            env=env,
            trainer_cfg=trainer_cfg,
            evaluators=evaluators,
            logger=logger,
            collector=collector,
        )

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

            best_objective = float("-inf")
            best_epoch = -1

            for epoch in range(epochs):
                epoch_start = time.time()
                epoch_losses = []
                epoch_returns = []
                epoch_grad_norms = []

                # Training phase: accumulate statistics across all batches
                for batch_idx in range(batches_per_epoch):
                    batch_log_probs = []  # List of tensors, one per instance
                    batch_rewards = []    # List of tensors, one per instance
                    batch_entropies = []  # List of tensors, one per instance

                    for _ in range(instances_per_batch):
                        self.env.retask(task_id)
                        batch_data = self.collector.collect(agent, self.env)
                        # Stack episodes for this instance
                        if batch_data["log_probs"]:
                            instance_log_probs = torch.stack(
                                [torch.as_tensor(lp, device=globals.DEVICE).squeeze(0) for lp in batch_data["log_probs"]]
                            )
                            instance_rewards = torch.tensor(
                                batch_data["rewards"], dtype=torch.float32, device=globals.DEVICE
                            )
                            instance_entropies = torch.stack(
                                [torch.as_tensor(ent, device=globals.DEVICE).squeeze(0) for ent in batch_data.get("entropies", [])]
                            )
                            batch_log_probs.append(instance_log_probs)
                            batch_rewards.append(instance_rewards)
                            batch_entropies.append(instance_entropies)

                    # Update after batch
                    if batch_log_probs:
                        batch = {
                            "log_probs": batch_log_probs,   # List of (num_starting_points_i,) tensors
                            "rewards": batch_rewards,       # List of (num_starting_points_i,) tensors
                            "entropies": batch_entropies,   # List of (num_starting_points_i,) tensors
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
                    for instance_rewards in batch_rewards:
                        if isinstance(instance_rewards, torch.Tensor):
                            epoch_returns.extend(instance_rewards.detach().cpu().tolist())
                        else:
                            epoch_returns.extend(instance_rewards)
                    self._total_instances += instances_per_batch

                # Compute per-epoch training statistics
                epoch_time = time.time() - epoch_start
                train_returns = np.array(epoch_returns)

                # Debug: epoch summary
                epoch_losses_array = np.array(epoch_losses) if epoch_losses else np.array([])
                epoch_grad_norms_array = np.array(epoch_grad_norms) if epoch_grad_norms else np.array([])
                print(f"Epoch {epoch + 1:3d} | "
                      f"loss: {np.mean(epoch_losses_array):8.6f}±{np.std(epoch_losses_array):8.6f} | "
                      f"return: {np.mean(train_returns):8.4f}±{np.std(train_returns):8.4f} | "
                      f"grad_norm: {np.mean(epoch_grad_norms_array):8.4f}±{np.std(epoch_grad_norms_array):8.4f} | "
                      f"time: {epoch_time:6.1f}s")

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

                    # Track best objective
                    mean_obj = eval_stats.get("mean_objective", float("-inf"))
                    if mean_obj > best_objective:
                        best_objective = mean_obj
                        best_epoch = epoch + 1
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
        print(
            f"\n{'=' * 64}\n"
            f"  Algorithm  : POMO (Multiple Optima)\n"
            f"  Tasks      : {len(self.env.tasks)}\n"
            f"  Device     : {globals.DEVICE}\n"
            f"{'=' * 64}"
        )

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
