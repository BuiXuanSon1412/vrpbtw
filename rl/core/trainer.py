"""
core/trainer.py
---------------
Training-loop implementations.

Trainers implement learning algorithms (PPO, MAML, etc.) and orchestrate
data collection, gradient updates, evaluation, and checkpointing.

Design principle:
  - Agents (policies) are algorithm-agnostic: select actions, save/load weights.
  - Trainers own all learning state: environment, buffer, and data collection via collect().
  - Estimators compute loss functions for parameter updates.

  MetaTrainer      — FOMAML meta-learning with inner/outer loops

Circular dependency avoidance
-----------------------------
Evaluator is injected as Any — trainer.py never imports evaluator.py.
"""

from __future__ import annotations

import random
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple
import copy
from abc import ABC, abstractmethod

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from core.agent import BaseAgent, _obs_to_tensor
from core.buffer import RolloutBuffer
from core.task import TaskManager, SimpleTask


# ---------------------------------------------------------------------------
# BaseTrainer
# ---------------------------------------------------------------------------


class BaseTrainer(ABC):
    """
    Abstract training-loop interface.

    Properties (for single-task trainers):
      env    — problem environment (with reset/step interface)
      agent  — policy agent (with select_action/update methods)
      buffer — rollout buffer for experience storage

    Methods:
      collect()  — fill buffer with trajectories from env and agent
      train()    — run the full training loop
    """

    def __init__(
        self,
        agent: BaseAgent,
        env: Any,
        buffer: RolloutBuffer,
    ):
        self.agent = agent
        self.env = env
        self.buffer = buffer

    def collect(
        self,
        instance_gen: Callable,
        rollout_len: Optional[int] = None,
    ) -> None:
        """
        Collect experience into the buffer from the environment.

        Args:
            instance_gen: callable that generates raw problem instances
            rollout_len: number of steps to collect (uses buffer capacity if None)
        """
        if rollout_len is None:
            rollout_len = self.buffer.capacity

        def _fresh_episode() -> Tuple[Any, np.ndarray]:
            for _ in range(100):
                raw = instance_gen()
                obs, info = self.env.reset(raw)
                mask = info["action_mask"]
                if mask.any():
                    return obs, mask
            raise RuntimeError("collect: 100 consecutive dead-start instances.")

        obs, action_mask = _fresh_episode()

        while self.buffer._ptr < rollout_len and not self.buffer.is_full:
            action, lp, val = self.agent.select_action(obs, action_mask, training=True)

            if not action_mask[action]:
                feasible = np.where(action_mask)[0]
                if len(feasible) > 0:
                    action = int(np.random.choice(feasible))
                    lp = 0.0
                else:
                    obs, action_mask = _fresh_episode()
                    continue

            next_obs, reward, terminated, truncated, info = self.env.step(action)

            self.buffer.add(
                obs=obs,
                action=action,
                reward=reward,
                done=(terminated or truncated),
                log_prob=lp,
                value=val,
                action_mask=action_mask,
            )

            obs = next_obs
            action_mask = info["action_mask"]

            if terminated or truncated or not action_mask.any():
                obs, action_mask = _fresh_episode()

    @abstractmethod
    def train(self) -> Dict[str, Any]:
        """Run the full training loop and return a summary dict."""
        ...


# ---------------------------------------------------------------------------
# InnerUpdater
# ---------------------------------------------------------------------------


class InnerUpdater:
    """
    Performs task-specific adaptation via gradient steps.

    Given a network (clone of the meta-policy) and support data, runs
    n_steps of gradient descent to adapt the network to the task.

    The result is a set of adapted parameters (or a modified network state)
    that can be evaluated on query data for meta-gradient computation.
    """

    def __init__(self, inner_lr: float, n_steps: int = 1):
        """
        Initialize the inner updater.

        Args:
            inner_lr: learning rate for inner-loop gradient steps.
            n_steps: number of gradient steps per inner update.
        """
        self.inner_lr = inner_lr
        self.n_steps = n_steps

    def update(
        self,
        network: nn.Module,
        loss_fn: Any,
        support_data: Any,
    ) -> Tuple[List[nn.Parameter], List[torch.Tensor]]:
        """
        Perform inner-loop adaptation.

        Computes loss on support_data and takes n_steps of gradient descent.

        Args:
            network: the cloned policy network to adapt.
            loss_fn: callable that takes (network, support_data) -> loss (scalar).
            support_data: data for support phase (typically a buffer or batch).

        Returns:
            (adapted_params, adapted_buffers): lists of adapted parameter tensors
            and buffers. These can be used to create a functional module for
            query evaluation.
        """
        params = list(network.parameters())

        for step_idx in range(self.n_steps):
            # Compute loss
            loss = loss_fn(network, support_data)

            # Compute gradients (with create_graph=True for meta-learning)
            grads = torch.autograd.grad(
                loss,
                params,
                create_graph=True,  # Critical: allows meta-gradients to flow through
                retain_graph=True,
                allow_unused=True,  # Some params may not contribute to loss
            )

            # Update parameters in-place
            # Note: This modifies network in-place; clone if you need the original
            with torch.no_grad():
                for param, grad in zip(params, grads):
                    if grad is not None:
                        param.data = param.data - self.inner_lr * grad.data

        # Return adapted parameters (still on the computation graph)
        adapted_params = [p for p in network.parameters()]
        # Buffers (e.g., batch norm statistics) are also updated in-place
        adapted_buffers = [b for b in network.buffers()]

        return adapted_params, adapted_buffers

    def update_with_sgd(
        self,
        network: nn.Module,
        optimizer: Any,
        loss_fn: Any,
        support_data: Any,
    ) -> None:
        """
        Alternative: perform inner update using an explicit optimizer.

        Useful when you want to use momentum, weight decay, or other
        optimizer features.  This version does not preserve the computation
        graph (no meta-gradients), so use only for the inner loop if you
        implement meta-learning differently.

        Args:
            network: policy network to adapt.
            optimizer: SGD or Adam optimizer initialized for this network.
            loss_fn: callable that takes (network, support_data) -> loss.
            support_data: data for support phase.
        """
        for _ in range(self.n_steps):
            loss = loss_fn(network, support_data)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


# ---------------------------------------------------------------------------
# MetaLearner
# ---------------------------------------------------------------------------


class MetaLearner:
    """
    Orchestrates FOMAML-style meta-learning.

    For each meta-step:
      1. Sample active tasks (from TaskManager)
      2. For each task:
         a. Clone the meta-policy
         b. Inner-loop adaptation on support data (InnerUpdater)
         c. Accumulate meta-gradients on query data
      3. Meta-optimizer step on accumulated gradients
    """

    def __init__(
        self,
        agent: Any,
        task_manager: Any,
        meta_optimizer: optim.Optimizer,
        inner_lr: float,
        n_inner_steps: int = 1,
        max_grad_norm: float = 1.0,
    ):
        """
        Initialize the meta-learner.

        Args:
            agent: the meta-policy agent (will be cloned for inner loops).
            task_manager: TaskManager providing active tasks.
            meta_optimizer: optimizer for the meta-policy (usually Adam).
            inner_lr: learning rate for inner-loop adaptation.
            n_inner_steps: number of gradient steps in inner loop per task.
            max_grad_norm: gradient clipping threshold.
        """
        self.agent = agent
        self.task_manager = task_manager
        self.meta_optimizer = meta_optimizer
        self.inner_updater = InnerUpdater(inner_lr=inner_lr, n_steps=n_inner_steps)
        self.max_grad_norm = max_grad_norm

    def meta_step(
        self,
        task_trajectories: Dict[int, Any],
        support_loss_fn: Callable[[Any, Any], torch.Tensor],
        query_loss_fn: Callable[[Any, Any], torch.Tensor],
    ) -> Dict[str, float]:
        """
        Perform one FOMAML meta-learning step.

        Args:
            task_trajectories: dict mapping task_id -> (support_data, query_data).
                Where support_data and query_data are buffers or batch dicts.
            support_loss_fn: callable (network, support_data) -> scalar loss.
            query_loss_fn: callable (network, query_data) -> scalar loss.

        Returns:
            Dictionary of metrics (meta_loss, num_tasks, etc).
        """
        self.meta_optimizer.zero_grad()

        active_tasks = self.task_manager.get_active_task_ids()
        task_losses: List[float] = []

        for task_id in active_tasks:
            if task_id not in task_trajectories:
                continue

            support_data, query_data = task_trajectories[task_id]

            # Inner loop: adapt meta-policy to this task
            fast_agent = self.agent.clone()
            self.inner_updater.update(
                network=fast_agent.network,
                loss_fn=lambda net, data: support_loss_fn(net, data),
                support_data=support_data,
            )

            # Outer loop: accumulate meta-gradients on query data
            query_loss = query_loss_fn(fast_agent.network, query_data)

            # Scale by number of tasks (averaging the meta-loss)
            scaled_loss = query_loss / len(active_tasks)

            # Accumulate gradient (do not step yet)
            scaled_loss.backward()

            task_losses.append(query_loss.item())

        # Gradient clipping
        nn.utils.clip_grad_norm_(
            self.agent.network.parameters(),
            self.max_grad_norm,
        )

        # Meta-optimizer step
        self.meta_optimizer.step()

        return {
            "meta_loss": float(sum(task_losses) / len(task_losses))
            if task_losses
            else 0.0,
            "num_active_tasks": len(active_tasks),
        }

    def meta_step_with_roles(
        self,
        tasks_with_splits: Dict[int, Tuple[Any, Any]],
        loss_fn: Callable[[Any, Any], torch.Tensor],
    ) -> Dict[str, float]:
        """
        Simplified version: same loss function for both support and query.

        Use this if you're doing standard meta-learning where support and query
        are just different samples from the same task distribution.

        Args:
            tasks_with_splits: dict mapping task_id -> (support_data, query_data).
            loss_fn: callable (network, data) -> scalar loss.

        Returns:
            Dictionary of metrics.
        """
        return self.meta_step(
            task_trajectories=tasks_with_splits,
            support_loss_fn=loss_fn,
            query_loss_fn=loss_fn,
        )

    def get_meta_params(self) -> Dict[str, Any]:
        """Return current meta-policy parameters (network state dict)."""
        return self.agent.network.state_dict()

    def set_meta_params(self, state_dict: Dict[str, Any]) -> None:
        """Load meta-policy parameters from a state dict."""
        self.agent.network.load_state_dict(state_dict)


class CurriculumScheduler:
    """
    Schedules curriculum expansion based on policy stability metrics.

    Initially, training focuses on the easiest task. Once the policy becomes
    stable (low entropy, low loss variance, etc.), harder tasks are gradually
    introduced.

    Decouples curriculum logic from the trainer/meta-learner.
    """

    def __init__(
        self,
        entropy_threshold: float,
        expand_interval: int = 1,
        check_metric: str = "entropy",
    ):
        """
        Initialize the curriculum scheduler.

        Args:
            entropy_threshold: if policy entropy on the hardest active task falls
                             below this, consider expanding curriculum.
            expand_interval: number of scheduler.check() calls between expansions.
            check_metric: which metric to use ("entropy", "loss_variance", etc).
        """
        self.entropy_threshold = entropy_threshold
        self.expand_interval = expand_interval
        self.check_metric = check_metric
        self._check_counter = 0

        # Track metrics over time for filtering/smoothing
        self.recent_entropies: Dict[int, list] = {}  # task_id -> [entropy, ...]
        self.recent_losses: Dict[int, list] = {}  # task_id -> [loss, ...]

    def should_expand(
        self,
        task_manager: Any,
        task_id: int,
        metric_value: float,
    ) -> bool:
        """
        Check if curriculum should expand.

        Args:
            task_manager: TaskManager instance (to check expansion feasibility).
            task_id: the task ID being evaluated.
            metric_value: the metric (entropy, loss, etc) for this task.

        Returns:
            True if expansion should happen, False otherwise.
        """
        if task_manager.is_fully_expanded():
            return False

        if self.check_metric == "entropy":
            # Expand if entropy is low (policy is confident/stable)
            return metric_value < self.entropy_threshold
        elif self.check_metric == "loss_variance":
            # Expand if loss variance is low (training is stable)
            return metric_value < self.entropy_threshold
        else:
            raise ValueError(f"Unknown check_metric: {self.check_metric}")

    def update(
        self,
        task_manager: Any,
        task_metrics: Dict[int, float],
    ) -> bool:
        """
        Check if any active task meets expansion criteria.

        Args:
            task_manager: TaskManager instance.
            task_metrics: dict mapping task_id -> metric_value for active tasks.

        Returns:
            True if curriculum was expanded, False otherwise.
        """
        self._check_counter += 1

        # Only check expansion every expand_interval steps
        if self._check_counter % self.expand_interval != 0:
            return False

        # Get the hardest active task
        active_ids = task_manager.get_active_task_ids()
        if not active_ids:
            return False

        hardest_task_id = max(active_ids)
        metric_value = task_metrics.get(hardest_task_id, float("inf"))

        # Check if we should expand
        if self.should_expand(task_manager, hardest_task_id, metric_value):
            expanded = task_manager.expand_curriculum()
            if expanded:
                print(
                    f"[Curriculum] Expanding: {task_manager.num_active_tasks() - 1} -> "
                    f"{task_manager.num_active_tasks()} active tasks "
                    f"({self.check_metric}={metric_value:.4f})"
                )
            return expanded

        return False

    def record_metric(self, task_id: int, metric_value: float) -> None:
        """
        Record a metric value for a task (for filtering/smoothing).

        Args:
            task_id: the task ID.
            metric_value: the metric value (entropy, loss, etc).
        """
        if task_id not in self.recent_entropies:
            self.recent_entropies[task_id] = []
        self.recent_entropies[task_id].append(metric_value)

        # Keep only the last 10 values
        if len(self.recent_entropies[task_id]) > 10:
            self.recent_entropies[task_id].pop(0)

    def get_smoothed_metric(self, task_id: int) -> Optional[float]:
        """
        Return a smoothed metric (e.g., moving average) for a task.

        Args:
            task_id: the task ID.

        Returns:
            The smoothed metric, or None if not enough data.
        """
        if task_id not in self.recent_entropies:
            return None
        values = self.recent_entropies[task_id]
        if not values:
            return None
        return float(np.mean(values))


def compute_policy_entropy(
    policy_logits: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> float:
    """
    Compute average entropy of policy distribution.

    Args:
        policy_logits: logits from policy head, shape (batch, n_actions).
        mask: optional binary mask for valid actions.

    Returns:
        Scalar entropy value.
    """
    # Convert logits to probabilities
    log_probs = torch.log_softmax(policy_logits, dim=-1)
    probs = torch.softmax(policy_logits, dim=-1)

    # Apply mask if provided
    if mask is not None:
        mask = mask.float()
        probs = probs * mask
        probs = probs / (probs.sum(dim=-1, keepdim=True) + 1e-8)

    # Compute entropy: -sum(p * log(p))
    entropy = -(probs * log_probs).sum(dim=-1)

    return float(entropy.mean().detach().cpu().item())


class FineTuner:
    """
    Fine-tunes task-specific policies from a meta-policy.

    Phase 2 of MAML training:
      1. Initialize by copying the meta-policy for each active task
      2. Train each copy independently on task data
      3. Save or return the specialized policies

    Unlike MetaLearner, there is no inner/outer loop or meta-gradients.
    Each task just does standard on-policy learning (e.g., PPO).
    """

    def __init__(
        self,
        base_agent: Any,
        task_manager: Any,
        cfg: Any,
    ):
        """
        Initialize the fine-tuner.

        Args:
            base_agent: the meta-policy agent (will be cloned for each task).
            task_manager: TaskManager with active tasks.
            cfg: experiment config (contains train, ppo settings).
        """
        self.base_agent = base_agent
        self.task_manager = task_manager
        self.cfg = cfg
        self.tcfg = cfg.train
        self.pcfg = cfg.ppo

        # One optimizer per task
        self.task_agents: Dict[int, Any] = {}
        self.task_optimizers: Dict[int, Any] = {}

    def initialize(self) -> None:
        """
        Create task-specific agent copies.

        For each active task, deep-copy the meta-policy and create an optimizer.
        """
        self.task_agents.clear()
        self.task_optimizers.clear()

        for task in self.task_manager.get_active_tasks():
            task_id = task.task_id

            # Deep copy the meta-policy
            task_agent = copy.deepcopy(self.base_agent)

            # Create optimizer for this task
            task_optimizer = optim.Adam(
                task_agent.network.parameters(),
                lr=self.pcfg.lr,
            )

            self.task_agents[task_id] = task_agent
            self.task_optimizers[task_id] = task_optimizer

    def get_task_agent(self, task_id: int) -> Any:
        """Return the agent (policy) for a given task."""
        if task_id not in self.task_agents:
            raise ValueError(f"Task {task_id} not initialized")
        return self.task_agents[task_id]

    def get_task_optimizer(self, task_id: int) -> Any:
        """Return the optimizer for a given task."""
        if task_id not in self.task_optimizers:
            raise ValueError(f"Task {task_id} not initialized")
        return self.task_optimizers[task_id]

    def train_step(
        self,
        task_id: int,
        rollout_buffer: Any,
        policy_update_fn: Callable[[Any, Any, Any], Dict[str, float]],
    ) -> Dict[str, float]:
        """
        Perform one training step for a specific task.

        Args:
            task_id: the task ID.
            rollout_buffer: buffer with collected trajectories for this task.
            policy_update_fn: callable that takes (agent, buffer, optimizer) -> metrics.

        Returns:
            Dictionary of metrics (loss, entropy, etc).
        """
        agent = self.get_task_agent(task_id)
        optimizer = self.get_task_optimizer(task_id)

        # Call the update function (typically PPO-style)
        metrics = policy_update_fn(agent, rollout_buffer, optimizer)
        metrics["task_id"] = task_id

        return metrics

    def save_task_policies(self, save_dir: str) -> None:
        """
        Save all task-specific policies to disk.

        Args:
            save_dir: directory to save policies (will be created if needed).
        """
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)

        for task_id, agent in self.task_agents.items():
            task_save_path = save_path / f"task_{task_id}_best.pt"
            torch.save(
                {"network_state": agent.network.state_dict()},
                task_save_path,
            )

    def load_task_policies(self, load_dir: str) -> None:
        """
        Load task-specific policies from disk.

        Args:
            load_dir: directory containing saved policies.
        """
        load_path = Path(load_dir)

        for task_id, agent in self.task_agents.items():
            task_load_path = load_path / f"task_{task_id}_best.pt"
            if task_load_path.exists():
                checkpoint = torch.load(task_load_path, map_location="cpu")
                agent.network.load_state_dict(checkpoint["network_state"])

    def get_all_agents(self) -> Dict[int, Any]:
        """Return all task agents (for deployment or evaluation)."""
        return dict(self.task_agents)


# ---------------------------------------------------------------------------
# MetaTrainer  — FOMAML meta-learning method
# ---------------------------------------------------------------------------


class MetaTrainer(BaseTrainer):
    """
    FOMAML (First-Order Model-Agnostic Meta-Learning) trainer.

    Implements Phase 1 of MAML training with curriculum learning support:
      1. Initialize TaskManager with all tasks (easiest first)
      2. For each iteration:
         a. Run meta-learning step on active tasks
         b. Check curriculum expansion criteria
         c. Log and evaluate
      3. Optionally: Phase 2 fine-tuning via FineTuner

    Uses new architecture:
      - TaskManager: controls active task set and curriculum
      - MetaLearner: orchestrates inner/outer loop updates
      - CurriculumScheduler: monitors entropy and expands curriculum
    """

    def __init__(
        self,
        agent: BaseAgent,
        task_pool: Dict[int, Tuple[Any, Callable]],
        eval_problem: Any,
        eval_gen: Callable,
        cfg: Any,
        evaluator: Any,
        logger: Any,
    ):
        super().__init__(agent=agent)
        self.cfg = cfg
        self.mcfg = cfg.maml
        self.tcfg = cfg.train
        self.evaluator = evaluator
        self.logger = logger

        # Build TaskManager from task_pool
        tasks = []
        for task_id in sorted(task_pool.keys()):
            problem, gen = task_pool[task_id]
            task = SimpleTask(task_id=task_id, problem=problem, generator=gen)
            tasks.append(task)

        self.task_manager = TaskManager(tasks)

        # Evaluation task
        self.eval_problem = eval_problem
        self.eval_gen = eval_gen

        # Meta-learner
        self._meta_optimizer = optim.Adam(
            agent.policy.parameters(), lr=cfg.maml.meta_lr
        )
        self.meta_learner = MetaLearner(
            agent=agent,
            task_manager=self.task_manager,
            meta_optimizer=self._meta_optimizer,
            inner_lr=cfg.maml.inner_lr,
            n_inner_steps=cfg.maml.n_inner_steps,
            max_grad_norm=cfg.maml.max_grad_norm,
        )

        # Curriculum scheduler
        entropy_threshold = getattr(cfg.maml, "entropy_threshold", 0.5)
        self.curriculum_scheduler = CurriculumScheduler(
            entropy_threshold=entropy_threshold,
            expand_interval=getattr(cfg.maml, "curriculum_check_interval", 1),
            check_metric="entropy",
        )

        # Training state
        self._timestep = 0
        self._iteration = 0
        self._best_objective = float("-inf")
        self._patience_counter = 0
        self._task_entropies: Dict[int, List[float]] = {}

    def train(self) -> Dict[str, Any]:
        self._print_header()
        start_time = time.time()
        stop_reason = "timestep_limit"

        while self._timestep < self.tcfg.total_timesteps:
            iter_start = time.time()
            self._iteration += 1

            metrics = self._meta_update()
            self._timestep += int(metrics.pop("_steps", 0))

            metrics["iter_time_s"] = time.time() - iter_start
            metrics["num_active_tasks"] = self.task_manager.num_active_tasks()

            if self._iteration % self.tcfg.log_interval == 0:
                self.logger.log_metrics(
                    metrics,
                    step=self._timestep,
                    print_keys=[
                        "train/meta_loss",
                        "train/update_count",
                        "num_active_tasks",
                    ],
                )

            if self._iteration % self.tcfg.eval_interval == 0:
                eval_stats = self.evaluator.evaluate(self.eval_gen)
                self.logger.log_metrics(eval_stats, step=self._timestep, prefix="eval")

                mean_obj = eval_stats.get("mean_objective", float("-inf"))
                if mean_obj > self._best_objective + self.tcfg.min_delta:
                    self._best_objective = mean_obj
                    self._patience_counter = 0
                    self._save_checkpoint("best")
                    self.logger.log_event(
                        "best_checkpoint", self._timestep, objective=f"{mean_obj:.4f}"
                    )
                else:
                    self._patience_counter += 1

                if self._patience_counter >= self.tcfg.patience:
                    stop_reason = "early_stopping"
                    self.logger.log_event(
                        "early_stop", self._timestep, patience=self.tcfg.patience
                    )
                    break

            if self._iteration % self.tcfg.checkpoint_interval == 0:
                self._save_checkpoint(f"iter_{self._iteration}")

        summary = {
            "stop_reason": stop_reason,
            "total_iterations": self._iteration,
            "total_timesteps": self._timestep,
            "best_objective": self._best_objective,
            "training_time_s": round(time.time() - start_time, 1),
            "final_num_active_tasks": self.task_manager.num_active_tasks(),
        }
        self._save_checkpoint("final")
        self.logger.log_event("training_complete", self._timestep, **summary)
        self.logger.close()
        self._print_footer(summary)
        return summary

    def _meta_update(self) -> Dict[str, float]:
        """One FOMAML meta-update over sampled active tasks with curriculum expansion."""
        active_task_ids = self.task_manager.get_active_task_ids()
        n_tasks = min(self.mcfg.n_tasks_per_update, len(active_task_ids))
        sampled_task_ids = random.sample(active_task_ids, n_tasks)

        total_steps = 0
        task_losses: List[float] = []
        task_entropies: Dict[int, float] = {}

        # Build trajectories for meta-learning
        task_trajectories: Dict[int, Tuple[Any, Any]] = {}

        for task_id in sampled_task_ids:
            task = self.task_manager.get_task(task_id)

            # Collect support data
            self.env = task.problem
            self.buffer = RolloutBuffer(capacity=self.mcfg.support_rollout_len)
            self.collect(instance_gen=task.generator)
            sup_buf = self.buffer
            total_steps += sup_buf._ptr

            # Collect query data
            self.env = task.problem
            self.buffer = RolloutBuffer(capacity=self.mcfg.support_rollout_len)
            self.collect(instance_gen=task.generator)
            qry_buf = self.buffer
            total_steps += qry_buf._ptr

            task_trajectories[task_id] = (sup_buf, qry_buf)

        # Define loss functions for meta-learner
        def support_loss_fn(
            network: nn.Module, sup_data: RolloutBuffer
        ) -> torch.Tensor:
            return self.agent.estimator.compute_loss(network, sup_data)

        def query_loss_fn(network: nn.Module, qry_data: RolloutBuffer) -> torch.Tensor:
            return self.agent.estimator.compute_loss(network, qry_data)

        # Meta-learning step
        meta_metrics = self.meta_learner.meta_step(
            task_trajectories=task_trajectories,
            support_loss_fn=support_loss_fn,
            query_loss_fn=query_loss_fn,
        )
        task_losses = [meta_metrics["meta_loss"]]

        # Curriculum expansion check
        if task_entropies:
            expanded = self.curriculum_scheduler.update(
                self.task_manager,
                task_entropies,
            )
            if expanded:
                print(
                    f"[MetaTrainer] Curriculum expanded to {self.task_manager.num_active_tasks()} tasks"
                )

        return {
            "train/meta_loss": float(np.mean(task_losses)) if task_losses else 0.0,
            "train/update_count": float(self._iteration),
            "_steps": total_steps,
        }

    def _save_checkpoint(self, tag: str) -> None:
        path = f"{self.tcfg.checkpoint_dir}/{self.cfg.name}_{tag}.pt"
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "network_state": self.agent.policy.state_dict(),
                "meta_optimizer_state": self._meta_optimizer.state_dict(),
                "iteration": self._iteration,
            },
            path,
        )

    def _print_header(self) -> None:
        task_ids = sorted(self.task_manager.get_active_task_ids())
        total_tasks = self.task_manager.num_total_tasks()
        print(
            f"\n{'=' * 64}\n"
            f"  Experiment : {self.cfg.name}\n"
            f"  Algorithm  : FOMAML (Phase 1: Meta-Learning)\n"
            f"  Tasks      : {task_ids} (of {total_tasks} total)\n"
            f"  Inner lr   : {self.mcfg.inner_lr}   "
            f"Meta lr : {self.mcfg.meta_lr}\n"
            f"  Inner steps: {self.mcfg.n_inner_steps}   "
            f"Tasks/update: {self.mcfg.n_tasks_per_update}\n"
            f"  Budget     : {self.tcfg.total_timesteps:,} steps\n"
            f"  Device     : {self.cfg.device}\n"
            f"{'=' * 64}"
        )

    def _print_footer(self, summary: Dict) -> None:
        print(
            f"\n{'=' * 64}\n"
            f"  Done ({summary['stop_reason']})\n"
            f"  Iterations : {summary['total_iterations']:,}\n"
            f"  Timesteps  : {summary['total_timesteps']:,}\n"
            f"  Best Obj   : {summary['best_objective']:.4f}\n"
            f"  Time       : {summary['training_time_s']:.1f}s\n"
            f"{'=' * 64}\n"
        )
