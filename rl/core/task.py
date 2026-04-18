"""
core/task.py
------------
Task abstraction and curriculum management for multi-task meta-learning.

Represents tasks with different complexity levels (e.g., different problem sizes).
TaskManager controls which tasks are active during training and expands the
curriculum as the policy becomes more stable.

Design principle:
  - Task is a thin wrapper: stores problem and generator for a given difficulty
  - TaskManager tracks active tasks and expansion logic
  - All curriculum logic lives here, decoupled from the optimizer/trainer
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Set, Optional, Union


# ---------------------------------------------------------------------------
# Task
# ---------------------------------------------------------------------------


class Task(ABC):
    """
    Abstract task: represents a problem instance of a given complexity.

    A concrete task holds:
      - A Problem instance (e.g., VRPBTWProblem with n_customers=10)
      - A generator function for sampling instances of this complexity
      - An ID (size or complexity identifier)
    """

    @property
    @abstractmethod
    def task_id(self) -> Union[int, str]:
        """Unique task identifier (e.g., problem size or "size_distribution")."""
        ...

    @property
    @abstractmethod
    def problem(self) -> Any:
        """The Problem instance for this task."""
        ...

    @property
    @abstractmethod
    def generator(self) -> Callable[[], Dict[str, Any]]:
        """Generator function: callable that returns a raw instance dict."""
        ...


class SimpleTask(Task):
    """
    Concrete task: stores problem, generator, and task ID.
    """

    def __init__(self, task_id: Union[int, str], problem: Any, generator: Callable[[], Dict[str, Any]]):
        self._task_id = task_id
        self._problem = problem
        self._generator = generator

    @property
    def task_id(self) -> Union[int, str]:
        return self._task_id

    @property
    def problem(self) -> Any:
        return self._problem

    @property
    def generator(self) -> Callable[[], Dict[str, Any]]:
        return self._generator


# ---------------------------------------------------------------------------
# TaskManager
# ---------------------------------------------------------------------------


class TaskManager:
    """
    Manages a set of tasks with curriculum learning logic.

    Initially, only the easiest task (smallest task_id) is active.
    As training progresses, harder tasks can be added via expand_curriculum().

    This decouples curriculum control from the meta-learning optimizer.
    """

    def __init__(self, tasks: List[Task]):
        """
        Initialize task manager.

        Args:
            tasks: ordered list of Task objects (easiest to hardest).
                   Assumed to be sorted by difficulty.
        """
        if not tasks:
            raise ValueError("tasks list cannot be empty")

        self.tasks = tasks
        self.task_map: Dict[Union[int, str], Task] = {t.task_id: t for t in tasks}

        # Start with only the easiest task
        self.active_task_ids: Set[Union[int, str]] = {self.tasks[0].task_id}

    def get_active_tasks(self) -> List[Task]:
        """Return list of active Task objects."""
        return [self.task_map[tid] for tid in sorted(self.active_task_ids)]

    def get_active_task_ids(self) -> List[Union[int, str]]:
        """Return sorted list of active task IDs."""
        return sorted(self.active_task_ids, key=lambda x: (isinstance(x, str), x))

    def expand_curriculum(self) -> bool:
        """
        Add the next harder task to the active set.

        Returns:
            True if a new task was added, False if all tasks are already active.
        """
        # Find the hardest currently active task
        max_active_idx = max(
            i for i, t in enumerate(self.tasks) if t.task_id in self.active_task_ids
        )

        # If there's a harder task, add it
        if max_active_idx + 1 < len(self.tasks):
            next_task = self.tasks[max_active_idx + 1]
            self.active_task_ids.add(next_task.task_id)
            return True

        return False

    def is_fully_expanded(self) -> bool:
        """Check if all tasks are active."""
        return len(self.active_task_ids) == len(self.tasks)

    def get_task(self, task_id: Union[int, str]) -> Task:
        """Retrieve a specific task by ID."""
        if task_id not in self.task_map:
            raise ValueError(f"Unknown task_id {task_id}")
        return self.task_map[task_id]

    def num_active_tasks(self) -> int:
        """Return the number of active tasks."""
        return len(self.active_task_ids)

    def num_total_tasks(self) -> int:
        """Return the total number of tasks."""
        return len(self.tasks)
