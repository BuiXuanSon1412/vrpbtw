# core/__init__.py
# Convenience re-exports so callers can write:
#   from core import BaseAgent, PPOAgent, RolloutBuffer, Environment, ...
# instead of drilling into sub-modules.

from core.agent import BaseAgent, Agent
from core.buffer import Transition, RolloutBuffer
from core.estimator import BaseEstimator, PPOEstimator
from core.evaluator import Evaluator
from core.logger import Logger
from core.module import BasePolicy
from core.environment import Environment, ActionMask, StepResult, Solution, SolutionPool
from core.task import Task, SimpleTask, TaskManager
from core.trainer import (
    BaseTrainer,
    MetaTrainer,
    CurriculumScheduler,
    FineTuner,
    InnerUpdater,
    MetaLearner,
    compute_policy_entropy,
)
from core.utils import SeedManager, RunningNormalizer

__all__ = [
    # agents
    "BaseAgent",
    "Agent",
    # buffers
    "Transition",
    "RolloutBuffer",
    # curriculum
    "CurriculumScheduler",
    "compute_policy_entropy",
    # estimators
    "BaseEstimator",
    "PPOEstimator",
    # evaluator
    "Evaluator",
    # fine tuner
    "FineTuner",
    # inner updater
    "InnerUpdater",
    # logger
    "Logger",
    # meta learner
    "MetaLearner",
    # networks
    "BasePolicy",
    # problem
    "Environment",
    "ActionMask",
    "StepResult",
    "Solution",
    "SolutionPool",
    # task management
    "Task",
    "SimpleTask",
    "TaskManager",
    # trainer
    "BaseTrainer",
    "MetaTrainer",
    # utils
    "SeedManager",
    "RunningNormalizer",
]
