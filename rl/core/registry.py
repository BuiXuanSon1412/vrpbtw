"""
registry.py
-----------
Factory functions that wire concrete implementations to abstract interfaces.

train.py and evaluate.py are the only callers.

Rules
-----
- No hyperparameter values live here; all come from ExperimentConfig.
- n_fleets is read from the built Problem (problem.n_fleets) and passed
  forward explicitly — never re-read from problem_kwargs with a string key.
- build_network does not receive action_space_size (it does not use it).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict
import torch.optim as optim

# Problems
from impl.vrpbtw import VRPBTWEnv
from impl.mvrpbtw import MVRPBTWEnv

# Networks
from impl.hgnn import HGNNActorCritic
from impl.geman import GEMANActorCritic

# Core
from core.agent import BaseAgent, PPOAgent, ReinforceAgent, POMOAgent
from core.network import ActorCritic
from core.trainer import BaseTrainer, MetaTrainer, POMOTrainer
from core.evaluator import Evaluator
from core.logger import Logger
from core.collector import (
    BaseCollector,
    GAECollector,
    MCCollector,
    EPCollector,
    POMOSampler,
)

# ---------------------------------------------------------------------------
# Registry: factory method dispatch tables
# ---------------------------------------------------------------------------

_NETWORK_REGISTRY: Dict[str, type] = {
    "hgnn": HGNNActorCritic,
    "geman": GEMANActorCritic,
}

_AGENT_REGISTRY: Dict[str, type] = {
    "ppo": PPOAgent,
    "reinforce": ReinforceAgent,
    "pomo": POMOAgent,
}

_TRAINER_REGISTRY: Dict[str, type] = {
    "meta": MetaTrainer,
    "pomo": POMOTrainer,
}

_ENVIRONMENT_REGISTRY: Dict[str, type] = {
    "vrpbtw": VRPBTWEnv,
    "mvrpbtw": MVRPBTWEnv,
}

_OPTIMIZER_REGISTRY: Dict[str, type | None] = {
    "adam": optim.Adam,
    "sgd": optim.SGD,
    "adamw": optim.AdamW,
    "unspecified": None,
}

_COLLECTOR_REGISTRY: Dict[str, type] = {
    "gae": GAECollector,
    "mc": MCCollector,
    "ep": EPCollector,
    "pomo": POMOSampler,
}


# ---------------------------------------------------------------------------
# Collector
# ---------------------------------------------------------------------------


def build_collector(cfg: Dict[str, Any]) -> BaseCollector:
    """Build collector from config.

    Dispatches to registry based on collector type, then calls from_config.

    Args:
        cfg: config dict with 'name' and algorithm-specific params

    Returns:
        BaseCollector instance
    """
    collector_type = cfg.get("name", "gae")

    if collector_type not in _COLLECTOR_REGISTRY:
        raise ValueError(
            f"Unknown collector type {collector_type!r}. "
            f"Register it in registry.py. Known: {list(_COLLECTOR_REGISTRY.keys())}"
        )

    cls = _COLLECTOR_REGISTRY[collector_type]
    return cls.from_config(cfg)


# ---------------------------------------------------------------------------
# Logger
# ---------------------------------------------------------------------------


def build_logger(cfg: Dict[str, Any]) -> Logger:
    """Build logger from config.

    Reads logger settings from logger section:
    - base_dir: base directory for experiments (e.g., experiment/train)
    - verbose: print to console
    - tensorboard: enable TensorBoard
    """
    exp_cfg = cfg.get("experiment", {})
    logger_cfg = cfg.get("logger", {})

    exp_name = exp_cfg.get("name", "experiment")
    base_dir = logger_cfg.get("base_dir", "experiment/train")
    exp_dir = str(Path(base_dir) / exp_name)

    return Logger(
        dir=exp_dir,
        verbose=logger_cfg.get("verbose", True),
        tensorboard=logger_cfg.get("tensorboard", False),
    )


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------


def build_evaluators(
    cfg: Dict[str, Any], agents: Dict[str, Any], env: Any
) -> Dict[str, Evaluator]:
    """Build evaluators from config.

    Reads trainer.evaluators and builds an Evaluator for each entry.
    Each evaluator specifies which agent to use via the 'agent' field.

    Returns dict of evaluators keyed by name.
    """
    trainer_cfg = cfg.get("trainer", {})
    evaluators_cfg = trainer_cfg.get("evaluators", {})

    evaluators = {}
    for eval_name, eval_cfg in evaluators_cfg.items():
        agent_name = eval_cfg.get("agent")
        if not agent_name or agent_name not in agents:
            raise ValueError(
                f"Evaluator '{eval_name}' specifies agent '{agent_name}' "
                f"but it does not exist in agents: {list(agents.keys())}"
            )

        evaluators[eval_name] = Evaluator(
            agent=agents[agent_name],
            env=env,
            n_episodes=eval_cfg.get("n_episodes", 20),
            deterministic=eval_cfg.get("deterministic", True),
            beam_width=eval_cfg.get("decoding", {}).get("beam_width", 1),
        )

    return evaluators


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------


def build_trainer(
    cfg: Dict[str, Any],
    agents: Dict[str, Any],
    env: Any,
    evaluators: Dict[str, Any],
    logger: Any,
) -> BaseTrainer:
    """Build trainer from config.

    Dispatches to registry based on trainer type, then calls from_config.
    """
    trainer_cfg = cfg.get("trainer", {})
    trainer_type = trainer_cfg.get("name", "meta")

    if trainer_type not in _TRAINER_REGISTRY:
        raise ValueError(
            f"Unknown trainer type {trainer_type!r}. "
            f"Register it in registry.py. Known: {list(_TRAINER_REGISTRY.keys())}"
        )

    cls = _TRAINER_REGISTRY[trainer_type]

    # Build collector from trainer config
    collector_cfg = trainer_cfg.get("collector", {})
    collector = build_collector(collector_cfg)

    return cls.from_config(
        trainer_cfg=trainer_cfg,
        agents=agents,
        env=env,
        evaluators=evaluators,
        logger=logger,
        collector=collector,
    )


# ---------------------------------------------------------------------------
# Network / Policy
# ---------------------------------------------------------------------------


def build_network(cfg: Dict[str, Any]) -> ActorCritic:
    """Build network from config.

    Dispatches to registry based on network name, then calls from_config.
    """
    network_cfg = cfg.get("network", cfg.get("policy", {}))  # Support both old and new
    net_type = network_cfg.get("name", "hgnn")

    if net_type not in _NETWORK_REGISTRY:
        raise ValueError(
            f"Unknown network type {net_type!r}. "
            f"Register it in registry.py. Known: {list(_NETWORK_REGISTRY.keys())}"
        )

    cls = _NETWORK_REGISTRY[net_type]
    return cls.from_config(network_cfg)


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------


def _build_single_agent(cfg: Dict[str, Any], agent_cfg: Dict[str, Any]) -> "BaseAgent":
    """Build a single agent from agent config.

    Builds network and optimizer, then instantiates agent.
    """
    # Build network
    network = build_network(cfg)

    # Build optimizer with agent-specific learning rate
    opt_type = agent_cfg.get("optimizer", "adam")
    opt_lr = agent_cfg.get("learning_rate", 0.001)

    if opt_type not in _OPTIMIZER_REGISTRY:
        raise ValueError(
            f"Unknown optimizer type {opt_type!r}. "
            f"Register it in registry.py. Known: {list(_OPTIMIZER_REGISTRY.keys())}"
        )

    opt_class = _OPTIMIZER_REGISTRY[opt_type]
    optimizer = (
        None if opt_class is None else opt_class(network.parameters(), lr=opt_lr)
    )

    # Build agent
    agent_type = agent_cfg.get("name", "policy")
    if agent_type not in _AGENT_REGISTRY:
        raise ValueError(
            f"Unknown agent type {agent_type!r}. "
            f"Register it in registry.py. Known: {list(_AGENT_REGISTRY.keys())}"
        )

    cls = _AGENT_REGISTRY[agent_type]
    return cls.from_config(agent_cfg, network, optimizer)


def build_agents(cfg: Dict[str, Any]) -> Dict[str, "BaseAgent"]:
    """Build agents from config.

    Reads trainer.agents and builds an agent for each entry.

    Returns dict of agents keyed by agent name.
    """
    trainer_cfg = cfg.get("trainer", {})
    agents_cfg = trainer_cfg.get("agents", {})

    return {
        agent_name: _build_single_agent(cfg, agent_cfg)
        for agent_name, agent_cfg in agents_cfg.items()
    }


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------


def build_environment(cfg: Dict[str, Any]) -> VRPBTWEnv:
    """Build environment from config.

    Dispatches to registry based on environment name, then calls from_config.
    Instance-specific parameters are determined dynamically in reset() via raw_instance.
    """
    # Support hierarchical config structure
    env_cfg = cfg.get("environment", {})
    env_name = env_cfg.get("name", "vrpbtw")

    if env_name not in _ENVIRONMENT_REGISTRY:
        raise ValueError(
            f"Unknown environment {env_name!r}. "
            f"Register it in registry.py. Known: {list(_ENVIRONMENT_REGISTRY.keys())}"
        )

    cls = _ENVIRONMENT_REGISTRY[env_name]
    return cls.from_config(env_cfg)


# ---------------------------------------------------------------------------
# Estimator
