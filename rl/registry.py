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

import json
import sys
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np

from config import EnvironmentConfig, ExperimentConfig

# Problems
from impl.environment import VRPBTWEnv

# Networks
from impl.network import VRPBTWPolicy

# Core
from core.agent import BaseAgent, Agent
from core.module import BasePolicy

_DEFAULT_DATA_ROOT = Path(__file__).resolve().parent.parent / "data"
_GENERATED_ROOT = _DEFAULT_DATA_ROOT / "generated"

# Add data/generated to sys.path for direct import
sys.path.insert(0, str(_GENERATED_ROOT))
from generate import create_instance  # type: ignore[import]


# ---------------------------------------------------------------------------
# Problem
# ---------------------------------------------------------------------------


def build_problem(env_cfg: EnvironmentConfig) -> VRPBTWEnv:
    name = env_cfg.problem_name
    kwargs = dict(env_cfg.problem_kwargs)

    if name == "vrpbtw":
        return VRPBTWEnv(
            n_customers=kwargs.get("n_customers", 10),
            n_fleets=kwargs.get("n_fleets", 2),
        )

    raise ValueError(
        f"Unknown problem {name!r}.  Register it in registry.py.  Known: ['vrpbtw']"
    )


# ---------------------------------------------------------------------------
# Instance generator
# ---------------------------------------------------------------------------


def _default_raw_instance(
    n_customers: int,
    n_fleets: int,
    lambda_weight: float,
) -> Dict[str, Any]:
    return {
        "depot": [50.0, 50.0],
        "customers": [],
        "n_fleets": n_fleets,
        "truck_capacity": 50.0,
        "drone_capacity": 15.0,
        "system_duration": 100.0,
        "trip_duration": 25.0,
        "truck_speed": 1.0,
        "drone_speed": 2.0,
        "truck_cost": 1.0,
        "drone_cost": 0.5,
        "launch_time": 2.0,
        "land_time": 3.0,
        "service_time": 5.0,
        "lambda_weight": lambda_weight,
    }


def _normalize_generated_instance(
    data: Dict[str, Any], kwargs: Dict[str, Any]
) -> Dict[str, Any]:
    general = data["Config"]["General"]
    vehicles = data["Config"]["Vehicles"]
    depot = data["Config"]["Depot"]

    raw = _default_raw_instance(
        n_customers=int(general["NUM_CUSTOMERS"]),
        n_fleets=int(vehicles["NUM_TRUCKS"]),
        lambda_weight=float(kwargs.get("lambda_weight", 0.5)),
    )
    raw.update(
        {
            "depot": list(depot["coord"]),
            "customers": [
                [
                    float(node["coord"][0]),
                    float(node["coord"][1]),
                    float(node["tw_h"][0]),
                    float(node["tw_h"][1]),
                    float(node["demand"]),
                ]
                for node in data["Nodes"]
            ],
            "n_fleets": int(vehicles["NUM_TRUCKS"]),
            "truck_capacity": float(vehicles["CAPACITY_TRUCK"]),
            "drone_capacity": float(vehicles["CAPACITY_DRONE"]),
            "system_duration": float(general["T_MAX_SYSTEM_H"]),
            "trip_duration": float(vehicles["DRONE_DURATION_H"]),
            "truck_speed": float(vehicles["V_TRUCK_KM_H"]),
            "drone_speed": float(vehicles["V_DRONE_KM_H"]),
            "launch_time": float(vehicles["DRONE_TAKEOFF_MIN"]) / 60.0,
            "land_time": float(vehicles["DRONE_LANDING_MIN"]) / 60.0,
            "service_time": float(vehicles["SERVICE_TIME_MIN"]) / 60.0,
        }
    )
    return raw


def _load_generated_config() -> Dict[str, Any]:
    with (_GENERATED_ROOT / "config.json").open() as fh:
        return json.load(fh)


def get_generator(env_cfg: EnvironmentConfig) -> Callable[..., Any]:
    name = env_cfg.problem_name

    if name == "vrpbtw":
        kwargs = dict(env_cfg.problem_kwargs)
        generated_cfg = _load_generated_config()

        def _generator(
            size: Optional[int] = None,
            rng=None,
            **extra_kwargs: Any,
        ) -> Dict[str, Any]:
            n_customers = int(
                size if size is not None else kwargs.get("n_customers", 10)
            )
            dist = str(extra_kwargs.get("dist", kwargs.get("coord_distribution", "RC")))
            ratio = float(extra_kwargs.get("ratio", kwargs.get("linehaul_ratio", 0.5)))
            seed_offset = int(
                extra_kwargs.get(
                    "generator_seed_offset",
                    kwargs.get("generator_seed_offset", 0),
                )
            )
            seed = (
                int(rng.integers(0, 2**31 - 1)) + seed_offset
                if rng is not None
                else int(extra_kwargs.get("seed", kwargs.get("seed", 42))) + seed_offset
            )
            data = create_instance(
                generated_cfg, n_customers, dist, ratio, seed
            )
            raw = _normalize_generated_instance(data, kwargs)

            # Allow RL-specific overrides while keeping the data/generated pattern.
            for key in (
                "n_fleets",
                "truck_capacity",
                "drone_capacity",
                "truck_speed",
                "drone_speed",
                "truck_cost",
                "drone_cost",
                "launch_time",
                "land_time",
                "service_time",
                "trip_duration",
                "lambda_weight",
            ):
                if key in kwargs:
                    raw[key] = kwargs[key]
                if key in extra_kwargs:
                    raw[key] = extra_kwargs[key]

            return raw

        return _generator

    raise ValueError(f"No generator for problem {name!r}.  Known: ['vrpbtw']")


# ---------------------------------------------------------------------------
# MAML task pool
# ---------------------------------------------------------------------------


def build_task_pool(cfg: ExperimentConfig) -> Dict[str, Tuple[Any, Callable]]:
    """
    Build a MAML task pool with multiple coordinate distributions.

    Task pool structure: {task_id: (VRPBTWEnv, generator_fn)}
    Task ID format: "{size}_{distribution}" (e.g., "10_R", "20_C", "50_RC")

    Fleet sizes are read from data/generated/config.json FLEET_SIZES.
    Each task-distribution combination gets its own seeded RNG for reproducibility.

    The returned dict is compatible with MetaTrainer, which internally converts
    the task pool into Task objects and a TaskManager for curriculum-based
    meta-learning.
    """
    task_sizes = cfg.maml.task_sizes
    task_distributions = cfg.maml.task_distributions
    data_cfg = _load_generated_config()
    fleet_map: Dict[int, int] = {
        int(k): int(v) for k, v in data_cfg["FLEET_SIZES"].items()
    }

    base_gen = get_generator(cfg.env)
    pool: Dict[str, Tuple[Any, Callable]] = {}

    for size in task_sizes:
        n_fleets = fleet_map.get(size, 2)

        for dist in task_distributions:
            task_id = f"{size}_{dist}"

            # Create problem instance for this size/distribution combo
            problem = VRPBTWEnv(n_customers=size, n_fleets=n_fleets)

            # Each task has a dedicated RNG: hash of (size, dist) ensures determinism
            # while keeping distinct streams for different tasks
            _rng = np.random.default_rng(
                cfg.seed.global_seed + hash((size, dist)) % (2**31 - 1)
            )

            # Closure captures size, distribution, fleet count, and RNG
            gen = _make_task_generator(size, dist, n_fleets, base_gen, _rng)
            pool[task_id] = (problem, gen)

    return pool


def _make_task_generator(
    size: int,
    dist: str,
    n_fleets: int,
    base_gen: Callable,
    rng: np.random.Generator,
) -> Callable:
    """Factory for task-specific generators with captured parameters."""

    def _gen() -> Dict[str, Any]:
        return base_gen(size=size, dist=dist, n_fleets=n_fleets, rng=rng)

    return _gen


def _parse_task_id(task_id: str) -> Tuple[int, str]:
    """Parse task ID string (format: "{size}_{dist}") into (size, dist) tuple.

    Used for proper sorting of task IDs by size (numeric) then distribution (alphabetic).
    """
    parts = task_id.rsplit("_", 1)
    if len(parts) != 2:
        raise ValueError(
            f"Invalid task_id format: {task_id!r}. Expected '{{size}}_{{dist}}'"
        )
    try:
        size = int(parts[0])
    except ValueError:
        raise ValueError(f"Invalid size in task_id {task_id!r}: {parts[0]!r}")
    return size, parts[1]


def sort_task_ids(task_ids: List[str]) -> List[str]:
    """Sort task IDs by size (numeric) then distribution (alphabetic).

    Examples:
        ["10_R", "100_C", "10_C"] → ["10_C", "10_R", "100_C"]
    """
    return sorted(task_ids, key=_parse_task_id)


# ---------------------------------------------------------------------------
# Network
# ---------------------------------------------------------------------------


def build_network(
    cfg: ExperimentConfig,
) -> BasePolicy:
    net_type = cfg.network.network_type

    if net_type == "hgnn":
        return VRPBTWPolicy(cfg=cfg.network)

    raise ValueError(
        f"Unknown network type {net_type!r}.  "
        f"Register it in registry.py.  Known: ['hgnn']"
    )


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------


def build_agent(cfg: ExperimentConfig) -> BaseAgent:
    network = build_network(cfg)
    network = network.to_device(cfg.device)
    return Agent(policy=network, device=cfg.device)
