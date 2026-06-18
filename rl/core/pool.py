"""
vec_env.py
----------
Vectorized environment wrapper for parallel rollout collection using multiprocessing.

SubprocVecEnv manages N environment instances in separate worker processes.
Each worker runs an event loop communicating via multiprocessing.Pipe.
Fan-out/fan-in pattern enables genuine parallelism during step and reset.
"""

from __future__ import annotations

import multiprocessing as mp
from multiprocessing import Process, Pipe
import numpy as np
import torch
from typing import Any, Dict, List, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from multiprocessing.connection import Connection

import globals


def _worker(
    worker_id: int,
    env_class: type,
    env_cfg: Dict[str, Any],
    pipe: Any,
    base_seed: int,
) -> None:
    """
    Subprocess worker: runs env event loop, communicates via pipe.

    Commands:
    - ("retask", task_id): calls env.retask(task_id), replies ("ok", None)
    - ("reset", None): calls env.reset(), replies ("obs", (obs, info))
    - ("step", action): calls env.step(action), auto-resets on terminal, replies ("step", result)
    - ("close", None): breaks event loop and exits
    """
    np.random.seed(base_seed + worker_id)

    env = env_class(env_cfg)

    while True:
        cmd, payload = pipe.recv()

        if cmd == "retask":
            env.retask(payload)
            pipe.send(("ok", None))

        elif cmd == "reset":
            obs, info = env.reset()
            pipe.send(("obs", (obs, info)))

        elif cmd == "step":
            obs, reward, terminated, truncated, info = env.step(payload)

            if terminated or truncated:
                info["terminal_obs"] = obs
                new_obs, new_info = env.reset()
                info["action_mask"] = new_info["action_mask"]
                pipe.send(("step", (new_obs, reward, terminated, truncated, info)))
            else:
                pipe.send(("step", (obs, reward, terminated, truncated, info)))

        elif cmd == "close":
            break


def stack_obs(obs_list: List[Dict[str, np.ndarray]]) -> Dict[str, np.ndarray]:
    """
    Stack list of per-env obs dicts into batched obs dict.

    Edge indices (shared across envs) are not stacked.
    Node/vehicle features and edge attributes are stacked on axis 0.
    """
    stacked = {}
    for key in obs_list[0].keys():
        if "edge_index" in key:
            stacked[key] = obs_list[0][key]
        else:
            stacked[key] = np.stack([obs[key] for obs in obs_list], axis=0)
    return stacked


def batch_obs_to_tensor(
    obs_dict: Dict[str, np.ndarray],
    device: str,
) -> Dict[str, torch.Tensor]:
    """
    Convert batched obs dict to tensors without adding extra batch dim.

    Assumes obs_dict already has batch dimension (from stack_obs).
    """
    result = {}
    for k, v in obs_dict.items():
        if isinstance(v, np.ndarray):
            dtype = torch.long if "edge_index" in k else torch.float32
            result[k] = torch.tensor(v, dtype=dtype, device=device)
        elif isinstance(v, torch.Tensor):
            result[k] = v.to(device)
    return result


class SubprocVecEnv:
    """
    Vectorized environment using multiprocessing for parallel rollouts.

    Spawns n_envs worker processes, each running an env instance.
    Fan-out/fan-in communication pattern for genuine parallelism.
    """

    def __init__(
        self,
        env_class: type,
        env_cfg: Dict[str, Any],
        n_envs: int,
        base_seed: int = 42,
    ):
        self.n_envs = n_envs
        self.env_class = env_class
        self.env_cfg = env_cfg
        self.base_seed = base_seed

        self.parent_conns = []
        self.processes = []

        for worker_id in range(n_envs):
            parent_conn, child_conn = Pipe()
            proc = Process(
                target=_worker,
                args=(worker_id, env_class, env_cfg, child_conn, base_seed),
                daemon=True,
            )
            proc.start()
            child_conn.close()

            self.parent_conns.append(parent_conn)
            self.processes.append(proc)

        # Extract tasks from config (support both flat and hierarchical structures)
        props = env_cfg.get("properties", env_cfg)
        self.tasks = props.get("tasks", [])
        self._last_dones = np.zeros(n_envs, dtype=bool)

    def retask(self, task_id: str) -> None:
        """Send retask command to all workers and wait for confirmation."""
        for conn in self.parent_conns:
            conn.send(("retask", task_id))

        for conn in self.parent_conns:
            cmd, _ = conn.recv()
            assert cmd == "ok"

    def reset(self) -> Tuple[List[Dict], List[Dict]]:
        """Reset all environments and return observations and infos."""
        for conn in self.parent_conns:
            conn.send(("reset", None))

        obs_list = []
        info_list = []
        for conn in self.parent_conns:
            cmd, payload = conn.recv()
            assert cmd == "obs"
            obs, info = payload
            obs_list.append(obs)
            info_list.append(info)

        self._last_dones = np.zeros(self.n_envs, dtype=bool)
        return obs_list, info_list

    def step(
        self, actions: np.ndarray
    ) -> Tuple[List[Dict], np.ndarray, np.ndarray, np.ndarray, List[Dict]]:
        """
        Step all environments with corresponding actions (fan-out/fan-in).

        Actions should be shape (n_envs,) int.

        Returns:
            obs_list: list of next obs dicts (auto-reset obs after terminal)
            rewards: (n_envs,) float
            terminateds: (n_envs,) bool
            truncateds: (n_envs,) bool
            infos: list of info dicts
        """
        for i, conn in enumerate(self.parent_conns):
            conn.send(("step", int(actions[i])))

        obs_list = []
        rewards = []
        terminateds = []
        truncateds = []
        infos = []

        for i, conn in enumerate(self.parent_conns):
            cmd, payload = conn.recv()
            assert cmd == "step"
            obs, reward, terminated, truncated, info = payload
            obs_list.append(obs)
            rewards.append(reward)
            terminateds.append(terminated)
            truncateds.append(truncated)
            infos.append(info)

        rewards = np.array(rewards, dtype=np.float32)
        terminateds = np.array(terminateds, dtype=bool)
        truncateds = np.array(truncateds, dtype=bool)
        self._last_dones = terminateds | truncateds

        return obs_list, rewards, terminateds, truncateds, infos

    def close(self) -> None:
        """Send close command to all workers and join processes."""
        for conn in self.parent_conns:
            conn.send(("close", None))

        for proc in self.processes:
            proc.join(timeout=5)
            if proc.is_alive():
                proc.terminate()
