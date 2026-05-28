#!/usr/bin/env python3
"""
Profile the training bottleneck: CPU vs GPU time breakdown
Measures actual time spent in:
  1. obs_to_tensor (CPU preprocessing)
  2. GPU transfers
  3. agent.act() (GPU inference)
  4. env.step() (CPU environment simulation)
  5. get_action_mask() specifically
  6. Other env operations
"""

import torch
import time
from pathlib import Path
import sys
from collections import defaultdict

# Add repo to path
repo_root = Path(__file__).parent
sys.path.insert(0, str(repo_root))

from impl.vrpbtw import VRPBTWEnv
from impl.geman import GEMANActorCritic
from config import load_config


def profile_single_trajectory(agent, env, num_steps=256, device='cuda'):
    """
    Profile a single trajectory collection with detailed timing.
    Returns dict with timing breakdown.
    """
    timings = defaultdict(list)

    # Reset environment
    obs, info = env.reset()

    for step in range(num_steps):
        # 1. Preprocessing (CPU)
        t0 = time.perf_counter()
        obs_tensor = torch.as_tensor(obs['node_feat'], dtype=torch.float32, device='cpu')
        mask_tensor = torch.as_tensor(obs['action_mask'], dtype=torch.bool, device='cpu')
        t1 = time.perf_counter()
        timings['cpu_preprocessing'].append((t1 - t0) * 1000)

        # 2. Transfer to GPU
        t0 = time.perf_counter()
        obs_gpu = obs_tensor.to(device)
        mask_gpu = mask_tensor.to(device)
        t1 = time.perf_counter()
        timings['cpu_to_gpu'].append((t1 - t0) * 1000)

        # 3. GPU inference
        t0 = time.perf_counter()
        with torch.no_grad():
            action_idx, logp, val, ent = agent.act(obs_gpu[None], mask_gpu[None])
        t1 = time.perf_counter()
        timings['gpu_inference'].append((t1 - t0) * 1000)

        # 4. Transfer back to CPU
        t0 = time.perf_counter()
        action_idx = action_idx.cpu().numpy()
        t1 = time.perf_counter()
        timings['gpu_to_cpu'].append((t1 - t0) * 1000)

        # 5. Environment step (with mask computation breakdown)
        t0 = time.perf_counter()

        # Time just the mask computation
        t_mask_start = time.perf_counter()
        mask = env.get_action_mask(env.state)
        t_mask_end = time.perf_counter()
        timings['get_action_mask'].append((t_mask_end - t_mask_start) * 1000)

        # Continue with rest of step
        obs, reward, done, truncated, info = env.step(action_idx[0])

        t1 = time.perf_counter()
        timings['env_step'].append((t1 - t0) * 1000)
        timings['env_step_without_mask'].append((t1 - t0 - (t_mask_end - t_mask_start)) * 1000)

        if done or truncated:
            break

    return timings


def format_timing_report(timings):
    """Format timing dict into readable report."""
    report = []
    report.append("\n" + "="*70)
    report.append("BOTTLENECK PROFILING REPORT")
    report.append("="*70)

    per_step = {}
    for key, values in timings.items():
        if values:
            mean_ms = sum(values) / len(values)
            min_ms = min(values)
            max_ms = max(values)
            per_step[key] = mean_ms
            report.append(f"\n{key:30s} {mean_ms:8.3f} ms/step (min={min_ms:.3f}, max={max_ms:.3f})")

    # Compute percentages
    total_cpu_time = per_step.get('cpu_preprocessing', 0) + \
                     per_step.get('env_step', 0) + \
                     per_step.get('cpu_to_gpu', 0) + \
                     per_step.get('gpu_to_cpu', 0)
    total_gpu_time = per_step.get('gpu_inference', 0)
    total_time = total_cpu_time + total_gpu_time

    report.append("\n" + "-"*70)
    report.append("SUMMARY")
    report.append("-"*70)
    report.append(f"CPU work:          {total_cpu_time:8.3f} ms ({100*total_cpu_time/total_time:5.1f}%)")
    report.append(f"GPU work:          {total_gpu_time:8.3f} ms ({100*total_gpu_time/total_time:5.1f}%)")
    report.append(f"Total per step:    {total_time:8.3f} ms")
    report.append(f"Total per 256 steps: {total_time * 256 / 1000:8.3f} sec")

    # Breakdown of env.step
    mask_time = per_step.get('get_action_mask', 0)
    other_env = per_step.get('env_step_without_mask', 0)
    report.append(f"\nenv.step breakdown:")
    report.append(f"  - get_action_mask: {mask_time:8.3f} ms ({100*mask_time/per_step.get('env_step', 1):5.1f}%)")
    report.append(f"  - other env ops:   {other_env:8.3f} ms ({100*other_env/per_step.get('env_step', 1):5.1f}%)")

    report.append("\n" + "="*70)
    return "\n".join(report)


def main():
    print("Loading model and environment...")

    # Load configs
    env_cfg = load_config(str(repo_root / "configs/environment/mvrpbtw.yaml"))
    net_cfg = load_config(str(repo_root / "configs/network/geman.yaml"))

    # Create environment using from_config (same as trainer)
    env = VRPBTWEnv.from_config(env_cfg['environment'])
    print(f"Available tasks: {env.tasks}")
    # Set up a task first
    if env.tasks:
        task_id = env.tasks[0]
        env.retask(task_id)
        env.reset()
    else:
        raise ValueError("No tasks available in environment config")

    # Create agent
    agent = GEMANActorCritic(net_cfg['network'])
    agent.eval()
    agent = agent.cuda()

    print("Profiling single 256-step trajectory...")
    timings = profile_single_trajectory(agent, env, num_steps=256, device='cuda')

    report = format_timing_report(timings)
    print(report)

    # Save report
    report_file = repo_root / "profile_report.txt"
    with open(report_file, 'w') as f:
        f.write(report)
    print(f"\nReport saved to {report_file}")


if __name__ == "__main__":
    main()
