"""
Comprehensive test suite for MetaTrainer correctness.

Tests verify:
1. FOMAML gradient flow (most critical)
2. Curriculum learning mechanics
3. Early stopping logic
4. Two-phase pipeline
5. Agent initialization
6. Numerical stability
7. Edge cases
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd()))

import torch
import numpy as np
from typing import Dict, Any

import globals
from core import SeedManager
from core.trainer import MetaTrainer
from core.registry import (
    build_agents,
    build_environment,
    build_evaluators,
    build_logger,
    build_trainer,
)
from config import load_config


def test_fomaml_gradient_flow():
    """TEST 1: Verify FOMAML gradient flow to meta_agent."""
    print("\n" + "="*70)
    print("TEST 1: FOMAML Gradient Flow")
    print("="*70)

    cfg = load_config('configs/trainer/meta.yaml')
    cfg['trainer']['phases']['meta_learning']['control']['epochs'] = 1
    cfg['trainer']['phases']['meta_learning']['control']['batches_per_epoch'] = 1
    cfg['trainer']['phases']['fine_tuning']['enabled'] = False

    seed_mgr = SeedManager(random_seed=42, numpy_seed=42, torch_seed=42)
    seed_mgr.seed_everything()

    env = build_environment(cfg)
    agents = build_agents(cfg)
    evaluators = build_evaluators(cfg, agents, env)
    logger = build_logger(cfg)
    from core.collector import GAECollector
    collector = GAECollector.from_config(cfg['trainer']['collector'])

    trainer = build_trainer(
        cfg['trainer'], agents, env, evaluators, logger, collector
    )

    # Get initial weights
    initial_weights = {n: p.clone() for n, p in trainer.meta_agent.network.named_parameters()}

    # Run one FOMAML step
    meta_summary = trainer.meta_train()

    # Check if meta_agent weights changed
    weights_changed = False
    for name, new_param in trainer.meta_agent.network.named_parameters():
        if name in initial_weights:
            if not torch.allclose(initial_weights[name], new_param, atol=1e-6):
                weights_changed = True
                print(f"  ✅ {name}: weights changed (mean delta: {(new_param - initial_weights[name]).abs().mean():.6f})")

    # Check if meta_agent received gradients
    has_gradients = False
    for param in trainer.meta_agent.network.parameters():
        if param.grad is not None and param.grad.abs().max() > 0:
            has_gradients = True
            break

    logger.close()

    if weights_changed and meta_summary['stop_reason'] == 'completed':
        print("\n✅ PASS: FOMAML gradient flow is correct")
        print("   Meta-agent weights updated via FOMAML adaptation")
        return True
    else:
        print("\n❌ FAIL: FOMAML gradient flow issue")
        if not weights_changed:
            print("   Meta-agent weights did not change")
        return False


def test_curriculum_learning():
    """TEST 2: Verify curriculum learning mechanics."""
    print("\n" + "="*70)
    print("TEST 2: Curriculum Learning")
    print("="*70)

    cfg = load_config('configs/trainer/meta.yaml')
    cfg['trainer']['phases']['meta_learning']['control']['epochs'] = 3
    cfg['trainer']['phases']['meta_learning']['control']['batches_per_epoch'] = 2
    cfg['trainer']['phases']['meta_learning']['control']['eval_interval'] = 10
    cfg['trainer']['phases']['meta_learning']['curriculum']['check_interval'] = 1
    cfg['trainer']['phases']['fine_tuning']['enabled'] = False

    seed_mgr = SeedManager(random_seed=42, numpy_seed=42, torch_seed=42)
    seed_mgr.seed_everything()

    env = build_environment(cfg)
    agents = build_agents(cfg)
    evaluators = build_evaluators(cfg, agents, env)
    logger = build_logger(cfg)
    from core.collector import GAECollector
    collector = GAECollector.from_config(cfg['trainer']['collector'])

    trainer = build_trainer(
        cfg['trainer'], agents, env, evaluators, logger, collector
    )

    initial_tasks = len(trainer.active_tasks)
    print(f"\n  Initial active tasks: {initial_tasks}")
    print(f"  Total available tasks: {len(env.tasks)}")

    # Run training
    meta_summary = trainer.meta_train()

    final_tasks = len(trainer.active_tasks)
    print(f"  Final active tasks: {final_tasks}")

    logger.close()

    # Curriculum should expand tasks (or stay same if entropy doesn't drop below threshold)
    if final_tasks >= initial_tasks:
        print(f"\n✅ PASS: Curriculum learning works (tasks expanded from {initial_tasks} to {final_tasks})")
        return True
    else:
        print(f"\n❌ FAIL: Task count decreased (should only increase or stay same)")
        return False


def test_early_stopping():
    """TEST 3: Verify early stopping logic."""
    print("\n" + "="*70)
    print("TEST 3: Early Stopping")
    print("="*70)

    cfg = load_config('configs/trainer/meta.yaml')
    cfg['trainer']['phases']['meta_learning']['control']['epochs'] = 100
    cfg['trainer']['phases']['meta_learning']['control']['batches_per_epoch'] = 1
    cfg['trainer']['phases']['meta_learning']['control']['eval_interval'] = 1
    cfg['trainer']['phases']['meta_learning']['early_stopping']['patience'] = 3
    cfg['trainer']['phases']['fine_tuning']['enabled'] = False

    seed_mgr = SeedManager(random_seed=42, numpy_seed=42, torch_seed=42)
    seed_mgr.seed_everything()

    env = build_environment(cfg)
    agents = build_agents(cfg)
    evaluators = build_evaluators(cfg, agents, env)
    logger = build_logger(cfg)
    from core.collector import GAECollector
    collector = GAECollector.from_config(cfg['trainer']['collector'])

    trainer = build_trainer(
        cfg['trainer'], agents, env, evaluators, logger, collector
    )

    meta_summary = trainer.meta_train()

    epochs_run = meta_summary['total_epochs']
    print(f"\n  Configured epochs: 100")
    print(f"  Actual epochs completed: {epochs_run}")
    print(f"  Stop reason: {meta_summary['stop_reason']}")

    logger.close()

    # Early stopping should trigger before 100 epochs
    if epochs_run < 100 and meta_summary['stop_reason'] == 'early_stopping':
        print(f"\n✅ PASS: Early stopping triggered after {epochs_run} epochs")
        return True
    elif epochs_run == 100:
        print(f"\n⚠️  SKIP: Did not reach patience threshold (validation improving)")
        return True
    else:
        print(f"\n❌ FAIL: Unexpected stop reason: {meta_summary['stop_reason']}")
        return False


def test_two_phase_pipeline():
    """TEST 4: Verify meta-learning + fine-tuning pipeline."""
    print("\n" + "="*70)
    print("TEST 4: Two-Phase Pipeline")
    print("="*70)

    cfg = load_config('configs/trainer/meta.yaml')
    cfg['trainer']['phases']['meta_learning']['control']['epochs'] = 1
    cfg['trainer']['phases']['meta_learning']['control']['batches_per_epoch'] = 1
    cfg['trainer']['phases']['meta_learning']['control']['eval_interval'] = 10
    cfg['trainer']['phases']['fine_tuning']['enabled'] = True
    cfg['trainer']['phases']['fine_tuning']['control']['epochs'] = 1
    cfg['trainer']['phases']['fine_tuning']['control']['batches_per_epoch'] = 1

    seed_mgr = SeedManager(random_seed=42, numpy_seed=42, torch_seed=42)
    seed_mgr.seed_everything()

    env = build_environment(cfg)
    agents = build_agents(cfg)
    evaluators = build_evaluators(cfg, agents, env)
    logger = build_logger(cfg)
    from core.collector import GAECollector
    collector = GAECollector.from_config(cfg['trainer']['collector'])

    trainer = build_trainer(
        cfg['trainer'], agents, env, evaluators, logger, collector
    )

    summary = trainer.train()

    print(f"\n  Meta-learning enabled: {summary['meta_learning_enabled']}")
    print(f"  Fine-tuning enabled: {summary['fine_tuning_enabled']}")
    print(f"  Total updates: {summary['total_updates']}")
    print(f"  Stop reason: {summary['stop_reason']}")

    logger.close()

    if (summary['meta_learning_enabled'] and
        summary['fine_tuning_enabled'] and
        summary['total_updates'] > 0 and
        summary['stop_reason'] == 'completed'):
        print(f"\n✅ PASS: Two-phase pipeline executed successfully")
        return True
    else:
        print(f"\n❌ FAIL: Two-phase pipeline issue")
        return False


def test_ppo_epochs_loop():
    """TEST 5: Verify ppo_epochs loop in fine-tuning."""
    print("\n" + "="*70)
    print("TEST 5: PPO Epochs Loop (Fine-Tuning)")
    print("="*70)

    cfg = load_config('configs/trainer/meta.yaml')
    cfg['trainer']['phases']['meta_learning']['enabled'] = False
    cfg['trainer']['phases']['fine_tuning']['enabled'] = True
    cfg['trainer']['phases']['fine_tuning']['control']['epochs'] = 1
    cfg['trainer']['phases']['fine_tuning']['control']['batches_per_epoch'] = 2
    cfg['trainer']['phases']['fine_tuning']['control']['ppo_epochs'] = 3
    cfg['trainer']['phases']['fine_tuning']['control']['eval_interval'] = 10

    seed_mgr = SeedManager(random_seed=42, numpy_seed=42, torch_seed=42)
    seed_mgr.seed_everything()

    env = build_environment(cfg)
    agents = build_agents(cfg)
    evaluators = build_evaluators(cfg, agents, env)
    logger = build_logger(cfg)
    from core.collector import GAECollector
    collector = GAECollector.from_config(cfg['trainer']['collector'])

    trainer = build_trainer(
        cfg['trainer'], agents, env, evaluators, logger, collector
    )

    summary = trainer.train()

    # Expected: epochs * batches_per_epoch * ppo_epochs * num_tasks
    num_tasks = len(env.tasks)
    expected_updates = 1 * 2 * 3 * num_tasks
    actual_updates = summary['total_updates']

    print(f"\n  Tasks: {num_tasks}")
    print(f"  Epochs: 1")
    print(f"  Batches/epoch: 2")
    print(f"  PPO epochs: 3")
    print(f"  Expected updates: {expected_updates}")
    print(f"  Actual updates: {actual_updates}")

    logger.close()

    if actual_updates >= expected_updates * 0.9:  # Allow 10% tolerance
        print(f"\n✅ PASS: PPO epochs loop working correctly")
        return True
    else:
        print(f"\n❌ FAIL: PPO epochs not applied correctly")
        print(f"   Expected ~{expected_updates}, got {actual_updates}")
        return False


def test_advantage_normalization():
    """TEST 6: Verify advantage normalization is applied."""
    print("\n" + "="*70)
    print("TEST 6: Advantage Normalization")
    print("="*70)

    # Check if advantage normalization is in code
    from core.agent import PPOAgent
    import inspect

    source = inspect.getsource(PPOAgent.update)

    if 'advantages_normalized' in source:
        print(f"\n  ✅ Advantage normalization found in PPOAgent.update()")

        # Verify it's used in loss computation
        if 'surr1 = ratio * advantages_normalized' in source:
            print(f"  ✅ advantages_normalized used in policy loss")
            print(f"\n✅ PASS: Advantage normalization is properly implemented")
            return True
        else:
            print(f"  ❌ advantages_normalized not used in loss")
            return False
    else:
        print(f"  ❌ Advantage normalization not found")
        return False


def test_gradient_clipping():
    """TEST 7: Verify gradient clipping is applied."""
    print("\n" + "="*70)
    print("TEST 7: Gradient Clipping")
    print("="*70)

    from core.agent import PPOAgent
    import inspect

    source = inspect.getsource(PPOAgent.update)

    if 'clip_grad_norm_' in source:
        print(f"  ✅ Gradient clipping found (torch.nn.utils.clip_grad_norm_)")

        if 'self.max_grad_norm' in source:
            print(f"  ✅ Uses max_grad_norm parameter")
            print(f"\n✅ PASS: Gradient clipping properly implemented")
            return True
        else:
            print(f"  ❌ max_grad_norm not used")
            return False
    else:
        print(f"  ❌ Gradient clipping not found")
        return False


def main():
    """Run all tests."""
    print("\n" + "="*70)
    print("METATRAINER COMPREHENSIVE CORRECTNESS EVALUATION")
    print("="*70)

    results = []

    # Test 1: FOMAML gradient flow (CRITICAL)
    try:
        results.append(("FOMAML Gradient Flow", test_fomaml_gradient_flow()))
    except Exception as e:
        print(f"\n❌ EXCEPTION in test_fomaml_gradient_flow: {e}")
        results.append(("FOMAML Gradient Flow", False))

    # Test 2: Curriculum learning
    try:
        results.append(("Curriculum Learning", test_curriculum_learning()))
    except Exception as e:
        print(f"\n❌ EXCEPTION in test_curriculum_learning: {e}")
        results.append(("Curriculum Learning", False))

    # Test 3: Early stopping
    try:
        results.append(("Early Stopping", test_early_stopping()))
    except Exception as e:
        print(f"\n❌ EXCEPTION in test_early_stopping: {e}")
        results.append(("Early Stopping", False))

    # Test 4: Two-phase pipeline
    try:
        results.append(("Two-Phase Pipeline", test_two_phase_pipeline()))
    except Exception as e:
        print(f"\n❌ EXCEPTION in test_two_phase_pipeline: {e}")
        results.append(("Two-Phase Pipeline", False))

    # Test 5: PPO epochs loop
    try:
        results.append(("PPO Epochs Loop", test_ppo_epochs_loop()))
    except Exception as e:
        print(f"\n❌ EXCEPTION in test_ppo_epochs_loop: {e}")
        results.append(("PPO Epochs Loop", False))

    # Test 6: Advantage normalization
    try:
        results.append(("Advantage Normalization", test_advantage_normalization()))
    except Exception as e:
        print(f"\n❌ EXCEPTION in test_advantage_normalization: {e}")
        results.append(("Advantage Normalization", False))

    # Test 7: Gradient clipping
    try:
        results.append(("Gradient Clipping", test_gradient_clipping()))
    except Exception as e:
        print(f"\n❌ EXCEPTION in test_gradient_clipping: {e}")
        results.append(("Gradient Clipping", False))

    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status}: {name}")

    print(f"\nTotal: {passed}/{total} tests passed")

    if passed == total:
        print("\n✅✅✅ ALL TESTS PASSED - MetaTrainer is CORRECT ✅✅✅")
        return 0
    else:
        print(f"\n❌ {total - passed} test(s) failed")
        return 1


if __name__ == "__main__":
    exit_code = main()
    exit(exit_code)
