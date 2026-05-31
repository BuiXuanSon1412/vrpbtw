# Batching Analysis: MetaTrainer vs POMOTrainer

## Summary
**Neither trainer batches multiple episodes together for vectorized forward passes.** Both process data serially.

---

## MetaTrainer

### Meta-Training Phase
**Per batch iteration:**
1. For each active task:
   - Collect **1 support episode** (length T_s)
   - PPOAgent processes via **T_s separate forward passes** (one per timestep)
   - Collect **1 query episode** (length T_q)  
   - PPOAgent processes via **T_q separate forward passes**
2. Aggregate query losses across tasks → 1 meta-update

**Forward passes per batch:** 2 × num_active_tasks × avg_episode_length

### Fine-Tuning Phase
**Per batch iteration:**
1. Collect **1 episode** (length T)
2. PPOAgent.update() processes via **T forward passes**
3. Repeat step 2 for **ppo_epochs times** on the same episode
4. No new episode collection between PPO epochs

**Forward passes per batch:** T × ppo_epochs (reprocessing same data)

**Config:** 100 epochs × 100 batches_per_epoch × 10 ppo_epochs = 100,000 forward passes on single-episode batches

---

## POMOTrainer

### Training Phase
**Per batch iteration:**
1. For each of `instances_per_batch` (default=1):
   - POMOSampler collects N episodes (N = number of feasible starting points, typically 10-50)
   - Log probs computed **during collection** (not in forward pass)
   - Returns lists of scalars: `[lp_1, lp_2, ..., lp_N]` and `[r_1, r_2, ..., r_N]`
2. Aggregate: batch_log_probs = list of N-length tensors (one per instance)
3. POMOAgent.update() processes each instance's losses, **no forward passes**

**Forward passes per batch:** 0 (log probs precomputed during POMO collection)

**Note:** Despite being called "instances_per_batch" = 1, the collector itself internally processes N starting points, but without batching them in a single forward pass.

---

## Data Flow Comparison

### MetaTrainer (PPO-based)
```
for task in active_tasks:
    batch = collector.collect(agent, env)  # 1 episode
    observations = batch["observations"]    # List of T observations
    for i, obs_t in enumerate(observations):
        output_t = network.forward(obs_t)   # 1 forward pass per timestep
        results.append(output_t)
    aggregated = torch.cat(results)         # Concatenate T outputs
```

### POMOTrainer (POMO-based)
```
for instance in range(instances_per_batch):
    batch = collector.collect(agent, env)  # Multiple episodes from starting points
    log_probs = batch["log_probs"]          # List of scalars [lp_1, lp_2, ...]
    # No forward passes here - log_probs already computed during collection
    loss = agent.update({"log_probs": log_probs, ...})  # Just loss computation
```

---

## Implications

| Aspect | MetaTrainer | POMOTrainer |
|--------|-------------|------------|
| **Episodes per batch** | 1 (meta) or 1 (fine-tune) | 1 logical batch = N episodes (starting points) |
| **Forward passes per batch** | T forward passes | 0 forward passes (computed during collection) |
| **GPU utilization** | Serial (per-timestep) | Not applicable (no forward passes) |
| **Memory pattern** | Sequential single-obs processing | List-based scalar operations |
| **Scalability** | Limited by episode length T | Limited by number of starting points N |

---

## Conclusion

Neither trainer currently uses **batch vectorization** (processing multiple episodes in parallel in a single forward pass). To add batching:

**For MetaTrainer:**
- Collect K episodes of similar length
- Pad to common length
- Stack observations into batches
- Process K timesteps in parallel: shape `(K, obs_features)` → `(K, n_actions)`
- Requires significant refactoring of episode collection and padding logic

**For POMOTrainer:**
- Less applicable since no forward passes occur during training
- Batching would only help during collection (parallel environment resets)
- Would require parallel environment management
