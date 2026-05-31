# Batching Opportunities: When and What

## 1. Fine-Tuning in MetaTrainer (HIGHEST OPPORTUNITY)

### Current Flow
```
Line 455-477 in trainer.py:
for batch_idx in range(batches_per_epoch):  # 100 batches
    batch = collector.collect(agent, env)   # Collect 1 episode
    # batch["observations"] = [obs_0, obs_1, ..., obs_T]  # LIST
    # batch["masks"] = (T, n_actions)
    # batch["actions"] = (T,)
    # batch["log_probs"] = (T,)
    # batch["advantages"] = (T,)
    # batch["returns"] = (T,)
    
    for ppo_epoch in range(10):  # 10 PPO epochs
        metrics = agent.update(batch)  # PPOAgent.update()
            # Inside: 
            # for i, obs_t in enumerate(observations):  # T iterations
            #     network.evaluate(obs_t, mask_t)  # 1 forward pass per timestep
            #     → T separate forward passes total
```

### Batching Opportunity
**Collect K episodes before updating:**
```python
episodes = []
for i in range(K):  # K=4 or 8
    ep = collector.collect(agent, env)
    episodes.append(ep)

# Option A: Concatenate flat (easiest, works with variable episode lengths)
batch = {
    "observations": sum([ep["observations"] for ep in episodes], []),
    "masks": torch.cat([ep["masks"] for ep in episodes], dim=0),  # (K*T_avg, n_actions)
    "actions": torch.cat([ep["actions"] for ep in episodes], dim=0),  # (K*T_avg,)
    "log_probs": torch.cat([ep["log_probs"] for ep in episodes], dim=0),  # (K*T_avg,)
    "advantages": torch.cat([ep["advantages"] for ep in episodes], dim=0),  # (K*T_avg,)
    "returns": torch.cat([ep["returns"] for ep in episodes], dim=0),  # (K*T_avg,)
    "entropies": torch.cat([ep["entropies"] for ep in episodes], dim=0),  # (K*T_avg,)
}

for ppo_epoch in range(ppo_epochs):
    metrics = agent.update(batch)  # Still K*T forward passes, but could optimize
```

### What's in the Batch

**Key insight:** Even with K episodes, you still do K*T forward passes (one per timestep across all episodes). But now you can:
1. Process all K episodes' timesteps in parallel
2. Vectorize advantage normalization
3. Better GPU utilization

**Batch structure (concatenated):**
```python
{
    "observations": [obs_0_ep0, obs_1_ep0, ..., obs_T_ep0,   # Episode 0
                     obs_0_ep1, obs_1_ep1, ..., obs_T_ep1,   # Episode 1
                     ...
                     obs_0_epK, obs_1_epK, ..., obs_T_epK],  # Episode K
    
    "masks": (K*T_avg, n_actions),  # Concatenated row-wise
    "actions": (K*T_avg,),
    "log_probs": (K*T_avg,),
    "advantages": (K*T_avg,),
    "returns": (K*T_avg,),
    "entropies": (K*T_avg,),
}
```

**Alternative: Padded option (if you want true batched forward passes):**
```python
max_len = max(len(ep["observations"]) for ep in episodes)

# Pad each episode to max_len
padded_obs = pad_sequence([ep["observations"] for ep in episodes], batch_first=True)
# shape: (K, max_len, obs_dim)

# But: observations are dicts (graph nodes), not tensors!
# This requires substantial refactoring

padded_masks = torch.stack([pad(ep["masks"], max_len) for ep in episodes])
# shape: (K, max_len, n_actions)

{
    "observations": padded_obs,  # (K, max_len, obs_dim) - if changed
    "masks": padded_masks,  # (K, max_len, n_actions)
    "actions": padded_actions,  # (K, max_len) with pad_id=-1
    "log_probs": padded_log_probs,
    "advantages": padded_advantages,
    "returns": padded_returns,
    "entropies": padded_entropies,
    "episode_mask": (K, max_len),  # Boolean: which positions are valid
}
```

---

## 2. Meta-Training in MetaTrainer (MEDIUM OPPORTUNITY)

### Current Flow
```
Line 262-277 in trainer.py:
for batch_idx in range(batches_per_epoch):  # 50 batches
    task_losses, task_metrics = _compute_task_losses()
        # Inside (line 652):
        for task_id in active_tasks:  # e.g., 2-10 tasks
            # Collect support batch (1 episode for this task)
            support_batch = collector.collect(...)  # T_s forward passes
            
            # Collect query batch (1 episode for this task)  
            query_batch = collector.collect(...)  # T_q forward passes
    
    # Update meta-agent on aggregated task losses
    meta_agent.update({"task_losses": task_losses})
```

### Batching Opportunity
**Batch across tasks (process all task timesteps together):**

```python
def _compute_task_losses_batched(self):
    """Process all active tasks in a single batch."""
    
    # Collect all support sets
    all_support_obs = []
    all_support_masks = []
    all_support_actions = []
    all_support_log_probs = []
    task_boundaries = [0]  # Track where each task's data starts
    
    for task_id in self.active_tasks:
        support_batch = self.collector.collect(self.sub_agent, self.env)
        all_support_obs.extend(support_batch["observations"])  # List concat
        all_support_masks.append(support_batch["masks"])
        all_support_actions.append(support_batch["actions"])
        all_support_log_probs.append(support_batch["log_probs"])
        task_boundaries.append(task_boundaries[-1] + len(support_batch["observations"]))
    
    # Single forward pass for all tasks
    # Process all_support_obs as one batch
    # Then split results back by task_boundaries
    
    # similar for query sets
```

### What's in the Batch

**Multi-task batch structure:**
```python
{
    # Concatenated across all active tasks
    "observations": [all_obs_from_all_tasks],  # Length = sum of all episode lengths
    
    "masks": torch.cat([masks_task0, masks_task1, ..., masks_taskN]),  
    # shape: (sum(T_i), n_actions)
    
    "actions": torch.cat([actions_task0, actions_task1, ..., actions_taskN]),
    # shape: (sum(T_i),)
    
    "log_probs": torch.cat([lp_task0, lp_task1, ..., lp_taskN]),
    # shape: (sum(T_i),)
    
    "advantages": torch.cat([adv_task0, adv_task1, ..., adv_taskN]),
    "returns": torch.cat([ret_task0, ret_task1, ..., ret_taskN]),
    "entropies": torch.cat([ent_task0, ent_task1, ..., ent_taskN]),
    
    # Metadata for splitting back to per-task
    "task_splits": [T_0, T_0+T_1, T_0+T_1+T_2, ...],
    "task_ids": [0, 0, ..., 0, 1, 1, ..., 1, 2, 2, ..., 2],  # For aggregation
}
```

**Expected benefit:** 
- Forward passes: 2N (one per task) → 2 (one for all support, one for all query)
- If 5 active tasks, avg 20 timesteps each: 200 forward passes → 40 forward passes

---

## 3. POMOTrainer (LOWER OPPORTUNITY)

### Current Flow
```
Line 1077-1113 in trainer.py:
for batch_idx in range(batches_per_epoch):  # 10 batches
    batch_log_probs = []
    batch_rewards = []
    
    for instance_idx in range(instances_per_batch):  # Default: 1
        batch_data = collector.collect(agent, env)
        # batch_data["log_probs"] = [lp_start0, lp_start1, ..., lp_startN]
        # batch_data["rewards"] = [r_start0, r_start1, ..., r_startN]
        # No forward passes here - log probs computed during collection
        
        if batch_data["log_probs"]:
            batch_log_probs.append(torch.stack(batch_data["log_probs"]))
            batch_rewards.append(torch.stack(batch_data["rewards"]))
    
    batch = {
        "log_probs": batch_log_probs,  # List of (N_starting_points_i,) tensors
        "rewards": batch_rewards,
    }
    metrics = agent.update(batch)  # No forward passes, just loss computation
```

### Batching Opportunity (Minor)
**Increase instances_per_batch:**
- Currently: 1 instance → 1 POMO collection → N_starting_points episodes
- If instances_per_batch = 4: 4 POMO collections → 4N episodes per batch
- Agent processes them in separate loops (lines 424-437), still no vectorization

**Better: Collect from multiple instances in parallel:**
```python
# Requires: Multiple environment instances or parallel collection
instances = []
for instance_idx in range(K):  # K parallel instances
    instances.append(collector.collect(agent, env))  # Could be parallel

batch = {
    "log_probs": list_of_K_collections,  # List[List[scalars]]
    "rewards": list_of_K_collections,
    "entropies": list_of_K_collections,
}
agent.update(batch)  # Still processes each instance in loop
```

### What's in the Batch

**Current POMO batch:**
```python
{
    "log_probs": [
        torch.tensor(ep1_sum),  # Episode from starting point 1
        torch.tensor(ep2_sum),  # Episode from starting point 2
        ...,
        torch.tensor(epN_sum),  # Episode from starting point N
    ],
    # Each is a scalar (summed log probabilities from start to end)
    
    "rewards": [
        torch.tensor(r1),
        torch.tensor(r2),
        ...,
        torch.tensor(rN),
    ],
    # Each is a scalar (total return from this starting point)
    
    "entropies": [
        torch.tensor(ent1_mean),  # Mean entropy over trajectory
        torch.tensor(ent2_mean),
        ...,
        torch.tensor(entN_mean),
    ],
}
```

**If you increase instances_per_batch to K:**
```python
{
    "log_probs": [
        # Instance 0
        [lp_s1_i0, lp_s2_i0, ..., lp_sN_i0],
        # Instance 1
        [lp_s1_i1, lp_s2_i1, ..., lp_sN_i1],
        # ...
        # Instance K-1
        [lp_s1_iK, lp_s2_iK, ..., lp_sN_iK],
    ],
    # Shape: List[List[scalars]] = K × N scalars
    
    "rewards": [...],  # Same structure
    "entropies": [...],  # Same structure
}
```

---

## 4. Collection Phase (NOT CURRENT PRIORITY)

### Opportunity: Parallel Episode Collection
Currently: `collector.collect(agent, env)` runs one episode sequentially

Potential:
- Maintain K parallel environments
- Collect from all K in parallel
- Return concatenated/batched data

**Challenge:** Environment state management, reset synchronization

---

## Summary: Where to Batch

| Phase | Current | Batch Size | Benefit | Difficulty |
|-------|---------|-----------|---------|-----------|
| **Fine-tune (MetaTrainer)** | 1 episode/batch | K=4-8 episodes | Vectorize advantage norm, better GPU util | Low |
| **Meta-train (MetaTrainer)** | 1 task/loop | All N tasks together | Reduce forward passes: 2N → 2 | Medium |
| **Training (POMOTrainer)** | instances=1 | instances=4-8 | Process more starting points | Low |
| **Collection** | Serial | K parallel envs | 10-50% speedup | High |

### Recommended Starting Point
**Fine-tuning in MetaTrainer (most impactful + easiest):**
```python
# Modify line 455 loop:
collected_episodes = []
for i in range(min(K, batches_remaining)):
    collected_episodes.append(self.collector.collect(agent, self.env))

batch = concatenate_episodes(collected_episodes)  # Custom function

for ppo_epoch in range(ppo_epochs):
    metrics = agent.update(batch)
    # Now processes K episodes per update instead of 1
```

Requires:
- New `concatenate_episodes()` helper function
- Modify `PPOAgent.update()` to handle larger batches (mostly already handles this)
- Adjust `batches_per_epoch` down by factor of K (if using same compute budget)
