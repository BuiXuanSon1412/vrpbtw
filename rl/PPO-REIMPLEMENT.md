
Implement a PPO training framework that supports batched environments where episodes may terminate at different timesteps but observations have identical tensor shapes.

Requirements:

1. Rollout Collection

* Use vectorized environments and process all environments simultaneously.
* Avoid looping over environments individually.
* At each timestep:

  * Compute policy outputs for the entire batch.
  * Sample actions for active environments.
  * Execute batched environment steps.
  * Store:

    * states
    * actions
    * rewards
    * dones
    * old log probabilities
    * value estimates
* Continue rollout until a predefined horizon T_max or until all environments are done.
* Store rollout tensors in shape:

  * states: [B, T, ...]
  * actions: [B, T]
  * rewards: [B, T]
  * dones: [B, T]
  * old_log_probs: [B, T]
  * values: [B, T]

2. Done Mask Handling

* Episodes may terminate at different timesteps.
* Once an episode is done:

  * Prevent future rewards and advantages from propagating into previous episodes.
  * Use done masks during return and GAE computation.
* Do not allow information leakage across episode boundaries.

3. Return Computation

* Compute returns for the entire batch on GPU.
* Use vectorized tensor operations whenever possible.
* Returns must respect episode termination masks.

4. Generalized Advantage Estimation (GAE)

* Implement batched GAE using:

  delta_t = reward_t + gamma * value_{t+1} * (1 - done_t) - value_t

  advantage_t =
  delta_t
  + gamma * lambda * (1 - done_t) * advantage_{t+1}

* Compute GAE backward along the time dimension.

* Parallelize across batch dimension.

* Do not compute GAE episode-by-episode.

Pseudo-implementation:

gae = zeros(B)

for t in reversed(range(T)):
delta =
rewards[:, t]
+ gamma * next_values[:, t] * (1 - dones[:, t])
- values[:, t]

```
gae =
    delta
    + gamma * lambda * (1 - dones[:, t]) * gae

advantages[:, t] = gae
```

returns = advantages + values

5. PPO Update

* After rollout:

  * Compute advantages.
  * Compute returns.
  * Flatten rollout tensors:

    [B, T] -> [B*T]

* Shuffle flattened samples.

* Create PPO mini-batches from the flattened dataset.

* Mini-batches are optimization batches, not environment batches.

Example:

B = 512
T = 100

Flatten:

512 x 100 = 51200 samples

Sample optimization mini-batches such as:

mini_batch_size = 4096

6. PPO Objective
   For each optimization mini-batch:

* Recompute current log probabilities.

* Compute:

  ratio =
  exp(new_log_prob - old_log_prob)

* Compute clipped PPO objective.

* Compute critic loss.

* Compute entropy bonus.

* Compute total loss:

  loss =
  actor_loss
  + value_coef * critic_loss
  - entropy_coef * entropy

* Update actor and critic parameters.

7. GPU Utilization Requirements

* Rollout collection must be batched.
* Advantage computation must be batched.
* Return computation must be batched.
* PPO optimization must be batched.
* Do not iterate over environments individually.
* The only acceptable sequential loop is over the time dimension when computing GAE.

Generate clean, production-quality PyTorch code following these requirements.
