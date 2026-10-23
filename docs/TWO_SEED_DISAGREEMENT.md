# 🎲 Two-Seed Action Disagreement

The guarded-v7 head predicts a diagonal Gaussian for **one** residual action, with tensors `mean, log_std` of shape `[1, 1, 7]`. The recovery endpoint reports the mean as `jepa_residual_action_chunk` and its log standard deviation as `action_log_std_chunk`. It also reports the five-native-step state-servo prior as `state_servo_prior`. Sampling does not require two server requests: one response fixes the image, task, retrieved goal, state, checkpoint and prior.

For independent seeds $s_1$ and $s_2$, draw $\epsilon_s \sim \mathcal N(0, I_7)$ and construct the **proposed** K=1 chunks:

$$a_s = D/5 + \mu_r + \exp(\log \sigma_r) \odot \epsilon_s.$$

The accompanying [two_seed_disagreement.py](../examples/two_seed_disagreement.py) uses `numpy.random.default_rng(seed)`; these seeds are **not** the training seed or the bank-trajectory selection seed. Its core is:

```python
mean = np.asarray(response["jepa_residual_action_chunk"], dtype=np.float32)  # [1, 7]
std = np.exp(np.asarray(response["action_log_std_chunk"], dtype=np.float32))
prior = np.asarray(response["state_servo_prior"], dtype=np.float32)        # [7]
chunk = prior[None, :] + mean + std * np.random.default_rng(seed).standard_normal((1, 7))
```

Run the service as described in the [README](../README.md), prepare an RGB `uint8` image `[H,W,3]` and a float32 7D state (XYZ metres, Euler XYZ radians, gripper closedness) as NumPy `.npy` files, then run **from the repository root**:

```bash
"$PYTHON" -m examples.two_seed_disagreement --image-npy /path/to/current_rgb.npy \
  --state-npy /path/to/current_state.npy --task "<LIBERO task text>" \
  --seeds 101 202 --output /path/to/two_seed_actions.json
```

`--goal-row N` optionally fixes a bank row. Without it the single request retrieves one row and both samples share that same row. The JSON records both seeds, the matched bank checkpoint hash, goal reference, deterministic mean action, two sampled `[1,7]` proposed chunks and component-wise disagreement:

| Metric | Definition | Unit |
| --- | --- | --- |
| `xyz_l2_m` | $\|a_{s_1,0,:3} - a_{s_2,0,:3}\|_2$ | metres |
| `rotation_l2_rad` | $\|a_{s_1,0,3:6} - a_{s_2,0,3:6}\|_2$ | Euler-delta radians |
| `gripper_abs` | $|a_{s_1,0,6} - a_{s_2,0,6}|$ | closedness |

Do not add these unlike units into a single uncalibrated score. Since both chunks share the same prior, disagreement measures **sample variability of the learned Gaussian residual**, not independent model/seed-training disagreement or robot success probability. In the service response, `jepa_diff_action_chunk` is the **mean proposed action**, not the action after the executor's native conversion, clipping, cosine/ratio guard or environment dynamics. Training includes native-action and guard *losses*; the HTTP endpoint does **not** enforce those physical guards. [GUARDED_EXECUTOR.md](GUARDED_EXECUTOR.md) shows the downstream LIBERO conversion and gate; never execute sampled actions without the full control and re-observation loop. Compare post-guard commands separately if actuator disagreement is the target. K=1 is one action per observation, not a two-step rollout; obtain a new observation and replan to build a temporal chunk.
