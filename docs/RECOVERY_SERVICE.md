# 🌐 Recovery Service and Runtime Contract

The recovery service returns a proposed correction for a caller to inspect and execute through the guarded native-action conversion. For setup, training, and bank encoding, start with the [quick start](../README.md#getting-started).

Run commands from the repository root:

```bash
bash scripts/serve.sh --device cuda:0 --host 127.0.0.1 --port 18082
```

The implementation lives in [app/recovery_server.py](../app/recovery_server.py). To inspect command-line options without loading a checkpoint:

```bash
python -m app.recovery_server --help
```

## HTTP API

The launcher uses `RESETTLE_CHECKPOINT` and `RESETTLE_BANK` configured during the quick start. The only planning route is `POST /plan-recovery-state-servo-residual`; `GET /health` reports checkpoint/bank metadata and the checkpoint-verified native calibration. The POST JSON needs `task` (LIBERO task text), `image` (RGB uint8 `[H,W,3]`) and `state` (float32 `[7]`: XYZ metres, Euler XYZ radians, gripper closedness). Send each array as `{"format":"npy_base64","data":"<base64 of NumPy .npy bytes>"}`. Optional `goal_reference: {"row": N}` pins a bank row; `retrieval_top_k` controls reported nearest candidates.

The response contains `goal_reference`, `goal_state`, `state_servo_prior [7]`, `jepa_residual_action_chunk [1,7]`, `action_log_std_chunk [1,7]` and `jepa_diff_action_chunk [1,7]`. The last field is the **mean proposed action** (prior plus mean residual), **not** a post-guard executed action. HTTP input/output and sampling usage are demonstrated in [TWO_SEED_DISAGREEMENT.md](TWO_SEED_DISAGREEMENT.md).

## Guarded Native Action

The server deliberately stops before physical execution. [GUARDED_EXECUTOR.md](GUARDED_EXECUTOR.md) and [guarded_executor.py](../src/recovery/guarded_executor.py) show the LIBERO-only downstream single-tick conversion: use the server's `/health` calibration, convert the full state-servo delta over five native steps and the masked motion residual over one step, gate XYZ and rotation independently, then add and clip or fall back to the servo baseline. The gripper always follows the baseline. A caller must supply the measured current state, trigger/schedule interventions externally and execute at most one tick before re-observation; this example has no robot or simulator connection.

## Contract and Limits

| Boundary | Compatibility condition |
| --- | --- |
| Training to checkpoint | `training_contract_version=asf1_guarded_s1_native_residual_k1_v7`, state-servo residual, conditioned prior, executed correction, `K=1`, goal frame 4 and 5 native steps. |
| Checkpoint to expert bank | Same trained encoder weights, deterministic 256px transform, mean pooled layer-normalized frame tokens with L2 normalization for nearest-neighbour retrieval; bank and server require the same SHA-256. |
| Checkpoint to server | Strict encoder/predictor/head state-dict loads; server rejects other action contracts and mismatched native calibration. Current and goal frame encoding uses the same encoder as the bank. |
| Prior and action | Both training and server use SO(3) fractional state-servo `D/5` and a learned K=1 residual. State is LIBERO's 7D XYZ/Euler/closedness convention, not raw LeRobot 8D observation state. |

Training applies image augmentation to 8-frame clips; bank and server encode individual frames as repeated tubelets with deterministic preprocessing. This is the original inference convention, **not** proof of identical train/inference feature distributions. Guard/native losses during training encourage suitable corrections, but the server does not execute the downstream native command clipping, direction or ratio guard. The optional [guarded executor example](GUARDED_EXECUTOR.md) computes these from the verified `/health` calibration; it does not trigger interventions or step a robot. Do not treat the raw HTTP action or the two-seed samples as safe to apply directly. Two seeds sample one K=1 Gaussian residual at a fixed observation/goal, not a temporal two-step chunk and not independently trained policies.
