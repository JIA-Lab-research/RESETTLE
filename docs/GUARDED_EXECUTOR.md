# 🛡️ LIBERO Guarded Execution

The [recovery server](../app/recovery_server.py) returns a **proposed** 7D state-servo residual action, not a LIBERO native command. [guarded_executor.py](../src/recovery/guarded_executor.py) extracts the LIBERO-only, single-tick conversion and gate used by the guarded-S1 evaluation path. It computes an action; it does not connect to a robot, step an environment, monitor a base policy or calibrate trigger thresholds.

## Input contract

Call `GET /health` and `POST /plan-recovery-state-servo-residual` on the **same server**. Require an `ok` recovery response and the same bank checkpoint hash in both replies. The executor uses the checkpoint-verified `guarded_native_config` from `/health` (5 native steps, XYZ metres per native command, 3x3 world-frame rotation gain, motion clamps, cosine minima, ratio bound). Use the **measured current** 7D LIBERO state supplied to the POST: XYZ metres, Euler XYZ radians, gripper closedness in `[0,1]`. This is not the raw 8D LeRobot state.

Run the following example and validation command **from the repository root**, using the environment configured in the [README](../README.md).

```python
import json
from urllib.request import urlopen

from src.recovery.guarded_executor import guarded_action

with urlopen("http://127.0.0.1:18082/health", timeout=5) as connection:
    health = json.load(connection)
# recovery: JSON reply from POST /plan-recovery-state-servo-residual.
# current_state: the same measured [7] used for that POST.
native_action, diagnostics = guarded_action(recovery, current_state, health)
```

Do not call the executor for `no_task_success_reference` or other non-`ok` responses. The caller decides **when** to intervene using base-policy proposals and offline calibrated thresholds, discards any pending base-policy chunk, submits the native command for **one** control tick, then re-observes and returns control to the base policy. This repository does not provide those base-policy, trigger, scheduling or simulator interfaces. The evaluation submission script's category-specific thresholds and gate settings are not baked into this executor.

## Conversion and gate

The implementation follows the LIBERO `guarded_s1_residual_one` path:

1. Convert `state_servo_delta` to a bounded native servo command: clamp XYZ and Euler deltas, divide translation by `5 * meters_per_native_command`, convert the Euler target change to a **world-frame relative rotation vector**, divide by 5 and multiply by the pseudoinverse of `rotation_gain_matrix`. Binarize `current_closedness + state_servo_delta[6]` to native gripper `-1/+1`.
2. Read `jepa_residual_action` only when `residual_representation=executed_correction` and `residual_gain_applied=false`. **Do not multiply it by an additional gain.** Zero the gripper residual, then convert the six motion coordinates to native command using **one** step (not five).
3. Independently gate XYZ and rotation native command groups. A negligible residual passes; nonzero correction against a negligible servo group fails. Otherwise require cosine similarity at least `min_translation_cosine`/`min_rotation_cosine` and residual-to-servo norm ratio at most `max_correction_ratio`.
4. If both groups pass, add their native commands and clip to `max_native_command`. If either fails, use the unmodified bounded native servo command. In both cases leave gripper on the servo baseline. `diagnostics` records the acceptance, per-group reason and fallback.

The packaged [config](../configs/libero.yaml) supplies `native_steps=5`, `max_translation_delta=0.03 m`, `max_rotation_delta=0.05 rad`, `max_native_command=1`, cosine minima `0.5` (translation) and `0.0` (rotation), and maximum correction ratio `0.5`; the actual values passed to the executor come from `/health`, which the server verifies against its checkpoint. The original evaluation also loads an external world-frame rotation calibration file: verify that the checkpoint's gain matrix matches the robot/simulator and action rate before sending any native command. A cosine/ratio guard is **not** collision avoidance or a safety guarantee.

`jepa_diff_action` and `state_servo_prior` in the server response are useful diagnostics in 7D state-delta coordinates; they are **not** the inputs to the two native conversions. In particular, the servo baseline uses the full `state_servo_delta` with divisor 5, while the residual uses divisor 1. Applying both in state-delta coordinates and converting the sum once would not reproduce the guarded controller.

Validate the pure conversion and accept/fallback behavior without hardware using `"$PYTHON" -m unittest discover -s tests -p test_build_bank.py -k GuardedExecutorTest -v`. Full simulator or real-robot behavior remains untested by that CPU check.
