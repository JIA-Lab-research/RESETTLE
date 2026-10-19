"""LIBERO native single-tick conversion for the guarded residual example."""

import numpy as np
from scipy.spatial.transform import Rotation


def native_from_delta(delta, state, steps, calibration):
    delta = np.asarray(delta, dtype=np.float32)
    state = np.asarray(state, dtype=np.float32)
    if delta.shape != (7,) or state.shape != (7,) or not np.all(np.isfinite(delta)) or not np.all(np.isfinite(state)):
        raise ValueError("delta and state must be finite LIBERO [7] vectors")
    if steps < 1:
        raise ValueError("steps must be positive")
    scale = np.asarray(calibration["meters_per_native_command"], dtype=np.float32)
    gain = np.asarray(calibration["rotation_gain_matrix"], dtype=np.float64)
    limit = float(calibration["max_native_command"])
    if (scale.shape != (3,) or not np.all(np.isfinite(scale)) or np.any(scale <= 0)
            or gain.shape != (3, 3) or not np.all(np.isfinite(gain)) or limit <= 0):
        raise ValueError("invalid native motion calibration")
    condition_number = float(np.linalg.cond(gain))
    if not np.isfinite(condition_number) or condition_number > 100.0:
        raise ValueError("ill-conditioned world-frame rotation gain matrix")
    translation = np.clip(
        np.clip(delta[:3], -calibration["max_translation_delta"], calibration["max_translation_delta"])
        / (steps * scale), -limit, limit,
    )
    physical_rotation = np.clip(delta[3:6], -calibration["max_rotation_delta"], calibration["max_rotation_delta"])
    current_rotation = Rotation.from_euler("xyz", state[3:6])
    target_rotation = Rotation.from_euler("xyz", state[3:6] + physical_rotation)
    world_rotation_vector = (target_rotation * current_rotation.inv()).as_rotvec()
    rotation = np.clip(np.linalg.pinv(gain) @ (world_rotation_vector / steps), -limit, limit)
    gripper = 1.0 if np.clip(float(state[6] + delta[6]), 0.0, 1.0) >= 0.5 else -1.0
    return np.concatenate((translation, rotation, [gripper])).astype(np.float32)


def guarded_action(recovery, state, health):
    if recovery.get("status") != "ok" or recovery.get("residual_representation") != "executed_correction":
        raise ValueError("requires a successful guarded executed-correction response")
    if recovery.get("residual_gain_applied") is not False or recovery.get("residual_execution_mode") != "guarded_single_tick":
        raise ValueError("unexpected residual scaling or execution contract")
    if (recovery.get("bank_checkpoint_sha256") != health.get("bank_checkpoint_sha256")
            or health.get("intent_state_servo_prior") != "state_goal_delta_guarded_single_tick_v7"):
        raise ValueError("recovery response does not match the guarded checkpoint/bank")
    calibration = health["guarded_native_config"]
    if int(calibration["native_steps"]) != 5:
        raise ValueError("this LIBERO executor requires five native steps")

    baseline = native_from_delta(recovery["state_servo_delta"], state, 5, calibration)
    residual = np.asarray(recovery["jepa_residual_action"], dtype=np.float32).copy()
    if residual.shape != (7,) or not np.all(np.isfinite(residual)):
        raise ValueError("expected finite residual [7]")
    residual[6] = 0.0
    correction = native_from_delta(residual, state, 1, calibration)

    def gate(base, change, minimum_cosine):
        base_norm = float(np.linalg.norm(base))
        change_norm = float(np.linalg.norm(change))
        if change_norm <= 1e-8:
            return {"accepted": True, "reason": "negligible_residual"}
        if base_norm <= 1e-8:
            return {"accepted": False, "reason": "negligible_servo_delta"}
        cosine = float(np.clip(np.dot(base, change) / (base_norm * change_norm), -1.0, 1.0))
        ratio = change_norm / base_norm
        if cosine < minimum_cosine:
            reason = "direction_mismatch"
        elif ratio > calibration["max_correction_ratio"]:
            reason = "correction_too_large"
        else:
            reason = "ok"
        return {"accepted": reason == "ok", "reason": reason, "cosine": cosine, "correction_ratio": ratio}

    translation = gate(baseline[:3], correction[:3], calibration["min_translation_cosine"])
    rotation = gate(baseline[3:6], correction[3:6], calibration["min_rotation_cosine"])
    accepted = translation["accepted"] and rotation["accepted"]
    action = baseline.copy()
    if accepted:
        action[:6] = np.clip(baseline[:6] + correction[:6], -calibration["max_native_command"], calibration["max_native_command"])
    return action, {"accepted": accepted, "translation": translation, "rotation": rotation,
                    "fallback": None if accepted else "exact_s1_native_action"}