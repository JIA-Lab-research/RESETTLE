#!/usr/bin/env python3
"""HTTP service for task-conditioned V-JEPA-2-AC nearest-ID recovery planning."""

from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import threading
import time
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
import torch
import yaml
import av
from decord import VideoReader, cpu
from scipy.spatial.transform import Rotation

from libero_bank_utils import canonical_task
from app.vjepa_intent_action.model import PooledGaussianIntentActionHead
from recovery_models import encode_frame, load_models
from app.vjepa_droid.transforms import make_transforms


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--bank-dir", required=True, type=Path)
    parser.add_argument("--intent-config", type=Path, default=Path(__file__).parent / "configs/libero.yaml")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18082)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--planning-horizon", type=int, default=4)
    parser.add_argument("--rotation-bound", type=float, default=0.10)
    parser.add_argument("--bank-trajectory-fraction", type=int, default=100)
    parser.add_argument("--bank-trajectory-seed", type=int, default=4042)
    parser.set_defaults(residual_endpoint_constraint="none")
    return parser.parse_args()


def decode_array(payload: dict) -> np.ndarray:
    if payload.get("format") != "npy_base64":
        raise ValueError("array payload must use npy_base64")
    return np.load(io.BytesIO(base64.b64decode(payload["data"])), allow_pickle=False)


def checkpoint_sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as checkpoint_file:
        for chunk in iter(lambda: checkpoint_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class RecoveryEngine:
    @staticmethod
    def _select_trajectory_rows(metadata: list[dict], fraction: int, seed: int) -> np.ndarray:
        if not 1 <= fraction <= 100:
            raise ValueError("bank trajectory fraction must be in 1..100")
        trajectories: dict[str, dict[tuple[str, str, int], list[int]]] = {}
        for index, row in enumerate(metadata):
            canonical_task = str(row["canonical_task"])
            key = (canonical_task, str(row["source_name"]), int(row["episode_index"]))
            trajectories.setdefault(canonical_task, {}).setdefault(key, []).append(index)

        selected_rows: list[int] = []
        for task_trajectories in trajectories.values():
            ranked_keys = sorted(
                task_trajectories,
                key=lambda key: hashlib.sha256(f"{seed}|{key[0]}|{key[1]}|{key[2]}".encode()).digest(),
            )
            keep = max(1, int(np.ceil(len(ranked_keys) * fraction / 100.0)))
            for key in ranked_keys[:keep]:
                selected_rows.extend(task_trajectories[key])
        return np.asarray(sorted(selected_rows), dtype=np.int64)

    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.device = torch.device(args.device)
        self.encoder, self.predictor, self.target_encoder, self.epoch, _ = load_models(args.checkpoint, self.device)
        self.direct_head = None
        self.direct_checkpoint_epoch = None
        self.intent_action_parameterization = None
        self.intent_state_servo_prior = None
        self.intent_condition_on_state_servo_prior = False
        self.intent_goal_consistency_loss_weight = None
        self.intent_residual_execution_mode = None
        self.intent_residual_representation = None
        self._load_direct_head(Path(args.checkpoint), args.intent_config)
        with np.load(args.bank_dir / "success_id_bank.npz", allow_pickle=False) as data:
            self.features = np.asarray(data["features"], dtype=np.float32)
            self.states = np.asarray(data["states"], dtype=np.float32)
        self.metadata = [json.loads(line) for line in (args.bank_dir / "success_id_bank.jsonl").read_text().splitlines() if line]
        summary = json.loads((args.bank_dir / "success_id_bank_summary.json").read_text())
        bank_hashes = {item.get("checkpoint_sha256") for item in summary["source_shards"]}
        if len(bank_hashes) != 1 or None in bank_hashes:
            raise ValueError("success ID bank must have one checkpoint_sha256 across all source shards")
        self.bank_checkpoint_sha256 = bank_hashes.pop()
        loaded_checkpoint_sha256 = checkpoint_sha256(args.checkpoint)
        if loaded_checkpoint_sha256 != self.bank_checkpoint_sha256:
            raise ValueError(
                "V-JEPA checkpoint does not match the success ID bank: "
                f"{loaded_checkpoint_sha256} != {self.bank_checkpoint_sha256}"
            )
        if len(self.features) != len(self.states) or len(self.features) != len(self.metadata):
            raise ValueError("inconsistent success ID bank arrays and metadata")
        self.source_bank_entries = len(self.metadata)
        selected_rows = self._select_trajectory_rows(
            self.metadata, args.bank_trajectory_fraction, args.bank_trajectory_seed
        )
        self.features = self.features[selected_rows]
        self.states = self.states[selected_rows]
        self.metadata = [self.metadata[index] for index in selected_rows]
        self.bank_trajectory_fraction = args.bank_trajectory_fraction
        self.bank_trajectory_seed = args.bank_trajectory_seed
        self.by_task: dict[str, np.ndarray] = {}
        for index, row in enumerate(self.metadata):
            self.by_task.setdefault(row["canonical_task"], []).append(index)
        self.by_task = {task: np.asarray(indices, dtype=np.int64) for task, indices in self.by_task.items()}
        self.transform = make_transforms(
            random_horizontal_flip=False, random_resize_aspect_ratio=(1.0, 1.0),
            random_resize_scale=(1.0, 1.0), crop_size=256,
        )
        self.plan_lock = threading.Lock()

    def _load_direct_head(self, checkpoint_path: Path, config_path: Path) -> None:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        with config_path.open() as config_file:
            config = yaml.safe_load(config_file)
        intent = config["intent_action"]
        action_parameterization = str(checkpoint.get("action_parameterization", intent.get("action_parameterization", "absolute_action")))
        chunk_size = int(checkpoint.get("action_chunk_size", intent.get("chunk_size", 0)))
        use_local_intent = bool(checkpoint.get("use_local_intent", intent.get("use_local_intent", False)))
        execution_mode = str(intent.get("residual_execution_mode", "receding_horizon"))
        guarded_single_tick = execution_mode == "guarded_single_tick"
        if (
            action_parameterization != "state_servo_residual"
            or chunk_size != 1
            or use_local_intent
            or not guarded_single_tick
            or checkpoint.get("training_contract_version") != "asf1_guarded_s1_native_residual_k1_v7"
            or checkpoint.get("residual_representation") != "executed_correction"
            or checkpoint.get("state_servo_prior") != "state_goal_delta_guarded_single_tick_v7"
        ):
            raise ValueError("Recovery requires a guarded K=1 v7 state-servo residual checkpoint")
        guarded_config = checkpoint.get("guarded_native_config")
        if not isinstance(guarded_config, dict) or guarded_config.get("native_steps") != intent.get("guarded_native_steps"):
            raise ValueError("checkpoint and intent config disagree on guarded native steps")
        for key in (
            "meters_per_native_command", "rotation_gain_matrix", "max_translation_delta",
            "max_rotation_delta", "max_native_command", "min_translation_cosine",
            "min_rotation_cosine", "max_correction_ratio",
        ):
            if guarded_config.get(key) != intent.get(key):
                raise ValueError(f"checkpoint and intent config disagree on guarded calibration: {key}")
        self.intent_guarded_native_config = guarded_config
        if (
            checkpoint.get("action_target_frame", 0) != 0
            or checkpoint.get("previous_action_frame") is not None
            or (not guarded_single_tick and chunk_size != self.args.planning_horizon)
            or (guarded_single_tick and chunk_size != 1)
            or (not guarded_single_tick and not use_local_intent)
            or (guarded_single_tick and use_local_intent)
        ):
            raise ValueError("Direct recovery requires a first-action A-SF1 checkpoint matching the planning horizon")
        if action_parameterization not in {"absolute_action", "state_servo_residual"}:
            raise ValueError(f"unsupported intent action parameterization: {action_parameterization}")
        if action_parameterization == "state_servo_residual":
            if int(checkpoint.get("goal_frame_index", intent.get("goal_frame_index", 0))) != self.args.planning_horizon:
                raise ValueError("state-servo residual checkpoint goal_frame_index must match the planning horizon")
            state_servo_prior = checkpoint.get("state_servo_prior", intent.get("state_servo_prior"))
            if state_servo_prior not in {
                "state_goal_delta_divided_by_condition_horizon",
                "full_state_goal_delta_v2",
                "state_goal_delta_fractional_so3_by_condition_horizon_v4",
                "state_goal_delta_guarded_single_tick_v7",
            }:
                raise ValueError("unsupported state-servo residual checkpoint prior contract")
            expected_training_version = {
                "state_goal_delta_fractional_so3_by_condition_horizon_v4": (
                    "asf1_state_servo_conditioned_residual_k4_v6_fractional_open_loop_goal_prior"
                    if checkpoint.get("condition_on_state_servo_prior", intent.get("condition_on_state_servo_prior", False))
                    else "asf1_state_servo_residual_k4_v4_fractional_goal_prior"
                ),
                "full_state_goal_delta_v2": (
                    "asf1_state_servo_conditioned_residual_k4_v5_full_goal_prior"
                    if checkpoint.get("condition_on_state_servo_prior", False)
                    else None
                ),
                "state_goal_delta_guarded_single_tick_v7": "asf1_guarded_s1_native_residual_k1_v7",
            }.get(state_servo_prior)
            if expected_training_version is not None and "training_contract_version" in checkpoint and checkpoint["training_contract_version"] != expected_training_version:
                raise ValueError("fractional-SO(3) residual checkpoint has an incompatible training contract")
            if guarded_single_tick:
                config_representation = intent.get("residual_representation")
                checkpoint_representation = checkpoint.get("residual_representation")
                if config_representation != "executed_correction" or checkpoint_representation != "executed_correction":
                    raise ValueError("guarded single-tick v7 requires directly executed residual corrections")
        config_condition_on_prior = bool(intent.get("condition_on_state_servo_prior", False))
        checkpoint_condition_on_prior = bool(checkpoint.get("condition_on_state_servo_prior", config_condition_on_prior))
        if config_condition_on_prior != checkpoint_condition_on_prior:
            raise ValueError("intent config and checkpoint disagree on state-servo prior conditioning")
        if action_parameterization == "state_servo_residual" and intent.get("state_servo_prior") != state_servo_prior:
            raise ValueError("intent config and checkpoint disagree on the state-servo prior contract")
        if action_parameterization == "state_servo_residual" and execution_mode not in {"receding_horizon", "open_loop_chunk", "guarded_single_tick"}:
            raise ValueError("unsupported residual_execution_mode")
        head = PooledGaussianIntentActionHead(
            encoder_dim=self.encoder.embed_dim,
            state_dim=7,
            action_dim=7,
            chunk_size=chunk_size,
            condition_dim=intent["condition_dim"],
            hidden_dim=intent["hidden_dim"],
            depth=intent["depth"],
            dropout=intent["dropout"],
            min_log_std=intent["min_log_std"],
            max_log_std=intent["max_log_std"],
            condition_on_state_servo_prior=checkpoint_condition_on_prior,
        ).to(self.device)
        head.load_state_dict(
            {key.removeprefix("module."): value for key, value in checkpoint["intent_action_head"].items()},
            strict=True,
        )
        self.direct_head = head.eval()
        self.direct_checkpoint_epoch = checkpoint.get("epoch")
        self.intent_action_parameterization = action_parameterization
        self.intent_state_servo_prior = state_servo_prior if action_parameterization == "state_servo_residual" else None
        self.intent_condition_on_state_servo_prior = checkpoint_condition_on_prior
        self.intent_goal_consistency_loss_weight = float(checkpoint.get("goal_consistency_loss_weight", config["optimization"].get("goal_consistency_loss_weight", 0.0)) or 0.0)
        self.intent_residual_execution_mode = execution_mode if action_parameterization == "state_servo_residual" else None
        self.intent_residual_representation = (
            checkpoint.get("residual_representation") if action_parameterization == "state_servo_residual" else None
        )
        self.intent_guarded_native_steps = int(
            checkpoint.get("guarded_native_steps", intent.get("guarded_native_steps", 5))
        )

    @staticmethod
    def _relative_euler_xyz(state: np.ndarray, goal_state: np.ndarray) -> np.ndarray:
        current_rotation = Rotation.from_euler("xyz", state[3:6], degrees=False).as_matrix()
        goal_rotation = Rotation.from_euler("xyz", goal_state[3:6], degrees=False).as_matrix()
        return Rotation.from_matrix(goal_rotation @ current_rotation.T).as_euler("xyz", degrees=False).astype(np.float32)

    def _state_servo_prior(self, state: np.ndarray, goal_state: np.ndarray) -> np.ndarray:
        delta = np.asarray(goal_state, dtype=np.float32) - np.asarray(state, dtype=np.float32)
        delta[3:6] = self._relative_euler_xyz(state, goal_state)
        if self.intent_state_servo_prior == "full_state_goal_delta_v2":
            return delta
        if self.intent_state_servo_prior == "state_goal_delta_divided_by_condition_horizon":
            return delta / float(self.args.planning_horizon)
        if self.intent_state_servo_prior == "state_goal_delta_fractional_so3_by_condition_horizon_v4":
            delta[:3] /= float(self.args.planning_horizon)
            delta[6] /= float(self.args.planning_horizon)
            relative_rotation = Rotation.from_euler("xyz", delta[3:6], degrees=False)
            delta[3:6] = Rotation.from_rotvec(
                relative_rotation.as_rotvec() / float(self.args.planning_horizon)
            ).as_euler("xyz", degrees=False)
            return delta
        if self.intent_state_servo_prior == "state_goal_delta_guarded_single_tick_v7":
            delta[:3] /= float(self.intent_guarded_native_steps)
            delta[6] /= float(self.intent_guarded_native_steps)
            relative_rotation = Rotation.from_euler("xyz", delta[3:6], degrees=False)
            delta[3:6] = Rotation.from_rotvec(
                relative_rotation.as_rotvec() / float(self.intent_guarded_native_steps)
            ).as_euler("xyz", degrees=False)
            return delta
        raise RuntimeError("state-servo residual prior is not configured")

    @staticmethod
    def _constrain_residual_endpoint(residual_chunk: np.ndarray, mode: str) -> np.ndarray:
        constrained = np.asarray(residual_chunk, dtype=np.float32).copy()
        if constrained.ndim != 2 or constrained.shape[1] != 7:
            raise ValueError(f"expected residual chunk [K,7], got {constrained.shape}")
        if mode == "none":
            return constrained
        if mode == "mean_projection":
            constrained -= constrained.mean(axis=0, keepdims=True)
        elif mode == "last_action_compensation":
            constrained[-1] -= constrained.sum(axis=0)
        else:
            raise ValueError(f"unsupported residual endpoint constraint: {mode}")
        return constrained

    @staticmethod
    def _chunk_terminal_error(state: np.ndarray, goal_state: np.ndarray, action_chunk: np.ndarray) -> dict:
        position = state[:3].astype(np.float32).copy()
        rotation = Rotation.from_euler("xyz", state[3:6], degrees=False).as_matrix()
        gripper = float(state[6])
        for action in action_chunk:
            position += action[:3]
            rotation = Rotation.from_euler("xyz", action[3:6], degrees=False).as_matrix() @ rotation
            gripper += float(action[6])
        target_rotation = Rotation.from_euler("xyz", goal_state[3:6], degrees=False).as_matrix()
        rotation_error = Rotation.from_matrix(rotation @ target_rotation.T).magnitude()
        return {
            "xyz_l2": float(np.linalg.norm(position - goal_state[:3])),
            "rotation_geodesic_rad": float(rotation_error),
            "gripper_abs": float(abs(gripper - float(goal_state[6]))),
        }

    @staticmethod
    def _close_chunk_at_goal(
        state: np.ndarray, goal_state: np.ndarray, action_chunk: np.ndarray
    ) -> np.ndarray:
        """Replace the last action with the exact ideal-state correction."""
        closed = np.asarray(action_chunk, dtype=np.float32).copy()
        if closed.ndim != 2 or closed.shape[0] < 1 or closed.shape[1] != 7:
            raise ValueError(f"expected action chunk [K,7], got {closed.shape}")
        position = np.asarray(state[:3], dtype=np.float64).copy()
        rotation = Rotation.from_euler("xyz", state[3:6], degrees=False)
        gripper = float(state[6])
        for action in closed[:-1]:
            position += action[:3]
            rotation = Rotation.from_euler("xyz", action[3:6], degrees=False) * rotation
            gripper += float(action[6])
        goal_rotation = Rotation.from_euler("xyz", goal_state[3:6], degrees=False)
        closed[-1, :3] = np.asarray(goal_state[:3], dtype=np.float64) - position
        closed[-1, 3:6] = (goal_rotation * rotation.inv()).as_euler("xyz", degrees=False)
        closed[-1, 6] = float(goal_state[6]) - gripper
        return closed

    def _feature(self, image: np.ndarray) -> np.ndarray:
        with torch.inference_mode(), torch.autocast(device_type=self.device.type, dtype=torch.bfloat16, enabled=self.device.type == "cuda"):
            tokens = encode_frame(self.encoder, image, self.transform, self.device)
            return torch.nn.functional.normalize(tokens.float().mean(dim=1), dim=-1)[0].cpu().numpy()

    def _goal_image(self, row: dict) -> np.ndarray:
        observation = row.get("observation")
        if observation is None:
            observation = {
                "type": "wmdata",
                "path": row["wmdata_path"],
                "frame_index": row["frame_index"],
            }
        if observation["type"] == "wmdata":
            with np.load(observation["path"], allow_pickle=False) as data:
                return np.asarray(data["images"][int(observation["frame_index"])], dtype=np.uint8)
        if observation["type"] == "lerobot_video":
            video_path = str(observation["video_path"])
            frame_index = int(observation["frame_index"])
            try:
                reader = VideoReader(video_path, ctx=cpu(0), num_threads=1)
                return np.asarray(reader[frame_index].asnumpy(), dtype=np.uint8)
            except Exception as decord_error:
                try:
                    with av.open(video_path) as container:
                        for index, frame in enumerate(container.decode(video=0)):
                            if index == frame_index:
                                return frame.to_ndarray(format="rgb24").astype(np.uint8, copy=False)
                except Exception as pyav_error:
                    raise RuntimeError(
                        f"failed to decode goal frame {frame_index} from {video_path}; "
                        f"decord={decord_error}; pyav={pyav_error}"
                    ) from pyav_error
                raise IndexError(f"goal frame {frame_index} is out of range in {video_path}")
        raise ValueError(f"Unsupported goal observation type: {observation['type']}")

    def _resolve_goal(self, image: np.ndarray, task: str, goal_reference: dict | None) -> tuple[int, dict, float, bool, np.ndarray, np.ndarray]:
        current_feature = self._feature(image)
        if goal_reference is not None:
            if not isinstance(goal_reference, dict) or "row" not in goal_reference:
                raise ValueError("goal_reference must be an object containing a bank row")
            selected = int(goal_reference["row"])
            if selected < 0 or selected >= len(self.metadata):
                raise ValueError(f"goal_reference row is out of range: {selected}")
            goal_row = self.metadata[selected]
            if goal_row["canonical_task"] != task:
                raise ValueError("goal_reference does not belong to the requested canonical task")
            return selected, goal_row, float(1.0 - self.features[selected] @ current_feature), True, np.asarray([selected]), current_feature
        indices = self.by_task.get(task)
        if indices is None:
            raise LookupError(task)
        distances = 1.0 - self.features[indices] @ current_feature
        selected = int(indices[int(np.argmin(distances))])
        return selected, self.metadata[selected], float(distances.min()), False, indices, current_feature

    def _goal_response(self, selected: int, goal_row: dict, candidate_indices: np.ndarray, current_feature: np.ndarray, retrieval_top_k: int) -> dict:
        ordered_candidates = candidate_indices[np.argsort(1.0 - self.features[candidate_indices] @ current_feature)]
        candidate_rows = ordered_candidates[:max(1, int(retrieval_top_k))]
        response = {
            "goal_reference": {"row": selected},
            "goal": {key: goal_row[key] for key in ("wmdata_path", "frame_index", "env_step", "episode_length") if key in goal_row},
            "goal_observation": goal_row["observation"],
            "goal_latent": self.features[selected].astype(float).tolist(),
            "goal_latent_dimension": int(self.features.shape[1]),
            "bank_checkpoint_sha256": self.bank_checkpoint_sha256,
            "goal_eef_position": self.states[selected, :3].astype(float).tolist(),
            "retrieval_candidates": [
                {
                    "row": int(index),
                    "distance": float(1.0 - self.features[index] @ current_feature),
                    "eef_position": self.states[index, :3].astype(float).tolist(),
                }
                for index in candidate_rows
            ],
        }
        return response

    @staticmethod
    def _request_state(payload: dict) -> tuple[np.ndarray, np.ndarray, str]:
        image = np.asarray(decode_array(payload["image"]), dtype=np.uint8)
        state = np.asarray(decode_array(payload["state"]), dtype=np.float32).reshape(-1)
        if state.shape != (7,):
            raise ValueError(f"expected state (7,), got {state.shape}")
        task = canonical_task(str(payload["task"]), str(payload.get("task_name", "")), str(payload.get("category", "")))
        return image, state, task

    def plan_state_servo_residual(self, payload: dict) -> dict:
        """Add a residual intent head action to the training-compatible state-servo prior."""
        if self.direct_head is None:
            return {"status": "state_servo_residual_not_configured"}
        if self.intent_action_parameterization != "state_servo_residual":
            return {"status": "state_servo_residual_checkpoint_required"}
        with self.plan_lock:
            try:
                image, state, task = self._request_state(payload)
                selected, goal_row, goal_distance, goal_is_fixed, candidate_indices, current_feature = self._resolve_goal(
                    image, task, payload.get("goal_reference")
                )
            except LookupError:
                return {"status": "no_task_success_reference", "canonical_task": task}
            goal_state = self.states[selected].astype(np.float32)
            servo_prior = self._state_servo_prior(state, goal_state)
            state_servo_delta = goal_state - state
            state_servo_delta[3:6] = self._relative_euler_xyz(state, goal_state)
            goal_image = self._goal_image(goal_row)
            with torch.inference_mode(), torch.autocast(device_type=self.device.type, dtype=torch.bfloat16, enabled=self.device.type == "cuda"):
                current = encode_frame(self.encoder, image, self.transform, self.device).mean(dim=1)
                goal = encode_frame(self.encoder, goal_image, self.transform, self.device).mean(dim=1)
                state_tensor = torch.as_tensor(state, device=self.device).unsqueeze(0)
                previous_action = torch.zeros((1, 7), device=self.device)
                previous_valid = torch.zeros((1, 1), device=self.device)
                residual_chunk, log_std_chunk = self.direct_head(
                    current,
                    goal - current,
                    state_tensor,
                    previous_action,
                    previous_valid,
                    servo_prior=torch.as_tensor(servo_prior, device=self.device).unsqueeze(0),
                )
            raw_residual_chunk_np = residual_chunk[0].float().cpu().numpy()
            residual_chunk_np = self._constrain_residual_endpoint(
                raw_residual_chunk_np, self.args.residual_endpoint_constraint
            )
            action_chunk = residual_chunk_np + servo_prior[None, :]
            if self.args.residual_endpoint_constraint != "none":
                action_chunk = self._close_chunk_at_goal(state, goal_state, action_chunk)
                residual_chunk_np = action_chunk - servo_prior[None, :]
            terminal_error = self._chunk_terminal_error(state, goal_state, action_chunk)
            return {
                "status": "ok",
                "planner": "retrieved_goal_state_servo_residual_asf1",
                "direct_checkpoint_epoch": self.direct_checkpoint_epoch,
                "canonical_task": task,
                "goal_distance": goal_distance,
                "goal_is_fixed": goal_is_fixed,
                "goal_state": goal_state.astype(float).tolist(),
                "state_servo_prior": servo_prior.astype(float).tolist(),
                "state_servo_delta": state_servo_delta.astype(float).tolist(),
                "goal_consistency_loss_weight": self.intent_goal_consistency_loss_weight,
                "residual_execution_mode": self.intent_residual_execution_mode,
                "residual_representation": self.intent_residual_representation,
                "residual_gain_applied": False,
                "residual_endpoint_constraint": self.args.residual_endpoint_constraint,
                "terminal_goal_projection": self.args.residual_endpoint_constraint != "none",
                "raw_residual_endpoint_sum": raw_residual_chunk_np.sum(axis=0).astype(float).tolist(),
                "constrained_residual_endpoint_sum": residual_chunk_np.sum(axis=0).astype(float).tolist(),
                "chunk_terminal_goal_error": terminal_error,
                "jepa_residual_action": residual_chunk_np[0].astype(float).tolist(),
                "jepa_residual_action_chunk": residual_chunk_np.astype(float).tolist(),
                "jepa_diff_action": action_chunk[0].astype(float).tolist(),
                "jepa_diff_action_chunk": action_chunk.astype(float).tolist(),
                "action_log_std": log_std_chunk[0, 0].float().cpu().tolist(),
                "action_log_std_chunk": log_std_chunk[0].float().cpu().tolist(),
                **self._goal_response(selected, goal_row, candidate_indices, current_feature, payload.get("retrieval_top_k", 1)),
            }

def handler(engine: RecoveryEngine):
    class RequestHandler(BaseHTTPRequestHandler):
        def _send(self, status: int, body: dict) -> None:
            encoded = json.dumps(body).encode("utf-8")
            try:
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(encoded)))
                self.end_headers()
                self.wfile.write(encoded)
            except BrokenPipeError:
                # The caller timed out while CEM was still finishing.
                return

        def do_GET(self) -> None:
            if self.path == "/health":
                self._send(200, {
                    "status": "ok", "checkpoint_epoch": engine.epoch,
                    "bank_entries": len(engine.features),
                    "source_bank_entries": engine.source_bank_entries,
                    "bank_trajectory_fraction": engine.bank_trajectory_fraction,
                    "bank_trajectory_seed": engine.bank_trajectory_seed,
                    "bank_checkpoint_sha256": engine.bank_checkpoint_sha256,
                    "direct_enabled": engine.direct_head is not None,
                    "direct_checkpoint_epoch": engine.direct_checkpoint_epoch,
                    "intent_action_parameterization": engine.intent_action_parameterization,
                    "intent_state_servo_prior": engine.intent_state_servo_prior,
                    "intent_condition_on_state_servo_prior": engine.intent_condition_on_state_servo_prior,
                    "intent_goal_consistency_loss_weight": engine.intent_goal_consistency_loss_weight,
                    "intent_residual_execution_mode": engine.intent_residual_execution_mode,
                    "intent_residual_representation": engine.intent_residual_representation,
                    "guarded_native_config": engine.intent_guarded_native_config,
                    "residual_endpoint_constraint": engine.args.residual_endpoint_constraint,
                    "terminal_goal_projection": engine.args.residual_endpoint_constraint != "none",
                })
            else:
                self._send(404, {"status": "not_found"})

        def do_POST(self) -> None:
            if self.path != "/plan-recovery-state-servo-residual":
                self._send(404, {"status": "not_found"})
                return
            try:
                payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])).decode("utf-8"))
                started = time.monotonic()
                result = engine.plan_state_servo_residual(payload)
                result["latency_s"] = time.monotonic() - started
                print(
                    "[plan-recovery-state-servo-residual] "
                    f"status={result.get('status')} "
                    f"latency_s={result['latency_s']:.3f}",
                    flush=True,
                )
                self._send(200, result)
            except Exception as error:
                traceback.print_exc()
                self._send(400, {"status": "error", "message": str(error)})

        def log_message(self, format: str, *args) -> None:
            return

    return RequestHandler


def main() -> None:
    args = parse_args()
    engine = RecoveryEngine(args)
    server = ThreadingHTTPServer((args.host, args.port), handler(engine))
    print(f"V-JEPA recovery service listening on http://{args.host}:{args.port}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()