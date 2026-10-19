import copy
import gc
import hashlib
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel

from app.vjepa_droid.libero import (
    LIBEROLeRobotVideoDataset,
)
from app.vjepa_droid.transforms import make_transforms
from app.vjepa_droid.utils import init_video_model
from app.vjepa_intent_action.model import PooledGaussianIntentActionHead, diagonal_gaussian_nll
from src.utils.checkpoint_loader import robust_checkpoint_loader
from src.utils.distributed import init_distributed
from src.utils.logging import AverageMeter, CSVLogger, get_logger, gpu_timer

log_timings = True
log_freq = 10
CHECKPOINT_FREQ = 1
GARBAGE_COLLECT_ITR_FREQ = 50

_GLOBAL_SEED = 0
random.seed(_GLOBAL_SEED)
np.random.seed(_GLOBAL_SEED)
torch.manual_seed(_GLOBAL_SEED)
torch.backends.cudnn.benchmark = True

logger = get_logger(__name__, force=True)


class StaticSubsetSampler(torch.utils.data.Sampler):
    def __init__(self, indices, rank, world_size, shuffle, seed):
        self.indices = [int(index) for index in indices]
        self.rank = rank
        self.world_size = world_size
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def __iter__(self):
        indices = list(self.indices)
        if self.shuffle:
            generator = torch.Generator()
            generator.manual_seed(self.seed + self.epoch)
            order = torch.randperm(len(indices), generator=generator).tolist()
            indices = [indices[position] for position in order]
        total = len(indices)
        per_rank = total // self.world_size
        remainder = total % self.world_size
        start = self.rank * per_rank + min(self.rank, remainder)
        length = per_rank + (1 if self.rank < remainder else 0)
        return iter(indices[start : start + length])

    def __len__(self):
        total = len(self.indices)
        per_rank = total // self.world_size
        remainder = total % self.world_size
        return per_rank + (1 if self.rank < remainder else 0)


def _sha256_tensor_parameters(module):
    digest = hashlib.sha256()
    for parameter in module.parameters():
        digest.update(parameter.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def _strip_state_dict_prefixes(state_dict):
    cleaned = {}
    for key, value in state_dict.items():
        new_key = key
        for prefix in ("module.", "backbone."):
            if new_key.startswith(prefix):
                new_key = new_key[len(prefix) :]
        cleaned[new_key] = value
    return cleaned


def _strip_ddp_prefix(state_dict):
    return {key.removeprefix("module."): value for key, value in state_dict.items()}


def _load_pretrained_components(*, checkpoint_path, encoder, predictor, target_encoder, context_encoder_key, target_encoder_key):
    logger.info(f"Loading pretrained model from {checkpoint_path}")
    checkpoint = robust_checkpoint_loader(checkpoint_path, map_location=torch.device("cpu"))
    epoch = checkpoint.get("epoch", "unknown")

    encoder_msg = encoder.load_state_dict(_strip_state_dict_prefixes(checkpoint[context_encoder_key]), strict=False)
    predictor_msg = predictor.load_state_dict(_strip_state_dict_prefixes(checkpoint["predictor"]), strict=False)
    target_msg = target_encoder.load_state_dict(_strip_state_dict_prefixes(checkpoint[target_encoder_key]), strict=False)

    logger.info(f"loaded pretrained encoder from epoch {epoch} with msg: {encoder_msg}")
    logger.info(f"loaded pretrained predictor from epoch {epoch} with msg: {predictor_msg}")
    logger.info(f"loaded pretrained target encoder from epoch {epoch} with msg: {target_msg}")


def _episode_id_from_sample(sample):
    if len(sample) == 4:
        source, root, episode_path, source_fps = sample
    else:
        source = "expert"
        root, episode_path, source_fps = sample
    del source_fps
    return f"{source}::{root}::{episode_path.relative_to(root)}"


def _build_episode_splits(samples, train_fraction, val_fraction, seed):
    episode_ids = [_episode_id_from_sample(sample) for sample in samples]
    ordered_episode_ids = sorted(episode_ids)
    generator = random.Random(seed)
    generator.shuffle(ordered_episode_ids)

    total = len(ordered_episode_ids)
    train_count = max(1, int(total * train_fraction))
    val_count = max(1, int(total * val_fraction))
    if train_count + val_count >= total:
        val_count = max(1, total - train_count - 1)
    train_ids = set(ordered_episode_ids[:train_count])
    val_ids = set(ordered_episode_ids[train_count : train_count + val_count])
    if not val_ids:
        raise ValueError("Validation split is empty; increase dataset size or adjust fractions")

    train_indices = [index for index, episode_id in enumerate(episode_ids) if episode_id in train_ids]
    val_indices = [index for index, episode_id in enumerate(episode_ids) if episode_id in val_ids]
    manifest = {
        "seed": seed,
        "num_total": total,
        "num_train": len(train_indices),
        "num_val": len(val_indices),
        "train_episode_ids": sorted(train_ids),
        "val_episode_ids": sorted(val_ids),
    }
    manifest["split_hash"] = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode("utf-8")).hexdigest()
    return train_indices, val_indices, manifest


def _make_loader(dataset, indices, batch_size, rank, world_size, shuffle, seed, num_workers, pin_mem, persistent_workers):
    sampler = StaticSubsetSampler(indices, rank=rank, world_size=world_size, shuffle=shuffle, seed=seed)
    loader = torch.utils.data.DataLoader(
        dataset,
        sampler=sampler,
        batch_size=batch_size,
        drop_last=shuffle,
        pin_memory=pin_mem,
        num_workers=num_workers,
        persistent_workers=(num_workers > 0) and persistent_workers,
    )
    return loader, sampler


def _pool_frame_tokens(frame_tokens, normalize_reps):
    if normalize_reps:
        frame_tokens = F.layer_norm(frame_tokens, (frame_tokens.size(-1),))
    return frame_tokens.mean(dim=2)


def _encode_clip_tokens(encoder, clips, batch_size, max_num_frames, normalize_reps):
    repeated = clips.permute(0, 2, 1, 3, 4).flatten(0, 1).unsqueeze(2).repeat(1, 1, 2, 1, 1)
    encoded = encoder(repeated)
    encoded = encoded.view(batch_size, max_num_frames, -1, encoded.size(-1))
    if normalize_reps:
        encoded = F.layer_norm(encoded, (encoded.size(-1),))
    return encoded


def _encode_clip_frames(encoder, clips, batch_size, max_num_frames, normalize_reps):
    return _pool_frame_tokens(
        _encode_clip_tokens(encoder, clips, batch_size, max_num_frames, normalize_reps), normalize_reps=False
    )


def _teacher_forced_dynamics_loss(context_tokens, target_tokens, predictor, actions, states, normalize_reps, loss_exp):
    """Predict frame t+1 tokens from context frame t tokens and its action/state."""
    batch_size, num_frames, tokens_per_frame, embedding_dim = context_tokens.shape
    if actions.size(1) != num_frames - 1 or states.size(1) != num_frames:
        raise ValueError("Dynamics tensors must contain T-1 actions and T states for T frame tokens")
    predicted = predictor(
        context_tokens[:, :-1].flatten(1, 2),
        actions,
        states[:, :-1],
    ).view(batch_size, num_frames - 1, tokens_per_frame, embedding_dim)
    if normalize_reps:
        predicted = F.layer_norm(predicted, (embedding_dim,))
    return torch.mean(torch.abs(predicted - target_tokens[:, 1:]) ** loss_exp) / loss_exp


def _prepare_batch(sample, device, dtype, goal_frame_index, chunk_size):
    clips = sample[0].to(device, non_blocking=True)
    actions = sample[1].to(device, dtype=torch.float32, non_blocking=True)
    states = sample[2].to(device, dtype=torch.float32, non_blocking=True)

    if chunk_size > actions.size(1):
        raise ValueError(f"action chunk_size {chunk_size} exceeds available actions {actions.size(1)}")
    previous_action = torch.zeros((clips.size(0), actions.size(-1)), device=device, dtype=torch.float32)
    previous_action_valid = torch.zeros((clips.size(0), 1), device=device, dtype=torch.float32)
    target_action = actions[:, :chunk_size]
    current_state = states[:, 0]
    del dtype
    return clips, actions, states, current_state, previous_action, previous_action_valid, target_action


def _euler_xyz_to_matrix(euler_xyz):
    """Convert intrinsic XYZ Euler angles using the same convention as SciPy Rotation."""
    x, y, z = euler_xyz.unbind(dim=-1)
    cx, cy, cz = x.cos(), y.cos(), z.cos()
    sx, sy, sz = x.sin(), y.sin(), z.sin()
    return torch.stack(
        (
            cy * cz,
            sx * sy * cz - cx * sz,
            cx * sy * cz + sx * sz,
            cy * sz,
            sx * sy * sz + cx * cz,
            cx * sy * sz - sx * cz,
            -sy,
            sx * cy,
            cx * cy,
        ),
        dim=-1,
    ).reshape(*euler_xyz.shape[:-1], 3, 3)


def _matrix_to_euler_xyz(rotation_matrix):
    """Convert rotation matrices to intrinsic XYZ Euler angles with SciPy's singular convention."""
    sin_y = (-rotation_matrix[..., 2, 0]).clamp(-1.0, 1.0)
    y = torch.asin(sin_y)
    regular = torch.cos(y).abs() > 1e-6
    regular_x = torch.atan2(rotation_matrix[..., 2, 1], rotation_matrix[..., 2, 2])
    regular_z = torch.atan2(rotation_matrix[..., 1, 0], rotation_matrix[..., 0, 0])
    singular_x = torch.where(
        sin_y >= 0,
        torch.atan2(rotation_matrix[..., 0, 1], rotation_matrix[..., 0, 2]),
        torch.atan2(-rotation_matrix[..., 0, 1], -rotation_matrix[..., 0, 2]),
    )
    x = torch.where(regular, regular_x, singular_x)
    z = torch.where(regular, regular_z, torch.zeros_like(regular_z))
    return torch.stack((x, y, z), dim=-1)


def _relative_euler_xyz(current_euler, goal_euler):
    current_rotation = _euler_xyz_to_matrix(current_euler)
    goal_rotation = _euler_xyz_to_matrix(goal_euler)
    relative_rotation = goal_rotation @ current_rotation.transpose(-1, -2)
    return _matrix_to_euler_xyz(relative_rotation)


def _rotation_matrix_power(rotation_matrix, exponent):
    """Return the principal SO(3) power using a differentiable Rodrigues map."""
    skew_vector = torch.stack(
        (
            rotation_matrix[..., 2, 1] - rotation_matrix[..., 1, 2],
            rotation_matrix[..., 0, 2] - rotation_matrix[..., 2, 0],
            rotation_matrix[..., 1, 0] - rotation_matrix[..., 0, 1],
        ),
        dim=-1,
    )
    sin_theta = 0.5 * torch.linalg.vector_norm(skew_vector, dim=-1, keepdim=True)
    cos_theta = ((rotation_matrix.diagonal(dim1=-2, dim2=-1).sum(dim=-1, keepdim=True) - 1.0) * 0.5).clamp(-1.0, 1.0)
    theta = torch.atan2(sin_theta, cos_theta)
    theta_squared = theta.square()
    scale = torch.where(theta > 1e-4, theta / (2.0 * sin_theta.clamp_min(1e-8)), 0.5 + theta_squared / 12.0)
    rotation_vector = scale * skew_vector * float(exponent)
    angle = torch.linalg.vector_norm(rotation_vector, dim=-1, keepdim=True)
    angle_squared = angle.square()
    sine_over_angle = torch.where(angle > 1e-4, angle.sin() / angle, 1.0 - angle_squared / 6.0)
    one_minus_cosine_over_angle_squared = torch.where(
        angle > 1e-4,
        (1.0 - angle.cos()) / angle_squared,
        0.5 - angle_squared / 24.0,
    )
    x, y, z = rotation_vector.unbind(dim=-1)
    skew = torch.stack(
        (
            torch.zeros_like(x), -z, y,
            z, torch.zeros_like(x), -x,
            -y, x, torch.zeros_like(x),
        ),
        dim=-1,
    ).reshape(*rotation_vector.shape[:-1], 3, 3)
    identity = torch.eye(3, dtype=rotation_matrix.dtype, device=rotation_matrix.device)
    return identity + sine_over_angle.unsqueeze(-1) * skew + one_minus_cosine_over_angle_squared.unsqueeze(-1) * (skew @ skew)


def _matrix_to_rotation_vector(rotation_matrix):
    skew_vector = torch.stack(
        (
            rotation_matrix[..., 2, 1] - rotation_matrix[..., 1, 2],
            rotation_matrix[..., 0, 2] - rotation_matrix[..., 2, 0],
            rotation_matrix[..., 1, 0] - rotation_matrix[..., 0, 1],
        ),
        dim=-1,
    )
    sin_theta = 0.5 * torch.linalg.vector_norm(skew_vector, dim=-1, keepdim=True)
    cos_theta = ((rotation_matrix.diagonal(dim1=-2, dim2=-1).sum(dim=-1, keepdim=True) - 1.0) * 0.5).clamp(-1.0, 1.0)
    theta = torch.atan2(sin_theta, cos_theta)
    scale = torch.where(theta > 1e-4, theta / (2.0 * sin_theta.clamp_min(1e-8)), 0.5 + theta.square() / 12.0)
    return scale * skew_vector


def _rollout_action_chunk(current_state, action_chunk):
    """Integrate loader-convention action deltas from a 7-D LIBERO state."""
    if current_state.shape[-1] != 7 or action_chunk.shape[-1] != 7:
        raise ValueError("goal-consistency rollout requires 7-D LIBERO states and actions")
    action_chunk = action_chunk.to(device=current_state.device, dtype=current_state.dtype)
    position = current_state[..., :3]
    rotation = _euler_xyz_to_matrix(current_state[..., 3:6])
    gripper = current_state[..., 6:7]
    for action in action_chunk.unbind(dim=1):
        position = position + action[..., :3]
        rotation = _euler_xyz_to_matrix(action[..., 3:6]) @ rotation
        gripper = gripper + action[..., 6:7]
    return position, rotation, gripper


def _rotation_chordal_loss(predicted_rotation, target_rotation):
    """Use a smooth SO(3) terminal error that remains well-conditioned at zero error."""
    return (predicted_rotation - target_rotation).square().mean(dim=(-2, -1))


def _goal_consistency_loss(current_state, goal_state, action_chunk):
    """Penalize terminal state error after executing a predicted LIBERO action chunk."""
    predicted_position, predicted_rotation, predicted_gripper = _rollout_action_chunk(current_state, action_chunk)
    target_rotation = _euler_xyz_to_matrix(goal_state[..., 3:6])
    position_loss = (predicted_position - goal_state[..., :3]).square().mean()
    rotation_loss = _rotation_chordal_loss(predicted_rotation, target_rotation).mean()
    gripper_loss = (predicted_gripper - goal_state[..., 6:7]).square().mean()
    return position_loss + rotation_loss + gripper_loss


def _state_servo_prior(current_state, goal_state, chunk_size, horizon_steps, prior_contract):
    """Construct a state-servo action prior under the declared checkpoint contract."""
    if horizon_steps < 1:
        raise ValueError("servo prior horizon_steps must be positive")
    delta = goal_state - current_state
    delta[..., 3:6] = _relative_euler_xyz(current_state[..., 3:6], goal_state[..., 3:6])
    if prior_contract == "state_goal_delta_divided_by_condition_horizon":
        delta = delta / float(horizon_steps)
    elif prior_contract == "state_goal_delta_fractional_so3_by_condition_horizon_v4":
        delta[..., :3] = delta[..., :3] / float(horizon_steps)
        delta[..., 6:] = delta[..., 6:] / float(horizon_steps)
        relative_rotation = _euler_xyz_to_matrix(delta[..., 3:6])
        delta[..., 3:6] = _matrix_to_euler_xyz(
            _rotation_matrix_power(relative_rotation, 1.0 / float(horizon_steps))
        )
    elif prior_contract != "full_state_goal_delta_v2":
        raise ValueError(f"unsupported state-servo prior contract: {prior_contract}")
    return delta.unsqueeze(1).expand(-1, chunk_size, -1)


def _intent_targets(
    actions,
    states,
    goal_frame_index,
    chunk_size,
    action_parameterization,
    state_servo_prior,
    guarded_native_steps=5,
):
    target_action = actions[:, :chunk_size]
    if action_parameterization == "absolute_action":
        return target_action, target_action
    if action_parameterization != "state_servo_residual":
        raise ValueError(f"unsupported action_parameterization: {action_parameterization}")
    if state_servo_prior == "state_goal_delta_guarded_single_tick_v7":
        return _guarded_s1_residual_targets(actions, states, goal_frame_index, guarded_native_steps)
    global_prior = _state_servo_prior(
        states[:, 0], states[:, goal_frame_index], chunk_size, goal_frame_index, state_servo_prior
    )
    local_prior = _state_servo_prior(states[:, 0], states[:, 1], chunk_size, 1, state_servo_prior)
    return target_action - global_prior, target_action - local_prior


def _guarded_s1_residual_targets(actions, states, goal_frame_index, native_steps):
    """Build the correction consumed directly by guarded D/N + r single-tick inference."""
    if native_steps < 1:
        raise ValueError("guarded s1 native_steps must be positive")
    prior_contract = "state_goal_delta_fractional_so3_by_condition_horizon_v4"
    expert_tick = _state_servo_prior(
        states[:, 0], states[:, 1], 1, native_steps, prior_contract
    )
    global_servo_tick = _state_servo_prior(
        states[:, 0], states[:, goal_frame_index], 1, native_steps, prior_contract
    )
    global_target = expert_tick - global_servo_tick
    global_target[..., 6] = 0.0
    local_target = torch.zeros_like(global_target)
    return global_target, local_target


def _native_motion_components(diff, current_state, native_steps, config):
    diff = diff.float()
    current_state = current_state.float()
    command_scale = torch.as_tensor(
        config["meters_per_native_command"], device=diff.device, dtype=diff.dtype
    )
    rotation_gain = torch.as_tensor(
        config["rotation_gain_matrix"], device=diff.device, dtype=diff.dtype
    )
    translation = torch.clamp(diff[..., :3], -config["max_translation_delta"], config["max_translation_delta"])
    translation = torch.clamp(
        translation / (float(native_steps) * command_scale),
        -config["max_native_command"],
        config["max_native_command"],
    )
    physical_rotation = torch.clamp(
        diff[..., 3:6], -config["max_rotation_delta"], config["max_rotation_delta"]
    )
    current_rotation = _euler_xyz_to_matrix(current_state[..., 3:6])
    target_rotation = _euler_xyz_to_matrix(current_state[..., 3:6] + physical_rotation)
    world_rotation_vector = _matrix_to_rotation_vector(target_rotation @ current_rotation.transpose(-1, -2))
    rotation = torch.matmul(world_rotation_vector / float(native_steps), torch.linalg.pinv(rotation_gain).transpose(0, 1))
    rotation = torch.clamp(rotation, -config["max_native_command"], config["max_native_command"])
    return torch.cat((translation, rotation), dim=-1)


def _guarded_s1_native_losses(residual_mean, actions, states, goal_frame_index, config):
    current_state = states[:, 0]
    servo_delta = states[:, goal_frame_index] - current_state
    servo_delta = servo_delta.clone()
    servo_delta[..., 3:6] = _relative_euler_xyz(
        current_state[..., 3:6], states[:, goal_frame_index, 3:6]
    )
    baseline = _native_motion_components(servo_delta, current_state, config["native_steps"], config)
    residual_delta = residual_mean[:, 0]
    correction = _native_motion_components(residual_delta, current_state, 1, config)
    desired = _native_motion_components(actions[:, 0], current_state, config["native_steps"], config)
    raw_candidate = baseline + correction
    candidate = torch.clamp(raw_candidate, -config["max_native_command"], config["max_native_command"])
    native_action_loss = F.smooth_l1_loss(candidate, desired)
    saturation_loss = torch.relu(raw_candidate.abs() - config["max_native_command"]).square().mean()

    gate_loss = torch.zeros((), device=residual_mean.device, dtype=torch.float32)
    for component, minimum_cosine in ((slice(0, 3), config["min_translation_cosine"]), (slice(3, 6), config["min_rotation_cosine"])):
        base_component = baseline[..., component]
        correction_component = correction[..., component]
        base_norm = torch.linalg.vector_norm(base_component, dim=-1)
        correction_norm = torch.linalg.vector_norm(correction_component, dim=-1)
        cosine = (base_component * correction_component).sum(dim=-1) / (
            base_norm * correction_norm
        ).clamp_min(1e-8)
        correction_active = correction_norm > 1e-8
        base_active = base_norm > 1e-8
        direction_penalty = torch.relu(float(minimum_cosine) - cosine).square()
        ratio = correction_norm / base_norm.clamp_min(1e-8)
        ratio_penalty = torch.relu(ratio - config["max_correction_ratio"]).square()
        regular_gate_penalty = torch.where(
            correction_active & base_active, direction_penalty + ratio_penalty, 0.0
        )
        negligible_servo_penalty = torch.where(
            correction_active & ~base_active, correction_norm.square(), 0.0
        )
        gate_loss = gate_loss + (regular_gate_penalty + negligible_servo_penalty).mean()
    return native_action_loss, gate_loss, saturation_loss


def _reconstruct_global_actions(goal_residual, current_state, goal_state, chunk_size, goal_frame_index, state_servo_prior):
    global_prior = _state_servo_prior(
        current_state, goal_state, chunk_size, goal_frame_index, state_servo_prior
    )
    return global_prior + goal_residual


def _run_validation(
    *,
    loader,
    device,
    encoder,
    target_encoder,
    predictor,
    head,
    batch_size,
    max_num_frames,
    normalize_reps,
    goal_frame_index,
    chunk_size,
    use_local_intent,
    dynamics_loss_weight,
    dynamics_loss_exp,
    action_parameterization,
    state_servo_prior,
    condition_on_state_servo_prior,
    goal_consistency_loss_weight,
    guarded_native_steps,
    guarded_native_config,
    native_action_loss_weight,
    guard_loss_weight,
    saturation_loss_weight,
):
    encoder.eval()
    predictor.eval()
    head.eval()
    meter = AverageMeter()
    nll_meter = AverageMeter()
    dynamics_meter = AverageMeter()
    goal_consistency_meter = AverageMeter()
    with torch.no_grad():
        for sample in loader:
            clips, actions, states, current_state, previous_action, previous_action_valid, target_action = _prepare_batch(
                sample,
                device=device,
                dtype=torch.float32,
                goal_frame_index=goal_frame_index,
                chunk_size=chunk_size,
            )
            context_tokens = _encode_clip_tokens(
                encoder, clips, batch_size=clips.size(0), max_num_frames=max_num_frames, normalize_reps=normalize_reps
            )
            pooled = _pool_frame_tokens(context_tokens, normalize_reps=False)
            current = pooled[:, 0]
            goal = pooled[:, goal_frame_index]
            goal_target, local_target = _intent_targets(
                actions,
                states,
                goal_frame_index,
                chunk_size,
                action_parameterization,
                state_servo_prior,
                guarded_native_steps,
            )
            goal_servo_prior = None
            local_servo_prior = None
            if condition_on_state_servo_prior:
                goal_servo_prior = _state_servo_prior(
                    current_state,
                    states[:, goal_frame_index],
                    chunk_size,
                    guarded_native_steps if state_servo_prior == "state_goal_delta_guarded_single_tick_v7" else goal_frame_index,
                    "state_goal_delta_fractional_so3_by_condition_horizon_v4"
                    if state_servo_prior == "state_goal_delta_guarded_single_tick_v7"
                    else state_servo_prior,
                )[:, 0]
                if use_local_intent:
                    local_servo_prior = _state_servo_prior(
                        current_state, states[:, 1], chunk_size, 1, state_servo_prior
                    )[:, 0]
            goal_mean, goal_log_std = head(
                current,
                goal - current,
                current_state,
                previous_action,
                previous_action_valid,
                servo_prior=goal_servo_prior,
            )
            nll = diagonal_gaussian_nll(goal_mean, goal_log_std, goal_target)
            native_action_loss = torch.zeros((), device=device)
            guard_loss = torch.zeros((), device=device)
            saturation_loss = torch.zeros((), device=device)
            if state_servo_prior == "state_goal_delta_guarded_single_tick_v7":
                native_action_loss, guard_loss, saturation_loss = _guarded_s1_native_losses(
                    goal_mean, actions, states, goal_frame_index, guarded_native_config
                )
            goal_consistency_loss = torch.zeros((), device=device)
            if goal_consistency_loss_weight > 0.0:
                reconstructed_actions = _reconstruct_global_actions(
                    goal_mean, current_state, states[:, goal_frame_index], chunk_size, goal_frame_index, state_servo_prior
                )
                goal_consistency_loss = _goal_consistency_loss(
                    current_state, states[:, goal_frame_index], reconstructed_actions
                )
            if use_local_intent:
                local_mean, local_log_std = head(
                    current,
                    pooled[:, 1] - current,
                    current_state,
                    previous_action,
                    previous_action_valid,
                    servo_prior=local_servo_prior,
                )
                nll = 0.5 * (nll + diagonal_gaussian_nll(local_mean, local_log_std, local_target))
            dynamics_loss = torch.zeros((), device=device)
            if dynamics_loss_weight > 0.0:
                target_tokens = _encode_clip_tokens(
                    target_encoder,
                    clips,
                    batch_size=clips.size(0),
                    max_num_frames=max_num_frames,
                    normalize_reps=normalize_reps,
                )
                dynamics_loss = _teacher_forced_dynamics_loss(
                    context_tokens, target_tokens, predictor, actions, states, normalize_reps, dynamics_loss_exp
                )
            loss = (
                nll
                + dynamics_loss_weight * dynamics_loss
                + goal_consistency_loss_weight * goal_consistency_loss
                + native_action_loss_weight * native_action_loss
                + guard_loss_weight * guard_loss
                + saturation_loss_weight * saturation_loss
            )
            meter.update(float(loss.detach()), n=clips.size(0))
            nll_meter.update(float(nll.detach()), n=clips.size(0))
            dynamics_meter.update(float(dynamics_loss.detach()), n=clips.size(0))
            goal_consistency_meter.update(float(goal_consistency_loss.detach()), n=clips.size(0))
    encoder.train()
    predictor.train()
    head.train()
    return meter.avg, nll_meter.avg, dynamics_meter.avg, goal_consistency_meter.avg


def _distributed_mean(value, device, world_size):
    if world_size == 1:
        return float(value)
    tensor = torch.tensor([value], device=device, dtype=torch.float64)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    tensor /= float(world_size)
    return float(tensor.item())


def _ema_update(source, target, momentum):
    with torch.no_grad():
        source_parameters = list(source.parameters())
        target_parameters = list(target.parameters())
        torch._foreach_mul_(target_parameters, momentum)
        torch._foreach_add_(target_parameters, source_parameters, alpha=1.0 - momentum)
        for source_buffer, target_buffer in zip(source.buffers(), target.buffers()):
            target_buffer.copy_(source_buffer)


def main(args, resume_preempt=False):
    del resume_preempt

    folder = args.get("folder")
    cfgs_meta = args.get("meta")
    r_file = cfgs_meta.get("resume_checkpoint", None)
    p_file = cfgs_meta.get("pretrain_checkpoint", None)
    load_predictor = cfgs_meta.get("load_predictor", True)
    context_encoder_key = cfgs_meta.get("context_encoder_key", "encoder")
    target_encoder_key = cfgs_meta.get("target_encoder_key", "target_encoder")
    load_encoder = cfgs_meta.get("load_encoder", True)
    seed = cfgs_meta.get("seed", _GLOBAL_SEED)
    save_every_freq = cfgs_meta.get("save_every_freq", -1)
    skip_batches = cfgs_meta.get("skip_batches", -1)
    sync_gc = cfgs_meta.get("sync_gc", False)
    which_dtype = cfgs_meta.get("dtype", "float32")
    if which_dtype.lower() == "bfloat16":
        dtype = torch.bfloat16
        mixed_precision = True
    elif which_dtype.lower() == "float16":
        dtype = torch.float16
        mixed_precision = True
    else:
        dtype = torch.float32
        mixed_precision = False

    cfgs_model = args.get("model")
    model_name = cfgs_model.get("model_name")
    pred_depth = cfgs_model.get("pred_depth")
    pred_num_heads = cfgs_model.get("pred_num_heads", None)
    pred_embed_dim = cfgs_model.get("pred_embed_dim")
    pred_is_frame_causal = cfgs_model.get("pred_is_frame_causal", True)
    uniform_power = cfgs_model.get("uniform_power", False)
    use_rope = cfgs_model.get("use_rope", False)
    use_silu = cfgs_model.get("use_silu", False)
    use_pred_silu = cfgs_model.get("use_pred_silu", False)
    wide_silu = cfgs_model.get("wide_silu", True)
    use_extrinsics = cfgs_model.get("use_extrinsics", False)
    use_sdpa = cfgs_meta.get("use_sdpa", False)
    use_activation_checkpointing = cfgs_model.get("use_activation_checkpointing", False)

    cfgs_data = args.get("data")
    dataset_type = cfgs_data.get("dataset_type")
    if dataset_type != "libero_lerobot":
        raise ValueError("intent-action training supports only data.dataset_type=libero_lerobot")
    datasets = cfgs_data.get("datasets", [])
    dataset_fpcs = cfgs_data.get("dataset_fpcs")
    max_num_frames = max(dataset_fpcs)
    predictor_max_num_frames = int(cfgs_model.get("predictor_max_num_frames", max_num_frames))
    camera_views = cfgs_data.get("camera_views", ["observation.images.image"])
    batch_size = cfgs_data.get("batch_size")
    tubelet_size = cfgs_data.get("tubelet_size")
    fps = cfgs_data.get("fps")
    crop_size = cfgs_data.get("crop_size", 256)
    patch_size = cfgs_data.get("patch_size")
    pin_mem = cfgs_data.get("pin_mem", False)
    num_workers = cfgs_data.get("num_workers", 1)
    persistent_workers = cfgs_data.get("persistent_workers", True)

    cfgs_data_aug = args.get("data_aug")
    transform = make_transforms(
        random_horizontal_flip=cfgs_data_aug.get("horizontal_flip", False),
        random_resize_aspect_ratio=cfgs_data_aug.get("random_resize_aspect_ratio", [0.9, 1.1]),
        random_resize_scale=cfgs_data_aug.get("random_resize_scale", [0.8, 1.0]),
        reprob=cfgs_data_aug.get("reprob", 0.0),
        auto_augment=cfgs_data_aug.get("auto_augment", False),
        motion_shift=cfgs_data_aug.get("motion_shift", False),
        crop_size=crop_size,
    )

    cfgs_loss = args.get("loss")
    normalize_reps = cfgs_loss.get("normalize_reps", True)

    cfgs_opt = args.get("optimization")
    ipe = cfgs_opt.get("ipe", None)
    wd = float(cfgs_opt.get("weight_decay", 1.0e-6))
    num_epochs = int(cfgs_opt.get("epochs"))
    warmup = int(cfgs_opt.get("warmup", 0))
    start_lr = float(cfgs_opt.get("start_lr"))
    lr = float(cfgs_opt.get("lr"))
    final_lr = float(cfgs_opt.get("final_lr", 0.0))
    train_context_encoder = bool(cfgs_opt.get("train_context_encoder", False))
    context_encoder_lr_scale = float(cfgs_opt.get("context_encoder_lr_scale", 1.0))
    train_predictor = bool(cfgs_opt.get("train_predictor", False))
    predictor_lr_scale = float(cfgs_opt.get("predictor_lr_scale", 1.0))
    dynamics_loss_weight = float(cfgs_opt.get("dynamics_loss_weight", 0.0))
    dynamics_loss_exp = float(cfgs_opt.get("dynamics_loss_exp", 1.0))
    goal_consistency_loss_weight = float(cfgs_opt.get("goal_consistency_loss_weight", 0.0))
    native_action_loss_weight = float(cfgs_opt.get("native_action_loss_weight", 0.0))
    guard_loss_weight = float(cfgs_opt.get("guard_loss_weight", 0.0))
    saturation_loss_weight = float(cfgs_opt.get("saturation_loss_weight", 0.0))
    target_encoder_ema = cfgs_opt.get("target_encoder_ema", None)
    if train_context_encoder:
        if context_encoder_lr_scale <= 0.0:
            raise ValueError("context_encoder_lr_scale must be positive when train_context_encoder=true")
        if not isinstance(target_encoder_ema, list) or len(target_encoder_ema) != 2:
            raise ValueError("target_encoder_ema must be [start, end] when train_context_encoder=true")
        target_encoder_ema = tuple(float(value) for value in target_encoder_ema)
        if not 0.0 <= target_encoder_ema[0] <= target_encoder_ema[1] < 1.0:
            raise ValueError("target_encoder_ema values must satisfy 0 <= start <= end < 1")
    elif target_encoder_ema is not None:
        raise ValueError("target_encoder_ema requires train_context_encoder=true")
    if train_predictor:
        if not train_context_encoder:
            raise ValueError("train_predictor requires train_context_encoder=true")
        if predictor_lr_scale <= 0.0 or dynamics_loss_weight <= 0.0 or dynamics_loss_exp <= 0.0:
            raise ValueError("joint predictor training requires positive predictor_lr_scale, dynamics_loss_weight, and dynamics_loss_exp")
    elif dynamics_loss_weight != 0.0:
        raise ValueError("dynamics_loss_weight requires train_predictor=true")
    if goal_consistency_loss_weight < 0.0:
        raise ValueError("goal_consistency_loss_weight must be non-negative")

    cfgs_intent = args.get("intent_action")
    goal_frame_index = int(cfgs_intent.get("goal_frame_index", 4))
    chunk_size = int(cfgs_intent.get("chunk_size", 1))
    use_local_intent = bool(cfgs_intent.get("use_local_intent", False))
    action_parameterization = str(cfgs_intent.get("action_parameterization", "absolute_action"))
    state_servo_prior = str(
        cfgs_intent.get("state_servo_prior", "state_goal_delta_divided_by_condition_horizon")
    )
    condition_on_state_servo_prior = bool(cfgs_intent.get("condition_on_state_servo_prior", False))
    residual_representation = cfgs_intent.get("residual_representation")
    guarded_native_steps = int(cfgs_intent.get("guarded_native_steps", 5))
    guarded_native_config = {
        "native_steps": guarded_native_steps,
        "meters_per_native_command": cfgs_intent.get("meters_per_native_command", [0.010465, 0.012178, 0.010764]),
        "rotation_gain_matrix": cfgs_intent.get("rotation_gain_matrix"),
        "max_translation_delta": float(cfgs_intent.get("max_translation_delta", 0.03)),
        "max_rotation_delta": float(cfgs_intent.get("max_rotation_delta", 0.05)),
        "max_native_command": float(cfgs_intent.get("max_native_command", 1.0)),
        "min_translation_cosine": float(cfgs_intent.get("min_translation_cosine", 0.5)),
        "min_rotation_cosine": float(cfgs_intent.get("min_rotation_cosine", 0.0)),
        "max_correction_ratio": float(cfgs_intent.get("max_correction_ratio", 0.5)),
    }
    train_fraction = float(cfgs_intent.get("train_fraction", 0.9))
    val_fraction = float(cfgs_intent.get("val_fraction", 0.1))
    if goal_frame_index <= 0 or goal_frame_index >= max_num_frames:
        raise ValueError(f"goal_frame_index must be in [1, {max_num_frames - 1}], got {goal_frame_index}")
    if chunk_size <= 0 or chunk_size > goal_frame_index:
        raise ValueError(f"chunk_size must be in [1, {goal_frame_index}], got {chunk_size}")
    if action_parameterization not in {"absolute_action", "state_servo_residual"}:
        raise ValueError("intent_action.action_parameterization must be absolute_action or state_servo_residual")
    if action_parameterization == "state_servo_residual" and state_servo_prior not in {
        "state_goal_delta_divided_by_condition_horizon",
        "full_state_goal_delta_v2",
        "state_goal_delta_fractional_so3_by_condition_horizon_v4",
        "state_goal_delta_guarded_single_tick_v7",
    }:
        raise ValueError("unsupported intent_action.state_servo_prior")
    if condition_on_state_servo_prior and action_parameterization != "state_servo_residual":
        raise ValueError("condition_on_state_servo_prior requires state_servo_residual action parameterization")
    if goal_consistency_loss_weight > 0.0 and action_parameterization != "state_servo_residual":
        raise ValueError("goal_consistency_loss_weight requires state_servo_residual action parameterization")
    if goal_consistency_loss_weight > 0.0 and chunk_size != goal_frame_index:
        raise ValueError("goal_consistency_loss_weight requires chunk_size to equal goal_frame_index")
    if state_servo_prior == "state_goal_delta_guarded_single_tick_v7":
        if chunk_size != 1 or use_local_intent:
            raise ValueError("guarded single-tick residual training requires chunk_size=1 and use_local_intent=false")
        if not condition_on_state_servo_prior:
            raise ValueError("guarded single-tick residual training requires state-servo prior conditioning")
        if residual_representation != "executed_correction":
            raise ValueError("guarded single-tick residual training requires residual_representation=executed_correction")
        if guarded_native_steps < 1:
            raise ValueError("guarded_native_steps must be positive")
        if goal_consistency_loss_weight != 0.0:
            raise ValueError("guarded single-tick residual training does not use four-step goal consistency")
        if native_action_loss_weight <= 0.0 or guard_loss_weight <= 0.0 or saturation_loss_weight <= 0.0:
            raise ValueError("guarded single-tick residual training requires positive native, gate, and saturation loss weights")
        if len(guarded_native_config["meters_per_native_command"]) != 3:
            raise ValueError("meters_per_native_command must have three values")
        if guarded_native_config["rotation_gain_matrix"] is None or np.asarray(guarded_native_config["rotation_gain_matrix"]).shape != (3, 3):
            raise ValueError("rotation_gain_matrix must be a 3x3 matrix")
    elif native_action_loss_weight != 0.0 or guard_loss_weight != 0.0 or saturation_loss_weight != 0.0:
        raise ValueError("native, gate, and saturation losses require the guarded single-tick prior contract")

    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = True

    world_size, rank = init_distributed()
    logger.info(f"Initialized (rank/world-size) {rank}/{world_size}")

    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    if torch.cuda.is_available():
        torch.cuda.set_device(device)

    log_file = os.path.join(folder, f"log_r{rank}.csv")
    latest_path = os.path.join(folder, "latest.pt")
    resume_path = os.path.join(folder, r_file) if r_file is not None else latest_path
    if not os.path.exists(resume_path):
        resume_path = None

    csv_logger = CSVLogger(
        log_file,
        ("%d", "epoch"),
        ("%d", "itr"),
        ("%.6f", "train_nll"),
        ("%.6f", "val_nll"),
        ("%d", "iter-time(ms)"),
        ("%d", "gpu-time(ms)"),
        ("%d", "dataload-time(ms)"),
        mode="+a",
    )

    encoder, predictor = init_video_model(
        uniform_power=uniform_power,
        device=device,
        patch_size=patch_size,
        max_num_frames=max_num_frames,
        tubelet_size=tubelet_size,
        model_name=model_name,
        crop_size=crop_size,
        pred_depth=pred_depth,
        pred_num_heads=pred_num_heads,
        pred_embed_dim=pred_embed_dim,
        action_embed_dim=7,
        pred_is_frame_causal=pred_is_frame_causal,
        use_extrinsics=use_extrinsics,
        use_sdpa=use_sdpa,
        use_silu=use_silu,
        use_pred_silu=use_pred_silu,
        wide_silu=wide_silu,
        use_rope=use_rope,
        use_activation_checkpointing=use_activation_checkpointing,
        predictor_max_num_frames=predictor_max_num_frames,
    )
    target_encoder = copy.deepcopy(encoder)

    for module in (encoder, predictor, target_encoder):
        for parameter in module.parameters():
            parameter.requires_grad = False

    if not load_encoder or not load_predictor:
        raise ValueError("Frozen intent-action training expects load_encoder=true and load_predictor=true")
    _load_pretrained_components(
        checkpoint_path=p_file,
        encoder=encoder,
        predictor=predictor,
        target_encoder=target_encoder,
        context_encoder_key=context_encoder_key,
        target_encoder_key=target_encoder_key,
    )

    if train_context_encoder:
        for parameter in encoder.parameters():
            parameter.requires_grad = True
    if train_predictor:
        for parameter in predictor.parameters():
            parameter.requires_grad = True
        if not use_extrinsics:
            for parameter in predictor.extrinsics_encoder.parameters():
                parameter.requires_grad = False
            logger.info("Frozen predictor.extrinsics_encoder because model.use_extrinsics=false")

    head = PooledGaussianIntentActionHead(
        encoder_dim=encoder.embed_dim,
        state_dim=7,
        action_dim=7,
        chunk_size=chunk_size,
        condition_dim=cfgs_intent.get("condition_dim", 128),
        hidden_dim=cfgs_intent.get("hidden_dim", 1024),
        depth=cfgs_intent.get("depth", 3),
        dropout=cfgs_intent.get("dropout", 0.0),
        min_log_std=cfgs_intent.get("min_log_std", -5.0),
        max_log_std=cfgs_intent.get("max_log_std", 2.0),
        condition_on_state_servo_prior=condition_on_state_servo_prior,
    ).to(device)

    optimizer_parameters = [{"params": head.parameters(), "lr": lr, "name": "intent_action_head"}]
    if train_context_encoder:
        optimizer_parameters.append(
            {
                "params": encoder.parameters(),
                "lr": lr * context_encoder_lr_scale,
                "name": "context_encoder",
            }
        )
    if train_predictor:
        optimizer_parameters.append(
            {
                "params": predictor.parameters(),
                "lr": lr * predictor_lr_scale,
                "name": "predictor",
            }
        )
    optimizer = torch.optim.AdamW(optimizer_parameters, weight_decay=wd)
    scaler = torch.cuda.amp.GradScaler() if mixed_precision else None

    def lr_for_step(step):
        if warmup > 0 and step < warmup:
            alpha = float(step + 1) / float(warmup)
            return start_lr + alpha * (lr - start_lr)
        if num_epochs <= warmup:
            return lr
        decay_steps = max(1, num_epochs - warmup)
        progress = min(1.0, max(0.0, float(step - warmup) / float(decay_steps)))
        return final_lr + 0.5 * (lr - final_lr) * (1.0 + np.cos(np.pi * progress))

    def ema_for_step(step):
        if not train_context_encoder:
            return None
        total_steps = max(1, num_epochs * ipe)
        progress = min(1.0, float(step) / float(total_steps))
        return target_encoder_ema[0] + progress * (target_encoder_ema[1] - target_encoder_ema[0])

    start_epoch = 0
    if resume_path is not None and os.path.exists(resume_path):
        checkpoint = torch.load(resume_path, map_location=torch.device("cpu"))
        action_target_frame = checkpoint.get("action_target_frame", goal_frame_index - 1)
        previous_action_frame = checkpoint.get("previous_action_frame", goal_frame_index - 2)
        saved_chunk_size = int(checkpoint.get("action_chunk_size", 1))
        saved_use_local_intent = bool(checkpoint.get("use_local_intent", False))
        saved_train_context_encoder = bool(checkpoint.get("train_context_encoder", False))
        saved_train_predictor = bool(checkpoint.get("train_predictor", False))
        saved_dynamics_loss_weight = float(checkpoint.get("dynamics_loss_weight", 0.0) or 0.0)
        saved_action_parameterization = str(checkpoint.get("action_parameterization", "absolute_action"))
        saved_state_servo_prior = str(
            checkpoint.get("state_servo_prior", "state_goal_delta_divided_by_condition_horizon")
        )
        saved_condition_on_state_servo_prior = bool(checkpoint.get("condition_on_state_servo_prior", False))
        saved_guarded_native_steps = int(checkpoint.get("guarded_native_steps", guarded_native_steps))
        if (
            action_target_frame != 0
            or previous_action_frame is not None
            or saved_chunk_size != chunk_size
            or saved_use_local_intent != use_local_intent
            or saved_train_context_encoder != train_context_encoder
            or saved_train_predictor != train_predictor
            or saved_dynamics_loss_weight != dynamics_loss_weight
            or saved_action_parameterization != action_parameterization
            or saved_state_servo_prior != state_servo_prior
            or saved_condition_on_state_servo_prior != condition_on_state_servo_prior
            or saved_guarded_native_steps != guarded_native_steps
        ):
            raise RuntimeError(
                "Refusing to resume a checkpoint with a different intent-action temporal or loss contract; "
                "start a new run from the frozen LIBERO predictor checkpoint instead."
            )
        msg = head.load_state_dict(_strip_ddp_prefix(checkpoint["intent_action_head"]), strict=True)
        logger.info(f"loaded intent action head from epoch {checkpoint['epoch']} with msg: {msg}")
        if train_context_encoder:
            encoder.load_state_dict(_strip_state_dict_prefixes(checkpoint["encoder"]), strict=True)
            target_encoder.load_state_dict(_strip_state_dict_prefixes(checkpoint["target_encoder"]), strict=True)
        if train_predictor:
            predictor.load_state_dict(_strip_state_dict_prefixes(checkpoint["predictor"]), strict=True)
        optimizer.load_state_dict(checkpoint["opt"])
        if scaler is not None and checkpoint.get("scaler") is not None:
            scaler.load_state_dict(checkpoint["scaler"])
        start_epoch = int(checkpoint["epoch"])

    encoder_hash_before = _sha256_tensor_parameters(encoder)
    predictor_hash_before = _sha256_tensor_parameters(predictor)
    target_encoder_hash_before = _sha256_tensor_parameters(target_encoder)

    if world_size > 1:
        if train_context_encoder:
            encoder = DistributedDataParallel(encoder, static_graph=False, find_unused_parameters=False)
        if train_predictor:
            predictor = DistributedDataParallel(predictor, static_graph=False, find_unused_parameters=False)
        head = DistributedDataParallel(head, static_graph=False, find_unused_parameters=False)

    expert_dataset = LIBEROLeRobotVideoDataset(
        data_roots=datasets,
        camera_views=camera_views,
        frames_per_clip=max_num_frames,
        fps=fps,
        transform=transform,
    )
    dataset = expert_dataset
    train_indices, val_indices, split_manifest = _build_episode_splits(
        dataset.samples, train_fraction=train_fraction, val_fraction=val_fraction, seed=seed
    )
    if rank == 0:
        folder_path = Path(folder)
        folder_path.mkdir(parents=True, exist_ok=True)
        (folder_path / "intent_action_split_manifest.json").write_text(json.dumps(split_manifest, indent=2) + "\n")

    train_loader, train_sampler = _make_loader(
        dataset,
        train_indices,
        batch_size=batch_size,
        rank=rank,
        world_size=world_size,
        shuffle=True,
        seed=seed,
        num_workers=num_workers,
        pin_mem=pin_mem,
        persistent_workers=persistent_workers,
    )
    val_loader, val_sampler = _make_loader(
        dataset,
        val_indices,
        batch_size=batch_size,
        rank=rank,
        world_size=world_size,
        shuffle=False,
        seed=seed,
        num_workers=num_workers,
        pin_mem=pin_mem,
        persistent_workers=persistent_workers,
    )
    del val_sampler

    if ipe is None:
        ipe = len(train_loader)
    logger.info(f"Intent-action iterations per epoch/dataset length: {ipe}/{len(train_loader)}")

    loader = iter(train_loader)
    if skip_batches > 0:
        for itr in range(skip_batches):
            if itr % 10 == 0:
                logger.info(f"Skip {itr}/{skip_batches} batches")
            try:
                _ = next(loader)
            except Exception:
                train_sampler.set_epoch(start_epoch)
                loader = iter(train_loader)
                _ = next(loader)

    if sync_gc:
        gc.disable()
        gc.collect()

    def save_checkpoint(epoch, path, val_loss, val_nll, val_dynamics, val_goal_consistency):
        if rank != 0:
            return
        save_dict = {
            "encoder": encoder.module.state_dict() if isinstance(encoder, DistributedDataParallel) else encoder.state_dict(),
            "predictor": predictor.module.state_dict() if isinstance(predictor, DistributedDataParallel) else predictor.state_dict(),
            "target_encoder": target_encoder.state_dict(),
            "intent_action_head": head.module.state_dict() if isinstance(head, DistributedDataParallel) else head.state_dict(),
            "opt": optimizer.state_dict(),
            "scaler": None if scaler is None else scaler.state_dict(),
            "epoch": epoch,
            "val_loss": val_loss,
            "val_nll": val_nll,
            "val_dynamics": val_dynamics,
            "val_goal_consistency": val_goal_consistency,
            "split_hash": split_manifest["split_hash"],
            "goal_frame_index": goal_frame_index,
            "training_contract_version": (
                (
                    "asf1_guarded_s1_native_residual_k1_v7"
                    if state_servo_prior == "state_goal_delta_guarded_single_tick_v7"
                    else "asf1_state_servo_conditioned_residual_k4_v5_full_goal_prior"
                    if state_servo_prior == "full_state_goal_delta_v2" and condition_on_state_servo_prior
                    else "asf1_state_servo_conditioned_residual_k4_v6_fractional_open_loop_goal_prior"
                    if state_servo_prior == "state_goal_delta_fractional_so3_by_condition_horizon_v4" and condition_on_state_servo_prior
                    else "asf1_state_servo_residual_k4_v4_fractional_goal_prior"
                    if state_servo_prior == "state_goal_delta_fractional_so3_by_condition_horizon_v4"
                    else "asf1_state_servo_residual_k4_v3_goal_consistency"
                    if state_servo_prior == "full_state_goal_delta_v2" and goal_consistency_loss_weight > 0.0
                    else "asf1_state_servo_residual_k4_v2"
                    if state_servo_prior == "full_state_goal_delta_v2"
                    else "asf1_state_servo_residual_k4_v1"
                )
                if action_parameterization == "state_servo_residual"
                else ("asf1_shared_local_goal_k4_v1" if use_local_intent else "asf0_first_action_v1")
            ),
            "action_parameterization": action_parameterization,
            "state_servo_prior": state_servo_prior if action_parameterization == "state_servo_residual" else None,
            "condition_on_state_servo_prior": condition_on_state_servo_prior,
            "guarded_native_steps": guarded_native_steps if state_servo_prior == "state_goal_delta_guarded_single_tick_v7" else None,
            "residual_representation": residual_representation if state_servo_prior == "state_goal_delta_guarded_single_tick_v7" else None,
            "guarded_native_config": guarded_native_config if state_servo_prior == "state_goal_delta_guarded_single_tick_v7" else None,
            "encoder_training_contract": "ema_target_joint_direct_dynamics_v1" if train_predictor else ("ema_target_joint_direct_v1" if train_context_encoder else "frozen_encoder_v1"),
            "train_context_encoder": train_context_encoder,
            "context_encoder_lr_scale": context_encoder_lr_scale if train_context_encoder else None,
            "target_encoder_ema": target_encoder_ema if train_context_encoder else None,
            "train_predictor": train_predictor,
            "predictor_lr_scale": predictor_lr_scale if train_predictor else None,
            "dynamics_loss_weight": dynamics_loss_weight if train_predictor else None,
            "dynamics_loss_exp": dynamics_loss_exp if train_predictor else None,
            "goal_consistency_loss_weight": goal_consistency_loss_weight,
            "native_action_loss_weight": native_action_loss_weight,
            "guard_loss_weight": guard_loss_weight,
            "saturation_loss_weight": saturation_loss_weight,
            "action_target_frame": 0,
            "previous_action_frame": None,
            "action_chunk_size": chunk_size,
            "use_local_intent": use_local_intent,
            "encoder_hash_before": encoder_hash_before,
            "predictor_hash_before": predictor_hash_before,
        }
        torch.save(save_dict, path)

    for epoch in range(start_epoch, num_epochs):
        logger.info("Epoch %d" % (epoch + 1))
        train_sampler.set_epoch(epoch)

        loss_meter = AverageMeter()
        iter_time_meter = AverageMeter()
        gpu_time_meter = AverageMeter()
        data_elapsed_time_meter = AverageMeter()

        for itr in range(ipe):
            itr_start_time = time.time()
            iter_successful = False
            while not iter_successful:
                try:
                    sample = next(loader)
                    iter_successful = True
                except StopIteration:
                    train_sampler.set_epoch(epoch)
                    loader = iter(train_loader)
                except Exception as error:
                    logger.warning(f"Encountered exception when loading data: {error}")
                    train_sampler.set_epoch(epoch)
                    loader = iter(train_loader)

            data_elapsed_time_ms = (time.time() - itr_start_time) * 1000.0

            if sync_gc and (itr + 1) % GARBAGE_COLLECT_ITR_FREQ == 0:
                gc.collect()

            def train_step():
                step_lr = lr_for_step(epoch)
                for param_group in optimizer.param_groups:
                    if param_group["name"] == "context_encoder":
                        param_group["lr"] = step_lr * context_encoder_lr_scale
                    elif param_group["name"] == "predictor":
                        param_group["lr"] = step_lr * predictor_lr_scale
                    else:
                        param_group["lr"] = step_lr

                clips, actions, states, current_state, previous_action, previous_action_valid, target_action = _prepare_batch(
                    sample,
                    device=device,
                    dtype=dtype,
                    goal_frame_index=goal_frame_index,
                    chunk_size=chunk_size,
                )
                if train_context_encoder:
                    context_tokens = _encode_clip_tokens(
                        encoder,
                        clips,
                        batch_size=clips.size(0),
                        max_num_frames=max_num_frames,
                        normalize_reps=normalize_reps,
                    )
                    pooled = _pool_frame_tokens(context_tokens, normalize_reps=False)
                else:
                    with torch.no_grad():
                        pooled = _encode_clip_frames(
                            encoder,
                            clips,
                            batch_size=clips.size(0),
                            max_num_frames=max_num_frames,
                            normalize_reps=normalize_reps,
                        )
                current = pooled[:, 0]
                goal = pooled[:, goal_frame_index]
                goal_target, local_target = _intent_targets(
                    actions,
                    states,
                    goal_frame_index,
                    chunk_size,
                    action_parameterization,
                    state_servo_prior,
                    guarded_native_steps,
                )
                goal_servo_prior = None
                local_servo_prior = None
                if condition_on_state_servo_prior:
                    goal_servo_prior = _state_servo_prior(
                        current_state,
                        states[:, goal_frame_index],
                        chunk_size,
                        guarded_native_steps if state_servo_prior == "state_goal_delta_guarded_single_tick_v7" else goal_frame_index,
                        "state_goal_delta_fractional_so3_by_condition_horizon_v4"
                        if state_servo_prior == "state_goal_delta_guarded_single_tick_v7"
                        else state_servo_prior,
                    )[:, 0]
                    if use_local_intent:
                        local_servo_prior = _state_servo_prior(
                            current_state, states[:, 1], chunk_size, 1, state_servo_prior
                        )[:, 0]

                with torch.cuda.amp.autocast(dtype=dtype, enabled=mixed_precision):
                    goal_mean, goal_log_std = head(
                        current,
                        goal - current,
                        current_state,
                        previous_action,
                        previous_action_valid,
                        servo_prior=goal_servo_prior,
                    )
                    loss = diagonal_gaussian_nll(goal_mean, goal_log_std, goal_target)
                    goal_consistency_loss = torch.zeros((), device=device)
                    if goal_consistency_loss_weight > 0.0:
                        reconstructed_actions = _reconstruct_global_actions(
                            goal_mean, current_state, states[:, goal_frame_index], chunk_size, goal_frame_index, state_servo_prior
                        )
                        goal_consistency_loss = _goal_consistency_loss(
                            current_state, states[:, goal_frame_index], reconstructed_actions
                        )
                    if use_local_intent:
                        local_mean, local_log_std = head(
                            current,
                            pooled[:, 1] - current,
                            current_state,
                            previous_action,
                            previous_action_valid,
                            servo_prior=local_servo_prior,
                        )
                        loss = 0.5 * (loss + diagonal_gaussian_nll(local_mean, local_log_std, local_target))
                    if state_servo_prior == "state_goal_delta_guarded_single_tick_v7":
                        native_action_loss, guard_loss, saturation_loss = _guarded_s1_native_losses(
                            goal_mean, actions, states, goal_frame_index, guarded_native_config
                        )
                        loss = (
                            loss
                            + native_action_loss_weight * native_action_loss
                            + guard_loss_weight * guard_loss
                            + saturation_loss_weight * saturation_loss
                        )
                    dynamics_loss = torch.zeros((), device=device)
                    if train_predictor:
                        with torch.no_grad():
                            target_tokens = _encode_clip_tokens(
                                target_encoder,
                                clips,
                                batch_size=clips.size(0),
                                max_num_frames=max_num_frames,
                                normalize_reps=normalize_reps,
                            )
                        dynamics_loss = _teacher_forced_dynamics_loss(
                            context_tokens, target_tokens, predictor, actions, states, normalize_reps, dynamics_loss_exp
                        )
                        loss = loss + dynamics_loss_weight * dynamics_loss
                    loss = loss + goal_consistency_loss_weight * goal_consistency_loss

                if mixed_precision:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()
                if train_context_encoder:
                    _ema_update(encoder.module if isinstance(encoder, DistributedDataParallel) else encoder, target_encoder, ema_for_step(epoch * ipe + itr + 1))
                optimizer.zero_grad(set_to_none=True)
                return float(loss.detach()), float(dynamics_loss.detach()), float(goal_consistency_loss.detach()), step_lr

            (loss, dynamics_loss, goal_consistency_loss, step_lr), gpu_etime_ms = gpu_timer(train_step)
            iter_elapsed_time_ms = (time.time() - itr_start_time) * 1000.0
            loss_meter.update(loss)
            iter_time_meter.update(iter_elapsed_time_ms)
            gpu_time_meter.update(gpu_etime_ms)
            data_elapsed_time_meter.update(data_elapsed_time_ms)

            if (itr % log_freq == 0) or (itr == ipe - 1) or np.isnan(loss) or np.isinf(loss):
                csv_logger.log(epoch + 1, itr, loss, -1.0, iter_elapsed_time_ms, gpu_etime_ms, data_elapsed_time_ms)
                logger.info(
                    "[%d, %5d] train_loss: %.6f dynamics: %.6f [lr: %.2e] [mem: %.2e] [iter: %.1f ms] [gpu: %.1f ms] [data: %.1f ms]"
                    % (
                        epoch + 1,
                        itr,
                        loss_meter.avg,
                        dynamics_loss,
                        step_lr,
                        torch.cuda.max_memory_allocated() / 1024.0**2,
                        iter_time_meter.avg,
                        gpu_time_meter.avg,
                        data_elapsed_time_meter.avg,
                    )
                )
            assert not np.isnan(loss), "loss is nan"

        val_loss, val_nll, val_dynamics, val_goal_consistency = _run_validation(
            loader=val_loader,
            device=device,
            encoder=encoder,
            target_encoder=target_encoder,
            predictor=predictor,
            head=head,
            batch_size=batch_size,
            max_num_frames=max_num_frames,
            normalize_reps=normalize_reps,
            goal_frame_index=goal_frame_index,
            chunk_size=chunk_size,
            use_local_intent=use_local_intent,
            dynamics_loss_weight=dynamics_loss_weight,
            dynamics_loss_exp=dynamics_loss_exp,
            action_parameterization=action_parameterization,
            state_servo_prior=state_servo_prior,
            condition_on_state_servo_prior=condition_on_state_servo_prior,
            goal_consistency_loss_weight=goal_consistency_loss_weight,
            guarded_native_steps=guarded_native_steps,
            guarded_native_config=guarded_native_config,
            native_action_loss_weight=native_action_loss_weight,
            guard_loss_weight=guard_loss_weight,
            saturation_loss_weight=saturation_loss_weight,
        )
        val_loss = _distributed_mean(val_loss, device=device, world_size=world_size)
        val_nll = _distributed_mean(val_nll, device=device, world_size=world_size)
        val_dynamics = _distributed_mean(val_dynamics, device=device, world_size=world_size)
        val_goal_consistency = _distributed_mean(val_goal_consistency, device=device, world_size=world_size)
        logger.info("avg. train_loss %.6f val_loss %.6f val_nll %.6f val_dynamics %.6f val_goal_consistency %.6f" % (loss_meter.avg, val_loss, val_nll, val_dynamics, val_goal_consistency))
        csv_logger.log(epoch + 1, ipe, loss_meter.avg, val_loss, iter_time_meter.avg, gpu_time_meter.avg, data_elapsed_time_meter.avg)

        encoder_hash_after = _sha256_tensor_parameters(encoder)
        predictor_hash_after = _sha256_tensor_parameters(predictor)
        target_encoder_hash_after = _sha256_tensor_parameters(target_encoder)
        if not train_context_encoder and encoder_hash_after != encoder_hash_before:
            raise RuntimeError("Encoder parameters changed during frozen A-SF0 training")
        if train_context_encoder and encoder_hash_after == encoder_hash_before:
            raise RuntimeError("Context encoder did not change during joint intent-action training")
        if train_context_encoder and target_encoder_hash_after == target_encoder_hash_before:
            raise RuntimeError("EMA target encoder did not change during joint intent-action training")
        if not train_predictor and predictor_hash_after != predictor_hash_before:
            raise RuntimeError("Predictor parameters changed during frozen A-SF0 training")
        if train_predictor and predictor_hash_after == predictor_hash_before:
            raise RuntimeError("Predictor parameters did not change during joint intent-action dynamics training")

        if epoch % CHECKPOINT_FREQ == 0 or epoch == (num_epochs - 1):
            save_checkpoint(epoch + 1, latest_path, val_loss, val_nll, val_dynamics, val_goal_consistency)
            if save_every_freq > 0 and epoch % save_every_freq == 0:
                save_every_path = os.path.join(folder, f"e{epoch}.pt")
                save_checkpoint(epoch + 1, save_every_path, val_loss, val_nll, val_dynamics, val_goal_consistency)