import math

import torch
import torch.nn as nn


class PooledGaussianIntentActionHead(nn.Module):
    """Map pooled current/intent features to a diagonal-Gaussian action chunk."""

    def __init__(
        self,
        encoder_dim=1408,
        state_dim=7,
        action_dim=7,
        chunk_size=1,
        condition_dim=128,
        hidden_dim=1024,
        depth=3,
        dropout=0.0,
        min_log_std=-5.0,
        max_log_std=2.0,
        condition_on_state_servo_prior=False,
    ):
        super().__init__()
        if depth < 1:
            raise ValueError(f"depth must be positive, got {depth}")
        self.action_dim = action_dim
        self.chunk_size = chunk_size
        self.min_log_std = min_log_std
        self.max_log_std = max_log_std
        self.condition_on_state_servo_prior = condition_on_state_servo_prior

        self.visual_norm = nn.LayerNorm(encoder_dim)
        self.state_encoder = nn.Sequential(
            nn.LayerNorm(state_dim),
            nn.Linear(state_dim, condition_dim),
            nn.GELU(),
        )
        self.previous_action_encoder = nn.Sequential(
            nn.LayerNorm(action_dim + 1),
            nn.Linear(action_dim + 1, condition_dim),
            nn.GELU(),
        )
        if condition_on_state_servo_prior:
            self.state_servo_prior_encoder = nn.Sequential(
                nn.LayerNorm(action_dim),
                nn.Linear(action_dim, condition_dim),
                nn.GELU(),
            )

        input_dim = 3 * encoder_dim + (3 if condition_on_state_servo_prior else 2) * condition_dim
        layers = []
        for layer_index in range(depth):
            layer_input_dim = input_dim if layer_index == 0 else hidden_dim
            layers.extend(
                [
                    nn.Linear(layer_input_dim, hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                ]
            )
        self.backbone = nn.Sequential(*layers)
        self.output = nn.Linear(hidden_dim, 2 * chunk_size * action_dim)

    def forward(self, current, intent, state, previous_action, previous_action_valid, servo_prior=None):
        if previous_action_valid.ndim == previous_action.ndim - 1:
            previous_action_valid = previous_action_valid.unsqueeze(-1)
        previous_action_condition = torch.cat(
            [previous_action, previous_action_valid.to(previous_action.dtype)], dim=-1
        )
        current = self.visual_norm(current)
        intent = self.visual_norm(intent)
        features = [
            current,
            intent,
            current * intent,
            self.state_encoder(state),
            self.previous_action_encoder(previous_action_condition),
        ]
        if self.condition_on_state_servo_prior:
            if servo_prior is None:
                raise ValueError("servo_prior is required when condition_on_state_servo_prior=true")
            features.append(self.state_servo_prior_encoder(servo_prior))
        features = torch.cat(features, dim=-1)
        distribution = self.output(self.backbone(features))
        distribution = distribution.view(*distribution.shape[:-1], self.chunk_size, 2 * self.action_dim)
        mean, log_std = distribution.chunk(2, dim=-1)
        log_std = log_std.clamp(self.min_log_std, self.max_log_std)
        return mean, log_std


def diagonal_gaussian_nll(mean, log_std, target):
    if mean.shape != target.shape or log_std.shape != target.shape:
        raise ValueError(
            f"mean/log_std/target must have identical shapes, got {mean.shape}, "
            f"{log_std.shape}, and {target.shape}"
        )
    inverse_variance = torch.exp(-2.0 * log_std)
    nll = 0.5 * (target - mean).square() * inverse_variance + log_std + 0.5 * math.log(2.0 * math.pi)
    return nll.mean()