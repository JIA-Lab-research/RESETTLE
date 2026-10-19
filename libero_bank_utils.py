"""Lightweight shared helpers for LIBERO success-bank builders."""

from __future__ import annotations

import re

import numpy as np
from scipy.spatial.transform import Rotation


def canonical_task(task: str, task_name: str, category: str) -> str:
    """Match the suffix-aware task grouping used by action-divergence analysis."""
    if category.strip().lower() == "language instructions" and task_name:
        raw = re.sub(r"^(?:[a-z]+_)*scene\d+_", "", task_name.lower())
        return re.sub(r"_language_.+$", "", raw).strip("_")
    raw = (task or task_name).strip().lower().replace("_", " ")
    suffix_patterns = (
        r"\s+view\s+\d+(?:\s+\d+){3,8}\s+initstate\s+\d+\s*$",
        r"\s+add\s+\d+\s*$",
        r"\s+moved\s+level\s*\d+(?:\s+sample\s*\d+)?\s*$",
        r"\s+level\s*\d+\s+sample\s*\d+\s*$",
        r"\s+noise\s+\S+(?:\s+\S+)*\s*$",
        r"\s+light\s+\S+(?:\s+\S+)*\s*$",
        r"\s+(?:tb|table)\s+\d+\s*$",
    )
    while True:
        stripped = raw
        for pattern in suffix_patterns:
            stripped = re.sub(pattern, "", stripped)
        if stripped == raw:
            break
        raw = stripped
    return re.sub(r"\s+", "_", raw).strip("_")


def convert_states(raw_states: np.ndarray) -> np.ndarray:
    positions = raw_states[:, :3]
    euler_xyz = Rotation.from_rotvec(raw_states[:, 3:6]).as_euler("xyz", degrees=False)
    finger_opening = np.mean(np.abs(raw_states[:, 6:8]), axis=1, keepdims=True)
    closedness = np.clip(1.0 - finger_opening / 0.04, 0.0, 1.0)
    return np.concatenate([positions, euler_xyz, closedness], axis=1).astype(np.float32)