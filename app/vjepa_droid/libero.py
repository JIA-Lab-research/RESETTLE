import json
from logging import getLogger
from math import ceil
from pathlib import Path

import av
import numpy as np
import pandas as pd
import torch
from scipy.spatial.transform import Rotation


logger = getLogger()


def init_data(
    data_path,
    batch_size,
    frames_per_clip=8,
    fps=4,
    crop_size=256,
    rank=0,
    world_size=1,
    camera_views=("observation.images.image",),
    stereo_view=False,
    drop_last=True,
    num_workers=10,
    pin_mem=True,
    persistent_workers=True,
    collator=None,
    transform=None,
    camera_frame=False,
    tubelet_size=1,
):
    del crop_size, stereo_view, camera_frame, tubelet_size
    dataset = LIBEROLeRobotVideoDataset(
        data_roots=data_path,
        camera_views=camera_views,
        frames_per_clip=frames_per_clip,
        fps=fps,
        transform=transform,
    )
    sampler = torch.utils.data.distributed.DistributedSampler(
        dataset, num_replicas=world_size, rank=rank, shuffle=True
    )
    loader = torch.utils.data.DataLoader(
        dataset,
        collate_fn=collator,
        sampler=sampler,
        batch_size=batch_size,
        drop_last=drop_last,
        pin_memory=pin_mem,
        num_workers=num_workers,
        persistent_workers=(num_workers > 0) and persistent_workers,
    )
    logger.info("LIBERO LeRobot data loader created with %d episodes", len(dataset))
    return loader, sampler


class LIBEROLeRobotVideoDataset(torch.utils.data.Dataset):
    """Sample V-JEPA2-AC clips from LeRobot-formatted LIBERO episodes."""

    def __init__(
        self,
        data_roots,
        camera_views=("observation.images.image",),
        frames_per_clip=8,
        fps=4,
        transform=None,
        gripper_open_qpos=0.04,
    ):
        if isinstance(data_roots, (str, Path)):
            data_roots = [data_roots]
        self.camera_views = tuple(camera_views)
        self.frames_per_clip = frames_per_clip
        self.fps = fps
        self.transform = transform
        self.gripper_open_qpos = gripper_open_qpos
        self.samples = []

        for root_value in data_roots:
            root = Path(root_value).expanduser().resolve()
            info_path = root / "meta" / "info.json"
            if not info_path.is_file():
                raise FileNotFoundError(f"Missing LeRobot metadata: {info_path}")
            with info_path.open() as info_file:
                info = json.load(info_file)
            source_fps = float(info["fps"])
            for parquet_path in sorted((root / "data").glob("chunk-*/episode_*.parquet")):
                self.samples.append((root, parquet_path, source_fps))

        if not self.samples:
            raise RuntimeError(f"No LIBERO parquet episodes found under {data_roots}")
        if self.fps <= 0:
            raise ValueError(f"fps must be positive, got {self.fps}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        last_error = None
        for _ in range(10):
            try:
                return self._load_episode_clip(index)
            except Exception as error:
                last_error = error
                logger.warning("Failed to load LIBERO sample %d: %s", index, error)
                index = int(torch.randint(0, len(self), (1,)).item())
        raise RuntimeError("Could not load a valid LIBERO clip after 10 attempts") from last_error

    def _load_episode_clip(self, index):
        root, parquet_path, source_fps = self.samples[index]
        frame_table = pd.read_parquet(
            parquet_path,
            columns=["observation.state", "timestamp", "frame_index"],
        )
        raw_states = np.stack(frame_table["observation.state"].to_numpy()).astype(np.float32)
        if raw_states.ndim != 2 or raw_states.shape[1] != 8:
            raise ValueError(f"Expected observation.state [T, 8], got {raw_states.shape} in {parquet_path}")

        camera_view = self.camera_views[int(torch.randint(0, len(self.camera_views), (1,)).item())]
        chunk_name = parquet_path.parent.name
        video_path = root / "videos" / chunk_name / camera_view / f"{parquet_path.stem}.mp4"
        if not video_path.is_file():
            raise FileNotFoundError(f"Missing video: {video_path}")

        stride = ceil(source_fps / self.fps)
        span = (self.frames_per_clip - 1) * stride + 1
        if len(raw_states) < span:
            raise ValueError(f"Episode too short for {self.frames_per_clip} frames at {self.fps} FPS")
        start = int(torch.randint(0, len(raw_states) - span + 1, (1,)).item())
        indices = start + np.arange(self.frames_per_clip, dtype=np.int64) * stride

        states = self._convert_states(raw_states[indices])
        actions = self.poses_to_diffs(states)
        extrinsics = np.zeros_like(states, dtype=np.float32)
        frames, video_fps, video_length = self._read_frames(video_path, indices)
        if not np.isclose(video_fps, source_fps, atol=0.1):
            raise ValueError(f"Metadata/video FPS mismatch for {video_path}: {source_fps} vs {video_fps}")
        if video_length and video_length != len(raw_states):
            raise ValueError(
                f"Parquet/video length mismatch for {parquet_path.stem}: {len(raw_states)} vs {video_length}"
            )
        if self.transform is not None:
            frames = self.transform(frames)
        return frames, actions, states, extrinsics, indices

    @staticmethod
    def _read_frames(video_path, indices):
        requested = [int(frame_index) for frame_index in indices]
        selected = set(requested)
        decoded = {}
        with av.open(str(video_path)) as container:
            stream = container.streams.video[0]
            video_fps = float(stream.average_rate)
            video_length = int(stream.frames)
            for frame_index, frame in enumerate(container.decode(stream)):
                if frame_index in selected:
                    decoded[frame_index] = frame.to_ndarray(format="rgb24")
                if frame_index >= max(selected):
                    break
        missing = selected.difference(decoded)
        if missing:
            raise RuntimeError(f"Failed to decode frame indices {sorted(missing)} from {video_path}")
        frames = [decoded[frame_index] for frame_index in requested]
        return np.stack(frames), video_fps, video_length

    def _convert_states(self, raw_states):
        positions = raw_states[:, :3]
        rotation_matrices = Rotation.from_rotvec(raw_states[:, 3:6]).as_matrix()
        euler_xyz = Rotation.from_matrix(rotation_matrices).as_euler("xyz", degrees=False)
        finger_opening = np.mean(np.abs(raw_states[:, 6:8]), axis=1, keepdims=True)
        closedness = np.clip(1.0 - finger_opening / self.gripper_open_qpos, 0.0, 1.0)
        return np.concatenate([positions, euler_xyz, closedness], axis=1).astype(np.float32)

    @staticmethod
    def poses_to_diffs(poses):
        xyz_diff = poses[1:, :3] - poses[:-1, :3]
        rotations = Rotation.from_euler("xyz", poses[:, 3:6], degrees=False).as_matrix()
        relative_rotations = rotations[1:] @ np.swapaxes(rotations[:-1], 1, 2)
        angle_diff = Rotation.from_matrix(relative_rotations).as_euler("xyz", degrees=False)
        closedness_delta = poses[1:, -1:] - poses[:-1, -1:]
        return np.concatenate([xyz_diff, angle_diff, closedness_delta], axis=1).astype(np.float32)