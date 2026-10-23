"""Build a guarded recovery goal bank from LIBERO LeRobot expert episodes."""

import argparse
import hashlib
import json
import os
import zipfile
from collections import Counter
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np


def checkpoint_hash(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def encode_suite(args):
    import av
    import pandas as pd
    import torch
    import torch.nn.functional as F

    from app.vjepa_droid.transforms import make_transforms
    from src.recovery.libero_bank_utils import canonical_task, convert_states
    from src.models.vision_transformer import vit_giant_xformers

    device = torch.device(args.device)
    encoder = vit_giant_xformers(
        patch_size=16, num_frames=512, tubelet_size=2, img_size=256,
        uniform_power=True, use_sdpa=True, use_rope=True,
        use_activation_checkpointing=False,
    ).to(device)
    weights = torch.load(args.checkpoint, map_location="cpu", mmap=True, weights_only=False)
    encoder.load_state_dict({key.removeprefix("module.").removeprefix("backbone."): value
                             for key, value in weights["encoder"].items()}, strict=True)
    encoder.eval()
    transform = make_transforms(
        random_horizontal_flip=False, random_resize_aspect_ratio=(1.0, 1.0),
        random_resize_scale=(1.0, 1.0), crop_size=256,
    )
    features, states, metadata = [], [], []
    suite = args.suite
    root = (args.data_root / f"libero_{suite}_no_noops_1.0.0_lerobot").resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)
    if args.batch_size < 1:
        raise ValueError("batch-size must be positive")
    tasks = {int(row["task_index"]): str(row["task"])
             for row in map(json.loads, (root / "meta/tasks.jsonl").read_text().splitlines())}
    for parquet in sorted((root / "data").rglob("episode_*.parquet")):
        table = pd.read_parquet(parquet, columns=["observation.state", "frame_index", "task_index"])
        task_index = int(table["task_index"].iloc[0])
        if not (table["task_index"] == task_index).all():
            raise ValueError(f"mixed tasks in {parquet}")
        frame_indices = table["frame_index"].to_numpy(dtype=np.int64)
        video = root / "videos" / parquet.parent.name / "observation.images.image" / f"{parquet.stem}.mp4"
        requested = set(frame_indices.tolist())
        decoded = {}
        with av.open(str(video)) as container:
            for index, frame in enumerate(container.decode(video=0)):
                if index in requested:
                    decoded[index] = frame.to_ndarray(format="rgb24")
                if len(decoded) == len(requested):
                    break
        if set(decoded) != requested:
            raise ValueError(f"missing video frames in {video}")
        episode_states = convert_states(np.stack(table["observation.state"].to_numpy()).astype(np.float32))
        task = tasks[task_index]
        with torch.inference_mode(), torch.autocast(
            device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"
        ):
            for start in range(0, len(frame_indices), args.batch_size):
                batch_indices = frame_indices[start:start + args.batch_size]
                frames = np.stack([decoded[int(index)] for index in batch_indices])
                clip = transform(frames).permute(1, 0, 2, 3)
                clip = clip.unsqueeze(2).repeat(1, 1, 2, 1, 1).to(device)
                tokens = encoder(clip)
                vectors = F.normalize(F.layer_norm(tokens, (tokens.size(-1),)).float().mean(dim=1), dim=-1)
                features.append(vectors.cpu().numpy())
        states.append(episode_states)
        for index in frame_indices:
            metadata.append({
                "row": len(metadata), "canonical_task": canonical_task(task, task, ""),
                "source_name": f"libero_{suite}", "episode_index": int(parquet.stem.removeprefix("episode_")),
                "observation": {"type": "lerobot_video", "video_path": str(video), "frame_index": int(index)},
            })
    if not features:
        raise ValueError(f"no LIBERO LeRobot episodes under {args.data_root}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    feature_array = np.concatenate(features)
    state_array = np.concatenate(states)
    if len(feature_array) != len(state_array) or len(metadata) != len(feature_array):
        raise ValueError("inconsistent expert bank rows")
    np.savez_compressed(args.output_dir / "bank_shard.npz", features=feature_array, states=state_array)
    (args.output_dir / "bank_shard.jsonl").write_text("".join(json.dumps(row) + "\n" for row in metadata))
    (args.output_dir / "bank_shard_manifest.json").write_text(json.dumps({
        "source_name": f"libero_{suite}", "source_kind": "lerobot",
        "checkpoint_sha256": checkpoint_hash(args.checkpoint), "entries": len(metadata),
        "feature_dimension": feature_array.shape[1],
    }, indent=2) + "\n")


def merge(args):
    source_dirs = [args.output_dir / f"libero_{suite}" for suite in ("goal", "spatial", "object", "10")]
    manifests = []
    for source in source_dirs:
        manifest = json.loads((source / "bank_shard_manifest.json").read_text())
        if manifest["source_name"] != source.name or manifest["source_kind"] != "lerobot":
            raise ValueError(f"unexpected bank source: {source}")
        manifests.append(manifest)
    hashes = {manifest["checkpoint_sha256"] for manifest in manifests}
    if len(hashes) != 1 or (args.checkpoint is not None and checkpoint_hash(args.checkpoint) not in hashes):
        raise ValueError("all bank shards must match the requested checkpoint")
    with np.load(source_dirs[0] / "bank_shard.npz", allow_pickle=False) as first:
        specs = {key: (first[key].shape[1:], first[key].dtype) for key in ("features", "states")}
    total = sum(manifest["entries"] for manifest in manifests)
    with NamedTemporaryFile("wb", dir=args.output_dir, prefix=".success_id_bank.", suffix=".npz", delete=False) as handle:
        array_path = Path(handle.name)
    with zipfile.ZipFile(array_path, "w", compression=zipfile.ZIP_DEFLATED, allowZip64=True) as archive:
        for key, (tail_shape, dtype) in specs.items():
            with archive.open(f"{key}.npy", "w", force_zip64=True) as output:
                np.lib.format.write_array_header_2_0(output, {
                    "descr": np.lib.format.dtype_to_descr(dtype), "fortran_order": False,
                    "shape": (total, *tail_shape),
                })
                for source, manifest in zip(source_dirs, manifests):
                    with np.load(source / "bank_shard.npz", allow_pickle=False) as shard:
                        values = shard[key]
                        if values.shape != (manifest["entries"], *tail_shape) or values.dtype != dtype:
                            raise ValueError(f"incompatible {key} shard: {source}")
                        byte_view = memoryview(values).cast("B")
                        for start in range(0, len(byte_view), 8 * 1024 * 1024):
                            output.write(byte_view[start:start + 8 * 1024 * 1024])
    counts = Counter()
    row_number = 0
    with NamedTemporaryFile("w", dir=args.output_dir, prefix=".success_id_bank.", suffix=".jsonl", delete=False) as output:
        metadata_path = Path(output.name)
        for source, manifest in zip(source_dirs, manifests):
            with (source / "bank_shard.jsonl").open() as rows:
                source_rows = 0
                for source_rows, line in enumerate(rows, start=1):
                    row = json.loads(line)
                    if row["row"] != source_rows - 1:
                        raise ValueError(f"noncontiguous row in {source}")
                    row["row"] = row_number
                    row_number += 1
                    counts[row["canonical_task"]] += 1
                    output.write(json.dumps(row) + "\n")
                if source_rows != manifest["entries"]:
                    raise ValueError(f"metadata count differs from arrays in {source}")
    if row_number != total or len(counts) != 40:
        raise ValueError(f"expected {total} rows across 40 LIBERO tasks; got {row_number} rows and {len(counts)} tasks")
    os.replace(array_path, args.output_dir / "success_id_bank.npz")
    os.replace(metadata_path, args.output_dir / "success_id_bank.jsonl")
    (args.output_dir / "success_id_bank_summary.json").write_text(json.dumps({
        "schema_version": 2, "entries": total, "source_shards": manifests,
        "tasks": sorted(counts), "task_entry_counts": dict(sorted(counts.items())),
    }, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--suite", choices=("object", "goal", "spatial", "10"))
    parser.add_argument("--merge", action="store_true")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()
    if args.merge:
        merge(args)
    else:
        if args.suite is None or args.checkpoint is None or args.data_root is None:
            parser.error("encoding requires --suite, --checkpoint and --data-root")
        encode_suite(args)


if __name__ == "__main__":
    main()