"""Sample two guarded K=1 residual chunks from one fixed recovery observation."""

import argparse
import base64
import io
import json
from pathlib import Path
from urllib.request import Request, urlopen

import numpy as np


def sample_two_chunks(response, seeds):
    if len(seeds) != 2 or seeds[0] == seeds[1]:
        raise ValueError("provide two distinct integer seeds")
    mean = np.asarray(response["jepa_residual_action_chunk"], dtype=np.float32)
    log_std = np.asarray(response["action_log_std_chunk"], dtype=np.float32)
    prior = np.asarray(response["state_servo_prior"], dtype=np.float32)
    if mean.shape != (1, 7) or log_std.shape != mean.shape or prior.shape != (7,):
        raise ValueError("guarded K=1 response requires residual/log_std [1,7] and prior [7]")
    chunks = [prior[None, :] + mean + np.exp(log_std) * np.random.default_rng(seed).standard_normal((1, 7))
              for seed in seeds]
    delta = chunks[0][0] - chunks[1][0]
    return chunks, {
        "xyz_l2_m": float(np.linalg.norm(delta[:3])),
        "rotation_l2_rad": float(np.linalg.norm(delta[3:6])),
        "gripper_abs": float(abs(delta[6])),
    }


def encode_array(array):
    buffer = io.BytesIO()
    np.save(buffer, array, allow_pickle=False)
    return {"format": "npy_base64", "data": base64.b64encode(buffer.getvalue()).decode("ascii")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:18082")
    parser.add_argument("--image-npy", required=True, type=Path)
    parser.add_argument("--state-npy", required=True, type=Path)
    parser.add_argument("--task", required=True)
    parser.add_argument("--goal-row", type=int)
    parser.add_argument("--seeds", type=int, nargs=2, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    image = np.load(args.image_npy, allow_pickle=False)
    state = np.load(args.state_npy, allow_pickle=False)
    if image.ndim != 3 or image.shape[-1] != 3 or image.dtype != np.uint8 or state.shape != (7,):
        parser.error("image must be RGB uint8 [H,W,3] and state must be [7]")
    payload = {"task": args.task, "image": encode_array(image),
               "state": encode_array(state.astype(np.float32))}
    if args.goal_row is not None:
        payload["goal_reference"] = {"row": args.goal_row}
    request = Request(args.url.rstrip("/") + "/plan-recovery-state-servo-residual",
                      data=json.dumps(payload).encode("utf-8"),
                      headers={"Content-Type": "application/json"})
    with urlopen(request, timeout=120) as connection:
        response = json.load(connection)
    if response.get("status") != "ok":
        raise RuntimeError(f"recovery request failed: {response}")
    chunks, disagreement = sample_two_chunks(response, args.seeds)
    result = {
        "seeds": args.seeds,
        "canonical_task": response["canonical_task"],
        "goal_reference": response["goal_reference"],
        "bank_checkpoint_sha256": response["bank_checkpoint_sha256"],
        "mean_action_chunk": response["jepa_diff_action_chunk"],
        "sampled_proposed_action_chunks": [chunk.tolist() for chunk in chunks],
        "disagreement": disagreement,
    }
    text = json.dumps(result, indent=2) + "\n"
    if args.output is not None:
        args.output.write_text(text)
    print(text, end="")


if __name__ == "__main__":
    main()