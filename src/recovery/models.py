"""Checkpoint-compatible visual encoding for the LIBERO success bank."""

import copy

import numpy as np
import torch
import torch.nn.functional as F

from app.vjepa_droid.utils import init_video_model


def load_models(checkpoint_path, device):
    encoder, predictor = init_video_model(
        device=device,
        patch_size=16,
        max_num_frames=512,
        tubelet_size=2,
        model_name="vit_giant_xformers",
        crop_size=256,
        pred_depth=24,
        pred_num_heads=16,
        pred_embed_dim=1024,
        action_embed_dim=7,
        pred_is_frame_causal=True,
        uniform_power=True,
        use_sdpa=True,
        use_rope=True,
        use_activation_checkpointing=False,
    )
    target_encoder = copy.deepcopy(encoder)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", mmap=True, weights_only=False)

    def strip_prefix(state_dict):
        clean = {}
        for key, value in state_dict.items():
            while key.startswith(("module.", "backbone.")):
                key = key.split(".", 1)[1]
            clean[key] = value
        return clean

    encoder.load_state_dict(strip_prefix(checkpoint["encoder"]), strict=True)
    predictor.load_state_dict(strip_prefix(checkpoint["predictor"]), strict=True)
    target_encoder.load_state_dict(strip_prefix(checkpoint["target_encoder"]), strict=True)
    return encoder.eval(), predictor.eval(), target_encoder.eval(), checkpoint.get("epoch"), checkpoint.get("loss")


def encode_frame(model, frame, transform, device):
    clip = transform(np.expand_dims(frame, axis=0)).permute(1, 0, 2, 3)
    clip = clip.unsqueeze(2).repeat(1, 1, 2, 1, 1).to(device, non_blocking=True)
    representation = model(clip)
    return F.layer_norm(representation, (representation.size(-1),))