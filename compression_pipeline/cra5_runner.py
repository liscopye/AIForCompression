from __future__ import annotations

import time
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from compression_pipeline.canonical import CanonicalSample
from compression_pipeline.metrics import base_metrics
from compression_pipeline.torch_codecs import nested_bytes, synchronize_if_needed

CRA5_IMG_SIZE = (721, 1440)
CRA5_IN_CHANS = 268


def _adapt_to_cra5(sample: CanonicalSample, device: str):
    """
    Adapt any sample to CRA5 input format (268ch, 721x1440).

    Returns:
        x: [1, 268, 721, 1440] tensor on device
        orig_c: original channel count
        orig_hw: original (H, W) tuple
        vmin, vmax: original min/max for denormalization
    """
    arr = sample.array.astype(np.float32, copy=False)
    c, h, w = arr.shape

    vmin = arr.min()
    vmax = arr.max()

    # Normalize to [0, 1]
    if vmax - vmin > 1e-8:
        arr_norm = (arr - vmin) / (vmax - vmin)
    else:
        arr_norm = arr

    # Resize to 721x1440
    t = torch.from_numpy(arr_norm).unsqueeze(0).to(device)  # [1, C, H, W]
    t = F.interpolate(t, size=CRA5_IMG_SIZE, mode="bilinear", align_corners=False)

    # Replicate channels to 268
    reps = CRA5_IN_CHANS // c
    rem = CRA5_IN_CHANS % c
    out = t.repeat(1, reps, 1, 1)
    if rem > 0:
        out = torch.cat([out, t[:, :rem, :, :]], dim=1)

    return out, c, (h, w), vmin, vmax


def _reconstruct_from_cra5(x_hat: torch.Tensor, orig_c: int, orig_hw: tuple, vmin: float, vmax: float) -> np.ndarray:
    """
    Map CRA5 output [1, 268, 721, 1440] back to original data space.

    Returns: numpy array [orig_c, orig_h, orig_w] in original value range
    """
    # Crop to original channels
    x_cropped = x_hat[:, :orig_c, :, :]

    # Resize to original spatial size
    x_resized = F.interpolate(x_cropped, size=orig_hw, mode="bilinear", align_corners=False)

    # Clamp to valid [0, 1] range
    x_clamped = torch.clamp(x_resized, 0.0, 1.0)

    # Denormalize
    x_denorm = x_clamped * (vmax - vmin) + vmin

    return x_denorm.detach().cpu().numpy()[0].astype(np.float32, copy=False)


@torch.no_grad()
def run_cra5_sample(
    sample: CanonicalSample,
    model: Any,
    device: str,
    allow_adapted: bool = False,
    valid_mask: np.ndarray | None = None,
) -> dict:
    sample.require_layout("channel_height_width")
    arr = sample.array.astype(np.float32, copy=False)
    c, h, w = arr.shape

    # ERA5 native path: directly passes, no resize/replicate needed
    is_era5 = (sample.kind == "scientific_field" and c == CRA5_IN_CHANS)

    if is_era5:
        x = torch.from_numpy(arr[None]).to(device)
        t0 = time.time()
        compressed = model.compress(x)
        synchronize_if_needed(x)
        t1 = time.time()
        decompressed = model.decompress(compressed["strings"], compressed["z_shape"])
        synchronize_if_needed(x)
        t2 = time.time()
        reconstruction = decompressed["x_hat"].detach().cpu().numpy()[0].astype(np.float32, copy=False)
        bitstream_bytes = nested_bytes(compressed["strings"])
        metrics = base_metrics(arr, reconstruction, bitstream_bytes, (t1 - t0, t2 - t1), group_count=1, valid_mask=valid_mask)
    elif allow_adapted:
        # Non-ERA5 path: adapt to CRA5 input, then map back
        cra5_input, orig_c, orig_hw, vmin, vmax = _adapt_to_cra5(sample, device)

        t0 = time.time()
        compressed = model.compress(cra5_input)
        synchronize_if_needed(cra5_input)
        t1 = time.time()
        decompressed = model.decompress(compressed["strings"], compressed["z_shape"])
        synchronize_if_needed(cra5_input)
        t2 = time.time()

        reconstruction = _reconstruct_from_cra5(decompressed["x_hat"], orig_c, orig_hw, vmin, vmax)
        bitstream_bytes = nested_bytes(compressed["strings"])
        metrics = base_metrics(arr, reconstruction, bitstream_bytes, (t1 - t0, t2 - t1), group_count=1, valid_mask=valid_mask)
    else:
        raise ValueError(
            f"CRA5 native runner requires an ERA5-style 268-channel scientific_field sample, "
            f"got kind={sample.kind!r}, channels={c}. Use allow_adapted=True only for explicit ablation runs."
        )

    metrics.update({
        "dataset_id": sample.dataset_id,
        "sample_id": sample.sample_id,
        "sample_kind": sample.kind,
        "model_view": "cra5_268",
        "shape": list(sample.array.shape),
    })
    return metrics
