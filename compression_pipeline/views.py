from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from compression_pipeline.canonical import CanonicalSample


@dataclass(frozen=True)
class ImageGroupView:
    sample_id: str
    group_index: int
    tensor: np.ndarray
    actual_channels: int
    normalization: dict[str, Any]
    source_channel_start: int
    source_shape: tuple[int, int, int]


@dataclass(frozen=True)
class CaesarView:
    sample_id: str
    tensor: np.ndarray
    layout: str
    metadata: dict[str, Any]


def build_image_groups(sample: CanonicalSample, group_size: int = 3) -> list[ImageGroupView]:
    sample.require_layout("channel_height_width")
    if group_size <= 0:
        raise ValueError(f"group_size must be positive, got {group_size}")

    array = sample.array
    if array.ndim != 3:
        raise ValueError(f"{sample.sample_id} must be CHW, got shape {array.shape}")

    groups: list[ImageGroupView] = []
    channels, height, width = array.shape
    for start in range(0, channels, group_size):
        chunk = array[start:start + group_size]
        actual_channels = chunk.shape[0]
        if actual_channels < group_size:
            pad = np.repeat(chunk[-1:], group_size - actual_channels, axis=0)
            chunk = np.concatenate([chunk, pad], axis=0)

        tensor, normalization = _normalize_for_image_model(
            chunk, sample.kind, sample.metadata, source_channel_start=start
        )
        groups.append(
            ImageGroupView(
                sample_id=sample.sample_id,
                group_index=len(groups),
                tensor=tensor[None].astype(np.float32, copy=False),
                actual_channels=actual_channels,
                normalization=normalization,
                source_channel_start=start,
                source_shape=(channels, height, width),
            )
        )
    return groups


def reconstruct_from_groups(groups: list[ImageGroupView], tensors: list[np.ndarray]) -> np.ndarray:
    if len(groups) != len(tensors):
        raise ValueError(f"Expected {len(groups)} tensors, got {len(tensors)}")
    restored: list[np.ndarray] = []
    for group, tensor in zip(groups, tensors):
        chw = _as_chw(tensor)[: group.actual_channels]
        restored.append(_denormalize_image_group(chw, group.normalization))
    return np.concatenate(restored, axis=0)


def build_caesar_view(sequence_vthw: np.ndarray, sample_id: str, n_frame: int) -> CaesarView:
    if sequence_vthw.ndim != 4:
        raise ValueError(f"CAESAR sequence must be [V,T,H,W], got {sequence_vthw.shape}")
    if n_frame <= 0:
        raise ValueError(f"n_frame must be positive, got {n_frame}")
    if sequence_vthw.shape[1] < n_frame:
        raise ValueError(f"CAESAR requires {n_frame} frames, got {sequence_vthw.shape[1]}")
    tensor = sequence_vthw[:, None, :n_frame].astype(np.float32, copy=False)
    return CaesarView(
        sample_id=sample_id,
        tensor=tensor,
        layout="variable_sample_time_height_width",
        metadata={
            "npz_key": "data",
            "n_frame": n_frame,
            "source_layout": "variable_time_height_width",
        },
    )


def _normalize_for_image_model(
    chunk: np.ndarray,
    sample_kind: str,
    metadata: dict[str, Any],
    source_channel_start: int = 0,
) -> tuple[np.ndarray, dict[str, Any]]:
    if sample_kind == "image" and chunk.dtype == np.uint8:
        return chunk.astype(np.float32) / 255.0, {"type": "uint8_255", "dtype": "uint8"}

    float_chunk = chunk.astype(np.float32, copy=False)
    n_ch = float_chunk.shape[0]

    if metadata.get("external_normalized") is True:
        return float_chunk, {
            "type": "dataset_fixed_identity",
            "dtype": metadata.get("dtype", str(chunk.dtype)),
            "normalization_id": metadata.get("normalization_id"),
        }

    # Z-score normalization if pre-computed mean/std available
    zscore_mean = metadata.get("zscore_mean")
    zscore_std = metadata.get("zscore_std")
    if zscore_mean is not None and zscore_std is not None:
        # Two-stage: z-score → minmax → [0,1]  (like CRA5/CompressAI)
        mean_arr = np.array(zscore_mean[source_channel_start:source_channel_start + n_ch], dtype=np.float32).reshape(-1, 1, 1)
        std_arr = np.maximum(np.array(zscore_std[source_channel_start:source_channel_start + n_ch], dtype=np.float32).reshape(-1, 1, 1), 1e-8)
        # Stage 1: z-score
        zscored = (float_chunk - mean_arr) / std_arr
        # Stage 2: per-channel minmax on z-scored data
        zmin = zscored.reshape(n_ch, -1).min(axis=1).reshape(-1, 1, 1).astype(np.float32)
        zmax = zscored.reshape(n_ch, -1).max(axis=1).reshape(-1, 1, 1).astype(np.float32)
        zscale = np.maximum(zmax - zmin, 1e-8).astype(np.float32)
        normalized = (zscored - zmin) / zscale
        return normalized, {
            "type": "per_channel_zscore",
            "dtype": metadata.get("dtype", str(chunk.dtype)),
            "z_min": zmin,
            "z_scale": zscale,
            "zscore_mean": mean_arr,
            "zscore_std": std_arr,
        }

    cmin = float_chunk.reshape(float_chunk.shape[0], -1).min(axis=1).reshape(-1, 1, 1)
    cmax = float_chunk.reshape(float_chunk.shape[0], -1).max(axis=1).reshape(-1, 1, 1)
    scale = np.maximum(cmax - cmin, 1e-8).astype(np.float32)
    return (float_chunk - cmin) / scale, {
        "type": "per_channel_minmax",
        "dtype": metadata.get("dtype", str(chunk.dtype)),
        "min": cmin.astype(np.float32),
        "scale": scale,
    }


def _denormalize_image_group(chw: np.ndarray, normalization: dict[str, Any]) -> np.ndarray:
    norm_type = normalization["type"]
    if norm_type == "uint8_255":
        return np.rint(np.clip(chw, 0.0, 1.0) * 255.0).astype(np.uint8)
    if norm_type == "dataset_fixed_identity":
        return chw.astype(np.float32, copy=False)
    if norm_type == "per_channel_minmax":
        return (chw.astype(np.float32) * normalization["scale"][: chw.shape[0]] + normalization["min"][: chw.shape[0]]).astype(np.float32)
    if norm_type == "per_channel_zscore":
        # Reverse minmax: [0,1] → z-score space
        z_val = chw.astype(np.float32) * normalization["z_scale"][: chw.shape[0]] + normalization["z_min"][: chw.shape[0]]
        # Reverse z-score: z-score → original
        return (z_val * normalization["zscore_std"][: chw.shape[0]] + normalization["zscore_mean"][: chw.shape[0]]).astype(np.float32)
    raise ValueError(f"Unsupported normalization type: {norm_type}")


def _as_chw(tensor: np.ndarray) -> np.ndarray:
    arr = np.asarray(tensor)
    if arr.ndim == 4:
        if arr.shape[0] != 1:
            raise ValueError(f"Expected batch size 1, got {arr.shape}")
        return arr[0]
    if arr.ndim == 3:
        return arr
    raise ValueError(f"Expected CHW or BCHW tensor, got {arr.shape}")
