from __future__ import annotations

import time
from typing import Protocol
from collections.abc import Callable

from compression_pipeline.canonical import CanonicalSample
from compression_pipeline.metrics import base_metrics
from compression_pipeline.torch_codecs import CodecResult
from compression_pipeline.views import build_image_groups, reconstruct_from_groups


class ImageGroupCodec(Protocol):
    def roundtrip(self, tensor_bchw): ...


def run_image_grouped_sample(
    sample: CanonicalSample,
    codec: ImageGroupCodec,
    lpips_fn: Callable[[object, object], float | None] | None = None,
    memory_fn: Callable[[], dict[str, float | None]] | None = None,
    valid_mask=None,
    metric_extras: dict | None = None,
) -> dict:
    wall_start = time.perf_counter()
    groups = build_image_groups(sample)
    results: list[CodecResult] = [codec.roundtrip(group.tensor) for group in groups]
    reconstruction = reconstruct_from_groups(groups, [result.reconstruction for result in results])
    wall_time = time.perf_counter() - wall_start
    bitstream_bytes = sum(result.bitstream_bytes for result in results)
    side_info_bytes = sum(_normalization_side_info_bytes(group.normalization, group.actual_channels) for group in groups)
    encode_time = sum(result.encode_time for result in results)
    decode_time = sum(result.decode_time for result in results)
    extra_metrics: dict[str, float | None] = {}
    if lpips_fn is not None:
        extra_metrics["lpips"] = lpips_fn(sample.array, reconstruction)
    if memory_fn is not None:
        extra_metrics.update(memory_fn())
    if metric_extras:
        extra_metrics.update(metric_extras)
    metrics = base_metrics(
        sample.array,
        reconstruction,
        bitstream_bytes,
        (encode_time, decode_time),
        group_count=len(groups),
        side_info_bytes=side_info_bytes,
        valid_mask=valid_mask,
        extra_metrics=extra_metrics,
    )
    metrics["sample_wall_time_total"] = wall_time
    metrics["sample_wall_throughput_MBps"] = metrics["original_bytes"] / wall_time / 1e6 if wall_time > 0 else None
    metrics.update({
        "dataset_id": sample.dataset_id,
        "sample_id": sample.sample_id,
        "sample_kind": sample.kind,
        "groups": len(groups),
        "shape": list(sample.array.shape),
    })
    return metrics


def _normalization_side_info_bytes(normalization: dict, actual_channels: int) -> int:
    """Bytes needed to store per-sample normalization parameters for exact denormalization."""
    norm_type = normalization.get("type")
    if norm_type == "per_channel_minmax":
        return int(actual_channels * 2 * 4)
    if norm_type == "per_channel_zscore":
        return int(actual_channels * 4 * 4)
    return 0
