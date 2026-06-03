from __future__ import annotations

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
) -> dict:
    groups = build_image_groups(sample)
    results: list[CodecResult] = [codec.roundtrip(group.tensor) for group in groups]
    reconstruction = reconstruct_from_groups(groups, [result.reconstruction for result in results])
    bitstream_bytes = sum(result.bitstream_bytes for result in results)
    encode_time = sum(result.encode_time for result in results)
    decode_time = sum(result.decode_time for result in results)
    extra_metrics: dict[str, float | None] = {}
    if lpips_fn is not None:
        extra_metrics["lpips"] = lpips_fn(sample.array, reconstruction)
    if memory_fn is not None:
        extra_metrics.update(memory_fn())
    metrics = base_metrics(
        sample.array,
        reconstruction,
        bitstream_bytes,
        (encode_time, decode_time),
        group_count=len(groups),
        extra_metrics=extra_metrics,
    )
    metrics.update({
        "dataset_id": sample.dataset_id,
        "sample_id": sample.sample_id,
        "sample_kind": sample.kind,
        "groups": len(groups),
        "shape": list(sample.array.shape),
    })
    return metrics
