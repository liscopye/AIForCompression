from __future__ import annotations

import numpy as np

from compression_pipeline.canonical import CanonicalSample
from compression_pipeline.metrics import base_metrics
from compression_pipeline.runner import run_image_grouped_sample
from compression_pipeline.torch_codecs import CodecResult


def test_base_metrics_preserves_optional_extra_metrics():
    original = np.ones((3, 2, 2), dtype=np.float32)
    reconstructed = original.copy()

    metrics = base_metrics(
        original,
        reconstructed,
        bitstream_bytes=8,
        elapsed=(1.0, 1.0),
        extra_metrics={"lpips": 0.125, "memory_usage_MB": 42.0},
    )

    assert metrics["lpips"] == 0.125
    assert metrics["memory_usage_MB"] == 42.0


def test_image_group_runner_records_optional_lpips_and_memory_metrics():
    class FakeCodec:
        def roundtrip(self, tensor):
            return CodecResult(
                reconstruction=tensor,
                bitstream_bytes=10,
                encode_time=0.25,
                decode_time=0.5,
            )

    sample = CanonicalSample(
        dataset_id="turb_rot_npz",
        sample_id="t0",
        kind="turb_rot_npz",
        array=np.zeros((3, 2, 2), dtype=np.float32),
        layout="channel_height_width",
        metadata={"dtype": "float32"},
    )

    result = run_image_grouped_sample(
        sample,
        FakeCodec(),
        lpips_fn=lambda original, reconstructed: 0.25,
        memory_fn=lambda: {"memory_usage_MB": 10.0, "memory_reserved_MB": 12.0},
    )

    assert result["lpips"] == 0.25
    assert result["memory_usage_MB"] == 10.0
    assert result["memory_reserved_MB"] == 12.0
