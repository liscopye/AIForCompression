from pathlib import Path

import numpy as np

from compression_pipeline.canonical import CanonicalSample
from compression_pipeline import nvjpeg_codecs


def test_fixed_unit_nvjpeg_decodes_back_to_unit_range(monkeypatch, tmp_path):
    original = np.full((3, 8, 8), 0.5, dtype=np.float32)
    decoded = np.full((3, 8, 8), 128, dtype=np.uint8)
    monkeypatch.setattr(nvjpeg_codecs, "build_nvjpeg_binary", lambda binary=None: Path("unused"))
    monkeypatch.setattr(
        nvjpeg_codecs,
        "_run_nvjpeg_roundtrip",
        lambda binary, rgb_u8, quality, root: (
            decoded,
            {"jpeg_bytes": 10, "encode_us": 1, "decode_us": 1},
        ),
    )
    sample = CanonicalSample("test", "unit", "image", original, "channel_height_width", {})

    result = nvjpeg_codecs.run_nvjpeg_sample(sample, 95, tmp_path, fixed_unit_range=True)

    assert result["mse"] < 2e-5
    assert result["psnr"] > 45
    assert result["original_bytes"] == original.nbytes
