import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from compression_pipeline.adapters.era5 import center_crop_vthw
from compression_pipeline.adapters.kodak import KodakAdapter
from compression_pipeline.caesar_runner import build_caesar_window, validate_regular_timestamps
from compression_pipeline.canonical import CanonicalSample
from compression_pipeline.cra5_runner import run_cra5_sample
from compression_pipeline.metrics import base_metrics
from compression_pipeline.runner import run_image_grouped_sample
from compression_pipeline.torch_codecs import CodecResult, ForwardLikelihoodCodec
from compression_pipeline.views import build_caesar_view, build_image_groups, reconstruct_from_groups


class CompressionPipelineTest(unittest.TestCase):
    def test_kodak_adapter_reads_rgb_images_as_canonical_samples(self):
        with tempfile.TemporaryDirectory() as tmp:
            image_path = Path(tmp) / "kodim01.png"
            pixels = np.zeros((2, 3, 3), dtype=np.uint8)
            pixels[..., 0] = 10
            pixels[..., 1] = 20
            pixels[..., 2] = 30
            Image.fromarray(pixels, mode="RGB").save(image_path)

            samples = list(KodakAdapter(tmp, dataset_id="kodak_test").iter_samples())

        self.assertEqual(len(samples), 1)
        self.assertEqual(samples[0].dataset_id, "kodak_test")
        self.assertEqual(samples[0].sample_id, "kodim01")
        self.assertEqual(samples[0].layout, "channel_height_width")
        self.assertEqual(samples[0].array.shape, (3, 2, 3))
        self.assertEqual(samples[0].metadata["source_format"], "png")

    def test_image_groups_pad_to_three_channels_and_reconstruct_original_channels(self):
        array = np.stack(
            [
                np.full((2, 2), 0.0, dtype=np.float32),
                np.full((2, 2), 10.0, dtype=np.float32),
                np.full((2, 2), 20.0, dtype=np.float32),
                np.full((2, 2), 30.0, dtype=np.float32),
            ],
            axis=0,
        )
        sample = CanonicalSample(
            dataset_id="era5",
            sample_id="t0",
            kind="scientific_field",
            array=array,
            layout="channel_height_width",
            metadata={"dtype": "float32"},
        )

        groups = build_image_groups(sample, group_size=3)
        reconstructed = reconstruct_from_groups(groups, [g.tensor for g in groups])

        self.assertEqual(len(groups), 2)
        self.assertEqual(groups[1].tensor.shape, (1, 3, 2, 2))
        self.assertEqual(groups[1].actual_channels, 1)
        np.testing.assert_allclose(reconstructed, array, atol=1e-6)

    def test_kodak_image_groups_use_uint8_scaling(self):
        array = np.full((3, 2, 2), 128, dtype=np.uint8)
        sample = CanonicalSample(
            dataset_id="kodak",
            sample_id="kodim01",
            kind="image",
            array=array,
            layout="channel_height_width",
            metadata={"dtype": "uint8"},
        )

        groups = build_image_groups(sample)
        reconstructed = reconstruct_from_groups(groups, [groups[0].tensor])

        self.assertEqual(groups[0].normalization["type"], "uint8_255")
        np.testing.assert_array_equal(reconstructed, array)

    def test_base_metrics_reports_total_time_and_mb_throughput(self):
        original = np.ones((4, 2, 2), dtype=np.float32)
        reconstructed = original.copy()

        metrics = base_metrics(original, reconstructed, bitstream_bytes=8, elapsed=(2.0, 4.0), group_count=2)

        self.assertEqual(metrics["encode_time_total"], 2.0)
        self.assertEqual(metrics["decode_time_total"], 4.0)
        self.assertEqual(metrics["encode_time_per_group_avg"], 1.0)
        self.assertEqual(metrics["decode_time_per_group_avg"], 2.0)
        self.assertEqual(metrics["encode_throughput"], 64.0 / 2.0)
        self.assertEqual(metrics["decode_throughput"], 64.0 / 4.0)
        self.assertEqual(metrics["encode_throughput_MBps"], 64.0 / 2.0 / 1e6)
        self.assertEqual(metrics["decode_throughput_MBps"], 64.0 / 4.0 / 1e6)
        self.assertEqual(metrics["encode_time_avg"], metrics["encode_time_total"])
        self.assertEqual(metrics["decode_time_avg"], metrics["decode_time_total"])

    def test_base_metrics_preserves_optional_extra_metrics(self):
        original = np.ones((3, 2, 2), dtype=np.float32)
        reconstructed = original.copy()

        metrics = base_metrics(
            original,
            reconstructed,
            bitstream_bytes=8,
            elapsed=(1.0, 1.0),
            extra_metrics={"lpips": 0.125, "memory_usage_MB": 42.0},
        )

        self.assertEqual(metrics["lpips"], 0.125)
        self.assertEqual(metrics["memory_usage_MB"], 42.0)

    def test_image_group_runner_reports_group_count_and_total_group_time(self):
        class FakeCodec:
            def __init__(self):
                self.calls = 0

            def roundtrip(self, tensor):
                self.calls += 1
                return CodecResult(
                    reconstruction=tensor,
                    bitstream_bytes=10,
                    encode_time=0.25,
                    decode_time=0.5,
                )

        sample = CanonicalSample(
            dataset_id="era5",
            sample_id="t0",
            kind="scientific_field",
            array=np.zeros((4, 2, 2), dtype=np.float32),
            layout="channel_height_width",
            metadata={"dtype": "float32"},
        )

        result = run_image_grouped_sample(sample, FakeCodec())

        self.assertEqual(result["groups"], 2)
        self.assertEqual(result["group_count"], 2)
        self.assertEqual(result["encode_time_total"], 0.5)
        self.assertEqual(result["decode_time_total"], 1.0)
        self.assertEqual(result["encode_time_per_group_avg"], 0.25)
        self.assertEqual(result["decode_time_per_group_avg"], 0.5)

    def test_image_group_runner_records_optional_lpips_and_memory_metrics(self):
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

        self.assertEqual(result["lpips"], 0.25)
        self.assertEqual(result["memory_usage_MB"], 10.0)
        self.assertEqual(result["memory_reserved_MB"], 12.0)

    def test_forward_likelihood_codec_uses_likelihoods_to_estimate_bitstream_size(self):
        torch = __import__("torch")

        class FakeForwardModel:
            def __call__(self, x):
                likelihood = torch.full_like(x, 0.5)
                return {
                    "x_hat": x.clone(),
                    "likelihoods": {"y": likelihood},
                }

        codec = ForwardLikelihoodCodec(FakeForwardModel(), device="cpu", divisor=2)
        tensor = np.full((1, 3, 2, 2), 0.5, dtype=np.float32)

        result = codec.roundtrip(tensor)

        self.assertEqual(result.reconstruction.shape, (1, 3, 2, 2))
        np.testing.assert_allclose(result.reconstruction, tensor, atol=1e-6)
        self.assertEqual(result.bitstream_bytes, 2)
        self.assertGreaterEqual(result.encode_time, 0.0)
        self.assertEqual(result.decode_time, 0.0)

    def test_caesar_view_uses_variable_sample_time_height_width_layout(self):
        sequence = np.zeros((4, 5, 2, 3), dtype=np.float32)

        view = build_caesar_view(sequence, sample_id="era5_seq", n_frame=3)

        self.assertEqual(view.tensor.shape, (4, 1, 3, 2, 3))
        self.assertEqual(view.layout, "variable_sample_time_height_width")

    def test_caesar_window_uses_contiguous_time_frames(self):
        sequence = np.arange(2 * 5 * 1 * 1, dtype=np.float32).reshape(2, 5, 1, 1)
        timestamps = [
            "2024-06-01T00:00:00",
            "2024-06-02T00:00:00",
            "2024-06-03T00:00:00",
            "2024-06-04T00:00:00",
            "2024-06-05T00:00:00",
        ]

        window = build_caesar_window(sequence, timestamps, n_frame=3, start_index=1)

        self.assertEqual(window.view.tensor.shape, (2, 1, 3, 1, 1))
        self.assertEqual(window.timestamps, timestamps[1:4])
        np.testing.assert_array_equal(window.view.tensor[:, 0, :, 0, 0], sequence[:, 1:4, 0, 0])

    def test_caesar_window_rejects_irregular_timestamps(self):
        timestamps = [
            "2024-06-01T00:00:00",
            "2024-06-02T00:00:00",
            "2024-06-04T00:00:00",
        ]

        with self.assertRaises(ValueError):
            validate_regular_timestamps(timestamps)

    def test_center_crop_vthw_preserves_variable_and_time_axes(self):
        sequence = np.arange(2 * 4 * 5 * 6, dtype=np.float32).reshape(2, 4, 5, 6)

        cropped = center_crop_vthw(sequence, (3, 4))

        self.assertEqual(cropped.shape, (2, 4, 3, 4))
        np.testing.assert_array_equal(cropped, sequence[:, :, 1:4, 1:5])

    def test_cra5_runner_uses_native_268_channel_shape(self):
        class FakeCRA5:
            def __init__(self):
                self.decompress_shape = None

            def compress(self, x):
                self.input_shape = tuple(x.shape)
                return {"strings": [[b"abc"], [b"de"]], "z_shape": (1, 2)}

            def decompress(self, strings, shape):
                self.decompress_shape = shape
                return {"x_hat": self.x_hat}

        model = FakeCRA5()
        model.x_hat = __import__("torch").zeros((1, 268, 2, 2), dtype=__import__("torch").float32)
        sample = CanonicalSample(
            dataset_id="era5",
            sample_id="t0",
            kind="scientific_field",
            array=np.ones((268, 2, 2), dtype=np.float32),
            layout="channel_height_width",
            metadata={"dtype": "float32"},
        )

        result = run_cra5_sample(sample, model, device="cpu")

        self.assertEqual(model.input_shape, (1, 268, 2, 2))
        self.assertEqual(model.decompress_shape, (1, 2))
        self.assertEqual(result["model_view"], "cra5_268")
        self.assertEqual(result["shape"], [268, 2, 2])

    def test_cra5_runner_rejects_non_268_channel_samples(self):
        sample = CanonicalSample(
            dataset_id="kodak",
            sample_id="kodim01",
            kind="image",
            array=np.zeros((3, 2, 2), dtype=np.uint8),
            layout="channel_height_width",
            metadata={"dtype": "uint8"},
        )

        with self.assertRaises(ValueError):
            run_cra5_sample(sample, object(), device="cpu")


if __name__ == "__main__":
    unittest.main()
