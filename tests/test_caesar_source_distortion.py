import argparse

import pytest
import torch

from scripts.finetune_caesar_era5 import vae_distortion_metrics


def _args(model_type="V"):
    return argparse.Namespace(
        distortion_domain="source",
        model_type=model_type,
        stage=1,
        d_stage1_all_frames=False,
        interpo_rate=3,
    )


def test_source_distortion_restores_patch_scale():
    full_x = torch.zeros(2, 1, 8, 2, 2)
    output = torch.ones_like(full_x)
    batch = {
        "input": full_x,
        "scale": torch.tensor([[[[2.0]]], [[[3.0]]]]),
    }

    distortion, normalized_mse, source_mse = vae_distortion_metrics(
        {"output": output},
        batch,
        full_x,
        full_x,
        _args(),
        torch.device("cpu"),
        False,
    )

    assert normalized_mse.item() == pytest.approx(1.0)
    assert source_mse.item() == pytest.approx((2.0**2 + 3.0**2) / 2)
    assert distortion.item() == pytest.approx(source_mse.item())


def test_source_distortion_selects_d_keyframe_scales():
    full_x = torch.zeros(1, 1, 8, 1, 1)
    model_x = full_x[:, :, [0, 3, 6]]
    frame_scales = torch.arange(1, 9, dtype=torch.float32).view(1, 8, 1, 1)

    distortion, _, source_mse = vae_distortion_metrics(
        {"output": torch.ones_like(model_x)},
        {"input": full_x, "scale": frame_scales},
        full_x,
        model_x,
        _args("D"),
        torch.device("cpu"),
        False,
    )

    expected = (1.0**2 + 4.0**2 + 7.0**2) / 3
    assert source_mse.item() == pytest.approx(expected)
    assert distortion.item() == pytest.approx(expected)


def test_normalized_distortion_remains_backward_compatible_without_scale():
    full_x = torch.zeros(1, 1, 8, 2, 2)
    args = _args()
    args.distortion_domain = "normalized"

    distortion, normalized_mse, source_mse = vae_distortion_metrics(
        {"output": torch.full_like(full_x, 2.0)},
        {"input": full_x},
        full_x,
        full_x,
        args,
        torch.device("cpu"),
        False,
    )

    assert distortion.item() == pytest.approx(4.0)
    assert normalized_mse.item() == pytest.approx(4.0)
    assert source_mse.item() == pytest.approx(4.0)


def test_normalized_distortion_logs_restored_source_mse_when_scale_is_available():
    full_x = torch.zeros(2, 1, 8, 2, 2)
    args = _args()
    args.distortion_domain = "normalized"
    batch = {
        "input": full_x,
        "scale": torch.tensor([[[[2.0]]], [[[3.0]]]]),
    }

    distortion, normalized_mse, source_mse = vae_distortion_metrics(
        {"output": torch.ones_like(full_x)},
        batch,
        full_x,
        full_x,
        args,
        torch.device("cpu"),
        False,
    )

    assert distortion.item() == pytest.approx(1.0)
    assert normalized_mse.item() == pytest.approx(1.0)
    assert source_mse.item() == pytest.approx((2.0**2 + 3.0**2) / 2)
