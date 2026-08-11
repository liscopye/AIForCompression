from types import SimpleNamespace

import torch

from compression_pipeline.caesar_runner import _count_caesar_params


def test_keyframe_only_caesar_d_counts_loaded_stage1_parameters():
    keyframe_model = torch.nn.Linear(3, 2)
    compressor = SimpleNamespace(
        keyframe_model=keyframe_model,
        diffusion_model=None,
    )

    assert _count_caesar_params(compressor, "caesar_d") == sum(
        parameter.numel() for parameter in keyframe_model.parameters()
    )
