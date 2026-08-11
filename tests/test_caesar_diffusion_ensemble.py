import sys
from pathlib import Path

import torch


CAESAR_ROOT = Path(__file__).resolve().parents[1] / "models" / "CAESAR"
if str(CAESAR_ROOT) not in sys.path:
    sys.path.insert(0, str(CAESAR_ROOT))

from CAESAR.compressor import CAESAR


class _KeyframeModel:
    def decompress(self, *args, device):
        del args
        return torch.full((1, 64, 15, 16, 16), 2.0, device=device)

    def decode(self, latent):
        return latent[:, :1]


class _DiffusionModel:
    def __init__(self):
        self.calls = 0

    def sample(self, condition, interpo_rate, batch_size):
        del interpo_rate, batch_size
        self.calls += 1
        value = 1.0 if self.calls == 1 else 3.0
        return condition.new_full((1, 64, 1, 16, 16), value)


def test_caesar_d_ensemble_averages_decoded_samples_in_pixel_space():
    compressor = CAESAR.__new__(CAESAR)
    compressor.device = "cpu"
    compressor.n_frame = 16
    compressor.interpo_rate = 1
    compressor.cond_idx = torch.arange(15)
    compressor.pred_idx = torch.tensor([False] * 15 + [True])
    compressor.diffusion_ensemble_size = 2
    compressor.keyframe_model = _KeyframeModel()
    compressor.diffusion_model = _DiffusionModel()

    compressed = {
        "compressed": (),
        "scale": torch.tensor(1.0),
        "offset": torch.tensor(0.0),
        "index": (
            torch.tensor([0]),
            torch.tensor([0]),
            torch.tensor([0]),
            torch.tensor([16]),
        ),
    }
    reconstructed = compressor.decompress_caesar_d(
        [compressed],
        (1, 1, 16, 16, 16),
        filtered_blocks=[],
    )

    assert compressor.diffusion_model.calls == 2
    assert torch.allclose(reconstructed[0, 0, 0], torch.full((16, 16), 2.0))
    assert torch.allclose(reconstructed[0, 0, 15], torch.full((16, 16), 3.0))
