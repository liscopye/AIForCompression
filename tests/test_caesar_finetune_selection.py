import argparse
import tempfile
import unittest
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

from scripts.finetune_caesar_era5 import (
    count_predicted_frames,
    configure_trainable_scope,
    diffusion_objective_loss,
    eval_diffusion,
    finetune_vae,
    resolve_channel_indices,
)


class _OneSampleDataset(Dataset):
    def __len__(self):
        return 1

    def __getitem__(self, index):
        del index
        return {"input": torch.ones(1, 1, 2, 2)}


class _TrainingMovesAwayFromValidationOptimum(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(0.9))

    def forward(self, x):
        offset = self.weight if self.training else self.weight - 1.0
        zero = self.weight * 0.0
        return {
            "output": x + offset,
            "frame_bit": zero.reshape(1),
            "bpp": zero.reshape(1),
        }


class _Logger:
    def __init__(self):
        self.records = []

    def log(self, metrics):
        self.records.append(metrics)


class _LatentIdentity(torch.nn.Module):
    def inference_qlatent(self, x):
        return x


class _StochasticDiffusion(torch.nn.Module):
    def forward(self, latent, interpo_rate):
        del interpo_rate
        return (latent + torch.randn_like(latent)).square().mean()


class _ObjectiveDiffusion(torch.nn.Module):
    num_timesteps = 4

    def q_sample(self, x_start, t, noise):
        del t
        return x_start + noise

    def denoise_fn(self, noisy, timesteps):
        del timesteps
        return torch.zeros_like(noisy)

    def predict_start_from_noise(self, noisy, timesteps, predicted_noise):
        del timesteps
        return noisy - predicted_noise


class _DStage1Modules(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = torch.nn.Linear(1, 1)
        self.dec = torch.nn.Linear(1, 1)
        self.hyper_dec = torch.nn.Linear(1, 1)


class CaesarFinetuneSelectionTest(unittest.TestCase):
    def test_channel_range_defaults_to_all_physical_channels(self):
        self.assertEqual(resolve_channel_indices(0, None, 268), tuple(range(268)))

    def test_channel_range_preserves_requested_physical_indices(self):
        self.assertEqual(resolve_channel_indices(37, 74, 268), tuple(range(37, 74)))

    def test_channel_range_rejects_empty_or_out_of_bounds_selection(self):
        with self.assertRaises(ValueError):
            resolve_channel_indices(37, 37, 268)
        with self.assertRaises(ValueError):
            resolve_channel_indices(259, 269, 268)

    def test_d_predicted_frame_count_tracks_interpolation_rate(self):
        self.assertEqual(count_predicted_frames(16, 3), 10)
        self.assertEqual(count_predicted_frames(16, 2), 8)
        self.assertEqual(count_predicted_frames(16, 1), 0)

    def test_d_stage1_decoder_scope_freezes_every_non_decoder_parameter(self):
        model = _DStage1Modules()
        args = argparse.Namespace(
            model_type="D", stage=1, trainable_scope="decoder"
        )

        trainable, total = configure_trainable_scope(model, args)

        self.assertEqual(trainable, sum(p.numel() for p in model.dec.parameters()))
        self.assertEqual(total, sum(p.numel() for p in model.parameters()))
        self.assertTrue(all(p.requires_grad for p in model.dec.parameters()))
        self.assertTrue(all(not p.requires_grad for p in model.enc.parameters()))
        self.assertTrue(all(not p.requires_grad for p in model.hyper_dec.parameters()))

    def test_initial_checkpoint_remains_best_when_training_degrades_validation(self):
        model = _TrainingMovesAwayFromValidationOptimum()
        loader = DataLoader(_OneSampleDataset(), batch_size=1)
        logger = _Logger()

        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "best.pt"
            args = argparse.Namespace(
                model_type="V",
                stage=1,
                d_stage1_all_frames=False,
                interpo_rate=3,
                rate_mode="bpp",
                lambda_rate=0.0,
                gradient_accumulation_steps=1,
                anchor_weight=0.0,
                lr=0.1,
                warmup_updates=0,
                max_grad_norm=1.0,
                iterations=1,
                log_interval=1,
                val_interval=1,
                save_interval=1,
                milestone_steps=[],
                output_ckpt=str(output),
            )

            finetune_vae(
                model,
                loader,
                loader,
                args,
                torch.device("cpu"),
                logger,
                "test",
            )

            self.assertAlmostEqual(model.weight.item(), 0.9)
            self.assertEqual(logger.records[0]["val/update_step"], 0)
            self.assertEqual(logger.records[0]["val/is_initial"], 1)
            self.assertGreater(logger.records[-1]["val/mse_change_from_initial_pct"], 0)

    def test_diffusion_validation_replays_the_same_rng_stream(self):
        loader = DataLoader(_OneSampleDataset(), batch_size=1)
        args = argparse.Namespace(
            seed=17,
            interpo_rate=3,
            diffusion_objective="noise",
            diffusion_x0_weight=0.1,
        )
        diffusion = _StochasticDiffusion()
        vae = _LatentIdentity()

        first = eval_diffusion(
            diffusion, vae, loader, args, torch.device("cpu"), False
        )
        second = eval_diffusion(
            diffusion, vae, loader, args, torch.device("cpu"), False
        )

        self.assertEqual(first, second)

    def test_hybrid_diffusion_objective_combines_noise_and_x0_losses(self):
        diffusion = _ObjectiveDiffusion()
        latent = torch.zeros(2, 1, 4, 2, 2)
        torch.manual_seed(9)

        total, noise_loss, x0_loss = diffusion_objective_loss(
            diffusion,
            latent,
            interpo_rate=2,
            objective="hybrid",
            x0_weight=0.25,
        )

        self.assertGreater(noise_loss.item(), 0)
        self.assertAlmostEqual(noise_loss.item(), x0_loss.item(), places=6)
        self.assertAlmostEqual(
            total.item(), noise_loss.item() + 0.25 * x0_loss.item(), places=6
        )


if __name__ == "__main__":
    unittest.main()
