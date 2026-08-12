from __future__ import annotations

import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from compression_pipeline.caesar_runner import _count_caesar_params, _resolve_caesar_checkpoint
from compression_pipeline.metrics import base_metrics, torch_memory_usage_mb


@dataclass
class PreparedCaesarCorpus:
    dataset: Any
    loader: DataLoader
    original: np.ndarray
    mask: np.ndarray | None
    npz_path: Path
    n_frame: int
    sections: int

    def close(self) -> None:
        self.npz_path.unlink(missing_ok=True)


def load_caesar_compressor(
    model_name: str,
    caesar_root: Path,
    checkpoint_root: Path,
    device: str = "cuda",
    interpo_rate: int = 3,
    diffusion_ensemble_size: int = 1,
):
    n_frame = 8 if model_name == "caesar_v" else 16
    if model_name == "caesar_d" and not 1 <= interpo_rate <= n_frame:
        raise ValueError(
            f"CAESAR-D interpo_rate must be in [1, {n_frame}], got {interpo_rate}"
        )
    if diffusion_ensemble_size <= 0:
        raise ValueError("CAESAR-D diffusion ensemble size must be positive")
    sys.path.insert(0, str(caesar_root))
    from CAESAR.compressor import CAESAR

    return CAESAR(
        model_path=str(_resolve_caesar_checkpoint(checkpoint_root, model_name)),
        use_diffusion=model_name == "caesar_d",
        device=device,
        n_frame=n_frame,
        interpo_rate=interpo_rate,
        diffusion_ensemble_size=(
            diffusion_ensemble_size if model_name == "caesar_d" else 1
        ),
    )


def prepare_caesar_corpus(
    normalized_vthw: np.ndarray,
    mask_vthw: np.ndarray | None,
    model_name: str,
    caesar_root: Path,
    output_dir: Path,
    sample_id: str,
    batch_size: int = 8,
    norm_type: str = "mean_range",
) -> PreparedCaesarCorpus:
    n_frame = 8 if model_name == "caesar_v" else 16
    variables, total_frames, height, width = normalized_vthw.shape
    if total_frames % n_frame:
        raise ValueError(f"{sample_id}: T={total_frames} is not divisible by CAESAR n_frame={n_frame}")
    sections = total_frames // n_frame
    tensor = np.ascontiguousarray(normalized_vthw.reshape(variables, sections, n_frame, height, width))
    reshaped_mask = (
        np.ascontiguousarray(mask_vthw.reshape(variables, sections, n_frame, height, width))
        if mask_vthw is not None else None
    )

    sys.path.insert(0, str(caesar_root))
    from dataset import ScientificDataset

    output_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(suffix=f"_{model_name}.npz", dir=output_dir, delete=False) as tmp:
        npz_path = Path(tmp.name)
    np.savez(npz_path, data=tensor)
    data_arg = {
        "data_path": str(npz_path),
        "name": f"objective-{sample_id}-{model_name}",
        "variable_idx": list(range(variables)),
        "section_range": [0, sections],
        "frame_range": [0, n_frame],
        "n_frame": n_frame,
        "train": False,
        "test_size": (256, 256),
        "inst_norm": True,
        "norm_type": norm_type,
    }
    dataset = ScientificDataset(data_arg)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    original = dataset.input_data().numpy()
    if original.shape != tensor.shape:
        npz_path.unlink(missing_ok=True)
        raise ValueError(f"CAESAR dataset changed corpus shape from {tensor.shape} to {original.shape}")
    return PreparedCaesarCorpus(dataset, loader, original, reshaped_mask, npz_path, n_frame, sections)


def caesar_corpus_roundtrip(
    compressor: Any,
    corpus: PreparedCaesarCorpus,
    model_name: str,
    eb: float,
    canonical_symbol_count: int,
    lpips_fn: Any | None = None,
) -> dict[str, Any]:
    recon, compressed_bytes, encode_seconds, decode_seconds, wall_seconds = caesar_corpus_raw_roundtrip(
        compressor, corpus, eb
    )
    metrics = base_metrics(
        corpus.original,
        recon,
        compressed_bytes,
        (encode_seconds, decode_seconds),
        group_count=len(corpus.dataset),
        side_info_bytes=0,
        valid_mask=corpus.mask,
        extra_metrics={
            **torch_memory_usage_mb("cuda"),
            "lpips": lpips_fn(corpus.original, recon) if lpips_fn is not None else None,
        },
    )
    metrics.update({
        "model_name": "CAESAR",
        "model_id": f"{model_name}-objective-eb{eb:g}",
        "control": float(eb),
        "eb": float(eb),
        "params": _count_caesar_params(compressor, model_name),
        "caesar_postprocess": "pca",
        "caesar_inference_batch_size": int(corpus.loader.batch_size),
        "partition_policy": f"{corpus.sections} sections x {corpus.n_frame} frames",
        "sample_wall_time_total": wall_seconds,
        "sample_wall_throughput_MBps": canonical_symbol_count * 4 / wall_seconds / 1e6,
    })
    return metrics


def caesar_corpus_raw_roundtrip(
    compressor: Any,
    corpus: PreparedCaesarCorpus,
    eb: float,
) -> tuple[np.ndarray, int, float, float, float]:
    """Run CAESAR once and expose its reconstruction for reversible corpus adapters."""
    if str(compressor.device).startswith("cuda"):
        torch.cuda.synchronize()
    wall_start = time.perf_counter()
    encode_start = time.perf_counter()
    compressed, compressed_size = compressor.compress(corpus.loader, eb=float(eb))
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    encode_end = time.perf_counter()
    reconstructed = compressor.decompress(compressed)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    decode_end = time.perf_counter()
    recon = corpus.dataset.recons_data(reconstructed).detach().cpu().numpy()
    wall_end = time.perf_counter()

    compressed_bytes = int(round(float(compressed_size.item() if hasattr(compressed_size, "item") else compressed_size)))
    return recon, compressed_bytes, encode_end - encode_start, decode_end - encode_end, wall_end - wall_start
