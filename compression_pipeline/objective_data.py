from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


WORKSPACE = Path("/workspace")


@dataclass(frozen=True)
class ObjectiveSample:
    dataset_id: str
    sample_id: str
    raw: np.ndarray
    mask: np.ndarray | None
    metadata: dict[str, Any]

    def __post_init__(self) -> None:
        if self.raw.ndim != 4:
            raise ValueError(f"Objective samples must be [V,T,H,W], got {self.raw.shape}")
        if self.mask is not None and self.mask.shape != self.raw.shape:
            raise ValueError(f"Mask {self.mask.shape} does not match data {self.raw.shape}")


@dataclass(frozen=True)
class DatasetNormalization:
    normalization_id: str
    minimum: np.ndarray
    scale: np.ndarray
    source: str

    def normalize(self, raw_vthw: np.ndarray) -> np.ndarray:
        if raw_vthw.shape[0] != self.minimum.size:
            raise ValueError(f"Expected {self.minimum.size} variables, got {raw_vthw.shape[0]}")
        minimum = self.minimum.reshape(-1, 1, 1, 1)
        scale = self.scale.reshape(-1, 1, 1, 1)
        normalized = (raw_vthw.astype(np.float32, copy=False) - minimum) / scale
        low = float(np.nanmin(normalized))
        high = float(np.nanmax(normalized))
        if low < -2e-5 or high > 1.00002:
            raise ValueError(f"Frozen normalization does not cover the objective corpus: [{low}, {high}]")
        return np.ascontiguousarray(normalized, dtype=np.float32)

    def denormalize(self, normalized_vthw: np.ndarray) -> np.ndarray:
        minimum = self.minimum.reshape(-1, 1, 1, 1)
        scale = self.scale.reshape(-1, 1, 1, 1)
        return normalized_vthw.astype(np.float32, copy=False) * scale + minimum

    def to_json(self) -> dict[str, Any]:
        return {
            "normalization_id": self.normalization_id,
            "scope": "dataset",
            "type": "fixed_affine_minmax",
            "minimum": self.minimum.astype(float).tolist(),
            "scale": self.scale.astype(float).tolist(),
            "source": self.source,
            "clipping": False,
            "dtype": "float32",
        }

    @classmethod
    def from_json(cls, payload: dict[str, Any]) -> "DatasetNormalization":
        return cls(
            normalization_id=str(payload["normalization_id"]),
            minimum=np.asarray(payload["minimum"], dtype=np.float32),
            scale=np.asarray(payload["scale"], dtype=np.float32),
            source=str(payload["source"]),
        )


def center_crop(array: np.ndarray, height: int, width: int) -> np.ndarray:
    source_h, source_w = array.shape[-2:]
    top = (source_h - height) // 2
    left = (source_w - width) // 2
    return array[..., top : top + height, left : left + width]


def load_objective_samples(dataset_id: str) -> list[ObjectiveSample]:
    if dataset_id == "e3sm_npz":
        path = WORKSPACE / "Data/E3SM/caesar_processed/e3sm_5vars_paperlike240_first800_caesar.npz"
        with np.load(path) as handle:
            data = handle["data"][:5, 0]
            return [
                _sample(dataset_id, "vars000-004_sec000_t000-015", data[:, 0:16], source=path),
                _sample(dataset_id, "vars000-004_sec000_t400-415", data[:, 400:416], source=path),
            ]

    if dataset_id == "era5_npy":
        path = WORKSPACE / "Data/ERA5/finetune_processed/era5_test.npy"
        data = np.load(path, mmap_mode="r")
        raw = center_crop(np.asarray(data[:, :16], dtype=np.float32), 240, 240)
        return [_sample(dataset_id, "vars000-267_t000-015_crop240", raw, source=path)]

    if dataset_id == "hurricane":
        path = WORKSPACE / "Data/SDRBENCH-Hurricane-ISABEL-100x500x500/100x500x500/PRECIPf48.log10.bin.f32"
        data = np.memmap(path, dtype=np.float32, mode="r", shape=(100, 500, 500))
        return [_sample(dataset_id, "precip_log10_t000-095", np.asarray(data[:96])[None], source=path)]

    if dataset_id == "nyx":
        path = WORKSPACE / "Data/SDRBENCH-EXASKY-NYX-512x512x512/SDRBENCH-EXASKY-NYX-512x512x512/baryon_density.f32"
        data = np.memmap(path, dtype=np.float32, mode="r", shape=(512, 512, 512))
        return [_sample(dataset_id, "baryon_density_z000-511", np.asarray(data)[None], source=path)]

    if dataset_id == "turb_rot_npz":
        path = WORKSPACE / "Turb_Rot_testset.npz"
        with np.load(path) as handle:
            data = handle["data"][0]
            return [
                _sample(dataset_id, "var000_sec000_t000-255", data[0][None], source=path),
                _sample(dataset_id, "var000_sec008_t000-255", data[8][None], source=path),
            ]

    if dataset_id == "tomo":
        import h5py

        path = WORKSPACE / "Data/tomo_00001.h5"
        with h5py.File(path, "r") as handle:
            data = handle["exchange/data"]
            first = center_crop(data[0:512], 512, 512).astype(np.float32)[None]
            second = center_crop(data[989:1501], 512, 512).astype(np.float32)[None]
        return [
            _sample(dataset_id, "projection0000-0511_crop512", first, source=path),
            _sample(dataset_id, "projection0989-1500_crop512", second, source=path),
        ]

    if dataset_id == "lysozyme":
        path = WORKSPACE / "Data/lysozyme_processed/mmap/lysozyme_test_nf16.npy"
        data = np.load(path, mmap_mode="r")[0]
        output = []
        for chunk_start, sample_id in [
            (0, "test_chunks000-030_frames000-495"),
            (31, "test_chunks031-061_frames000-495"),
        ]:
            raw = np.asarray(data[chunk_start : chunk_start + 31], dtype=np.float32).reshape(496, 1024, 1024)[None]
            mask = raw < 4_294_967_000.0
            cleaned = np.array(raw, copy=True)
            cleaned[~mask] = 0.0
            output.append(_sample(dataset_id, sample_id, cleaned, mask=mask, source=path))
        return output

    if dataset_id == "s2c":
        from types import SimpleNamespace
        from scripts.run_external_scientific_codecs import iter_s2c_band_volume_samples

        path = WORKSPACE / (
            "Data/S2C_MSIL2A_20260509T022531_N0512_R046_T51RUQ_20260509T055911/"
            "S2C_MSIL2A_20260509T022531_N0512_R046_T51RUQ_20260509T055911.SAFE"
        )
        args = SimpleNamespace(data_root=str(path), s2c_bands=["B02", "B03", "B04", "B08"], tile_size=1024, max_samples=4)
        return [
            _sample(dataset_id, item.sample_id, np.asarray(item.array, dtype=np.float32)[:, None], source=path)
            for item in iter_s2c_band_volume_samples(args, None)
        ]

    if dataset_id == "kodak":
        from PIL import Image

        root = WORKSPACE / "Data/Kodac"
        output = []
        for path in sorted(root.glob("kodim*.png")):
            hwc = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32)
            output.append(_sample(dataset_id, path.stem, np.moveaxis(hwc, -1, 0)[:, None], source=path))
        return output

    if dataset_id == "uvg_twilight_1080p":
        from compression_pipeline.adapters.uvg import UVGAdapter

        path = WORKSPACE / "Data/UVG_Twilight_1080p"
        sequence, _ = UVGAdapter(path).load_sequence(max_samples=30)
        return [_sample(dataset_id, "twilight_frames000-029_1080p", sequence, source=path)]

    raise ValueError(f"Unsupported objective dataset: {dataset_id}")


def derive_dataset_normalization(dataset_id: str, samples: list[ObjectiveSample]) -> DatasetNormalization:
    if not samples:
        raise ValueError(f"No objective samples for {dataset_id}")
    variables = samples[0].raw.shape[0]
    if any(sample.raw.shape[0] != variables for sample in samples):
        raise ValueError(f"Variable count changes across {dataset_id} samples")

    fixed_ranges = {
        "kodak": (0.0, 255.0),
        "uvg_twilight_1080p": (0.0, 1.0),
        "tomo": (0.0, 65535.0),
    }
    if dataset_id in fixed_ranges:
        minimum, maximum = fixed_ranges[dataset_id]
        mins = np.full(variables, minimum, dtype=np.float32)
        maxs = np.full(variables, maximum, dtype=np.float32)
        source = f"fixed dataset convention [{minimum:g},{maximum:g}]"
    else:
        mins = np.full(variables, np.inf, dtype=np.float64)
        maxs = np.full(variables, -np.inf, dtype=np.float64)
        for sample in samples:
            for variable in range(variables):
                values = sample.raw[variable]
                if sample.mask is not None:
                    values = values[sample.mask[variable]]
                mins[variable] = min(mins[variable], float(np.min(values)))
                maxs[variable] = max(maxs[variable], float(np.max(values)))
        mins = mins.astype(np.float32)
        maxs = maxs.astype(np.float32)
        source = "complete objective evaluation corpus"
    scales = np.maximum(maxs - mins, np.float32(1e-12)).astype(np.float32)
    return DatasetNormalization(
        normalization_id=f"{dataset_id}-objective-corpus-minmax-v1",
        minimum=mins,
        scale=scales,
        source=source,
    )


def checksum(array: np.ndarray, mask: np.ndarray | None = None) -> str:
    digest = hashlib.sha256()
    digest.update(np.ascontiguousarray(array).view(np.uint8))
    if mask is not None:
        digest.update(np.ascontiguousarray(mask).view(np.uint8))
    return digest.hexdigest()


def save_normalization(path: Path, normalization: DatasetNormalization) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(normalization.to_json(), indent=2), encoding="utf-8")


def load_normalization(path: Path) -> DatasetNormalization:
    return DatasetNormalization.from_json(json.loads(path.read_text(encoding="utf-8")))


def _sample(
    dataset_id: str,
    sample_id: str,
    raw: np.ndarray,
    *,
    source: Path,
    mask: np.ndarray | None = None,
) -> ObjectiveSample:
    return ObjectiveSample(
        dataset_id=dataset_id,
        sample_id=sample_id,
        raw=np.ascontiguousarray(raw, dtype=np.float32),
        mask=np.ascontiguousarray(mask, dtype=bool) if mask is not None else None,
        metadata={"source": str(source), "layout": "variable,time_or_depth,height,width"},
    )
