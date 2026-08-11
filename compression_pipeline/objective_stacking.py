from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class PackedObjectiveCorpus:
    volume: np.ndarray
    mask: np.ndarray | None
    metadata: dict[str, Any]


def pack_objective_corpus(
    dataset_id: str,
    normalized_samples: list[np.ndarray],
    masks: list[np.ndarray | None],
) -> PackedObjectiveCorpus:
    """Build the reversible 3D view used by corpus-level scientific codecs."""
    if not normalized_samples:
        raise ValueError("Cannot pack an empty objective corpus")

    if dataset_id == "s2c":
        planes = [np.ascontiguousarray(array[:, 0]) for array in normalized_samples]
        volume = np.concatenate(planes, axis=0)[None]
        packed_mask = _concatenate_optional_masks(
            [None if mask is None else np.ascontiguousarray(mask[:, 0]) for mask in masks], axis=0
        )
        return PackedObjectiveCorpus(
            np.ascontiguousarray(volume),
            None if packed_mask is None else packed_mask[None],
            {
                "packing": "tile-major_band-minor",
                "source_shapes": [list(array.shape) for array in normalized_samples],
                "unpadded_depth": int(volume.shape[1]),
            },
        )

    if dataset_id == "kodak":
        planes = []
        packed_masks = []
        rotations = []
        for array, mask in zip(normalized_samples, masks):
            chw = np.ascontiguousarray(array[:, 0])
            rotate = chw.shape[-2] > chw.shape[-1]
            if rotate:
                chw = np.rot90(chw, k=1, axes=(-2, -1)).copy()
            planes.append(chw)
            rotations.append(bool(rotate))
            if mask is not None:
                mask_chw = np.ascontiguousarray(mask[:, 0])
                packed_masks.append(np.rot90(mask_chw, k=1, axes=(-2, -1)).copy() if rotate else mask_chw)
        if len({plane.shape[-2:] for plane in planes}) != 1:
            raise ValueError("Reversible Kodak orientation normalization did not produce one spatial shape")
        volume = np.concatenate(planes, axis=0)[None]
        packed_mask = np.concatenate(packed_masks, axis=0)[None] if packed_masks else None
        return PackedObjectiveCorpus(
            np.ascontiguousarray(volume),
            None if packed_mask is None else np.ascontiguousarray(packed_mask),
            {
                "packing": "image-major_rgb-minor",
                "portrait_rotation_k": 1,
                "rotated_samples": rotations,
                "source_shapes": [list(array.shape) for array in normalized_samples],
                "unpadded_depth": int(volume.shape[1]),
            },
        )

    if dataset_id == "uvg_twilight_1080p":
        if len(normalized_samples) != 1:
            raise ValueError("UVG objective corpus must contain exactly one sequence")
        return PackedObjectiveCorpus(
            np.ascontiguousarray(normalized_samples[0]),
            None if masks[0] is None else np.ascontiguousarray(masks[0]),
            {
                "packing": "rgb-variable_time-depth",
                "source_shapes": [list(normalized_samples[0].shape)],
                "unpadded_depth": int(normalized_samples[0].shape[1]),
            },
        )

    raise ValueError(f"No corpus stacking contract for {dataset_id}")


def pad_corpus_depth(corpus: PackedObjectiveCorpus, multiple: int) -> PackedObjectiveCorpus:
    if multiple <= 0:
        raise ValueError(f"Depth multiple must be positive, got {multiple}")
    depth = int(corpus.volume.shape[1])
    padded_depth = ((depth + multiple - 1) // multiple) * multiple
    if padded_depth == depth:
        return corpus
    pad_count = padded_depth - depth
    volume = np.concatenate(
        [corpus.volume, np.repeat(corpus.volume[:, -1:], pad_count, axis=1)], axis=1
    )
    mask = None
    if corpus.mask is not None:
        mask = np.concatenate([corpus.mask, np.repeat(corpus.mask[:, -1:], pad_count, axis=1)], axis=1)
    metadata = dict(corpus.metadata)
    metadata.update({"depth_padding": "repeat-last", "padded_depth": padded_depth, "padding_planes": pad_count})
    return PackedObjectiveCorpus(np.ascontiguousarray(volume), None if mask is None else np.ascontiguousarray(mask), metadata)


def crop_corpus_depth(corpus: PackedObjectiveCorpus, reconstruction: np.ndarray) -> np.ndarray:
    depth = int(corpus.metadata["unpadded_depth"])
    return np.ascontiguousarray(reconstruction[:, :depth])


def unpack_objective_corpus(
    dataset_id: str,
    corpus: PackedObjectiveCorpus,
    volume: np.ndarray,
) -> list[np.ndarray]:
    volume = crop_corpus_depth(corpus, volume)
    source_shapes = corpus.metadata["source_shapes"]
    if dataset_id == "s2c":
        output = []
        start = 0
        for shape in source_shapes:
            bands = int(shape[0])
            output.append(np.ascontiguousarray(volume[0, start : start + bands, :, :][:, None]))
            start += bands
        return output
    if dataset_id == "kodak":
        output = []
        for index, (shape, rotated) in enumerate(zip(source_shapes, corpus.metadata["rotated_samples"])):
            chw = volume[0, index * 3 : (index + 1) * 3]
            if rotated:
                chw = np.rot90(chw, k=-1, axes=(-2, -1)).copy()
            restored = np.ascontiguousarray(chw[:, None])
            if list(restored.shape) != list(shape):
                raise ValueError(f"Unpacked Kodak shape {restored.shape} does not match {shape}")
            output.append(restored)
        return output
    if dataset_id == "uvg_twilight_1080p":
        return [np.ascontiguousarray(volume)]
    raise ValueError(f"No corpus unpacking contract for {dataset_id}")


def _concatenate_optional_masks(masks: list[np.ndarray | None], axis: int) -> np.ndarray | None:
    if all(mask is None for mask in masks):
        return None
    if any(mask is None for mask in masks):
        raise ValueError("A packed corpus cannot mix masked and unmasked samples")
    return np.concatenate([mask for mask in masks if mask is not None], axis=axis)
