from __future__ import annotations

import numpy as np

from compression_pipeline.objective_data import ObjectiveSample, checksum, derive_dataset_normalization


def test_dataset_normalization_is_shared_across_samples_and_invertible() -> None:
    first = ObjectiveSample("demo", "a", np.array([[[[0.0, 2.0]]]], dtype=np.float32), None, {})
    second = ObjectiveSample("demo", "b", np.array([[[[1.0, 4.0]]]], dtype=np.float32), None, {})

    normalization = derive_dataset_normalization("demo", [first, second])

    np.testing.assert_allclose(normalization.minimum, [0.0])
    np.testing.assert_allclose(normalization.scale, [4.0])
    np.testing.assert_allclose(normalization.denormalize(normalization.normalize(first.raw)), first.raw)
    np.testing.assert_allclose(normalization.denormalize(normalization.normalize(second.raw)), second.raw)


def test_masked_sentinel_does_not_set_normalization_range() -> None:
    raw = np.array([[[[1.0, 4_294_967_295.0, 3.0]]]], dtype=np.float32)
    mask = np.array([[[[True, False, True]]]])
    cleaned = raw.copy()
    cleaned[~mask] = 0.0
    sample = ObjectiveSample("masked", "a", cleaned, mask, {})

    normalization = derive_dataset_normalization("masked", [sample])

    np.testing.assert_allclose(normalization.minimum, [1.0])
    np.testing.assert_allclose(normalization.scale, [2.0])


def test_checksum_includes_mask() -> None:
    array = np.zeros((1, 1, 2, 2), dtype=np.float32)
    first = np.ones_like(array, dtype=bool)
    second = first.copy()
    second[..., 0, 0] = False
    assert checksum(array, first) != checksum(array, second)
