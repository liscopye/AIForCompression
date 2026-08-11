import numpy as np

from compression_pipeline.objective_stacking import (
    crop_corpus_depth,
    pack_objective_corpus,
    pad_corpus_depth,
    unpack_objective_corpus,
)


def test_s2c_pack_is_tile_major_band_minor_and_reversible():
    samples = [np.full((4, 1, 3, 5), index, dtype=np.float32) for index in range(4)]
    packed = pack_objective_corpus("s2c", samples, [None] * 4)
    assert packed.volume.shape == (1, 16, 3, 5)
    restored = unpack_objective_corpus("s2c", packed, packed.volume)
    assert all(np.array_equal(before, after) for before, after in zip(samples, restored))


def test_kodak_pack_uses_all_rgb_planes_and_reverses_portrait_rotation():
    landscape = np.arange(3 * 1 * 4 * 6, dtype=np.float32).reshape(3, 1, 4, 6)
    portrait = np.arange(3 * 1 * 6 * 4, dtype=np.float32).reshape(3, 1, 6, 4)
    packed = pack_objective_corpus("kodak", [landscape, portrait], [None, None])
    assert packed.volume.shape == (1, 6, 4, 6)
    restored = unpack_objective_corpus("kodak", packed, packed.volume)
    assert np.array_equal(restored[0], landscape)
    assert np.array_equal(restored[1], portrait)


def test_caesar_depth_padding_is_reversible_and_declared():
    sample = np.arange(3 * 30 * 2 * 2, dtype=np.float32).reshape(3, 30, 2, 2)
    packed = pack_objective_corpus("uvg_twilight_1080p", [sample], [None])
    padded = pad_corpus_depth(packed, 16)
    assert padded.volume.shape == (3, 32, 2, 2)
    assert padded.metadata["padding_planes"] == 2
    assert np.array_equal(crop_corpus_depth(padded, padded.volume), sample)
