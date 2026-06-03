from __future__ import annotations

import numpy as np

from compression_pipeline.adapters.turb_rot_npz import TurbRotNPZAdapter


def _write_tiny_turb_rot(path):
    data = np.arange(1 * 4 * 5 * 6 * 7, dtype=np.float32).reshape(1, 4, 5, 6, 7)
    variable_name = np.array(["vx"], dtype="<U2")
    np.savez(path, data=data, variable_name=variable_name)
    return data


def test_iter_samples_stacks_neighboring_sections_as_three_channel_images(tmp_path):
    npz_path = tmp_path / "tiny_turb_rot.npz"
    data = _write_tiny_turb_rot(npz_path)

    samples = list(TurbRotNPZAdapter(npz_path).iter_samples(max_samples=2))

    assert len(samples) == 2
    assert samples[0].dataset_id == "turb_rot_npz"
    assert samples[0].sample_id == "section000-002_t0000"
    assert samples[0].array.shape == (3, 6, 7)
    np.testing.assert_array_equal(samples[0].array, data[0, 0:3, 0])
    assert samples[0].metadata["variable_name"] == ["vx"]
    assert samples[0].metadata["section_indices"] == [0, 1, 2]


def test_iter_samples_pads_last_section_group_by_repeating_last_section(tmp_path):
    npz_path = tmp_path / "tiny_turb_rot.npz"
    data = _write_tiny_turb_rot(npz_path)

    samples = list(TurbRotNPZAdapter(npz_path, section_start=2).iter_samples(max_samples=1))

    assert samples[0].sample_id == "section002-003_t0000"
    assert samples[0].array.shape == (3, 6, 7)
    np.testing.assert_array_equal(samples[0].array[0], data[0, 2, 0])
    np.testing.assert_array_equal(samples[0].array[1], data[0, 3, 0])
    np.testing.assert_array_equal(samples[0].array[2], data[0, 3, 0])


def test_load_sequence_selects_one_section_for_caesar_vthw(tmp_path):
    npz_path = tmp_path / "tiny_turb_rot.npz"
    data = _write_tiny_turb_rot(npz_path)

    sequence, timestamps = TurbRotNPZAdapter(npz_path, section_index=2).load_sequence(max_samples=4)

    assert sequence.shape == (1, 4, 6, 7)
    np.testing.assert_array_equal(sequence, data[:, 2, :4])
    assert timestamps == [
        "turb_rot_t0000",
        "turb_rot_t0001",
        "turb_rot_t0002",
        "turb_rot_t0003",
    ]
