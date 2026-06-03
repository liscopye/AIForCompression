from __future__ import annotations

import json

import pytest

from utils.plot_turb_rot_results import aggregate_records, load_labeled_records, plot_label


def test_load_labeled_records_marks_caesar_original_and_tuned(tmp_path):
    orig = tmp_path / "orig.json"
    tuned = tmp_path / "tuned.json"
    orig.write_text(json.dumps([{"model_name": "CAESAR", "model_id": "caesar_v", "psnr": 30.0}]))
    tuned.write_text(json.dumps([{"model_name": "CAESAR", "model_id": "caesar_v", "psnr": 31.0}]))

    records = []
    records.extend(load_labeled_records(orig, "caesar_original"))
    records.extend(load_labeled_records(tuned, "caesar_tuned"))

    assert [plot_label(r) for r in records] == ["CAESAR-V original", "CAESAR-V tuned"]


def test_aggregate_records_averages_quality_and_memory_fields():
    records = [
        {
            "model_name": "DCAE",
            "model_id": "DCAE_a",
            "psnr": 30.0,
            "lpips": 0.2,
            "compression_ratio": 10.0,
            "bpp": 1.0,
            "memory_usage_MB": 100.0,
        },
        {
            "model_name": "DCAE",
            "model_id": "DCAE_a",
            "psnr": 34.0,
            "lpips": 0.4,
            "compression_ratio": 14.0,
            "bpp": 3.0,
            "memory_usage_MB": 120.0,
        },
    ]

    rows = aggregate_records(records)

    assert len(rows) == 1
    assert rows[0]["psnr"] == 32.0
    assert rows[0]["lpips"] == pytest.approx(0.3)
    assert rows[0]["compression_ratio"] == 12.0
    assert rows[0]["bpp"] == 2.0
    assert rows[0]["memory_usage_MB"] == 110.0
    assert rows[0]["sample_count"] == 2
