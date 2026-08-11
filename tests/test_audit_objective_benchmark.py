from __future__ import annotations

import json
from pathlib import Path

from scripts.audit_objective_benchmark import audit, expected_curves, load_rows, model_family, row_gates


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = json.loads((ROOT / "benchmark_protocols/objective_v1.json").read_text())


def compliant_row() -> dict:
    repetitions = [{"roundtrip_seconds": 1.0} for _ in range(5)]
    return {
        "protocol_id": "aifc-objective-v1",
        "track_id": "scientific_numeric",
        "dataset_id": "e3sm_npz",
        "model_name": "DCAE",
        "model_id": "DCAE-test",
        "canonical_sample_id": "vars000-004_sec000_t000-015",
        "canonical_sha256": "abc",
        "normalized_canonical_sha256": "def",
        "canonical_shape": [5, 16, 240, 240],
        "canonical_symbol_count": 4_608_000,
        "canonical_valid_symbol_count": 4_608_000,
        "external_input_manifest": {
            "scope": "dataset",
            "normalization_id": "e3sm-section5-minmax-v1",
            "name": "per-variable-fixed-scale"
        },
        "payload_bytes": 1000,
        "side_info_bytes": 20,
        "total_bytes_with_side_info": 1020,
        "scientific_bpp_with_side_info": 8 * 1020 / 4_608_000,
        "metric_protocol": "aifc-objective-v1",
        "normalized_mse": 1e-4,
        "normalized_psnr": 40.0,
        "fixed_scale_data_range": 1.0,
        "timing_protocol": "aifc-objective-v1",
        "timing_repetitions": repetitions,
        "hardware_manifest": {"gpu": "test"},
        "psnr": 40.0,
    }


def test_compliant_row_passes_every_gate() -> None:
    gates = row_gates(compliant_row(), PROTOCOL)
    assert all(gates.values()), gates


def test_rate_gate_rejects_payload_side_info_mismatch() -> None:
    row = compliant_row()
    row["total_bytes_with_side_info"] = 999
    assert not row_gates(row, PROTOCOL)["rate_complete"]


def test_metric_gate_rejects_inconsistent_fixed_scale_psnr() -> None:
    row = compliant_row()
    row["normalized_psnr"] = 39.0
    assert not row_gates(row, PROTOCOL)["metric_fixed_scale"]


def test_caesar_gate_requires_authors_batch_size() -> None:
    row = compliant_row()
    row["model_name"] = "CAESAR"
    row["model_id"] = "caesar_v-objective-eb0.1"
    row["caesar_inference_batch_size"] = 8
    assert not row_gates(row, PROTOCOL)["codec_execution_declared"]
    row["caesar_inference_batch_size"] = 64
    assert row_gates(row, PROTOCOL)["codec_execution_declared"]


def test_cusz_requires_verified_error_bound() -> None:
    row = compliant_row()
    row["model_name"] = "cuSZ-Hi"
    row["model_id"] = "cuSZ-Hi-eb0.01"
    assert not row_gates(row, PROTOCOL)["codec_valid"]
    row["error_bound_satisfied"] = True
    assert row_gates(row, PROTOCOL)["codec_valid"]


def test_lysozyme_requires_shared_mask_declaration() -> None:
    row = compliant_row()
    row["dataset_id"] = "lysozyme"
    assert not row_gates(row, PROTOCOL)["mask_policy_declared"]
    row["external_input_manifest"]["validity_mask_policy"] = "shared_benchmark_metadata"
    assert row_gates(row, PROTOCOL)["mask_policy_declared"]


def test_model_family_distinguishes_video_and_ablation() -> None:
    assert model_family({"model_id": "DCMVC_Pframe_q2"}) == "DCMVC-IP"
    assert model_family({"model_id": "DCVC_RT_Intra_q42"}) == "DCVC-RT-I"
    assert model_family({"model_id": "caesar_v_no_pca"}) == "CAESAR-no-PCA"
    assert model_family({"model_id": "DCAE_caesar_pca_eb0.01"}) == "PCA-hybrid"
    assert model_family({"model_id": "caesar_v_turb_tuned-objective-eb0.01"}) == "CAESAR-V-Turb-tuned"
    assert model_family({"model_id": "caesar_d_turb_tuned-objective-eb0.01"}) == "CAESAR-D-Turb-tuned"


def test_turb_rot_requires_declared_tuned_ablation_curves() -> None:
    curves = expected_curves(PROTOCOL, "turb_rot_npz")
    assert curves["CAESAR-V-Turb-tuned"] == 7
    assert curves["CAESAR-D-Turb-tuned"] == 7
    assert "CAESAR-V-Turb-tuned" not in expected_curves(PROTOCOL, "era5_npy")


def test_load_rows_accepts_combined_schema(tmp_path: Path) -> None:
    path = tmp_path / "combined.json"
    path.write_text(json.dumps({"records": [compliant_row()]}))
    assert load_rows(path)[0]["model_id"] == "DCAE-test"


def test_dataset_is_incomplete_until_all_contract_samples_exist() -> None:
    report = audit([compliant_row()], PROTOCOL)
    e3sm = next(item for item in report["datasets"] if item["dataset_id"] == "e3sm_npz")
    assert not e3sm["objective_sample_coverage"]
    assert not e3sm["complete"]


def test_corpus_row_covers_declared_objective_samples() -> None:
    row = compliant_row()
    second = "vars000-004_sec000_t400-415"
    row.update({
        "canonical_sample_id": "__objective_corpus__",
        "covered_canonical_sample_ids": [row["canonical_sample_id"], second],
        "covered_canonical_sha256": {
            "vars000-004_sec000_t000-015": "abc",
            second: "ghi",
        },
        "covered_normalized_canonical_sha256": {
            "vars000-004_sec000_t000-015": "def",
            second: "jkl",
        },
    })
    row["covered_canonical_sample_ids"][0] = "vars000-004_sec000_t000-015"
    report = audit([row], PROTOCOL)
    e3sm = next(item for item in report["datasets"] if item["dataset_id"] == "e3sm_npz")
    assert e3sm["objective_sample_coverage"]
