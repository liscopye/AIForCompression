from scripts.analyze_objective_benchmark import aggregate_point, pareto_partition


def point(model_id, bpp, psnr):
    return {
        "curve": "codec",
        "model_id": model_id,
        "scientific_bpp_with_side_info": bpp,
        "normalized_psnr": psnr,
    }


def test_pareto_partition_rejects_dominated_and_duplicate_points():
    kept, rejected = pareto_partition([
        point("a", 0.1, 30.0),
        point("duplicate", 0.1, 30.0),
        point("dominated", 0.2, 29.0),
        point("b", 0.2, 35.0),
    ])

    assert {row["model_id"] for row in kept} == {"a", "b"}
    assert {row["model_id"]: row["pareto_reason"] for row in rejected} == {
        "duplicate": "duplicate",
        "dominated": "dominated",
    }


def test_aggregate_point_accepts_one_exact_corpus_row():
    samples = {"a", "b"}
    row = {
        "model_name": "CAESAR",
        "model_id": "caesar_v-objective-eb0.01",
        "covered_canonical_sample_ids": ["a", "b"],
        "canonical_symbol_count": 100,
        "canonical_valid_symbol_count": 100,
        "total_bytes_with_side_info": 10,
        "normalized_mse": 0.01,
        "lpips": None,
        "timing_repetitions": [
            {"roundtrip_seconds": 2.0, "encode_seconds": 1.0, "decode_seconds": 0.5}
            for _ in range(5)
        ],
    }
    point = aggregate_point([row], samples)
    assert point is not None
    assert point["sample_count"] == 2
    assert point["scientific_bpp_with_side_info"] == 0.8
