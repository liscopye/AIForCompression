from scripts.select_objective_eb_schedule import aggregate_candidates, pareto, select


def row(eb, bpp, psnr, *, valid=True, sample_id=None):
    return {
        "model_name": "cuSZ-Hi",
        "model_id": f"cuSZ-Hi-objective-eb{eb}",
        "eb": eb,
        "scientific_bpp_with_side_info": bpp,
        "normalized_psnr": psnr,
        "error_bound_satisfied": valid,
        "canonical_sample_id": sample_id,
    }


def test_invalid_and_dominated_eb_points_are_not_selected():
    candidates = aggregate_candidates([
        row(0.5, 0.1, 20),
        row(0.1, 0.5, 30),
        row(0.01, 1.0, 29),
        row(0.001, 2.0, 50, valid=False),
    ])
    frontier = pareto(candidates)
    assert [point["control"] for point in frontier] == [0.5, 0.1]


def test_selection_keeps_both_bpp_endpoints():
    points = [{"control": float(index), "bpp": float(2**index), "psnr": float(index)} for index in range(1, 11)]
    chosen = select(points, 7, target_max_bpp=1e9)
    assert chosen[0] == points[0]
    assert chosen[-1] == points[-1]
    assert len(chosen) == 7


def test_candidate_must_cover_every_objective_sample():
    candidates = aggregate_candidates([
        row(0.1, 0.5, 30, sample_id="a"),
        row(0.01, 1.0, 40, sample_id="a"),
        row(0.01, 1.2, 39, sample_id="b"),
    ], {"a", "b"})
    assert [point["control"] for point in candidates] == [0.01]
