from scripts.select_caesar_era5_hourly_checkpoint import candidate_norm_type


def test_candidate_norm_type_tracks_training_variant():
    assert candidate_norm_type("v_lr1e4_lam1e4_hw_update500") == "mean_range_hw"
    assert candidate_norm_type("d_s2_hw_lr1e6_update1000") == "mean_range_hw"
    assert candidate_norm_type("original_v") == "mean_range"
    assert candidate_norm_type("d_s2_mr_lr1e6_update1000") == "mean_range"
