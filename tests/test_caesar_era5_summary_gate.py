import pytest

from scripts.summarize_caesar_era5_stable_tuning import (
    improvement_failures,
    validate_curve_norm,
)


def test_improvement_gate_accepts_both_negative_bd_rates():
    assert improvement_failures(-1.0, -0.01) == []


def test_improvement_gate_reports_each_non_improving_model():
    failures = improvement_failures(0.0, 1.5)

    assert len(failures) == 2
    assert failures[0].startswith("CAESAR-V")
    assert failures[1].startswith("CAESAR-D")


def test_curve_norm_validation_rejects_stale_or_mixed_results():
    validate_curve_norm(
        [{"caesar_norm_type": "mean_range"}],
        "mean_range",
        "original",
    )
    with pytest.raises(ValueError, match="observed"):
        validate_curve_norm(
            [
                {"caesar_norm_type": "mean_range"},
                {"caesar_norm_type": "mean_range_hw"},
            ],
            "mean_range_hw",
            "tuned",
        )
