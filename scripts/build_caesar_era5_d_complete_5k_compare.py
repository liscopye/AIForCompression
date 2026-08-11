#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Callable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


RATE_KEY = "scientific_bpp_with_side_info"
EXPECTED_EBS = [0.3, 0.1, 0.05, 0.03, 0.025, 0.02, 0.015, 0.01, 0.003, 0.001, 1e-4, 3e-6, 1e-9]
STYLES = {
    "DCAE": ("#0072B2", "o", "-", 1.9),
    "LIC-HPCM-base": ("#009E73", "D", "-", 1.8),
    "LIC-HPCM-large": ("#56B4E9", "^", "-", 1.8),
    "CAESAR-D original": ("#6B7280", "o", "--", 2.0),
    "CAESAR-D Stage2 5k probe": ("#D55E00", "X", "-", 2.8),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--original", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--candidate-variant", required=True)
    parser.add_argument("--candidate-label", default="CAESAR-D Stage2 5k probe")
    parser.add_argument(
        "--status", default="Stage2 5k probe; not the final 200k CAESAR-D result"
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def load_rows(path: Path) -> list[dict[str, Any]]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise ValueError(f"Expected a row list in {path}")
    return rows


def quality(row: dict[str, Any]) -> float:
    value = row.get("normalized_psnr", row.get("psnr"))
    if value is None:
        value = row.get("psnr")
    return float(value)


def valid(row: dict[str, Any]) -> bool:
    try:
        return (
            "error" not in row
            and float(row[RATE_KEY]) > 0
            and math.isfinite(float(row[RATE_KEY]))
            and math.isfinite(quality(row))
        )
    except (KeyError, TypeError, ValueError):
        return False


def select(rows: list[dict[str, Any]], predicate: Callable[[dict[str, Any]], bool]) -> list[dict[str, Any]]:
    selected = [dict(row) for row in rows if valid(row) and predicate(row)]
    selected.sort(key=lambda row: float(row[RATE_KEY]))
    if not selected:
        raise ValueError("No rows matched the requested curve")
    return selected


def assert_expected_ebs(rows: list[dict[str, Any]], label: str) -> None:
    actual = sorted(float(row["eb"]) for row in rows)
    expected = sorted(EXPECTED_EBS)
    if len(actual) != len(expected) or any(
        not math.isclose(left, right, rel_tol=1e-8, abs_tol=1e-12)
        for left, right in zip(actual, expected)
    ):
        raise ValueError(f"{label} has EB values {actual}, expected {expected}")


def monotonic(rows: list[dict[str, Any]]) -> bool:
    points = [(float(row[RATE_KEY]), quality(row)) for row in rows]
    return all(
        right_rate > left_rate and right_quality >= left_quality
        for (left_rate, left_quality), (right_rate, right_quality) in zip(points, points[1:])
    )


def interpolate_quality(rows: list[dict[str, Any]], rate: float) -> float | None:
    points = [(float(row[RATE_KEY]), quality(row)) for row in rows]
    rates = np.asarray([point[0] for point in points])
    if rate < rates[0] or rate > rates[-1]:
        return None
    qualities = np.asarray([point[1] for point in points])
    return float(np.interp(math.log(rate), np.log(rates), qualities))


def interpolate_rate(rows: list[dict[str, Any]], target_quality: float) -> float | None:
    points = sorted((quality(row), math.log(float(row[RATE_KEY]))) for row in rows)
    qualities = np.asarray([point[0] for point in points])
    if target_quality < qualities[0] or target_quality > qualities[-1]:
        return None
    return math.exp(float(np.interp(target_quality, qualities, [point[1] for point in points])))


def same_eb_comparison(
    original: list[dict[str, Any]], candidate: list[dict[str, Any]]
) -> list[dict[str, float]]:
    original_by_eb = {float(row["eb"]): row for row in original}
    output = []
    for row in candidate:
        eb = float(row["eb"])
        reference = original_by_eb[eb]
        original_rate = float(reference[RATE_KEY])
        candidate_rate = float(row[RATE_KEY])
        output.append(
            {
                "eb": eb,
                "original_bpp": original_rate,
                "candidate_bpp": candidate_rate,
                "bpp_change_pct": (candidate_rate / original_rate - 1.0) * 100.0,
                "original_psnr_db": quality(reference),
                "candidate_psnr_db": quality(row),
                "psnr_change_db": quality(row) - quality(reference),
            }
        )
    return sorted(output, key=lambda row: row["eb"], reverse=True)


def matched_bpp_comparison(
    original: list[dict[str, Any]], candidate: list[dict[str, Any]]
) -> tuple[list[dict[str, float]], dict[str, float]]:
    original_rates = [float(row[RATE_KEY]) for row in original]
    candidate_rates = [float(row[RATE_KEY]) for row in candidate]
    overlap_min = max(min(original_rates), min(candidate_rates))
    overlap_max = min(max(original_rates), max(candidate_rates))

    requested = [overlap_min, 0.243493081, 0.3, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 30.0, overlap_max]
    rates = []
    for rate in requested:
        if overlap_min <= rate <= overlap_max and not any(math.isclose(rate, found) for found in rates):
            rates.append(rate)

    comparison = []
    for rate in rates:
        original_quality = interpolate_quality(original, rate)
        candidate_quality = interpolate_quality(candidate, rate)
        assert original_quality is not None and candidate_quality is not None
        comparison.append(
            {
                "bpp": rate,
                "original_psnr_db": original_quality,
                "candidate_psnr_db": candidate_quality,
                "psnr_gain_db": candidate_quality - original_quality,
            }
        )

    knots = sorted(
        {
            overlap_min,
            overlap_max,
            *(rate for rate in original_rates if overlap_min < rate < overlap_max),
            *(rate for rate in candidate_rates if overlap_min < rate < overlap_max),
        }
    )
    log_rates = np.log(knots)
    gains = np.asarray(
        [
            float(interpolate_quality(candidate, rate)) - float(interpolate_quality(original, rate))
            for rate in knots
        ]
    )
    summary = {
        "overlap_min_bpp": overlap_min,
        "overlap_max_bpp": overlap_max,
        "mean_psnr_gain_db_log_bpp": float(
            np.trapezoid(gains, log_rates) / (log_rates[-1] - log_rates[0])
        ),
        "minimum_psnr_gain_db_at_union_knots": float(gains.min()),
        "maximum_psnr_gain_db_at_union_knots": float(gains.max()),
    }
    return comparison, summary


def image_codec_comparison(
    image_curves: dict[str, list[dict[str, Any]]], candidate: list[dict[str, Any]]
) -> list[dict[str, float | str | None]]:
    output = []
    for label, rows in image_curves.items():
        endpoint = max(rows, key=lambda row: float(row[RATE_KEY]))
        endpoint_rate = float(endpoint[RATE_KEY])
        endpoint_quality = quality(endpoint)
        candidate_quality = interpolate_quality(candidate, endpoint_rate)
        candidate_rate = interpolate_rate(candidate, endpoint_quality)
        output.append(
            {
                "model": label,
                "baseline_bpp": endpoint_rate,
                "baseline_psnr_db": endpoint_quality,
                "candidate_psnr_at_baseline_bpp_db": candidate_quality,
                "candidate_psnr_delta_at_baseline_bpp_db": (
                    candidate_quality - endpoint_quality if candidate_quality is not None else None
                ),
                "candidate_bpp_at_baseline_psnr": candidate_rate,
                "candidate_bpp_change_at_baseline_psnr_pct": (
                    (candidate_rate / endpoint_rate - 1.0) * 100.0
                    if candidate_rate is not None
                    else None
                ),
            }
        )
    return output


def plot(
    curves: dict[str, list[dict[str, Any]]],
    original: list[dict[str, Any]],
    candidate: list[dict[str, Any]],
    candidate_label: str,
    output: Path,
) -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9.5,
            "axes.titlesize": 11.5,
            "axes.labelsize": 10.5,
            "legend.fontsize": 8.3,
            "axes.linewidth": 0.9,
            "lines.solid_capstyle": "round",
        }
    )
    fig = plt.figure(figsize=(13.5, 6.3))
    grid = fig.add_gridspec(2, 2, width_ratios=[1.05, 0.95], hspace=0.34, wspace=0.20)
    full_axis = fig.add_subplot(grid[:, 0])
    zoom_axis = fig.add_subplot(grid[0, 1])
    gain_axis = fig.add_subplot(grid[1, 1])

    for axis in (full_axis, zoom_axis):
        for label, rows in curves.items():
            color, marker, linestyle, width = STYLES[label]
            axis.plot(
                [row[RATE_KEY] for row in rows],
                [quality(row) for row in rows],
                label=label,
                color=color,
                marker=marker,
                linestyle=linestyle,
                linewidth=width,
                markersize=6.4 if "5k" in label else 4.8,
                markeredgecolor="white",
                markeredgewidth=0.65,
                zorder=7 if "5k" in label else 3,
            )
        axis.set_xscale("log")
        axis.grid(True, which="major", color="#D1D5DB", linewidth=0.7, alpha=0.78)
        axis.grid(True, which="minor", color="#E5E7EB", linewidth=0.45, alpha=0.62)
        axis.spines[["top", "right"]].set_visible(False)
        axis.set_xlabel("BPP including required side information")

    full_axis.set_xlim(0.018, 35)
    full_axis.set_ylabel("Dataset-normalized PSNR (dB)")
    full_axis.set_title("Complete rate-distortion curves")

    zoom_axis.set_xlim(0.018, 0.65)
    zoom_axis.set_ylim(20, 46)
    zoom_axis.set_ylabel("Dataset-normalized PSNR (dB)")
    zoom_axis.set_title("Image-codec transition region")

    overlap_min = max(float(original[0][RATE_KEY]), float(candidate[0][RATE_KEY]))
    overlap_max = min(float(original[-1][RATE_KEY]), float(candidate[-1][RATE_KEY]))
    rate_grid = np.geomspace(overlap_min, overlap_max, 400)
    gains = [
        float(interpolate_quality(candidate, rate)) - float(interpolate_quality(original, rate))
        for rate in rate_grid
    ]
    gain_axis.plot(rate_grid, gains, color=STYLES[candidate_label][0], linewidth=2.4)
    gain_axis.axhline(0, color="#6B7280", linewidth=0.9)
    gain_axis.fill_between(rate_grid, 0, gains, color="#F3B49F", alpha=0.35)
    gain_axis.set_xscale("log")
    gain_axis.set_xlim(overlap_min, overlap_max)
    gain_axis.set_xlabel("Matched BPP")
    gain_axis.set_ylabel("PSNR gain vs original (dB)")
    gain_axis.set_title("Matched-BPP improvement")
    gain_axis.grid(True, which="major", color="#D1D5DB", linewidth=0.7, alpha=0.78)
    gain_axis.grid(True, which="minor", color="#E5E7EB", linewidth=0.45, alpha=0.62)
    gain_axis.spines[["top", "right"]].set_visible(False)

    handles, labels = full_axis.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.005),
        ncol=3,
        frameon=False,
        columnspacing=1.2,
        handlelength=2.6,
    )
    fig.suptitle(f"ERA5 Objective-v1: {candidate_label}", fontsize=14)
    fig.subplots_adjust(left=0.07, right=0.99, top=0.88, bottom=0.19)
    fig.savefig(output / "caesar_d_stage2_5k_vs_original_and_image_codecs.png", dpi=300, bbox_inches="tight")
    fig.savefig(output / "caesar_d_stage2_5k_vs_original_and_image_codecs.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    STYLES[args.candidate_label] = STYLES["CAESAR-D Stage2 5k probe"]
    baseline_rows = load_rows(args.baseline)
    original_rows = load_rows(args.original)
    candidate_rows = load_rows(args.candidate)

    original = select(
        original_rows,
        lambda row: str(row.get("model_id", "")).lower().startswith("caesar_d")
        and str(row.get("checkpoint_variant", "")).lower() == "original",
    )
    candidate = select(
        candidate_rows,
        lambda row: str(row.get("model_id", "")).lower().startswith("caesar_d")
        and str(row.get("checkpoint_variant", "")).lower() == args.candidate_variant.lower(),
    )
    assert_expected_ebs(original, "original")
    assert_expected_ebs(candidate, "candidate")

    image_curves = {
        "DCAE": select(baseline_rows, lambda row: row.get("dataset_id") == "era5_npy" and row.get("model_name") == "DCAE"),
        "LIC-HPCM-base": select(
            baseline_rows,
            lambda row: row.get("dataset_id") == "era5_npy"
            and row.get("model_name") == "LIC-HPCM"
            and "base" in str(row.get("model_id", "")).lower(),
        ),
        "LIC-HPCM-large": select(
            baseline_rows,
            lambda row: row.get("dataset_id") == "era5_npy"
            and row.get("model_name") == "LIC-HPCM"
            and "large" in str(row.get("model_id", "")).lower(),
        ),
    }
    curves = {
        **image_curves,
        "CAESAR-D original": original,
        args.candidate_label: candidate,
    }

    matched_bpp, matched_bpp_summary = matched_bpp_comparison(original, candidate)
    comparison = {
        "protocol_id": "aifc-objective-v1",
        "dataset": "era5_npy",
        "status": args.status,
        "sources": {
            "baseline": str(args.baseline.resolve()),
            "original": str(args.original.resolve()),
            "candidate": str(args.candidate.resolve()),
        },
        "candidate_variant": args.candidate_variant,
        "monotonic": {label: monotonic(rows) for label, rows in curves.items()},
        "same_eb_candidate_vs_original": same_eb_comparison(original, candidate),
        "matched_bpp_candidate_vs_original": matched_bpp,
        "matched_bpp_summary": matched_bpp_summary,
        "image_codec_endpoint_comparison": image_codec_comparison(image_curves, candidate),
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "comparison.json").write_text(
        json.dumps(comparison, indent=2) + "\n", encoding="utf-8"
    )
    plot(curves, original, candidate, args.candidate_label, args.output)
    print(json.dumps(matched_bpp_summary, indent=2))


if __name__ == "__main__":
    main()
