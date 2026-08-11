#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


EXPECTED_FINAL_EBS = [0.3, 0.1, 0.05, 0.03, 0.025, 0.02, 0.015, 0.01, 0.003, 0.001, 1e-4, 3e-6, 1e-9]
RATE_KEY = "scientific_bpp_with_side_info"
QUALITY_KEY = "normalized_psnr"

STYLES = {
    "DCAE": ("#0072B2", "o", "-", 1.9),
    "LIC-HPCM-base": ("#009E73", "D", "-", 1.8),
    "LIC-HPCM-large": ("#56B4E9", "^", "-", 1.8),
    "CAESAR-V original": ("#6B7280", "o", "--", 1.8),
    "CAESAR-V low-rate base": ("#E69F00", "s", ":", 2.0),
    "CAESAR-V decoder FT": ("#D55E00", "X", "-", 2.8),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--lowrate-base", type=Path, required=True)
    parser.add_argument("--lowrate-probe", type=Path, required=True)
    parser.add_argument("--final", type=Path, required=True)
    parser.add_argument("--final-variant", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def load_rows(path: Path) -> list[dict[str, Any]]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise ValueError(f"Expected a row list in {path}")
    return rows


def finite(row: dict[str, Any]) -> bool:
    rate = row.get(RATE_KEY)
    quality = row.get(QUALITY_KEY)
    return (
        "error" not in row
        and isinstance(rate, (int, float))
        and isinstance(quality, (int, float))
        and math.isfinite(float(rate))
        and math.isfinite(float(quality))
        and float(rate) > 0
    )


def baseline_name(row: dict[str, Any]) -> str | None:
    if row.get("dataset_id") != "era5_npy" or not finite(row):
        return None
    model_name = str(row.get("model_name", ""))
    model_id = str(row.get("model_id", "")).lower()
    if model_name == "DCAE":
        return "DCAE"
    if model_name == "LIC-HPCM":
        return "LIC-HPCM-large" if "large" in model_id else "LIC-HPCM-base"
    if model_id.startswith("caesar_v") and str(row.get("checkpoint_variant", "")).lower() in {
        "",
        "none",
        "original",
    }:
        return "CAESAR-V original"
    return None


def selected_caesar_rows(
    rows: list[dict[str, Any]], variant: str, label: str
) -> list[dict[str, Any]]:
    selected = [
        dict(row)
        for row in rows
        if finite(row)
        and row.get("dataset_id") == "era5_npy"
        and str(row.get("model_id", "")).lower().startswith("caesar_v")
        and str(row.get("checkpoint_variant", "")).lower() == variant.lower()
    ]
    if not selected:
        raise ValueError(f"No valid {label} rows for checkpoint variant {variant}")
    selected.sort(key=lambda row: float(row[RATE_KEY]))
    return selected


def assert_expected_final_ebs(rows: list[dict[str, Any]]) -> None:
    found = sorted(float(row["eb"]) for row in rows)
    expected = sorted(EXPECTED_FINAL_EBS)
    if len(found) != len(expected) or any(
        abs(actual - wanted) > max(1e-12, abs(wanted) * 1e-8)
        for actual, wanted in zip(found, expected)
    ):
        raise ValueError(f"Final curve has EB values {found}, expected {expected}")


def merge_by_eb(*row_groups: list[dict[str, Any]]) -> list[dict[str, Any]]:
    merged: dict[float, dict[str, Any]] = {}
    for rows in row_groups:
        for row in rows:
            merged[float(row["eb"])] = dict(row)
    return sorted(merged.values(), key=lambda row: float(row[RATE_KEY]))


def is_monotonic(rows: list[dict[str, Any]]) -> bool:
    ordered = sorted(rows, key=lambda row: float(row[RATE_KEY]))
    return all(
        float(right[QUALITY_KEY]) >= float(left[QUALITY_KEY])
        for left, right in zip(ordered, ordered[1:])
    )


def interpolate_quality(rows: list[dict[str, Any]], rate: float) -> float | None:
    points = sorted((float(row[RATE_KEY]), float(row[QUALITY_KEY])) for row in rows)
    rates = np.asarray([point[0] for point in points])
    if rate < rates[0] or rate > rates[-1]:
        return None
    qualities = np.asarray([point[1] for point in points])
    return float(np.interp(math.log(rate), np.log(rates), qualities))


def common_eb_gains(
    base: list[dict[str, Any]], final: list[dict[str, Any]]
) -> list[dict[str, float]]:
    base_by_eb = {float(row["eb"]): row for row in base}
    gains = []
    for row in final:
        eb = float(row["eb"])
        if eb not in base_by_eb:
            continue
        baseline = base_by_eb[eb]
        gains.append(
            {
                "eb": eb,
                "base_bpp": float(baseline[RATE_KEY]),
                "final_bpp": float(row[RATE_KEY]),
                "psnr_gain_db": float(row[QUALITY_KEY]) - float(baseline[QUALITY_KEY]),
            }
        )
    return gains


def common_eb_comparison(
    reference: list[dict[str, Any]], candidate: list[dict[str, Any]]
) -> list[dict[str, float]]:
    reference_by_eb = {float(row["eb"]): row for row in reference}
    comparison = []
    for row in candidate:
        eb = float(row["eb"])
        if eb not in reference_by_eb:
            continue
        baseline = reference_by_eb[eb]
        baseline_bpp = float(baseline[RATE_KEY])
        candidate_bpp = float(row[RATE_KEY])
        comparison.append(
            {
                "eb": eb,
                "original_bpp": baseline_bpp,
                "final_bpp": candidate_bpp,
                "bpp_change_pct": (candidate_bpp / baseline_bpp - 1.0) * 100.0,
                "original_psnr_db": float(baseline[QUALITY_KEY]),
                "final_psnr_db": float(row[QUALITY_KEY]),
                "psnr_change_db": float(row[QUALITY_KEY]) - float(baseline[QUALITY_KEY]),
            }
        )
    return sorted(comparison, key=lambda row: row["eb"], reverse=True)


def interpolate_rate(rows: list[dict[str, Any]], quality: float) -> float | None:
    points = sorted((float(row[QUALITY_KEY]), math.log(float(row[RATE_KEY]))) for row in rows)
    qualities = np.asarray([point[0] for point in points])
    if quality < qualities[0] or quality > qualities[-1]:
        return None
    log_rates = np.asarray([point[1] for point in points])
    return math.exp(float(np.interp(quality, qualities, log_rates)))


def matched_quality_comparison(
    original: list[dict[str, Any]],
    lowrate: list[dict[str, Any]],
    final: list[dict[str, Any]],
    qualities: list[float],
) -> list[dict[str, float]]:
    rows = []
    for quality in qualities:
        original_bpp = interpolate_rate(original, quality)
        lowrate_bpp = interpolate_rate(lowrate, quality)
        final_bpp = interpolate_rate(final, quality)
        if original_bpp is None or lowrate_bpp is None or final_bpp is None:
            continue
        rows.append(
            {
                "psnr_db": quality,
                "original_bpp": original_bpp,
                "lowrate_base_bpp": lowrate_bpp,
                "final_bpp": final_bpp,
                "lowrate_vs_original_bpp_change_pct": (lowrate_bpp / original_bpp - 1.0)
                * 100.0,
                "final_vs_original_bpp_change_pct": (final_bpp / original_bpp - 1.0)
                * 100.0,
            }
        )
    return rows


def matched_bpp_comparison(
    original: list[dict[str, Any]], candidate: list[dict[str, Any]]
) -> tuple[list[dict[str, float]], dict[str, float]]:
    original_rates = [float(row[RATE_KEY]) for row in original]
    candidate_rates = [float(row[RATE_KEY]) for row in candidate]
    overlap_min = max(min(original_rates), min(candidate_rates))
    overlap_max = min(max(original_rates), max(candidate_rates))

    requested = [overlap_min, 0.3, 0.4, 0.5, 1.0, 2.0, 5.0, 10.0, 15.0, 30.0, overlap_max]
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
                "final_psnr_db": candidate_quality,
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
        "interpolation": "piecewise linear PSNR over log(BPP)",
        "overlap_min_bpp": overlap_min,
        "overlap_max_bpp": overlap_max,
        "mean_psnr_gain_db_log_bpp": float(
            np.trapezoid(gains, log_rates) / (log_rates[-1] - log_rates[0])
        ),
        "minimum_psnr_gain_db_at_union_knots": float(gains.min()),
        "maximum_psnr_gain_db_at_union_knots": float(gains.max()),
    }
    return comparison, summary


def plot(curves: dict[str, list[dict[str, Any]]], output: Path) -> None:
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
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 5.5), gridspec_kw={"width_ratios": [1.0, 1.08]})
    for axis in axes:
        for label, rows in curves.items():
            color, marker, linestyle, width = STYLES[label]
            ordered = sorted(rows, key=lambda row: float(row[RATE_KEY]))
            axis.plot(
                [row[RATE_KEY] for row in ordered],
                [row[QUALITY_KEY] for row in ordered],
                label=label,
                color=color,
                marker=marker,
                linestyle=linestyle,
                linewidth=width,
                markersize=6.4 if label == "CAESAR-V decoder FT" else 4.8,
                markeredgecolor="white",
                markeredgewidth=0.65,
                zorder=7 if label == "CAESAR-V decoder FT" else 3,
            )
        axis.grid(True, which="major", color="#D1D5DB", linewidth=0.7, alpha=0.78)
        axis.spines[["top", "right"]].set_visible(False)
        axis.set_xlabel("BPP including required side information")

    axes[0].set_xscale("log")
    axes[0].set_xlim(0.018, 35)
    axes[0].set_ylabel("Dataset-normalized PSNR (dB)")
    axes[0].set_title("Full rate-distortion range")

    axes[1].set_xlim(0.015, 0.32)
    axes[1].set_ylim(30, 46)
    axes[1].set_title("DCAE/HPCM transition region")

    dcae_end = max(curves["DCAE"], key=lambda row: float(row[RATE_KEY]))
    axes[1].annotate(
        "DCAE endpoint",
        xy=(dcae_end[RATE_KEY], dcae_end[QUALITY_KEY]),
        xytext=(12, -24),
        textcoords="offset points",
        color=STYLES["DCAE"][0],
        fontsize=8.5,
        arrowprops={"arrowstyle": "-", "color": STYLES["DCAE"][0], "lw": 0.8},
    )

    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.01),
        ncol=3,
        frameon=False,
        columnspacing=1.2,
        handlelength=2.6,
    )
    fig.suptitle("ERA5 Objective-v1: CAESAR-V Decoder Fine-tuning", fontsize=14)
    fig.subplots_adjust(left=0.07, right=0.99, top=0.84, bottom=0.25, wspace=0.17)
    fig.savefig(output / "caesar_v_decoder_finetune_vs_image_codecs.png", dpi=300, bbox_inches="tight")
    fig.savefig(output / "caesar_v_decoder_finetune_vs_image_codecs.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_caesar_original_comparison(
    curves: dict[str, list[dict[str, Any]]], output: Path
) -> None:
    labels = ["CAESAR-V original", "CAESAR-V low-rate base", "CAESAR-V decoder FT"]
    fig = plt.figure(figsize=(13.4, 6.2))
    grid = fig.add_gridspec(
        2,
        2,
        width_ratios=[1.02, 0.98],
        height_ratios=[1.08, 0.92],
        hspace=0.34,
        wspace=0.20,
    )
    full_axis = fig.add_subplot(grid[:, 0])
    zoom_axis = fig.add_subplot(grid[0, 1])
    gain_axis = fig.add_subplot(grid[1, 1])

    for axis in (full_axis, zoom_axis):
        for label in labels:
            rows = sorted(curves[label], key=lambda row: float(row[RATE_KEY]))
            color, marker, linestyle, width = STYLES[label]
            axis.plot(
                [row[RATE_KEY] for row in rows],
                [row[QUALITY_KEY] for row in rows],
                label=label,
                color=color,
                marker=marker,
                linestyle=linestyle,
                linewidth=width,
                markersize=6.5 if label == "CAESAR-V decoder FT" else 5.2,
                markeredgecolor="white",
                markeredgewidth=0.65,
                zorder=6 if label == "CAESAR-V decoder FT" else 3,
            )
        axis.set_xscale("log")
        axis.grid(True, which="major", color="#D1D5DB", linewidth=0.7, alpha=0.78)
        axis.grid(True, which="minor", color="#E5E7EB", linewidth=0.45, alpha=0.62)
        axis.spines[["top", "right"]].set_visible(False)
        axis.set_xlabel("BPP including required side information")

    full_axis.set_xlim(0.09, 35)
    full_axis.set_ylabel("Dataset-normalized PSNR (dB)")
    full_axis.set_title("Complete CAESAR-V RD curves")

    zoom_axis.set_xlim(0.09, 2.7)
    zoom_axis.set_ylim(35, 65)
    zoom_axis.set_ylabel("Dataset-normalized PSNR (dB)")
    zoom_axis.set_title("Low- and mid-rate detail")

    original = curves["CAESAR-V original"]
    for label in labels[1:]:
        overlap_min = max(
            min(float(row[RATE_KEY]) for row in original),
            min(float(row[RATE_KEY]) for row in curves[label]),
        )
        overlap_max = min(
            max(float(row[RATE_KEY]) for row in original),
            max(float(row[RATE_KEY]) for row in curves[label]),
        )
        rate_grid = np.geomspace(overlap_min, overlap_max, 320)
        gains = np.asarray(
            [
                float(interpolate_quality(curves[label], rate))
                - float(interpolate_quality(original, rate))
                for rate in rate_grid
            ]
        )
        color, _, linestyle, width = STYLES[label]
        gain_axis.plot(
            rate_grid,
            gains,
            label=label,
            color=color,
            linestyle=linestyle,
            linewidth=width,
        )
    gain_axis.axhline(0, color="#6B7280", linewidth=0.9)
    gain_axis.set_xscale("log")
    gain_axis.set_xlabel("Matched BPP")
    gain_axis.set_ylabel("PSNR gain vs original (dB)")
    gain_axis.set_title("Matched-BPP quality gain")
    gain_axis.grid(True, which="major", color="#D1D5DB", linewidth=0.7, alpha=0.78)
    gain_axis.grid(True, which="minor", color="#E5E7EB", linewidth=0.45, alpha=0.62)
    gain_axis.spines[["top", "right"]].set_visible(False)

    handles, legend_labels = full_axis.get_legend_handles_labels()
    fig.legend(
        handles,
        legend_labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.01),
        ncol=3,
        frameon=False,
        columnspacing=1.4,
        handlelength=2.8,
    )
    fig.suptitle("ERA5 Objective-v1: Complete CAESAR-V Comparison", fontsize=14)
    fig.subplots_adjust(left=0.07, right=0.99, top=0.88, bottom=0.16)
    fig.savefig(output / "caesar_v_decoder_finetune_vs_original.png", dpi=300, bbox_inches="tight")
    fig.savefig(output / "caesar_v_decoder_finetune_vs_original.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in load_rows(args.baseline):
        name = baseline_name(row)
        if name is not None:
            grouped[name].append(dict(row))

    lowrate_standard_rows = selected_caesar_rows(
        load_rows(args.lowrate_base), "v_lowrate_lam1em3_100k", "low-rate base"
    )
    lowrate_probe_rows = selected_caesar_rows(
        load_rows(args.lowrate_probe), "lowrate_base_100k", "low-rate base probe"
    )
    lowrate_rows = merge_by_eb(lowrate_standard_rows, lowrate_probe_rows)
    final_rows = selected_caesar_rows(
        load_rows(args.final), args.final_variant, "final decoder fine-tune"
    )
    assert_expected_final_ebs(final_rows)
    curves = {
        "DCAE": grouped["DCAE"],
        "LIC-HPCM-base": grouped["LIC-HPCM-base"],
        "LIC-HPCM-large": grouped["LIC-HPCM-large"],
        "CAESAR-V original": grouped["CAESAR-V original"],
        "CAESAR-V low-rate base": lowrate_rows,
        "CAESAR-V decoder FT": final_rows,
    }
    if any(not rows for rows in curves.values()):
        missing = [name for name, rows in curves.items() if not rows]
        raise ValueError(f"Missing curves: {missing}")

    args.output.mkdir(parents=True, exist_ok=True)
    combined = []
    for label, rows in curves.items():
        for row in rows:
            item = dict(row)
            item["comparison_curve"] = label
            combined.append(item)
    (args.output / "summary.json").write_text(json.dumps(combined, indent=2), encoding="utf-8")

    image_endpoints = {
        label: max(rows, key=lambda row: float(row[RATE_KEY]))
        for label, rows in curves.items()
        if label in {"DCAE", "LIC-HPCM-base", "LIC-HPCM-large"}
    }
    highest_image_quality = max(float(row[QUALITY_KEY]) for row in image_endpoints.values())
    extension_point = next(
        row
        for row in sorted(final_rows, key=lambda row: float(row[RATE_KEY]))
        if float(row[QUALITY_KEY]) > highest_image_quality
    )
    gains = common_eb_gains(lowrate_rows, final_rows)
    original_to_final = common_eb_comparison(
        curves["CAESAR-V original"], final_rows
    )
    matched_quality = matched_quality_comparison(
        curves["CAESAR-V original"],
        lowrate_rows,
        final_rows,
        [38.0, 40.0, 45.0, 50.0, 60.0, 80.0, 110.0, 150.0, 168.0],
    )
    matched_bpp, matched_bpp_summary = matched_bpp_comparison(
        curves["CAESAR-V original"], final_rows
    )
    manifest = {
        "protocol_id": "aifc-objective-v1",
        "dataset": "era5_npy",
        "baseline_summary": str(args.baseline.resolve()),
        "lowrate_base_summary": str(args.lowrate_base.resolve()),
        "lowrate_probe_summary": str(args.lowrate_probe.resolve()),
        "final_summary": str(args.final.resolve()),
        "final_checkpoint_variant": args.final_variant,
        "final_checkpoint": final_rows[0].get("checkpoint_root"),
        "final_eb_values": EXPECTED_FINAL_EBS,
        "monotonic": {label: is_monotonic(rows) for label, rows in curves.items()},
        "image_codec_endpoints": {
            label: {"bpp": row[RATE_KEY], "psnr_db": row[QUALITY_KEY]}
            for label, row in image_endpoints.items()
        },
        "first_final_point_above_highest_image_endpoint_psnr": {
            "eb": extension_point["eb"],
            "bpp": extension_point[RATE_KEY],
            "psnr_db": extension_point[QUALITY_KEY],
        },
        "final_quality_at_dcae_max_bpp_db": interpolate_quality(
            final_rows, float(image_endpoints["DCAE"][RATE_KEY])
        ),
        "common_eb_base_to_final": gains,
        "common_eb_original_to_final": original_to_final,
        "matched_quality_original_comparison": matched_quality,
        "matched_bpp_original_comparison": matched_bpp,
        "matched_bpp_summary": matched_bpp_summary,
        "mean_psnr_gain_at_common_eb_db": float(np.mean([row["psnr_gain_db"] for row in gains])),
        "low_rate_mean_psnr_gain_at_common_eb_db": float(
            np.mean([row["psnr_gain_db"] for row in gains if row["final_bpp"] <= 0.3])
        ),
        "comparison_scope": (
            "A final point above and to the right of the image-codec endpoints establishes a "
            "high-quality extension. It is not evidence of equal-rate dominance; use the "
            "interpolated quality at the DCAE endpoint BPP for that comparison."
        ),
        "timing_status": "RD-only: 0 warmups and 1 repetition; excluded from throughput ranking",
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    plot(curves, args.output)
    plot_caesar_original_comparison(curves, args.output)


if __name__ == "__main__":
    main()
