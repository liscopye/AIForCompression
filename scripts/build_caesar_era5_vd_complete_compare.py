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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot complete ERA5 CAESAR-V/D curves against their original weights."
    )
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--v-final", type=Path, required=True)
    parser.add_argument("--v-variant", required=True)
    parser.add_argument("--d-original", type=Path, required=True)
    parser.add_argument("--d-final", type=Path, required=True)
    parser.add_argument("--d-variant", required=True)
    parser.add_argument("--d-secondary", type=Path)
    parser.add_argument("--d-secondary-variant")
    parser.add_argument("--d-keyframe", type=Path, required=True)
    parser.add_argument("--d-keyframe-variant", required=True)
    parser.add_argument("--d-ensemble", type=Path)
    parser.add_argument("--d-ensemble-variant")
    parser.add_argument("--d-lowrate-ensemble", type=Path)
    parser.add_argument("--d-lowrate-ensemble-variant")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    for path_name, variant_name in (
        ("d_secondary", "d_secondary_variant"),
        ("d_ensemble", "d_ensemble_variant"),
        ("d_lowrate_ensemble", "d_lowrate_ensemble_variant"),
    ):
        if (getattr(args, path_name) is None) != (getattr(args, variant_name) is None):
            parser.error(
                f"--{path_name.replace('_', '-')} and "
                f"--{variant_name.replace('_', '-')} must be provided together"
            )
    return args


def load_rows(path: Path) -> list[dict[str, Any]]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise ValueError(f"Expected a row list in {path}")
    return rows


def quality(row: dict[str, Any]) -> float:
    value = row.get("normalized_psnr")
    if value is None:
        value = row["psnr"]
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


def select(
    rows: list[dict[str, Any]],
    predicate: Callable[[dict[str, Any]], bool],
    label: str,
) -> list[dict[str, Any]]:
    selected = [dict(row) for row in rows if valid(row) and predicate(row)]
    selected.sort(key=lambda row: float(row[RATE_KEY]))
    if not selected:
        raise ValueError(f"No valid rows found for {label}")
    return selected


def interpolate_quality(rows: list[dict[str, Any]], rate: float) -> float:
    rates = np.asarray([float(row[RATE_KEY]) for row in rows])
    qualities = np.asarray([quality(row) for row in rows])
    if not rates[0] <= rate <= rates[-1]:
        raise ValueError(f"BPP {rate} is outside [{rates[0]}, {rates[-1]}]")
    return float(np.interp(math.log(rate), np.log(rates), qualities))


def matched_bpp(
    original: list[dict[str, Any]], candidate: list[dict[str, Any]]
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    overlap_min = max(float(original[0][RATE_KEY]), float(candidate[0][RATE_KEY]))
    overlap_max = min(float(original[-1][RATE_KEY]), float(candidate[-1][RATE_KEY]))
    if overlap_min >= overlap_max:
        raise ValueError("Original and candidate curves do not overlap in BPP")

    original_rates = [float(row[RATE_KEY]) for row in original]
    candidate_rates = [float(row[RATE_KEY]) for row in candidate]
    knots = sorted(
        {
            overlap_min,
            overlap_max,
            *(rate for rate in original_rates if overlap_min < rate < overlap_max),
            *(rate for rate in candidate_rates if overlap_min < rate < overlap_max),
        }
    )
    log_knots = np.log(knots)
    knot_gains = np.asarray(
        [
            interpolate_quality(candidate, rate) - interpolate_quality(original, rate)
            for rate in knots
        ]
    )
    rate_grid = np.geomspace(overlap_min, overlap_max, 500)
    gain_grid = np.asarray(
        [
            interpolate_quality(candidate, rate) - interpolate_quality(original, rate)
            for rate in rate_grid
        ]
    )
    summary = {
        "overlap_min_bpp": overlap_min,
        "overlap_max_bpp": overlap_max,
        "mean_psnr_gain_db_log_bpp": float(
            np.trapezoid(knot_gains, log_knots) / (log_knots[-1] - log_knots[0])
        ),
        "minimum_psnr_gain_db_at_union_knots": float(knot_gains.min()),
        "maximum_psnr_gain_db_at_union_knots": float(knot_gains.max()),
    }
    return rate_grid, gain_grid, summary


def matched_bpp_frontier(
    original: list[dict[str, Any]], candidates: list[list[dict[str, Any]]]
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    overlap_min = max(
        float(original[0][RATE_KEY]),
        min(float(candidate[0][RATE_KEY]) for candidate in candidates),
    )
    overlap_max = min(
        float(original[-1][RATE_KEY]),
        max(float(candidate[-1][RATE_KEY]) for candidate in candidates),
    )
    rate_grid = np.geomspace(overlap_min, overlap_max, 5000)

    def frontier_quality(rate: float) -> float:
        available = [
            interpolate_quality(candidate, rate)
            for candidate in candidates
            if float(candidate[0][RATE_KEY]) <= rate <= float(candidate[-1][RATE_KEY])
        ]
        if not available:
            raise ValueError(f"No candidate covers frontier BPP {rate}")
        return max(available)

    gain_grid = np.asarray(
        [
            frontier_quality(rate) - interpolate_quality(original, rate)
            for rate in rate_grid
        ]
    )
    summary = {
        "overlap_min_bpp": overlap_min,
        "overlap_max_bpp": overlap_max,
        "mean_psnr_gain_db_log_bpp": float(
            np.trapezoid(gain_grid, np.log(rate_grid))
            / (math.log(overlap_max) - math.log(overlap_min))
        ),
        "minimum_psnr_gain_db_on_dense_grid": float(gain_grid.min()),
        "maximum_psnr_gain_db_on_dense_grid": float(gain_grid.max()),
        "grid_points": len(rate_grid),
    }
    return rate_grid, gain_grid, summary


def monotonic(rows: list[dict[str, Any]]) -> bool:
    return all(
        float(right[RATE_KEY]) > float(left[RATE_KEY])
        and quality(right) >= quality(left)
        for left, right in zip(rows, rows[1:])
    )


def plot_curve(
    axis: plt.Axes,
    rows: list[dict[str, Any]],
    label: str,
    color: str,
    marker: str,
    linestyle: str,
    linewidth: float,
    zorder: int,
) -> None:
    axis.plot(
        [float(row[RATE_KEY]) for row in rows],
        [quality(row) for row in rows],
        label=label,
        color=color,
        marker=marker,
        linestyle=linestyle,
        linewidth=linewidth,
        markersize=5.2,
        markeredgecolor="white",
        markeredgewidth=0.7,
        zorder=zorder,
    )


def style_axis(axis: plt.Axes) -> None:
    axis.set_xscale("log")
    axis.grid(True, which="major", color="#CBD5E1", linewidth=0.7, alpha=0.78)
    axis.grid(True, which="minor", color="#E2E8F0", linewidth=0.45, alpha=0.62)
    axis.spines[["top", "right"]].set_visible(False)


def plot(
    image_curves: dict[str, list[dict[str, Any]]],
    v_original: list[dict[str, Any]],
    v_final: list[dict[str, Any]],
    d_original: list[dict[str, Any]],
    d_final: list[dict[str, Any]],
    d_secondary: list[dict[str, Any]] | None,
    d_keyframe: list[dict[str, Any]],
    d_additional: list[tuple[list[dict[str, Any]], str, str, str, str]],
    v_gain: tuple[np.ndarray, np.ndarray],
    d_gain: tuple[np.ndarray, np.ndarray],
    d_keyframe_gain: tuple[np.ndarray, np.ndarray],
    summaries: dict[str, dict[str, float]],
    output: Path,
) -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9.2,
            "axes.titlesize": 11.2,
            "axes.labelsize": 10.2,
            "legend.fontsize": 8.2,
            "axes.linewidth": 0.9,
            "lines.solid_capstyle": "round",
        }
    )
    fig, axes = plt.subplots(
        2,
        3,
        figsize=(15.8, 8.4),
        gridspec_kw={"width_ratios": [1.1, 0.95, 0.9], "hspace": 0.34, "wspace": 0.24},
    )

    image_styles = {
        "DCAE": ("#117733", "o"),
        "LIC-HPCM-base": ("#44AA99", "D"),
        "LIC-HPCM-large": ("#88CCEE", "^"),
    }
    family_rows = [
        (
            "CAESAR-V",
            v_original,
            v_final,
            "#6B7280",
            "#CC6677",
            v_gain,
            summaries["caesar_v"],
        ),
        (
            "CAESAR-D",
            d_original,
            d_final,
            "#6B7280",
            "#4477AA",
            d_gain,
            summaries["caesar_d"],
        ),
    ]

    for row_index, (
        family,
        original,
        final,
        original_color,
        final_color,
        gain,
        summary,
    ) in enumerate(family_rows):
        full_axis, zoom_axis, gain_axis = axes[row_index]
        for axis in (full_axis, zoom_axis):
            for label, rows in image_curves.items():
                color, marker = image_styles[label]
                plot_curve(axis, rows, label, color, marker, "-", 1.55, 2)
            plot_curve(
                axis,
                original,
                f"{family} original",
                original_color,
                "o",
                "--",
                2.0,
                4,
            )
            final_label = f"{family} tuned" if row_index == 0 else "CAESAR-D tuned (5k)"
            plot_curve(axis, final, final_label, final_color, "X", "-", 2.7, 6)
            if row_index == 1:
                if d_secondary is not None:
                    plot_curve(
                        axis,
                        d_secondary,
                        "CAESAR-D tuned (secondary)",
                        "#EE7733",
                        "s",
                        "--",
                        2.0,
                        5,
                    )
                for rows, label, color, marker, linestyle in d_additional:
                    plot_curve(
                        axis,
                        rows,
                        label,
                        color,
                        marker,
                        linestyle,
                        2.25,
                        6,
                    )
                plot_curve(
                    axis,
                    d_keyframe,
                    "CAESAR-D keyframe-only (ablation)",
                    "#AA4499",
                    "P",
                    "-.",
                    2.2,
                    5,
                )
            style_axis(axis)
            axis.set_xlabel("BPP including side information")

        full_axis.set_xlim(0.06, 35)
        full_axis.set_ylim(18, 174)
        full_axis.set_ylabel("Dataset-normalized PSNR (dB)")
        full_axis.set_title(f"{family}: complete RD curve")

        zoom_axis.set_xlim(0.06, 0.7)
        zoom_axis.set_ylim(20, 48)
        zoom_axis.set_ylabel("Dataset-normalized PSNR (dB)")
        zoom_axis.set_title("Low-rate transition")

        gain_rates, gain_values = gain
        gain_axis.plot(gain_rates, gain_values, color=final_color, linewidth=2.5)
        if row_index == 1:
            keyframe_rates, keyframe_values = d_keyframe_gain
            gain_axis.plot(
                keyframe_rates,
                keyframe_values,
                color="#AA4499",
                linewidth=2.1,
                linestyle="-.",
                label="keyframe-only ablation",
            )
        gain_axis.axhline(0, color="#64748B", linewidth=0.9)
        gain_axis.fill_between(
            gain_rates,
            0,
            gain_values,
            where=gain_values >= 0,
            color=final_color,
            alpha=0.18,
            interpolate=True,
        )
        gain_axis.fill_between(
            gain_rates,
            0,
            gain_values,
            where=gain_values < 0,
            color="#AA4499",
            alpha=0.16,
            interpolate=True,
        )
        style_axis(gain_axis)
        gain_axis.set_xlim(summary["overlap_min_bpp"], summary["overlap_max_bpp"])
        gain_axis.set_xlabel("Matched BPP")
        gain_axis.set_ylabel("PSNR gain vs original (dB)")
        if row_index == 0:
            gain_axis.set_title(
                f"Same-BPP gain: mean {summary['mean_psnr_gain_db_log_bpp']:+.3f} dB"
            )
        else:
            keyframe_mean = summaries["caesar_d_keyframe"][
                "mean_psnr_gain_db_log_bpp"
            ]
            d_gain_label = (
                "formal frontier"
                if d_secondary is not None or d_additional
                else "complete candidate"
            )
            gain_axis.set_title(
                "Same-BPP gain: "
                f"{d_gain_label} {summary['mean_psnr_gain_db_log_bpp']:+.3f}, "
                f"ablation {keyframe_mean:+.3f} dB"
            )

    handles, labels = axes[0, 0].get_legend_handles_labels()
    d_handles, d_labels = axes[1, 0].get_legend_handles_labels()
    for handle, label in zip(d_handles, d_labels):
        if label not in labels:
            handles.append(handle)
            labels.append(label)
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.012),
        ncol=5,
        frameon=False,
        columnspacing=1.25,
        handlelength=2.7,
    )
    fig.suptitle("ERA5 Objective-v1: Complete CAESAR vs Original", fontsize=14.5)
    fig.subplots_adjust(left=0.065, right=0.99, top=0.91, bottom=0.115)
    fig.savefig(output / "caesar_vd_complete_vs_original.png", dpi=300, bbox_inches="tight")
    fig.savefig(output / "caesar_vd_complete_vs_original.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    baseline = load_rows(args.baseline)
    v_final_source = load_rows(args.v_final)
    d_original_source = load_rows(args.d_original)
    d_final_source = load_rows(args.d_final)
    d_secondary_source = load_rows(args.d_secondary) if args.d_secondary is not None else None
    d_keyframe_source = load_rows(args.d_keyframe)

    image_curves = {
        "DCAE": select(
            baseline,
            lambda row: row.get("dataset_id") == "era5_npy"
            and row.get("model_name") == "DCAE",
            "DCAE",
        ),
        "LIC-HPCM-base": select(
            baseline,
            lambda row: row.get("dataset_id") == "era5_npy"
            and row.get("model_name") == "LIC-HPCM"
            and "base" in str(row.get("model_id", "")).lower(),
            "LIC-HPCM-base",
        ),
        "LIC-HPCM-large": select(
            baseline,
            lambda row: row.get("dataset_id") == "era5_npy"
            and row.get("model_name") == "LIC-HPCM"
            and "large" in str(row.get("model_id", "")).lower(),
            "LIC-HPCM-large",
        ),
    }
    v_original = select(
        baseline,
        lambda row: row.get("dataset_id") == "era5_npy"
        and str(row.get("model_id", "")).lower().startswith("caesar_v")
        and str(row.get("checkpoint_variant", "")).lower() in {"", "none", "original"},
        "CAESAR-V original",
    )
    v_final = select(
        v_final_source,
        lambda row: row.get("dataset_id") == "era5_npy"
        and str(row.get("model_id", "")).lower().startswith("caesar_v")
        and str(row.get("checkpoint_variant", "")).lower() == args.v_variant.lower(),
        "CAESAR-V tuned",
    )
    d_original = select(
        d_original_source,
        lambda row: row.get("dataset_id") == "era5_npy"
        and str(row.get("model_id", "")).lower().startswith("caesar_d")
        and str(row.get("checkpoint_variant", "")).lower() == "original",
        "CAESAR-D original",
    )
    d_final = select(
        d_final_source,
        lambda row: row.get("dataset_id") == "era5_npy"
        and str(row.get("model_id", "")).lower().startswith("caesar_d")
        and str(row.get("checkpoint_variant", "")).lower() == args.d_variant.lower(),
        "CAESAR-D tuned",
    )
    d_secondary = (
        select(
            d_secondary_source,
            lambda row: row.get("dataset_id") == "era5_npy"
            and str(row.get("model_id", "")).lower().startswith("caesar_d")
            and str(row.get("checkpoint_variant", "")).lower()
            == args.d_secondary_variant.lower(),
            "CAESAR-D tuned secondary",
        )
        if d_secondary_source is not None and args.d_secondary_variant is not None
        else None
    )
    d_keyframe = select(
        d_keyframe_source,
        lambda row: row.get("dataset_id") == "era5_npy"
        and str(row.get("model_id", "")).lower().startswith("caesar_d")
        and str(row.get("checkpoint_variant", "")).lower()
        == args.d_keyframe_variant.lower(),
        "CAESAR-D keyframe-only ablation",
    )

    d_additional: list[tuple[list[dict[str, Any]], str, str, str, str]] = []
    d_additional_manifest: dict[str, dict[str, Any]] = {}
    optional_specs = (
        (
            "d_ensemble",
            args.d_ensemble,
            args.d_ensemble_variant,
            "CAESAR-D tuned ensemble-4",
            "#009988",
            "D",
            "-",
        ),
        (
            "d_lowrate_ensemble",
            args.d_lowrate_ensemble,
            args.d_lowrate_ensemble_variant,
            "CAESAR-D low-rate ensemble-4",
            "#AA3377",
            "v",
            "--",
        ),
    )
    for key, path, variant, label, color, marker, linestyle in optional_specs:
        if path is None or variant is None:
            continue
        rows = select(
            load_rows(path),
            lambda row, expected=variant: row.get("dataset_id") == "era5_npy"
            and str(row.get("model_id", "")).lower().startswith("caesar_d")
            and str(row.get("checkpoint_variant", "")).lower() == expected.lower(),
            label,
        )
        d_additional.append((rows, label, color, marker, linestyle))
        _, _, individual_summary = matched_bpp(d_original, rows)
        d_additional_manifest[key] = {
            "source": str(path.resolve()),
            "variant": variant,
            "point_count": len(rows),
            "monotonic": monotonic(rows),
            "matched_bpp_summary": individual_summary,
        }

    v_rates, v_gains, v_summary = matched_bpp(v_original, v_final)
    d_rates, d_gains, d_summary = matched_bpp_frontier(
        d_original,
        [
            d_final,
            *([d_secondary] if d_secondary is not None else []),
            *(rows for rows, *_ in d_additional),
        ],
    )
    d_keyframe_rates, d_keyframe_gains, d_keyframe_summary = matched_bpp(
        d_original, d_keyframe
    )
    summaries = {
        "caesar_v": v_summary,
        "caesar_d": d_summary,
        "caesar_d_keyframe": d_keyframe_summary,
    }

    args.output.mkdir(parents=True, exist_ok=True)
    manifest = {
        "protocol_id": "aifc-objective-v1",
        "dataset": "era5_npy",
        "sources": {
            "baseline": str(args.baseline.resolve()),
            "v_final": str(args.v_final.resolve()),
            "d_original": str(args.d_original.resolve()),
            "d_final": str(args.d_final.resolve()),
            **(
                {"d_secondary": str(args.d_secondary.resolve())}
                if args.d_secondary is not None
                else {}
            ),
            "d_keyframe": str(args.d_keyframe.resolve()),
        },
        "variants": {
            "v_final": args.v_variant,
            "d_final": args.d_variant,
            **(
                {"d_secondary": args.d_secondary_variant}
                if args.d_secondary_variant is not None
                else {}
            ),
            "d_keyframe": args.d_keyframe_variant,
        },
        "curve_point_counts": {
            "v_original": len(v_original),
            "v_final": len(v_final),
            "d_original": len(d_original),
            "d_final": len(d_final),
            **({"d_secondary": len(d_secondary)} if d_secondary is not None else {}),
            "d_keyframe": len(d_keyframe),
        },
        "monotonic": {
            "v_original": monotonic(v_original),
            "v_final": monotonic(v_final),
            "d_original": monotonic(d_original),
            "d_final": monotonic(d_final),
            **(
                {"d_secondary": monotonic(d_secondary)}
                if d_secondary is not None
                else {}
            ),
            "d_keyframe": monotonic(d_keyframe),
        },
        "matched_bpp_summary": summaries,
        "additional_d_curves": d_additional_manifest,
        "interpolation": "piecewise linear PSNR over log(BPP)",
    }
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    plot(
        image_curves,
        v_original,
        v_final,
        d_original,
        d_final,
        d_secondary,
        d_keyframe,
        d_additional,
        (v_rates, v_gains),
        (d_rates, d_gains),
        (d_keyframe_rates, d_keyframe_gains),
        summaries,
        args.output,
    )
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
