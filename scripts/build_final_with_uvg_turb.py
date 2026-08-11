#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OLD_FINAL = ROOT / "unified_results/final/all_models_fullstack_cuszhi_nvjpeg_pca_pcaanchored_nozero_zoom6"
BASE_FINAL = ROOT / "unified_results/final/no_pca_cuszhi3d_packz_full_n1_nvjpeg_lpips"
PREFIX_SUFFIX = "all_models_fullstack_cuszhi_nvjpeg_pca_pcaanchored_nozero_zoom6_uvg_turb"

OLD_DATASETS = [
    ("e3sm_npz", "E3SM"),
    ("era5_npy", "ERA5"),
    ("hurricane", "Hurricane"),
    ("kodak", "Kodak"),
    ("lysozyme", "Lysozyme"),
    ("nyx", "NYX"),
    ("s2c", "S2C"),
    ("tomo", "Tomo"),
]

NEW_DATASETS = [
    ("turb_rot_npz", "Turb_Rot"),
    ("uvg_twilight_1080p", "UVG Twilight 1080p"),
]

DATASET_CATEGORIES = {
    "uvg_twilight_1080p": ("general", "General Image / Video"),
    "kodak": ("general", "General Image / Video"),
    "tomo": ("scientific_images", "Scientific Images"),
    "s2c": ("scientific_images", "Scientific Images"),
    "lysozyme": ("scientific_images", "Scientific Images"),
    "era5_npy": ("scientific_fields", "Scientific Fields"),
    "nyx": ("scientific_fields", "Scientific Fields"),
    "hurricane": ("scientific_fields", "Scientific Fields"),
    "turb_rot_npz": ("scientific_fields", "Scientific Fields"),
    "e3sm_npz": ("scientific_fields", "Scientific Fields"),
    "jhtd": ("scientific_fields", "Scientific Fields"),
}

CATEGORY_ORDER = {
    "general": 0,
    "scientific_images": 1,
    "scientific_fields": 2,
    "uncategorized": 99,
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default=str(ROOT / f"unified_results/final/{PREFIX_SUFFIX}"))
    parser.add_argument("--skip_plots", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    clean_generated_dataset_dirs(output_dir)

    built = []
    for dataset_id, title in OLD_DATASETS:
        source_dir = find_old_dataset_dir(dataset_id)
        if source_dir is None:
            print(f"[skip] missing old final dir for {dataset_id}")
            continue
        dest_dir = dataset_output_dir(output_dir, dataset_id)
        rows = load_json(source_dir / "summary_no_visemz_graphcomp.json")
        rows = replace_caesar_no_pca_lpips(dataset_id, rows)
        write_dataset(dest_dir, rows)
        write_json(dest_dir / "summary.json", rows)
        if not args.skip_plots:
            plot_dataset(dest_dir / "summary_no_visemz_graphcomp.json", dest_dir / "plots", f"{dataset_id}_{PREFIX_SUFFIX}", title)
        built.append(dataset_record(dataset_id, title))

    turb_rows = build_turb_rot()
    if turb_rows:
        dest_dir = dataset_output_dir(output_dir, "turb_rot_npz")
        write_dataset(dest_dir, turb_rows)
        if not args.skip_plots:
            plot_dataset(dest_dir / "summary_no_visemz_graphcomp.json", dest_dir / "plots", f"turb_rot_npz_{PREFIX_SUFFIX}", "Turb_Rot")
        built.append(dataset_record("turb_rot_npz", "Turb_Rot"))

    uvg_1080p_rows = build_uvg_1080p()
    if uvg_1080p_rows:
        dest_dir = dataset_output_dir(output_dir, "uvg_twilight_1080p")
        write_dataset(dest_dir, uvg_1080p_rows)
        if not args.skip_plots:
            plot_dataset(dest_dir / "summary_no_visemz_graphcomp.json", dest_dir / "plots", f"uvg_twilight_1080p_{PREFIX_SUFFIX}", "UVG Twilight 1080p")
        built.append(dataset_record("uvg_twilight_1080p", "UVG Twilight 1080p"))

    write_index(output_dir, built)
    print(f"wrote {len(built)} datasets to {output_dir}")


def dataset_record(dataset_id: str, title: str) -> tuple[str, str, str, str]:
    category_id, category_label = DATASET_CATEGORIES.get(dataset_id, ("uncategorized", "Uncategorized"))
    return dataset_id, title, category_id, category_label


def dataset_output_dir(output_dir: Path, dataset_id: str) -> Path:
    category_id, _ = DATASET_CATEGORIES.get(dataset_id, ("uncategorized", "Uncategorized"))
    return output_dir / category_id / f"{dataset_id}_{PREFIX_SUFFIX}"


def clean_generated_dataset_dirs(output_dir: Path) -> None:
    for dataset_id, _ in OLD_DATASETS + NEW_DATASETS:
        flat = output_dir / f"{dataset_id}_{PREFIX_SUFFIX}"
        if flat.exists():
            shutil.rmtree(flat)


def find_old_dataset_dir(dataset_id: str) -> Path | None:
    matches = sorted(OLD_FINAL.glob(f"{dataset_id}_*"))
    return matches[0] if matches else None


def copy_summary_only(source_dir: Path, dest_dir: Path) -> None:
    dest_dir.mkdir(parents=True, exist_ok=True)
    for name in ("summary_no_visemz_graphcomp.json", "summary.json"):
        src = source_dir / name
        if src.exists():
            shutil.copy2(src, dest_dir / name)


def replace_caesar_no_pca_lpips(dataset_id: str, rows: list[dict]) -> list[dict]:
    fill_paths = {
        "e3sm_npz": ROOT / "unified_results/no_pca_lpips_fill/e3sm_npz/summary.json",
        "era5_npy": ROOT / "unified_results/no_pca_lpips_fill/era5_npy/summary.json",
        "hurricane": ROOT / "unified_results/no_pca_lpips_fill/hurricane/summary.json",
        "kodak": ROOT / "unified_results/no_pca_lpips_fill/kodak/summary.json",
        "lysozyme": ROOT / "unified_results/no_pca_lpips_fill/lysozyme_raw_lpips/summary.json",
        "nyx": ROOT / "unified_results/no_pca_lpips_fill/nyx/summary.json",
    }
    fill = fill_paths.get(dataset_id)
    if fill is None or not fill.exists():
        return rows
    replacement = [
        r for r in load_json(fill)
        if r.get("model_name") == "CAESAR" and str(r.get("model_id", "")).endswith("_no_pca") and r.get("lpips") is not None
    ]
    if not replacement:
        return rows
    kept = [r for r in rows if not (r.get("model_name") == "CAESAR" and str(r.get("model_id", "")).endswith("_no_pca"))]
    for row in replacement:
        item = dict(row)
        item["label"] = caesar_no_pca_label(item)
        item["source"] = "caesar_no_pca_lpips_fill"
        kept.append(item)
    return kept


def build_turb_rot() -> list[dict]:
    base = BASE_FINAL / "turb_rot_npz_all_models_with_no_pca_and_cuszhi3d_packz_full_n1_with_nvjpeg_lpips/summary_no_visemz_graphcomp.json"
    pca = ROOT / "unified_results/experiments/image_codec_caesar_pca_auto_rightfill/turb_rot_npz/summary.json"
    caesar_no_pca_lpips = ROOT / "unified_results/turb_rot_full/caesar_no_pca_lpips/summary.json"
    if not base.exists():
        print(f"[skip] missing {base}")
        return []
    rows = load_json(base)
    if caesar_no_pca_lpips.exists():
        rows = [r for r in rows if not str(r.get("model_id", "")).endswith("_no_pca")]
        rows.extend(load_json(caesar_no_pca_lpips))
    if pca.exists():
        pca_rows = load_json(pca)
        rows.extend(pca_selected_rows(pca_rows))
        rows.extend(pca_anchor_rows(pca_rows))
    else:
        print(f"[warn] missing {pca}")
    return dedupe_rows(strip_vis_graph(rows))


def build_uvg() -> list[dict]:
    paths = [
        ROOT / "unified_results/uvg_twilight_full/all_models_image_video_n30/summary.json",
        ROOT / "unified_results/uvg_twilight_full/caesar_no_pca_n30/summary.json",
        ROOT / "unified_results/uvg_twilight_full/cuszhi3d_rgbstack_n30/summary.json",
    ]
    missing = [p for p in paths if not p.exists()]
    if missing:
        for path in missing:
            print(f"[skip] missing {path}")
        return []

    rows = []
    for path in paths:
        rows.extend(load_json(path))
    for path in [
        ROOT / "unified_results/uvg_twilight_full/cuszhi3d_rgbstack_n30_chunk2_loweb_probe/summary.json",
    ]:
        if path.exists():
            rows.extend(load_json(path))
    rows = aggregate_uvg_rows(rows)

    pca = ROOT / "unified_results/experiments/image_codec_caesar_pca_auto_rightfill/uvg_twilight/summary.json"
    if pca.exists():
        pca_rows = load_json(pca)
        rows.extend(pca_selected_rows(pca_rows))
        rows.extend(pca_anchor_rows(pca_rows))
    else:
        print(f"[warn] missing {pca}")
    for row in rows:
        row["dataset_id"] = "uvg_twilight"
    return dedupe_rows(strip_vis_graph(rows))


def build_uvg_1080p() -> list[dict]:
    base_dir = ROOT / "unified_results/uvg_twilight_1080p"
    paths = [
        base_dir / "all_models_image_video_n30/summary.json",
        base_dir / "caesar_pca_eb7_n30_merged/summary.json",
        base_dir / "caesar_no_pca_n30/summary.json",
        base_dir / "cuszhi3d_rgbstack_n30/summary.json",
        base_dir / "cuszhi3d_rgbstack_n30_eb001/summary.json",
        base_dir / "cuszhi3d_rgbstack_n30_eb0005/summary.json",
        base_dir / "cuszhi3d_rgbstack_n30_eb0002/summary.json",
        base_dir / "cuszhi3d_rgbstack_n30_eb0001/summary.json",
        base_dir / "video_intra_n30/summary.json",
        base_dir / "dcvc_rt_pframe_n30/summary.json",
        base_dir / "dcmvc_pframe_n30_memory_bitstream/summary.json",
    ]
    required = [
        base_dir / "all_models_image_video_n30/summary.json",
        base_dir / "video_intra_n30/summary.json",
        base_dir / "caesar_no_pca_n30/summary.json",
        base_dir / "dcvc_rt_pframe_n30/summary.json",
        base_dir / "dcmvc_pframe_n30_memory_bitstream/summary.json",
    ]
    missing = [p for p in required if not p.exists()]
    if missing:
        for path in missing:
            print(f"[skip] missing {path}")
        return []

    rows = []
    for path in paths:
        if path.exists():
            rows.extend(load_json(path))
    rows = aggregate_uvg_rows(rows)

    pca = ROOT / "unified_results/experiments/image_codec_caesar_pca_auto_rightfill/uvg_twilight_1080p/summary.json"
    if pca.exists():
        pca_rows = load_json(pca)
        rows.extend(pca_selected_rows(pca_rows))
        rows.extend(pca_anchor_rows(pca_rows))
    else:
        print(f"[warn] missing {pca}")
    for row in rows:
        row["dataset_id"] = "uvg_twilight_1080p"
    return dedupe_rows(strip_vis_graph(rows))


def aggregate_uvg_rows(rows: list[dict]) -> list[dict]:
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        if row.get("error"):
            continue
        item = dict(row)
        if str(item.get("model_id", "")).endswith("_no_pca"):
            item["eb"] = None
            item.setdefault("label", caesar_no_pca_label(item))
        key = (
            item.get("model_name"),
            item.get("model_id"),
            item.get("label"),
            item.get("quality"),
            item.get("eb"),
            item.get("pca_eb"),
            item.get("checkpoint"),
        )
        groups[key].append(item)
    return [aggregate_group(group) for _, group in sorted(groups.items(), key=lambda kv: str(kv[0]))]


def aggregate_group(group: list[dict]) -> dict:
    if len(group) == 1:
        return dict(group[0])
    first = dict(group[0])
    scalars = [scalar_count(r) for r in group]
    weights = [s if s and s > 0 else 1.0 for s in scalars]
    total_weight = sum(weights)

    out = dict(first)
    out["sample_count"] = len(group)
    out["success_count"] = len(group)
    out["error_count"] = 0
    out["sample_id"] = f"{first.get('model_id', first.get('model_name', 'uvg'))}_aggregate_n{len(group)}"
    out["dataset_id"] = "uvg_twilight"

    for key in ("bitstream_bytes", "side_info_bytes", "total_bytes_with_side_info", "original_bytes", "encode_time_total", "decode_time_total"):
        values = [r.get(key) for r in group if isinstance(r.get(key), (int, float))]
        if values:
            out[key] = float(sum(values))

    mse_values = [r.get("mse") for r in group]
    if all(isinstance(v, (int, float)) for v in mse_values):
        mse = sum(float(v) * w for v, w in zip(mse_values, weights)) / total_weight
        out["mse"] = mse
        out["rmse"] = math.sqrt(max(mse, 0.0))
        peak_sq = infer_peak_sq(group)
        if peak_sq and mse > 0:
            psnr = 10.0 * math.log10(peak_sq / mse)
            out["psnr"] = psnr
            out["average_frame_psnr"] = psnr
            out["average_variable_psnr"] = psnr

    scalar_total = sum(s for s in scalars if s)
    total_bytes = out.get("total_bytes_with_side_info", out.get("bitstream_bytes"))
    bitstream_bytes = out.get("bitstream_bytes")
    if scalar_total and isinstance(bitstream_bytes, (int, float)):
        out["bpp"] = float(bitstream_bytes) * 8.0 / scalar_total
        out["scientific_bpp"] = out["bpp"]
    if scalar_total and isinstance(total_bytes, (int, float)):
        out["scientific_bpp_with_side_info"] = float(total_bytes) * 8.0 / scalar_total
    if isinstance(out.get("original_bytes"), (int, float)) and isinstance(total_bytes, (int, float)) and total_bytes > 0:
        out["compression_ratio"] = float(out["original_bytes"]) / float(total_bytes)

    mean_fields = ["lpips", "memory_usage_MB", "memory_reserved_MB", "params"]
    for key in mean_fields:
        values = [r.get(key) for r in group if isinstance(r.get(key), (int, float))]
        if values:
            out[key] = float(sum(values) / len(values))

    for prefix in ("encode", "decode"):
        total_time = out.get(f"{prefix}_time_total")
        original_bytes = out.get("original_bytes")
        if isinstance(total_time, (int, float)) and total_time > 0 and isinstance(original_bytes, (int, float)):
            out[f"{prefix}_time_avg"] = float(total_time) / len(group)
            out[f"{prefix}_throughput"] = float(original_bytes) / float(total_time)
            out[f"{prefix}_throughput_MBps"] = out[f"{prefix}_throughput"] / 1e6

    return out


def scalar_count(row: dict) -> float | None:
    shape = row.get("shape") or row.get("input_shape")
    if isinstance(shape, list) and shape and all(isinstance(x, (int, float)) for x in shape):
        total = 1
        for dim in shape:
            total *= int(dim)
        return float(total)
    value = row.get("voxel_count")
    if isinstance(value, (int, float)):
        return float(value)
    original = row.get("original_bytes")
    if isinstance(original, (int, float)):
        bpp_value = row.get("bpp")
        bytes_value = row.get("bitstream_bytes")
        if isinstance(bpp_value, (int, float)) and bpp_value > 0 and isinstance(bytes_value, (int, float)):
            return float(bytes_value) * 8.0 / float(bpp_value)
    return None


def infer_peak_sq(rows: list[dict]) -> float | None:
    values = []
    for row in rows:
        mse = row.get("mse")
        psnr = row.get("psnr") or row.get("average_frame_psnr")
        if isinstance(mse, (int, float)) and mse > 0 and isinstance(psnr, (int, float)):
            values.append(float(mse) * (10.0 ** (float(psnr) / 10.0)))
    if not values:
        return None
    values.sort()
    return values[len(values) // 2]


def pca_selected_rows(rows: list[dict]) -> list[dict]:
    selected = []
    for row in rows:
        if not str(row.get("model_name", "")).endswith("+CAESAR-PCA"):
            continue
        if row.get("pca_bytes") == 0:
            continue
        item = dict(row)
        item["label"] = pca_label(item)
        selected.append(item)
    return selected


def pca_anchor_rows(rows: list[dict]) -> list[dict]:
    anchors = []
    for row in rows:
        if row.get("model_name") not in {"DCAE", "LIC-HPCM"}:
            continue
        label = pca_label(row)
        item = dict(row)
        item["model_name"] = f"{row.get('model_name')}+CAESAR-PCA"
        item["model_id"] = f"{row.get('model_id')}_pca_anchor_original"
        item["label"] = label
        item["pca_eb"] = None
        item["pca_bytes"] = 0
        item["base_bitstream_bytes"] = row.get("bitstream_bytes")
        item["pca_postprocess"] = "caesar_pca_anchor"
        anchors.append(item)
    return anchors


def pca_label(row: dict) -> str:
    model_name = str(row.get("model_name", ""))
    model_id = str(row.get("model_id", ""))
    label = str(row.get("label") or "")
    if label.endswith("+CAESAR-PCA"):
        return label
    if "DCAE" in model_name or model_id.startswith("DCAE"):
        return "DCAE+CAESAR-PCA"
    if "large" in model_id:
        return "HPCM-large+CAESAR-PCA"
    return "HPCM-base+CAESAR-PCA"


def caesar_no_pca_label(row: dict) -> str:
    model_id = str(row.get("model_id", ""))
    return "CAESAR-D no PCA" if model_id.startswith("caesar_d") else "CAESAR-V no PCA"


def strip_vis_graph(rows: list[dict]) -> list[dict]:
    return [r for r in rows if r.get("model_name") not in {"visemz", "GraphComp"}]


def dedupe_rows(rows: list[dict]) -> list[dict]:
    seen = set()
    out = []
    for row in rows:
        key = (
            row.get("model_name"),
            row.get("model_id"),
            row.get("label"),
            row.get("eb"),
            row.get("pca_eb"),
            row.get("quality"),
            row.get("bpp"),
            row.get("psnr"),
            row.get("average_frame_psnr"),
        )
        if key in seen:
            continue
        seen.add(key)
        out.append(row)
    return out


def write_dataset(dest_dir: Path, rows: list[dict]) -> None:
    dest_dir.mkdir(parents=True, exist_ok=True)
    write_json(dest_dir / "summary_no_visemz_graphcomp.json", rows)
    write_json(dest_dir / "summary.json", rows)


def plot_dataset(summary: Path, plot_dir: Path, prefix: str, title: str) -> None:
    subprocess.run(
        [
            "python",
            "scripts/plot_combined_external_results.py",
            "--summary",
            str(summary),
            "--output_dir",
            str(plot_dir),
            "--prefix",
            prefix,
            "--title",
            title,
            "--zoom_bpp",
            "6",
            "--extra_metrics",
        ],
        cwd=ROOT,
        check=True,
    )


def write_index(output_dir: Path, datasets: list[tuple[str, str, str, str]]) -> None:
    datasets = sorted(datasets, key=lambda item: (CATEGORY_ORDER.get(item[2], 99), item[1].lower()))
    dataset_js = json.dumps(
        [
            {"id": dataset_id, "label": title, "category": category_id, "categoryLabel": category_label}
            for dataset_id, title, category_id, category_label in datasets
        ],
        indent=6,
    )
    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Compression RD Results</title>
  <style>
    :root {{ color-scheme: light; --bg: #f7f8fa; --panel: #fff; --text: #171a1f; --muted: #667085; --line: #d9dee7; --accent: #0b6bcb; }}
    * {{ box-sizing: border-box; }}
    body {{ margin: 0; background: var(--bg); color: var(--text); font: 14px/1.45 system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }}
    header {{ position: sticky; top: 0; z-index: 10; background: rgba(247,248,250,.94); border-bottom: 1px solid var(--line); backdrop-filter: blur(10px); }}
    .wrap {{ max-width: 1440px; margin: 0 auto; padding: 18px 22px; }}
    h1 {{ margin: 0 0 4px; font-size: 22px; font-weight: 650; letter-spacing: 0; }}
    .subtitle, .note {{ color: var(--muted); font-size: 13px; }}
    .controls {{ display: grid; grid-template-columns: minmax(0,1fr) auto; gap: 14px; align-items: end; margin-top: 16px; }}
    .button-row {{ display: flex; flex-direction: column; gap: 10px; }}
    .category-row {{ display: flex; flex-wrap: wrap; gap: 8px; align-items: center; }}
    .category-label {{ min-width: 142px; color: var(--muted); font-size: 12px; font-weight: 650; text-transform: uppercase; letter-spacing: .03em; }}
    button, select {{ border: 1px solid var(--line); background: var(--panel); color: var(--text); border-radius: 7px; font: inherit; min-height: 36px; }}
    button {{ padding: 8px 10px; cursor: pointer; }}
    button:hover {{ border-color: #aeb8c7; }}
    button.active {{ border-color: var(--accent); background: #e8f2ff; color: #064d92; }}
    select {{ min-width: 190px; padding: 0 10px; }}
    main.wrap {{ padding-top: 18px; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(520px, 1fr)); gap: 18px; }}
    .figure {{ background: var(--panel); border: 1px solid var(--line); border-radius: 8px; overflow: hidden; }}
    .figure-header {{ display: flex; justify-content: space-between; gap: 12px; align-items: center; padding: 10px 12px; border-bottom: 1px solid var(--line); }}
    .figure-title {{ font-weight: 620; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }}
    .links {{ display: flex; gap: 10px; font-size: 12px; white-space: nowrap; }}
    a {{ color: var(--accent); text-decoration: none; }}
    a:hover {{ text-decoration: underline; }}
    .figure img {{ display: block; width: 100%; height: auto; background: #fff; }}
    .note {{ margin: 0 0 14px; }}
    @media (max-width: 760px) {{ .controls {{ grid-template-columns: 1fr; }} .grid {{ grid-template-columns: 1fr; }} .wrap {{ padding-left: 12px; padding-right: 12px; }} }}
  </style>
</head>
<body>
  <header>
    <div class="wrap">
      <h1>All-Model Compression Results</h1>
      <div class="subtitle">full-stack cuSZ-Hi-3D, nvJPEG/nvJPEG2000, DCAE/HPCM+CAESAR-PCA, no visemz/GraphComp, LPIPS enabled where evaluated</div>
      <div class="controls">
        <div><div class="button-row" id="datasetButtons"></div></div>
        <select id="viewSelect" aria-label="View">
          <option value="all" selected>All figures</option>
          <option value="rd_all_models">RD</option>
          <option value="rd_all_models_zoom">RD zoom, BPP &lt;= 6</option>
          <option value="lpips_all_models">LPIPS</option>
          <option value="throughput_all_models">Throughput</option>
          <option value="memory_all_models">Memory</option>
          <option value="params_throughput">Params vs throughput</option>
          <option value="all_models_metrics">Metric ranges</option>
        </select>
      </div>
    </div>
  </header>
  <main class="wrap">
    <p class="note" id="note"></p>
    <section class="grid" id="figures"></section>
  </main>
  <script>
    const datasets = {dataset_js};
    const views = [
      ["rd_all_models", "RD"],
      ["rd_all_models_zoom", "RD zoom"],
      ["lpips_all_models", "LPIPS"],
      ["throughput_all_models", "Throughput"],
      ["memory_all_models", "Memory"],
      ["params_throughput", "Params vs throughput"],
      ["all_models_metrics", "Metric ranges"]
    ];
    let activeDataset = "all";
    function datasetInfo(dataset) {{ return datasets.find(d => d.id === dataset); }}
    function prefix(dataset) {{ return `${{dataset}}_{PREFIX_SUFFIX}`; }}
    function imagePath(dataset, view, ext = "png") {{
      const info = datasetInfo(dataset);
      const p = prefix(dataset);
      return `${{info.category}}/${{p}}/plots/${{p}}_${{view}}.${{ext}}`;
    }}
    function renderButtons() {{
      const holder = document.getElementById("datasetButtons");
      holder.innerHTML = "";
      const allRow = document.createElement("div");
      allRow.className = "category-row";
      const allButton = document.createElement("button");
      allButton.textContent = "All";
      allButton.className = activeDataset === "all" ? "active" : "";
      allButton.addEventListener("click", () => {{ activeDataset = "all"; render(); }});
      allRow.appendChild(allButton);
      holder.appendChild(allRow);
      const groups = [];
      for (const item of datasets) {{
        let group = groups.find(g => g.id === item.category);
        if (!group) {{
          group = {{ id: item.category, label: item.categoryLabel, items: [] }};
          groups.push(group);
        }}
        group.items.push(item);
      }}
      groups.forEach(group => {{
        const row = document.createElement("div");
        row.className = "category-row";
        const label = document.createElement("div");
        label.className = "category-label";
        label.textContent = group.label;
        row.appendChild(label);
        group.items.forEach(({{id, label}}) => {{
        const button = document.createElement("button");
        button.textContent = label;
        button.className = id === activeDataset ? "active" : "";
        button.addEventListener("click", () => {{ activeDataset = id; render(); }});
          row.appendChild(button);
        }});
        holder.appendChild(row);
      }});
    }}
    function addFigure(holder, dataset, view) {{
      const viewLabel = views.find(v => v[0] === view)[1];
      const datasetLabel = datasetInfo(dataset).label;
      const png = imagePath(dataset, view, "png");
      const pdf = imagePath(dataset, view, "pdf");
      const article = document.createElement("article");
      article.className = "figure";
      article.innerHTML = `<div class="figure-header"><div class="figure-title">${{datasetLabel}} - ${{viewLabel}}</div><div class="links"><a href="${{png}}" target="_blank">PNG</a><a href="${{pdf}}" target="_blank">PDF</a></div></div><img src="${{png}}" loading="lazy" alt="${{datasetLabel}} ${{viewLabel}}">`;
      holder.appendChild(article);
    }}
    function render() {{
      renderButtons();
      const holder = document.getElementById("figures");
      const selectedView = document.getElementById("viewSelect").value;
      holder.innerHTML = "";
      const selectedDatasets = activeDataset === "all" ? datasets.map(d => d.id) : [activeDataset];
      const selectedViews = selectedView === "all" ? views.map(v => v[0]) : [selectedView];
      for (const dataset of selectedDatasets) for (const view of selectedViews) addFigure(holder, dataset, view);
      document.getElementById("note").textContent = `${{selectedDatasets.length}} dataset(s), ${{selectedViews.length}} view(s)`;
    }}
    document.getElementById("viewSelect").addEventListener("change", render);
    render();
  </script>
</body>
</html>
"""
    (output_dir / "index.html").write_text(html, encoding="utf-8")


def load_json(path: Path) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        for key in ("results", "records", "summary"):
            if isinstance(data.get(key), list):
                return data[key]
    raise ValueError(f"unsupported JSON shape: {path}")


def write_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
