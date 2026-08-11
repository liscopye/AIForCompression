# CLAUDE.md

This file provides guidance to Claude Code when working in this repository.

# AIForCompression 项目

目标是搭建一个可复用的学习型/传统压缩 benchmark 框架，用统一的数据中间层评测多种数据集和多种模型，而不是为每个“数据集 x 模型”写一次性脚本。

当前主框架是：

```text
Dataset Adapter -> CanonicalSample -> Model View -> Codec Runner -> Metrics -> summary.json
```

主入口：

```bash
python scripts/run_dataset_compression.py \
  --dataset <dataset> \
  --data_root <path> \
  --output_dir unified_results/<run_name> \
  --models <model...> \
  --max_samples <n>
```

## 当前项目结构

```text
AIForCompression/
├── compression_pipeline/           # shared benchmark framework
│   ├── adapters/                   # dataset -> CanonicalSample
│   ├── canonical.py                # CanonicalSample / DatasetManifest
│   ├── views.py                    # image groups / CAESAR views / inverse transforms
│   ├── runner.py                   # grouped image-model runner
│   ├── torch_codecs.py             # model API wrappers
│   ├── model_registry.py           # model checkpoint discovery and loaders
│   ├── cra5_runner.py              # CRA5 native ERA5 runner
│   └── caesar_runner.py            # CAESAR sequence runner
├── scripts/                        # benchmark and Slurm entrypoints
├── utils/                          # plotting, aggregation, download, statistics tools
├── tests/                          # regression tests for framework behavior
├── docs/                           # design notes and usage docs
├── models/                         # upstream model source trees
├── checkpoints/                    # model weights, organized by model
├── normalization/                  # ERA5 daily mean/std files used by adapter
└── unified_results/                # benchmark outputs and plots
```

## Supported Datasets

`scripts/run_dataset_compression.py --dataset` currently supports:

- `kodak`: RGB image folder, canonical `[C,H,W] uint8`.
- `uvg`: YUV420 video frames, canonical RGB frames.
- `era5`: paired `{timestamp}_pressure.nc` and `{timestamp}_single.nc`, canonical 268-channel float fields.
- `tomo`: reconstructed tomography HDF5 volume, slices or grouped slices.
- `hurricane`: `.bin.f32` time series fields.
- `s2c`: Sentinel-2 SAFE/SAFE.zip JP2 band tiles.
- `nyx`: SDRBench NYX `.f32` 3D volume.
- `shanghai_xray`: synchrotron X-ray TIFF images.
- `isot1024`: isotropic turbulence HDF5 velocity data.
- `lysozyme`: CHESS lysozyme HDF5 diffraction frames.

Detailed dataset/model processing rules are documented in
`docs/dataset_model_processing_spec.md`.  Use that document as the benchmark
contract for channel grouping, normalization, CAESAR sequence shape, bpp, PSNR,
LPIPS, memory, and throughput.

Adapters must emit `CanonicalSample(layout="channel_height_width")` for image-style models. Sequence-capable adapters may also implement `load_sequence()` returning `[V,T,H,W]` plus timestamps for CAESAR.

## Supported Model Families

### General image/intra models

These go through `build_image_groups()` and `run_image_grouped_sample()`:

- `DCAE`
- `LIC_TCM`
- `LIC-HPCM`
- `RwkvCompress`
- `WeConvene`
- `DCVC-RT` intra wrapper
- `DCMVC` intra wrapper

Rules:

- Image/uint8 data uses `/255 -> [0,1]`.
- Float scientific data uses per-channel min/max normalization, with inverse transform before metrics.
- Multi-channel samples are split into 3-channel groups; the last group is padded by repeating the last real channel.
- Metrics are computed in the reconstructed original data space, not in normalized model input space.

### CRA5

CRA5 is an ERA5-native model. In the main benchmark path it only accepts ERA5-style 268-channel scientific samples and should be run without `--max_channels`:

```bash
python scripts/run_dataset_compression.py \
  --dataset era5 \
  --data_root /data/run01/scxj523/zsh/project/Data/ERA5/2024 \
  --output_dir unified_results/era5_cra5 \
  --models CRA5 \
  --max_samples 1
```

Non-ERA5 CRA5 runs require the explicit `--allow_cra5_adapted` flag. That path resizes and channel-replicates samples to CRA5's 268-channel input and is only an ablation/diagnostic baseline, not a fair default benchmark.

### CAESAR

CAESAR uses native sequence views and does not go through the image-group runner:

- `caesar_v` requires 8 contiguous frames.
- `caesar_d` requires 16 contiguous frames.
- Input to CAESAR runner is `[V,T,H,W]`; `build_caesar_view()` converts it to `[V,S,T,H,W]` with `S=1`.
- `--caesar_start_index` selects the contiguous window.
- `--caesar_eb` controls error-bound sweeps.

Example:

```bash
python scripts/run_dataset_compression.py \
  --dataset era5 \
  --data_root /data/run01/scxj523/zsh/project/Data/ERA5/2024 \
  --output_dir unified_results/era5_caesar_v \
  --models caesar_v \
  --max_samples 8 \
  --max_channels 6 \
  --caesar_eb 1e-4 1e-3 1e-2
```

## Metrics Contract

Each successful result should include:

- `mse`, `rmse`, `psnr`
- `bpp`, `bitstream_bytes`, `original_bytes`, `compression_ratio`
- `encode_time_total`, `decode_time_total`
- `encode_time_per_group_avg`, `decode_time_per_group_avg`
- `encode_throughput_MBps`, `decode_throughput_MBps`
- legacy aliases: `encode_time_avg`, `decode_time_avg`, `encode_throughput`, `decode_throughput`
- `model_name`, `model_id`, `sample_id`, `dataset_id` where applicable

PSNR is computed as `10 * log10(data_range^2 / mse)` in original data space. `data_range` is per-sample `original.max() - original.min()`, falling back to `1.0` for constant samples.

## HPC / Slurm Rules

Current 5090 host environment:

```bash
source /workspace/ai4cp/bin/activate
cd /workspace/AIForCompression
```

Use `/workspace/Turb_Rot_testset.npz` for the local Turb_Rot benchmark data and
`/workspace/AIForCompression/checkpoints/caesar_tuned` for the official
Turb_Rot-tuned CAESAR checkpoint directory.

Legacy cluster environment:

Use the allocated GPU from Slurm. Job scripts must not unset, export, or overwrite `CUDA_VISIBLE_DEVICES` unless debugging outside Slurm.

Standard environment:

```bash
eval "$(/data/home/scxj523/run/miniconda3/bin/conda shell.bash hook)"
conda activate /data/run01/scxj523/zsh/envs/zsh
cd /data/run01/scxj523/zsh/project/AIForCompression
```

Smoke entrypoint:

```bash
sbatch scripts/run_framework_smoke_model.sh <case> [max_model_jobs|all] [max_samples|all]
```

Examples:

```bash
sbatch scripts/run_framework_smoke_model.sh kodak_dcae all all
sbatch scripts/run_framework_smoke_model.sh era5_caesar_v 1 8
```

## Development Rules

- Treat `models/` as upstream/vendor model code unless a model-specific fix is required.
- Prefer adding reusable adapter/view/codec code under `compression_pipeline/` over creating new one-off scripts.
- Add new datasets by implementing an adapter in `compression_pipeline/adapters/`, registering it in `scripts/run_dataset_compression.py`, and documenting its canonical layout.
- Add new models through `compression_pipeline/model_registry.py` plus a codec wrapper if the model does not expose CompressAI-style `compress()`/`decompress()`.
- Keep tests focused on reusable framework behavior. Run at least:

```bash
pytest -q tests/test_compression_pipeline.py tests/test_model_registry.py
```

## Key Paths

- Project root: `/workspace/AIForCompression`
- Turb_Rot data: `/workspace/Turb_Rot_testset.npz`
- Checkpoints: `/workspace/AIForCompression/checkpoints`
- CAESAR tuned checkpoints: `/workspace/AIForCompression/checkpoints/caesar_tuned`
- Model source: `/workspace/AIForCompression/models`
- ERA5 normalization: `/workspace/AIForCompression/normalization`
