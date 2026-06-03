# Turb Rot Benchmark Design

## Goal

Add the Turb_Rot CAESAR turbulence NPZ dataset to the unified benchmark path, evaluate image-style codecs and CAESAR original/tuned weights on the same data, and report LPIPS plus memory usage in the generated summaries and plots.

## Dataset Contract

`Turb_Rot_testset.npz` contains `data` in CAESAR-style `[V,S,T,H,W]` layout and a `variable_name` array used as metadata. The current file is `(1,16,256,256,256)`: one variable channel, sixteen section slices, 256 time frames, and 256x256 spatial fields.

The adapter exposes two views:

- Image models: `iter_samples()` yields `[3,H,W]` float32 samples by stacking neighboring section slices at one time index. This gives DCAE, LIC-HPCM, DCMVC, and DCVC-RT natural three-channel inputs without inventing missing variables.
- CAESAR: `load_sequence()` selects one section slice and returns `[V,T,H,W]`, preserving contiguous time frames for `caesar_v` and `caesar_d`.

## Metrics

The shared metrics layer adds optional `lpips` and `memory_usage_MB` fields. LPIPS is computed in normalized visual space and skipped with `None` if the dependency is unavailable. Memory is measured around each model/sample roundtrip: CUDA runs report peak allocated/reserved memory, and CPU runs report RSS when `psutil` or `resource` is available.

## CAESAR Weights

The CAESAR runner resolves checkpoint files flexibly. Original weights continue to use `checkpoints/caesar/caesar_v.pth` and `caesar_d.pth`; tuned weights may use files like `caesar_v_tuning_Turb-Rot (1).pt` and `caesar_d_tuning_Turb-Rot (1).pt` under `checkpoints/caesar_tuned`.

## Outputs

Add a Slurm/local script for Turb_Rot that runs:

- Image models: `DCAE LIC-HPCM DCMVC DCVC-RT`
- CAESAR original: `caesar_v caesar_d` with `checkpoints/caesar`
- CAESAR tuned: `caesar_v caesar_d` with `checkpoints/caesar_tuned`

Add a plotting utility that creates:

- CAESAR original vs tuned RD comparison
- Combined image-model and CAESAR result plot, including PSNR, LPIPS, compression ratio, and memory where available

## Testing

Unit tests cover adapter shape handling without reading the 2GB dataset, checkpoint resolution for tuned CAESAR names, and metrics fields when LPIPS/memory collection is disabled or unavailable.
