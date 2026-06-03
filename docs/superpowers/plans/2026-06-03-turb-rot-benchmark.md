# Turb Rot Benchmark Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a unified Turb_Rot NPZ benchmark path with LPIPS, memory usage, CAESAR tuned checkpoints, run scripts, and plots.

**Architecture:** Implement a focused dataset adapter for CAESAR-style `[V,S,T,H,W]` NPZ files, keep model execution in the existing unified runner, and extend metrics through optional helper functions. CAESAR checkpoint lookup becomes flexible but remains backward compatible with `caesar_v.pth` and `caesar_d.pth`.

**Tech Stack:** Python, NumPy, PyTorch, optional `lpips`, matplotlib, pytest, existing `compression_pipeline`.

---

### Task 1: Turb_Rot Adapter

**Files:**
- Create: `compression_pipeline/adapters/turb_rot_npz.py`
- Modify: `compression_pipeline/adapters/__init__.py`
- Modify: `scripts/run_dataset_compression.py`
- Test: `tests/test_turb_rot_npz_adapter.py`

- [ ] Write tests that create a tiny `[V,S,T,H,W]` NPZ and assert `iter_samples()` returns `[3,H,W]`, `load_sequence()` returns `[V,T,H,W]`, and metadata preserves `variable_name`.
- [ ] Run `pytest -q tests/test_turb_rot_npz_adapter.py` and confirm the adapter import fails.
- [ ] Implement `TurbRotNPZAdapter` with `section_index`, `image_group_axis`, and `max_samples` support.
- [ ] Register `turb_rot_npz` in `run_dataset_compression.py`.
- [ ] Re-run `pytest -q tests/test_turb_rot_npz_adapter.py`.

### Task 2: LPIPS and Memory Metrics

**Files:**
- Modify: `compression_pipeline/metrics.py`
- Modify: `compression_pipeline/runner.py`
- Modify: `compression_pipeline/caesar_runner.py`
- Test: `tests/test_compression_pipeline.py`

- [ ] Add tests for `base_metrics(..., extra_metrics={...})` preserving optional `lpips` and memory keys.
- [ ] Add tests for image runner accepting disabled LPIPS/memory collectors without changing existing numeric metrics.
- [ ] Implement optional LPIPS helper with lazy import and graceful `None` fallback.
- [ ] Implement memory snapshots around image and CAESAR roundtrips.
- [ ] Re-run focused tests.

### Task 3: Flexible CAESAR Checkpoints

**Files:**
- Modify: `compression_pipeline/caesar_runner.py`
- Test: `tests/test_caesar_era5_options.py`

- [ ] Add tests for resolving `caesar_v.pth`, `caesar_v*.pt`, and `caesar_d*.pt`.
- [ ] Implement `_resolve_caesar_checkpoint()`.
- [ ] Use the resolved path in `CAESAR(model_path=...)`.
- [ ] Re-run `pytest -q tests/test_caesar_era5_options.py`.

### Task 4: Turb_Rot Run Script

**Files:**
- Create: `scripts/run_turb_rot_benchmark.sh`

- [ ] Add a script that runs image models, CAESAR original, and CAESAR tuned into separate output directories.
- [ ] Keep Slurm-compatible environment activation and do not overwrite `CUDA_VISIBLE_DEVICES`.
- [ ] Syntax-check with `bash -n scripts/run_turb_rot_benchmark.sh`.

### Task 5: Plotting

**Files:**
- Create: `utils/plot_turb_rot_results.py`
- Test: `tests/test_plot_turb_rot_results.py`

- [ ] Add tests for loading multiple summaries and labeling CAESAR original/tuned separately.
- [ ] Implement plots for CAESAR original vs tuned and combined model comparison.
- [ ] Re-run `pytest -q tests/test_plot_turb_rot_results.py`.

### Task 6: Verification

**Files:**
- Existing tests and changed files

- [ ] Run `pytest -q tests/test_turb_rot_npz_adapter.py tests/test_caesar_era5_options.py tests/test_compression_pipeline.py tests/test_plot_turb_rot_results.py`.
- [ ] Run `python scripts/run_dataset_compression.py --help`.
- [ ] Run `bash -n scripts/run_turb_rot_benchmark.sh`.
- [ ] Report exact commands and any remaining limitations.
