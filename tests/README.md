# tests 目录索引

测试只覆盖当前框架、正式协议和保留的训练/数据工具，不依赖历史结果目录或本地大 checkpoint。

| 范围 | 测试文件 |
|---|---|
| 通用 pipeline、模型注册和 codec | `test_compression_pipeline.py`、`test_model_registry.py`、`test_nvjpeg_codecs.py` |
| objective 数据、堆叠、视频、审计和分析 | `test_objective_data.py`、`test_objective_stacking.py`、`test_objective_video.py`、`test_audit_objective_benchmark.py`、`test_analyze_objective_benchmark.py`、`test_select_objective_eb_schedule.py` |
| CAESAR 推理与最佳 ERA5 微调实现 | `test_caesar_checkpoint_resolution.py`、`test_caesar_diffusion_ensemble.py`、`test_caesar_finetune_selection.py`、`test_caesar_keyframe_only.py`、`test_caesar_pca_stability.py`、`test_caesar_source_distortion.py` |
| ERA5 下载与 shard | `test_download_era5.py`、`test_era5_netcdf_dataset.py`、`test_audit_era5_hourly_shards.py` |
| Turb_Rot | `test_turb_rot_npz_adapter.py`、`test_plot_turb_rot_results.py` |
| 绘图与 Slurm smoke | `test_plot_results_summary.py`、`test_framework_smoke_submit_env.py` |

运行全部测试：

```bash
pytest -q
```
