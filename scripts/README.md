# scripts 目录索引

本目录只保留当前可执行工作流及其直接依赖。一次性实验、旧集群 wrapper、已废弃模型入口和历史结果修复脚本不放在这里。

## 通用开发与正式评测

| 文件 | 用途 |
|---|---|
| `run_dataset_compression.py` | 通用 Dataset Adapter → codec → metrics 开发与 smoke 入口。 |
| `run_framework_smoke_model.sh` | 在 Slurm 上按模型/数据集 case 调用通用入口。 |
| `prepare_objective_inputs.py` | 生成 objective-v1 canonical 输入、manifest 和 checksum。 |
| `run_objective_benchmark.py` | 正式图像、科学数据、帧内模型与传统 codec 评测。 |
| `run_objective_video.py` | 正式 UVG I+P 视频评测。 |
| `run_uvg_pframe_codecs.py` | `run_objective_video.py` 使用的 DCVC-RT/DCMVC 底层 runner。 |
| `export_objective_uvg_frames.py` | 将 objective UVG tensor 导出成经过校验的 canonical PNG。 |
| `run_objective_caesar_bs64_schedule.py` | 多 GPU 调度 CAESAR objective batch-size 64 作业。 |
| `merge_objective_shards.py` | 合并并行 objective 结果 shard。 |
| `select_objective_eb_schedule.py` | 从完整候选曲线选择正式 EB。 |
| `build_objective_all_to_all.py` | 合并正式来源并应用完整性/有效性筛选。 |
| `audit_objective_benchmark.py` | 检查 objective 协议、覆盖率、指标和误差界。 |
| `analyze_objective_benchmark.py` | 生成正式图表、表格和 HTML 总览。 |

## Codec 与验证依赖

| 文件 | 用途 |
|---|---|
| `run_external_scientific_codecs.py` | cuSZ-Hi 3D、Visemz 和 GraphComp 的共享外部 codec 实现；cuSZ 只走 whole-volume 3D。 |
| `run_matched_codec_validation.py` | matched-input 验证及 objective 入口复用的数据集常量/聚合函数。 |
| `analyze_matched_validation.py` | 汇总 matched-input 验证结果。 |
| `analyze_dataset_structure.py` | 跨 adapter 检查数据 shape、range、样本组织和 3D packing。 |
| `analyze_benchmark_observations.py` | 从正式结果重建 benchmark 观察报告和 CSV。 |

## 最佳 ERA5 CAESAR 微调链

以下四个 shell 脚本按顺序运行，共同调用 `finetune_caesar_era5.py`；最后一步调用 `package_caesar_d_stage1.py` 生成完整 D checkpoint。

| 文件 | 用途 |
|---|---|
| `run_caesar_era5_vd_lowrate_100k.sh` | 训练最终 V/D 使用的低码率 Stage1 起点。 |
| `run_caesar_era5_v_decoder_quality_100k.sh` | 训练最终 CAESAR-V decoder。 |
| `run_caesar_era5_d_decoder_quality_100k.sh` | 训练最终 CAESAR-D Stage1 decoder。 |
| `run_caesar_era5_d_stage2_overlap_5k.sh` | 训练 matching Stage2 并打包最终完整 D。 |
| `finetune_caesar_era5.py` | ERA5 shard 数据、Stage1/Stage2、验证和 checkpoint 保存的共享实现。 |
| `package_caesar_d_stage1.py` | 将选定 VAE 与 diffusion 合并为完整 CAESAR-D checkpoint。 |

## 数据获取与预处理

| 文件 | 用途 |
|---|---|
| `download_jhtdb_local.py` | 下载 JHTDB 湍流数据供 isotropic 数据集使用。 |
| `run_era5_hourly_download.sh` | 下载 ERA5 hourly pressure/single-level 数据并持续生成 shard。 |
| `prepare_lysozyme_data.py` | 将 Lysozyme HDF5 转为训练/评测数组。 |
| `run_turb_rot_benchmark.sh` | 本机 Turb_Rot 原始/tuned CAESAR 与图像模型整套评测及绘图。 |

正式工作流的参数和数据协议以 `docs/benchmark_reproduction_manifest.md` 与 `benchmark_protocols/objective_v1.json` 为准。
