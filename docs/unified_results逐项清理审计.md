# `unified_results` 逐项清理审计

> 审计日期：2026-08-11
> 范围：`unified_results/` 当前 95 个一级条目，共约 220.52 MiB。
> 原则：正式 index、机器汇总、ERA5 最终 V/D、Stage2 选择证据和失败记录必须保留；已被这些结果替代且可由脚本重跑的中间目录可删除。

## 结论

| 分类 | 数量 | 大小 | 含义 |
|---|---:|---:|---|
| 必须保留 | 9 | 28.41 MiB | 当前正式交付和必要机器证据 |
| 建议保留 | 11 | 58.00 MiB | 协议验证、选择依据和仍有解释价值的消融 |
| 可删除 | 75 | 134.11 MiB | 已被最终结果替代、重复、旧协议或可重跑中间物 |

如果目标是“保留完整交接证据”，保留前两类，删除 75 项后约剩 86.41 MiB。如果只保留最终发表/交付结果，可以进一步审核“建议保留”的 11 项，最小保留集约 28.41 MiB。

## 1. 必须保留（9 项）

| 条目 | 内容与保留原因 |
|---|---|
| `objective_all_to_all_v1/` | 正式 10 数据集全模型结果；包含单文件 index、合并 JSON、图、normalization 和严格审计 |
| `objective_era5_caesar_v_decoder_final_rd/` | ERA5 CAESAR-V 最终 decoder 微调 RD 曲线 |
| `objective_era5_caesar_v_decoder_final_compare/` | CAESAR-V 最终版本与 original/图像 codec 对比 |
| `objective_era5_caesar_d_decoder100k_stage2_overlap5k_rd/` | ERA5 CAESAR-D 最终 RD 曲线 |
| `objective_era5_caesar_d_decoder100k_stage2_overlap5k_compare/` | CAESAR-D 最终版本对比图和 comparison JSON |
| `objective_era5_caesar_vd_complete_compare/` | V/D 最终统一对比图与 manifest |
| `diagnostic_caesar_d_stage2_cpu_full268/` | 证明 Stage2 5k 优于后续里程碑的 268 变量真实 sampling 证据 |
| `diagnostic_caesar_d_hard_channel_specialists/` | 已删除 hard-channel 训练的机器可读失败依据 |
| `diagnostics/` | 已删除训练、量化路径和 full268 诊断的最终 JSON 记录 |

## 2. 建议保留（11 项）

这些不是主排名，但仍能解释协议或最终权重为什么这样选择。

| 条目 | 内容与建议 |
|---|---|
| `matched_validation_20260721/` | objective-v1 前的统一输入、BPP 和 wall-time 验证；建议保留 |
| `caesar_era5_hourly_selection/` | ERA5 hourly checkpoint 选择结果；建议保留 selection provenance |
| `caesar_era5_hourly_final/` | hourly 阶段获选 V/D 的汇总；建议保留 |
| `caesar_era5_daily_cadence_real_codec/` | daily cadence 对真实 codec 的影响证据；失败记录引用 |
| `caesar_era5_daily_v_100k_eb_compare/` | 初始 100k V 版本的 268 变量比较；用于解释后续 decoder 微调 |
| `caesar_quantization_path_diagnostic/` | CAESAR-D 量化/扩散路径根因证据 |
| `diagnostic_caesar_d_temporal_reconstruction/` | D 时序重建诊断；可在不再研究 D 根因时删除 |
| `diagnostic_caesar_d_stage2_cpu_ensemble4_full268/` | ensemble-4 消融；可在不保留消融时删除 |
| `objective_era5_caesar_d_original_13pt_rd/` | 最终 D 对比使用的 original 13 点基线；建议与最终 compare 一起保留 |
| `experiments/` | 图像 codec + PCA 等历史消融，约 50 MiB；若只保留正式结果可删除 |
| `no_pca_lpips_fill/` | CAESAR no-PCA 消融；很小，建议随消融说明保留 |

## 3. 可删除（75 项）

### 3.1 已过期的目录说明与顶层分析（2 项）

- `README.md`：仍描述已删除的 `final/raw/merged` 布局，内容过期。
- `analysis/`：旧顶层分析；正式图已经进入 `objective_all_to_all_v1/analysis/`。

### 3.2 早期 CAESAR/ERA5 调参和诊断（22 项）

这些结果已由最终 decoder100k、Stage2 overlap5k 和保留的诊断 JSON 替代：

- `caesar_diag_10k_20260624/`
- `caesar_diag_10k_era5_eval_20260624/`
- `caesar_era5_d_stage2_overlap_eval/`
- `caesar_era5_daily_v_continuation_real_codec/`
- `caesar_era5_daily_v_full_100k_real_codec/`
- `caesar_era5_direct_reconstruction/`
- `caesar_era5_hourly_day2_rd_diagnostic/`
- `caesar_era5_hourly_day3_d_rd_diagnostic/`
- `caesar_era5_hourly_norm_path_diagnostic/`
- `caesar_era5_hourly_pilot_eval/`
- `caesar_era5_hourly_quality_eval/`
- `caesar_era5_selected_7eb_20260723/`
- `caesar_era5_source_direct_reconstruction/`
- `caesar_era5_stability_eval_20260723/`
- `caesar_era5_stable_tuning_20260723/`
- `caesar_era5_stage2_eval_20260723/`
- `caesar_hw_bpp_lambda_sweep_20260624/`
- `caesar_per_channel_diagnostic/`
- `caesar_v_hw_bpp_eval_era5_20260624/`
- `retest_caesar_tomo_s2c_20260624/`
- `turb_rot_full/`
- `uvg_twilight_full/`

### 3.3 旧外部 codec 专项结果（14 项）

正式 cuSZ-Hi、nvJPEG/nvJPEG2000 点已经进入 `objective_all_to_all_v1`：

- `e3sm_npz_external_models_n64/`
- `era5_npy_external_models_c3_t16_240/`
- `hurricane_external_models_n16/`
- `kodak_external_models/`
- `lysozyme_external_models_n16/`
- `nyx_external_models_n16/`
- `s2c_external_models_n16/`
- `tomo_external_models_n16/`
- `turb_rot_npz_external_models_n64/`
- `nvjpeg2k_packz_extreme7_n1/`
- `nvjpeg2k_packz_extreme7_n1_lpips/`
- `nvjpeg_kodak_q7_n24/`
- `nvjpeg_kodak_q7_n24_lpips/`
- `nvjpeg_kodak_q7_n24_lpips_full/`

### 3.4 已被最终 ERA5 结果替代的候选（15 项）

- `objective_era5_caesar_d_complete_5k_compare/`
- `objective_era5_caesar_d_complete_5k_rd/`
- `objective_era5_caesar_d_decoder100k_original_stage2_compare/`
- `objective_era5_caesar_d_decoder100k_original_stage2_rd/`
- `objective_era5_caesar_d_decoder100k_stage2_5k_compare/`
- `objective_era5_caesar_d_decoder100k_stage2_5k_rd/`
- `objective_era5_caesar_d_lam1em3_decoder_best_original_stage2_compare/`
- `objective_era5_caesar_d_lam1em3_decoder_best_original_stage2_rd/`
- `objective_era5_caesar_d_lam1em3_keyframe_only_compare/`
- `objective_era5_caesar_d_lam1em3_keyframe_only_rd/`
- `objective_era5_caesar_d_stage2_50000_compare/`
- `objective_era5_caesar_d_stage2_50000_rd/`
- `objective_era5_caesar_v_100k_best_compare/`
- `objective_era5_caesar_v_finetuned_100k_rd/`
- `objective_era5_caesar_vd_lowrate_100k_rd/`

这些候选的关键结论已经写入交接、复现和失败记录；最终选中的 V/D 目录位于“必须保留”。

### 3.5 objective-v1 聚合输入和补跑中间物（20 项）

最终 `combined_summary.json` 是自包含的，严格审计可以只针对 `objective_all_to_all_v1` 运行。删除这些项目会失去“无需重跑即可重新合并”的便利，但不会影响查看或审计正式结果：

- `objective_all_to_all_v1_eb_schedule.json`：正式目录内已有 `eb_schedule.json`，属于重复副本。
- `objective_v1/`
- `objective_v1_caesar_bs64_recovery/`
- `objective_v1_caesar_bs64_shards/`
- `objective_v1_caesar_bs64_tail/`
- `objective_v1_cusz_monotonic/`
- `objective_v1_cusz_recovery/`
- `objective_v1_eb_schedule_draft.json`
- `objective_v1_eb_schedule_draft2.json`
- `objective_v1_eb_schedules_partial.json`
- `objective_v1_era5_hourly_original/`
- `objective_v1_era5_hourly_quality_d/`
- `objective_v1_era5_hourly_quality_v/`
- `objective_v1_era5_hourly_tuned_d/`
- `objective_v1_era5_hourly_tuned_v/`
- `objective_v1_era5_tuned_stable_d100/`
- `objective_v1_era5_tuned_stable_v1000/`
- `objective_v1_turb_tuned_formal_d/`
- `objective_v1_turb_tuned_formal_v/`
- `objective_v1_turb_tuned_schedule.json`

### 3.6 其他被替代结果（2 项）

- `metadata/`：旧 index 和旧目录布局 metadata，引用已删除目录。
- `uvg_twilight_1080p/`：早期 UVG 单独运行；正式 UVG 结果已聚合。

## 4. 删除前验证门槛

执行“可删除”清单前应再次满足：

1. `objective_all_to_all_v1/index.html`、`combined_summary.json` 和 10 个数据集子目录存在；
2. 严格审计仍为 1642/1642、10/10；
3. 5 个 ERA5 最终 RD/compare 目录存在；
4. `diagnostic_caesar_d_stage2_cpu_full268/`、`diagnostic_caesar_d_hard_channel_specialists/` 和 `diagnostics/` 存在；
5. 删除目标中没有 Git 跟踪文件；
6. 删除后更新 `结果目录索引.md`、`结果清理清单.md` 和复现文档中的历史路径说明。

## 5. 恢复性

- `objective_v1*`、外部 codec 和旧候选：可由保留的数据、checkpoint、协议和脚本重跑，但成本较高。
- 过期 HTML、图和 metadata：可以从 JSON 或正式分析脚本重建。
- 删除操作若执行，将是永久删除；当前没有为这些未跟踪结果建立单独备份。
