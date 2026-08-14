# `unified_results` 文件说明

本目录只保留三类结果：Objective-v1 正式全模型结果、ERA5 CAESAR-V 有效微调结果，以及 Lysozyme CAESAR-V/D 微调结果。这里保存的是指标、协议、图表和网页，不保存模型权重或原始压缩 bitstream。

```text
unified_results/
├── README.md
├── objective_all_to_all_v1/  # 正式全模型 benchmark 与 PCA 消融
├── era5_caesar_v/            # ERA5 CAESAR-V 有效微调专项
└── lysozyme_caesar_tuned/    # Lysozyme CAESAR-V/D 微调专项
```

## 1. `objective_all_to_all_v1`

这是正式结果入口。最快的查看方式是直接打开 `index.html`。

### 根目录文件

| 文件 | 内容 |
|---|---|
| `index.html` | 单文件中文结果页面。全部图表以内嵌 base64 保存，因此复制这一个文件也能离线查看。每个数据集含主 RD、吞吐量、峰值显存、LPIPS，以及“PCA 消融与全部模型对比”图。 |
| `combined_summary.json` | 十个数据集正式主结果的逐记录合集；不含 `pca_hybrids/` 消融记录。适合程序读取。 |
| `protocol.json` | 本次结果使用的 Objective-v1 机器可读协议副本。 |
| `eb_schedule.json` | 各数据集 CAESAR、cuSZ-Hi、nvJPEG/nvJPEG2000 等方法最终采用的控制点。 |
| `objective_protocol_audit.json` | 正式主结果的逐 gate 审计汇总，包括数据集覆盖、模型族覆盖、canonical identity、码率、指标、计时和硬件声明。 |
| `objective_protocol_audit.md` | 上述审计的简短 Markdown 版。 |

### 数据集目录

以下十个目录结构相同：

```text
e3sm_npz/  era5_npy/  hurricane/  kodak/  lysozyme/
nyx/       s2c/       tomo/       turb_rot_npz/  uvg_twilight_1080p/
```

每个目录包含：

| 文件 | 内容 |
|---|---|
| `samples.json` | 正式 canonical sample 清单：sample ID、形状、dtype、总 symbol 数、有效 symbol 数、原始与归一化 checksum。 |
| `normalization.json` | 数据集级冻结外部归一化：minimum、scale、来源、是否 clipping。所有 codec 使用同一份。 |
| `summary.json` | 该数据集所有正式模型的逐 sample、逐控制点原始记录。包含 BPP、PSNR、LPIPS、bitstream/side-info 字节、重复计时、吞吐量、峰值显存、checkpoint、硬件和协议字段。 |

`lysozyme/samples.json` 还记录有效像素数量。Lysozyme 中 `raw >= 4294967000` 的探测器 sentinel 被 mask；所有 codec 共享相同 mask，invalid 输入替换为 0，指标只统计有效位置。

### `pca_hybrids`

```text
pca_hybrids/<dataset_id>/summary.json
```

十个 `summary.json` 分别保存该数据集的 DCAE/HPCM + CAESAR-PCA 消融原始记录。每个完整数据集有 21 个 corpus 点：

- DCAE + CAESAR-PCA：7 个 EB；
- HPCM-base + CAESAR-PCA：7 个 EB；
- HPCM-large + CAESAR-PCA：7 个 EB。

每条记录同时计算图像模型 bitstream、PCA residual payload 和必要 side information。它们在额外消融图中与全部正式模型对照，但不进入主模型 Pareto 排名。

### `analysis`

| 文件或命名规则 | 内容 |
|---|---|
| `objective_analysis.json` | 从逐 sample 记录聚合得到的 corpus 点、Pareto 点、被支配点、不完整点和消融点。 |
| `objective_analysis.md` | 每个数据集的记录数量与完整性摘要。 |
| `<dataset>_objective_rd.png` | 正式主模型的 BPP–normalized PSNR 曲线，不含 PCA hybrid。 |
| `<dataset>_pca_hybrid_rd.png` | 全部正式模型加三条 PCA hybrid 曲线；PCA hybrid 使用虚线。 |
| `<dataset>_throughput_range.png` | 各正式模型端到端吞吐量均值及控制点范围。 |
| `<dataset>_memory_range.png` | 各正式模型峰值 allocated GPU memory 均值及范围。 |
| `<dataset>_lpips_rd.png` | 各正式模型 BPP–LPIPS 图；LPIPS 越低越好。 |

这里的 `<dataset>` 是上述十个 dataset ID，因此每类图各有十张。`index.html` 内嵌的图片来自这里。

### `summary.json` 常用字段

| 字段 | 含义 |
|---|---|
| `dataset_id`, `canonical_sample_id` | 数据集和 canonical sample 身份。 |
| `model_name`, `model_id`, `control` | 模型族、具体 checkpoint/配置和控制点。 |
| `scientific_bpp_with_side_info` | 包含必要辅助信息的正式 BPP。 |
| `normalized_mse`, `normalized_psnr` | 在数据集固定单位范围上的主质量指标。 |
| `lpips` | 感知指标；越低越好。 |
| `bitstream_bytes`, `side_info_bytes`, `total_bytes_with_side_info` | payload、辅助信息和总字节数。 |
| `timing_repetitions` | 两次预热后五次正式测量的逐次计时。 |
| `encode_throughput_MBps`, `decode_throughput_MBps` | 编码和解码吞吐量。 |
| `memory_usage_MB`, `memory_reserved_MB` | PyTorch 峰值 allocated/reserved 显存。 |
| `canonical_sha256`, `normalized_canonical_sha256` | 输入一致性校验。 |
| `external_input_manifest` | 外部归一化和共享 mask 约定。 |
| `hardware_manifest`, `protocol_id` | 测试硬件和协议版本。 |

## 2. `era5_caesar_v`

这里只保存 ERA5 上确认有效的 CAESAR-V 微调证据。

### `daily_real_codec`

| 文件 | 内容 |
|---|---|
| `v_original/summary.json` | Original CAESAR-V 在 `EB=0.001` 的真实 codec 单点结果。 |
| `v_daily_rd/summary.json` | 率失真目标 daily 微调权重在相同输入和 EB 下的单点结果。 |
| `v_daily_quality/summary.json` | 偏重重建质量的 daily 微调权重在相同输入和 EB 下的单点结果。 |

这三份文件用于证明 daily 微调相对 original 的方向是正向的，不是最终完整 RD 曲线。

### `daily_v_100k_eb_compare`

| 文件 | 内容 |
|---|---|
| `comparison.json` | Original 与 early daily full-100k 权重在七个 EB 上的合并对比。 |
| `caesar_era5_original_vs_finetuned_100k_eb.png` | PNG 率失真对比图。 |
| `caesar_era5_original_vs_finetuned_100k_eb.pdf` | 同一张图的矢量 PDF。 |
| `complete` | 空完成标记；存在表示该 sweep 当时完整结束。 |
| `raw/original_eb<编码>/summary.json` | Original 权重某个 EB 的原始结果。 |
| `raw/finetuned_eb<编码>/summary.json` | Full-100k 微调权重相同 EB 的原始结果。 |

`eb0p1`、`eb0p03`、`eb0p01`、`eb0p003`、`eb0p001`、`eb0p0003`、`eb0p0001` 分别代表 `0.1`、`0.03`、`0.01`、`0.003`、`0.001`、`0.0003`、`0.0001`。

### `decoder_final_rd`

| 文件 | 内容 |
|---|---|
| `summary.json` | 当前最终 ERA5 CAESAR-V decoder-only 微调权重的 13 点完整 RD 结果，含 Objective-v1 canonical、计时、显存和 checkpoint 元数据。 |
| `_run_files/complete` | 该正式 sweep 的完成标记。 |

当前最终权重不在结果目录，而在：

```text
checkpoints/caesar_era5_v_decoder_quality_100k/from_lowrate_lr3em4.pt
```

## 3. `lysozyme_caesar_tuned`

该目录比较 Lysozyme 上 original 与 finetuned CAESAR-V/D。

```text
lysozyme_caesar_tuned/
├── v/  # CAESAR-V，8-frame
└── d/  # CAESAR-D，16-frame
```

`v/` 和 `d/` 的结构相同：

| 文件或目录 | 内容 |
|---|---|
| `sweep_results.json` | 七个 EB 下 original/finetuned 的合并曲线数据。 |
| `psnr_vs_bpp.png` | 该模型 V 或 D 的 original/finetuned BPP–PSNR 对比。 |
| `time_vs_eb.png` | 编码和解码时间随 EB 的变化。 |
| `eb_<编码>/CAESAR-V_original.json` 或 `CAESAR-D_original.json` | Original 权重在该 EB 的单独指标。 |
| `eb_<编码>/CAESAR-V_finetuned.json` 或 `CAESAR-D_finetuned.json` | 微调权重在该 EB 的单独指标。 |
| `eb_<编码>/all_results.json` | 同一 EB 的 original 和 finetuned 合并文件。 |

EB 目录编码：

| 目录 | EB |
|---|---:|
| `eb_1em2` | `1e-2` |
| `eb_5em3` | `5e-3` |
| `eb_1em3` | `1e-3` |
| `eb_5em4` | `5e-4` |
| `eb_1em4` | `1e-4` |
| `eb_5em5` | `5e-5` |
| `eb_1em5` | `1e-5` |

根目录的 `psnr_vs_bpp_combined.png` 把 CAESAR-V/D 的 original 和 finetuned 四条曲线画在同一张图中。

对应最终权重：

```text
checkpoints/caesar_lysozyme/caesar_v_tuning_lysozyme.pt
checkpoints/caesar_lysozyme/caesar_d_tuning_lysozyme_vae.pt  # D Stage 1
checkpoints/caesar_lysozyme/caesar_d_tuning_lysozyme.pt      # D Stage 2 / 最终推理
```

## 4. 哪个文件优先看

1. 看全模型结论：`objective_all_to_all_v1/index.html`。
2. 做程序分析：各数据集 `summary.json`，或主结果合集 `combined_summary.json`。
3. 看 PCA 消融：`analysis/<dataset>_pca_hybrid_rd.png` 和 `pca_hybrids/<dataset>/summary.json`。
4. 看 ERA5 微调最终曲线：`era5_caesar_v/decoder_final_rd/summary.json`。
5. 看 Lysozyme 微调：`lysozyme_caesar_tuned/psnr_vs_bpp_combined.png`。

重新生成 Objective 图表和页面：

```bash
source /workspace/ai4cp/bin/activate
python scripts/analyze_objective_benchmark.py \
  --root unified_results/objective_all_to_all_v1
```
