# CAESAR ERA5 微调失败根因与验证结果

日期：2026-07-27

## 结论

过去多轮微调没有得到独立测试收益，并不是 range coder 或量化解码路径错误。主要原因有三个：

1. **时间采样分布不一致**：训练使用连续小时窗口，而最终
   `era5_test.npy` 是 2024-06-01 至 2024-06-16 每日 00:00 的 16 帧。
2. **raw/source 全局 MSE 不适合 268 个异质量纲变量**：loss 被少数 patch
   range 很大的变量支配，能够降低全局 MSE，却会牺牲大多数变量的 PSNR。
3. **CAESAR-D 不能只微调 Stage 1**：Stage 1 只处理 6 个 keyframe，
   其余 10 帧由 diffusion 预测。只改善 VAE keyframe 不会自动改善完整
   16 帧结果；现有短程 diffusion noise-loss 微调又与最终 sampling PSNR
   缺乏一致性。

CAESAR-V 已通过匹配 daily cadence 的训练得到独立真实 codec 改善。
CAESAR-D Stage 2 的短程微调目前没有通过真实 codec 门槛，不应发布为改进权重。

## 排除项

`scripts/diagnose_caesar_quantization_paths.py` 对 forward quantization 和真实
range compress/decompress 做了逐 latent 对比：

- 重建路径 MSE 约 `1e-11`
- 最大绝对差约 `1e-4`
- 真实 bit 数只比理论 bit 数高约 `0.6%~1.3%`

因此，之前 forward 验证改善、真实 codec 退化不是 range coder 路径造成的。
证据位于：

`unified_results/caesar_quantization_path_diagnostic/`

## Cadence 对照

在同一个 5 月、相同空间裁剪上，只改变时间 cadence：

- hourly 训练权重在连续小时验证上表现为 MSE 降低
- 改为 daily-00 后，source-loss V 权重 MSE 增加 `5.11%`
- 268 个变量中 264 个退化
- normalized-loss 旧权重也从 hourly 的 240 个变量改善，变为 daily 的
  83 个改善、185 个退化

这排除了月份和空间区域变化，直接证明 cadence mismatch。

daily 采样通过 `frame_step=24` 实现，训练使用前 74 天，验证使用随后
16 天；每个序列在同一个 hour-of-day 轨道上跨日采样。

## Source Loss 问题

source-loss V 权重在 June daily 测试上：

- 只有 12/268 个变量改善
- 256/268 个变量退化
- 改善集中在 patch scale 约 40 到 100 的 `w` 变量

source loss 等价于将 normalized error 乘以每个 patch 的 `scale^2`。
这会让高 range 变量获得远大于其他变量的梯度权重。若目标是平均变量
PSNR 和跨变量公平性，应使用 normalized MSE 训练，并把反归一化 MSE
作为报告指标，而不是唯一反向传播目标。

## CAESAR-V 有效结果

独立数据：

`/workspace/Data/ERA5/finetune_processed/era5_test.npy`

协议：

- 268 个变量
- 16 个 daily 帧
- 240 x 240
- no PCA
- 真实 range compress/decompress

| 权重 | Average-variable PSNR | Global PSNR | BPP |
|---|---:|---:|---:|
| Original | 43.4202 | 60.1541 | 0.285881 |
| Daily quality (`lr=3e-5, lambda=3e-6`) | 44.3421 | 60.6420 | 0.273186 |
| Daily RD (`lr=3e-5, lambda=3e-5`) | 44.2362 | 60.6187 | 0.256018 |

相对原权重：

- Daily quality：average-variable PSNR `+0.9219 dB`，BPP `-4.44%`
- Daily RD：average-variable PSNR `+0.8160 dB`，BPP `-10.45%`

推荐权重：

- 质量优先：
  `checkpoints/caesar_era5_daily_cadence_pilot/daily_v_lr3e5_lam3e6.pt`
- RD 平衡：
  `checkpoints/caesar_era5_daily_cadence_pilot/daily_v_lr3e5_lam3e5.pt`

真实 codec 结果：

`unified_results/caesar_era5_daily_cadence_real_codec/`

## CAESAR-D 结果

Stage 1 keyframe 的最佳独立结果为 268/268 变量改善、平均 `+1.011 dB`，
但完整 D codec 中 10 个未适配 diffusion 预测帧抵消了收益。

Stage 2 做了以下隔离实验：

- 两种 Stage 1 VAE：quality 和 low-rate
- diffusion LR：`3e-7`、`1e-6`、`3e-6`
- 250 和 1000 step 真实 codec 验证
- 固定原始 VAE，额外测试 LR `3e-8`、`1e-7`、`3e-7`

所有 Stage 2 权重都低于原始完整 D codec。即使固定原始 VAE、仅用
`3e-8` 微调 250 step，average-variable PSNR 仍从 `25.7697` 变为
`25.7607 dB`。这证明短程 diffusion noise-loss 下降不能作为 sampling
质量选择标准。

当前可保留的 D 结果只有 Stage 1 low-rate 与原始 diffusion 的组合：

- average-variable PSNR `+0.0114 dB`
- BPP `-4.54%`

该增益很小，不足以宣称 diffusion 微调成功。若继续训练 D，需要按论文
完整 diffusion schedule 重训，并在训练中加入固定种子的实际 sampling
重建验证；不能再根据 noise prediction loss 选 checkpoint。

## 代码修复

- `utils/era5_netcdf_dataset.py`
  - 新增 `frame_step`，支持同一 hour-of-day 的 daily 序列。
- `scripts/finetune_caesar_era5.py`
  - 新增 `--frame_step`
  - DataLoader 使用 `spawn`，避免 worker 继承 CUDA context
  - normalized 训练时也正确记录恢复 patch scale 后的 source MSE
- `tests/test_era5_netcdf_dataset.py`
  - 覆盖 daily frame-step 索引
- `tests/test_caesar_source_distortion.py`
  - 覆盖 normalized distortion 与真实 source-MSE 日志分离

## 后续训练门槛

任何新权重必须同时满足：

1. 与最终测试相同 cadence 的独立时间段验证；
2. 268 个变量逐变量统计，而非只看 global MSE；
3. no-PCA 真实 range compress/decompress；
4. 同时报告 average-variable PSNR、global PSNR 和真实 BPP；
5. CAESAR-D 必须验证完整 16 帧 sampling，不能只验证 6 个 keyframe 或
   diffusion noise loss。
