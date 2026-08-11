# ERA5 上 CAESAR 微调失败尝试记录

> 状态日期：2026-08-11
> 目的：记录没有进入最终交付的训练方向、失败证据和可复用经验，避免接手人重复消耗算力。
> 最终有效权重和完整结论见 `项目交接总览.md` 第 4 节。
> 对应实验 checkpoint 和一次性脚本已清理；文中的旧路径仅是历史标识，不表示文件仍存在。

## 1. 如何定义“失败”

本项目不以训练 loss 或内部 validation loss 是否下降判断成功。一个候选只有在以下正式门槛上改善，才可进入最终结果：

1. 使用与正式 ERA5 测试一致的 cadence 和 268 个变量；
2. 在独立时间段验证；
3. 运行真实 range compress/decompress，而不是只看 forward reconstruction；
4. 同时报告真实 BPP、normalized PSNR、逐变量质量；
5. CAESAR-D 必须运行完整 16 帧、固定 seed 的 diffusion sampling；
6. 最终 checkpoint 由真实 codec RD 或 sampling PSNR 选择，不能由 surrogate loss 选择。

因此，下文中的“失败”包括三类：

- 真实 codec 指标退化；
- 改善只发生在错误的数据分布、错误指标或局部组件上；
- 候选有效但被后续结果严格替代，不再值得保留大量中间 checkpoint。

## 2. 失败尝试总表

| 尝试 | 原始动机 | 结果 | 处置 |
|---|---|---|---|
| 连续小时训练直接评价 daily-00 测试 | 增加时间样本并学习小时变化 | cadence 不匹配，多数变量退化 | checkpoint 已清理，保留日志与分析 |
| raw/source 全局 MSE | 直接优化物理量误差 | 被少数大 range 变量支配，256/268 变量退化 | 路径淘汰 |
| 只微调 CAESAR-D Stage1 | 先改善 keyframe VAE | keyframe 改善被 10 个 diffusion 预测帧抵消 | 不作为完整 D 成果 |
| 短程 D Stage2 noise-loss 微调 | 让 diffusion 适配新 VAE latent | noise loss 下降但真实 sampling 退化 | 早期 Stage2 网格已清理 |
| Stage2 延长到 35k/75k/100k | 判断长训练是否超过 5k | 三个里程碑均低于 5k | 长训练 checkpoint 已清理 |
| x0-only / noise+x0 hybrid | 让目标更贴近 reconstructed latent | 四组真实 sampling 均低于 original diffusion | checkpoint 已清理 |
| hard-channel specialist | 针对 humidity/single-level 难变量继续训练 | 三组 specialist 均退化 | checkpoint 已清理，逐候选 JSON 保留 |
| 更强 D Stage1 码率惩罚 `lambda=1e-3` | 进入 DCAE/HPCM 低 BPP 区 | BPP 更低但共同区间平均低于 D-original `1.164 dB` | 仅保留消融完整权重/结果 |
| 历史 50k Stage2 曲线 | 检查长 Stage2 的 13 点 RD | watcher 使用了较早 decoder snapshot，不能做固定 decoder 对照 | 结果只作历史证据 |
| hourly quality/source 大网格与 10k recovery | 搜索 loss、LR、lambda、冻结范围 | 筛选价值已完成，均被最终 100k + decoder-only 路径替代 | 中间 checkpoint 已清理 |

## 3. Cadence 不匹配

### 设置

早期训练使用连续小时窗口，但正式 `era5_test.npy` 是 2024-06-01 至 2024-06-16 每日 00:00 的 16 帧。空间区域和月份相近，但时间采样分布不同。

### 失败证据

- hourly 权重在连续小时验证上 MSE 下降；
- 同一权重切换到 daily-00 验证后，source-loss V 的 MSE 增加 `5.11%`；
- 268 个变量中 264 个退化；
- normalized-loss 旧权重在 hourly 上有 240 个变量改善，切换 daily 后只剩 83 个改善、185 个退化。

### 结论

训练/验证必须沿相同 hour-of-day 轨道跨日采样，即 `frame_step=24`。不能把连续小时 validation 的改善外推到 daily 正式测试。

证据位于 `unified_results/caesar_era5_daily_cadence_real_codec/`，复现参数见 `docs/benchmark_reproduction_manifest.md`。

## 4. Raw/source 全局 MSE 失败

### 设置

直接在反归一化物理量上使用全局 MSE，希望改善 source-domain reconstruction。

### 失败证据

- June daily 测试上只有 `12/268` 个变量改善，`256/268` 个退化；
- 改善集中在 patch scale 约 40–100 的 `w` 变量；
- source loss 等价于将 normalized error 乘以每个 patch 的 `scale²`，使大 range 变量获得远大于其他变量的梯度权重。

### 结论

如果目标是平均变量质量和跨变量公平性，应以 normalized MSE 反向传播；source MSE 作为报告指标记录，不能作为唯一训练目标。

## 5. 只微调 CAESAR-D Stage1 失败

### 设置

先微调 D 的 Stage1 VAE/keyframe decoder，再与原始 diffusion 组合，希望完整 16 帧同时改善。

### 失败证据

- Stage1 keyframe 独立评价曾达到 268/268 变量改善，平均约 `+1.011 dB`；
- 完整 D 中只有部分 keyframe 由 Stage1 重建，其余帧由 Stage2 diffusion 生成；
- 未适配 diffusion 的预测帧抵消了 keyframe 收益；
- 早期可保留组合仅有 `+0.0114 dB`、BPP `-4.54%`，不足以宣称 diffusion 或完整 D 成功。

### 结论

D 的 VAE 与 diffusion 必须匹配训练并作为完整 checkpoint 打包。不能用 Stage1 的独立改善代表完整 codec 改善。

## 6. 短程 Stage2 noise-loss 微调失败

### 设置

- 两种 Stage1 VAE：quality 和 low-rate；
- diffusion LR：`3e-7, 1e-6, 3e-6`；
- 250/1000 step 真实 codec 验证；
- 固定原始 VAE，另测 `3e-8, 1e-7, 3e-7`。

### 失败证据

所有 Stage2 候选都低于原始完整 D。即使固定原始 VAE，只用 `lr=3e-8` 微调 250 step，average-variable PSNR 仍从 `25.7697` 降至 `25.7607 dB`。

### 结论

diffusion noise-prediction loss 与最终固定 seed sampling PSNR 不一致。任何 Stage2 筛选都必须实际运行 sampling reconstruction。

早期 checkpoint 已删除，日志和入口仍保留：

```text
logs/caesar_era5_daily_d_stage2_pilot/
logs/caesar_era5_stage2_20260723/
scripts/archive/caesar_experiments/run_caesar_era5_daily_d_stage2_pilot.sh
scripts/archive/caesar_experiments/run_caesar_era5_stage2_grid.sh
```

## 7. Stage2 长训练未超过 5k

### 设置

固定 decoder-100k、condition latent、完整 `[268,16,240,240]` 输入和相同 seed，对 5k、validation-best 35k、75k、100k 做真实 32-step sampling。

### 结果

| Stage2 checkpoint | latent BPP | 全帧 PSNR | predicted-frame PSNR | 相对 5k |
|---|---:|---:|---:|---:|
| 5k | `0.111297` | `21.591 dB` | `19.556 dB` | baseline |
| validation-best 35k | `0.111297` | `21.439 dB` | `19.404 dB` | `-0.152 dB` |
| 75k | `0.111297` | `21.495 dB` | `19.460 dB` | `-0.096 dB` |
| 100k | `0.111297` | `21.516 dB` | `19.481 dB` | `-0.075 dB` |

四者 keyframe PSNR 均为 `45.746 dB`，差异完全来自 Stage2 diffusion。

### 结论

35k 的 validation loss 虽最低，但真实生成质量不如 5k。最终 D 必须保留 5k，而不是 validation-best 或训练最久的 checkpoint。

长训练 checkpoint 已删除，证据保留于：

```text
unified_results/diagnostic_caesar_d_stage2_cpu_full268/
scripts/archive/caesar_experiments/run_caesar_era5_d_cpu_sampling_audit.sh
logs/caesar_era5_d_stage2_full_200k/
```

## 8. x0-only 与 hybrid objective 失败

### 设置

在数据组织、Stage1、归一化不变的条件下，比较 x0-only 和 `noise + weight*x0`，每组训练 500 updates。

### 结果

original diffusion 为 `29.812 dB`：

| 目标 | 最佳真实 sampling PSNR |
|---|---:|
| hybrid，`x0_weight=0.01` | `28.303 dB` |
| hybrid，`x0_weight=0.1` | `28.919 dB` |
| hybrid，`x0_weight=1.0` | `28.917 dB` |
| x0-only | `28.202 dB` |

`x0_weight=0.1` 甚至从 update 50 的 `29.585 dB` 下降到 update 250 的 `28.680 dB`。

### 结论

修改 surrogate objective 没有自动带来 codec sampling 改善，不再扩展成长训练。

checkpoint 已删除，保留：

```text
logs/caesar_era5_d_x0_objective_pilot/
scripts/archive/caesar_experiments/run_caesar_era5_d_x0_objective_pilot.sh
unified_results/diagnostics/caesar_era5_discarded_training_manifest.json
```

## 9. Hard-channel specialist 失败

### 设置

分别对 specific humidity（变量 `37:74`）、relative humidity（`185:222`）和 single-level variables（`259:268`）继续训练 original noise objective。

### 结果

| 变量组 | Original sampling | Specialist sampling | 变化 |
|---|---:|---:|---:|
| Specific humidity | `19.827 dB` | `19.186 dB` | `-0.641 dB` |
| Relative humidity | `17.762 dB` | `17.108 dB` | `-0.654 dB` |
| Single-level | `19.375 dB` | `18.221 dB` | `-1.154 dB` |

### 结论

局部 validation loss 下降并不代表固定 seed sampling 改善。三个 specialist 均不再延长训练。

checkpoint 已删除，逐候选机器可读 JSON 和日志保留：

```text
unified_results/diagnostic_caesar_d_hard_channel_specialists/
logs/caesar_era5_d_hard_channel_specialists/
scripts/evaluate_caesar_era5_d_hard_channel_specialists.sh
```

## 10. 更强 D Stage1 码率惩罚未成为最终权重

`lambda_rate=1e-3` 的 D Stage1 把曲线扩展到 `0.07465–31.22213 BPP`，但与 D-original 的共同 BPP 区间平均为 `-1.164 dB`，最差 `-4.911 dB`；在 DCAE endpoint `0.13262 BPP` 处只有约 `32.632 dB`。

它证明可以进一步降低 latent BPP，但质量损失过大，不能替代最终 `lambda_rate=3e-4` 的完整 D。仅作为低码率消融保留：

```text
checkpoints/caesar_era5_d_complete_candidates/lam1em3_decoder_best_original_stage2.pt
unified_results/objective_era5_caesar_d_lam1em3_decoder_best_original_stage2_rd/
unified_results/objective_era5_caesar_d_lam1em3_decoder_best_original_stage2_compare/
```

## 11. 历史 50k 曲线不能作为严格 Stage2 对照

历史 50k 曾完成 13 点真实 codec 审计，但随后通过 tensor hash 发现 watcher 使用了同一路径下较早保存的 decoder snapshot，而不是最终 decoder-100k。该结果本身可解码，但不能支持“固定 decoder，只比较 Stage2 里程碑”的结论。

历史证据保留：

```text
unified_results/objective_era5_caesar_d_stage2_50000_rd/
unified_results/objective_era5_caesar_d_stage2_50000_compare/
```

## 12. 被后续结果替代的筛选网格

下列目录主要用于搜索 cadence、normalization、loss、LR、lambda、rate penalty、冻结范围和早期里程碑。它们并非全部“训练完全无效”，但筛选任务已完成，最终候选已被独立保存，因此中间 checkpoint 已删除：

```text
caesar_era5_hourly_pilot
caesar_era5_hourly_quality_sweep
caesar_era5_hourly_source_sweep
caesar_era5_daily_d_stage1_pilot
caesar_era5_vd_lowrate_10k
caesar_era5_v_lowrate_quality_recovery_10k
caesar_era5_v_decoder_quality_10k
caesar_era5_v_decoder_quality_highlr_10k
caesar_era5_v_decoder_quality_extreme_5k
caesar_era5_d_decoder_quality_10k
caesar_era5_d_stage2_rate2_pilot
caesar_era5_d_lam1em3_recovery_pilot
```

选择依据和机器记录统一保存在：

```text
unified_results/diagnostics/caesar_era5_discarded_training_manifest.json
```

## 13. 已排除的伪故障：range coder/量化路径

`scripts/archive/caesar_experiments/diagnose_caesar_quantization_paths.py` 对 forward quantization 和真实 range compress/decompress 做了逐 latent 对比：

- reconstruction path MSE 约 `1e-11`；
- 最大绝对差约 `1e-4`；
- 真实 bit 数仅比理论 bit 数高约 `0.6%–1.3%`。

因此，forward validation 改善而真实 codec 退化，并不是 range coder 或量化解码实现错误。证据：

```text
unified_results/caesar_quantization_path_diagnostic/
```

## 14. 接手人应避免重复的做法

1. 不要用连续小时 validation 选择 daily 测试权重。
2. 不要用 3-variable 输入代替正式 268-variable ERA5。
3. 不要只看 global/source MSE；必须检查逐变量 normalized 指标。
4. 不要用 Stage1 keyframe 改善代表完整 CAESAR-D 改善。
5. 不要按 diffusion noise loss 或 validation-best 自动选择 Stage2。
6. 不要在没有真实 codec/sampling gate 的情况下盲目延长训练。
7. 不要把 keyframe-only、ensemble-4 或其他消融标成默认正式曲线。
