# CAESAR ERA5 微调问题与 CRA5 数据/训练流程分析

> 2026-07-23 更新：第 1-5 节保留了早期失败实验的历史记录，其中关于
> 61 时刻空间切分和“CRA5 z-score 与 CAESAR mean-range 二次归一化冲突”的
> 判断不再适用于当前实验。当前使用 45/16 时刻的 chronological
> train/validation split；新的审计、消融和结论见下方“最新复核”。

## 1. 结论摘要

本次调查的核心结论如下。

1. 当前 CAESAR 在 ERA5 上的微调退化，不是 `resume` 导致的主要问题。V 和 D 在最早可测试的微调权重处就已明显劣于 pre-tune；D 的主要退化发生在 Stage 1 VAE 微调，而不是 Stage 2 diffusion resume。
2. 当前 CAESAR 的 ERA5 输入方式与 CRA5 有本质区别。CRA5 每次处理一个完整气象场，输入形状为 `[B, 268, 721, 1440]`，同时利用变量、气压层之间的相关性；CAESAR 当前把 268 个变量通道拆开，每个样本为单通道时序块 `[1, T, 256, 256]`。
3. 两者虽然使用了同一套 ERA5 变量和 CRA5 统计量，但 CAESAR 在已经完成 channel z-score 的数据上，又执行了每个单通道 patch 的 `mean_range` 归一化。这是为适配原始 CAESAR 数据接口而保留的操作，不等价于 CRA5 的输入分布。
4. 当前 CAESAR 的 checkpoint 选择指标与最终评测目标不对齐。训练验证用模型内部 MSE/bit 或 diffusion loss；最终 ERA5 测试还包含 EB 约束下的 residual postprocessing 字节，因此内部验证更优并不保证最终 RD 曲线更优。
5. 短期内应先修正训练/选择策略，而不是继续长时间微调：增加非常早期 checkpoint、降低学习率、用最终 EB sweep 指标选择权重、采用时间独立验证集，并做归一化消融。若要真正复用 CRA5 的主要优势，需要修改 CAESAR 为多通道联合建模，属于结构级改动。

## 最新复核（2026-07-23）

### 6.1 论文步数不能直接照搬

CAESAR 论文中的 domain fine-tuning 配置为：

- CAESAR-V：`100k` updates，学习率 `1e-4`。
- CAESAR-D：keyframe compressor `100k` updates，随后 latent diffusion
  `200k` updates，学习率均为 `1e-4`。
- 论文 ERA5 数据包含约 `2160` 个时间点。

当前训练集位于
`/workspace/Data/ERA5/finetune_processed_time_split_t45_v16`，采用前 45
个时间点训练、后 16 个时间点验证。batch size 为 32 时：

| 模型 | 每轮训练样本 | 每个 epoch 的 optimizer updates |
| --- | ---: | ---: |
| CAESAR-V | 1608 | 约 51 |
| CAESAR-D Stage 1 | 804 | 约 26 |

因此，在当前小时间跨度上训练 `100k` updates，相当于 V 约 2000 个 epoch、
D 约 4000 个 epoch。论文步数描述的是特定数据规模下的训练预算，不是与数据
规模无关的固定配方。当前实验必须按 held-out 最终 RD 曲线早停。

### 6.2 归一化更正

当前 `.npy` 先使用 CRA5 的固定 channel mean/std 做正仿射 z-score；CAESAR
随后对每个单变量时空 patch 做 `mean_range` 或 `mean_range_hw`。由于每个
CAESAR 样本只包含一个物理变量，正仿射 z-score 会被后续的 mean/range
标准化代数抵消：

```text
((x - mu_c) / sigma_c - patch_mean) / patch_range
    == (x - raw_patch_mean) / raw_patch_range
```

所以旧实验的主要问题不是“做了两次归一化”。真正需要比较的是：

- `mean_range`：整个 `[T,H,W]` patch 共用均值和范围，符合发布代码默认行为。
- `mean_range_hw`：每帧分别计算均值和范围，更接近论文中“each frame”
  的字面描述。

消融结果显示两者在非常早期的最终 RD 上接近；`mean_range` 略优，因此保留
发布代码的默认路径作为主搜索，同时正式 90 天网格包含论文配方
`mean_range_hw + lr=1e-4`。每个候选的内部归一化类型会写入 selection
manifest，并在 Stage 1 完整 codec 筛选、Stage 2 训练/筛选以及最终 objective
复测中保持一致。CRA5 外部固定 channel z-score 对所有候选仍完全相同。

### 6.3 稳定性消融

8 张 GPU 并行测试了 V/D、两种归一化、`lambda_rate` 和 L2-SP 参数锚定。
所有 run 均已同步到 W&B：

- project：`caesar-era5-stable-tuning`
- group：`stage1-2k-grid-20260723`
- Stage 2 group：`stage2-500-grid-20260723`

稳定配置为：

```text
batch_size=32
learning_rate=1e-5
warmup_updates=500
rate_mode=bpp
lambda_rate=1e-4
norm_type=mean_range
```

训练在 `100/250/500/1000/2000` updates 保存 checkpoint，并使用最终
`latent + PCA/Zstd residual` 的 7 点 EB 曲线选择权重，而不是内部
`MSE + lambda * rate` validation loss。

### 6.4 选中的权重

快速 held-out 3-variable 筛选结果：

| 模型 | 选中 checkpoint | 7 点重叠区 BD-rate | 结论 |
| --- | --- | ---: | --- |
| CAESAR-V | update 1000 | `-1.13%` | 中低 BPP 改善，继续到 2000 开始退化 |
| CAESAR-D Stage 1 | update 100 | `-3.45%` | 早期改善最明显，后续逐渐变差 |

权重位置：

```text
checkpoints/caesar_era5_stability_20260723/v_mr_lam1e4_anchor0_update1000.pt
checkpoints/caesar_era5_stability_20260723/packaged_d/d_mr_lam1e4_anchor0_update100.pt
```

以 D Stage 1 update 100 为基础继续进行了 4 组 Stage 2 diffusion 实验：
学习率 `1e-6/3e-6/1e-5`，并包含参数锚定消融。最早的 update 50 已未能改善
最终 RD，后续和更高学习率进一步退化。因此最终 D 使用微调后的 Stage 1 VAE
和原始 diffusion 权重，不继续 Stage 2。

### 6.5 当前正式验证

最终验证使用 objective-v1 ERA5 输入：

```text
[268 variables, 16 times, 240, 240]
```

覆盖 7 个 EB，计时使用 2 次 warmup 和 5 次端到端重复。输出目录为：

```text
unified_results/objective_v1_era5_tuned_stable_v1000
unified_results/objective_v1_era5_tuned_stable_d100
```

完整结果和原始权重对比图位于：

```text
unified_results/caesar_era5_stable_tuning_20260723/summary.json
unified_results/caesar_era5_stable_tuning_20260723/caesar_era5_stable_rd.png
unified_results/caesar_era5_stable_tuning_20260723/caesar_era5_stable_rd.pdf
```

正式全变量曲线的 piecewise log-rate BD-rate 为：

| 模型 | BD-rate | 解释 |
| --- | ---: | --- |
| CAESAR-V update 1000 | `-2.05%` | 整体小幅左移，中低码率改善更明显 |
| CAESAR-D Stage 1 update 100 | `-1.66%` | 改善较小，曲线与原始权重基本重合 |

这些结果证明新配置避免了旧实验在最早 checkpoint 即严重退化的问题，但收益
不足以称为显著的 domain fine-tuning。当前 45 个训练时刻均为每日 00:00
采样；论文 ERA5 数据有约 2160 个时刻。有限且稀疏的天气状态是继续训练很快
过拟合、收益上限较低的主要嫌疑。

下一轮数据范围为 2024-03-01 至 2024-05-29，共 2160 个逐小时 ERA5 时刻。
CDS 对同一 collection 限制并发排队，因此采用每天 24 小时合并为一次请求，
并下载中央 128°×128° 区域（ERA5 0.25° 网格上为 513×513）：

```text
/workspace/Data/ERA5/hourly_center512_20240301_90d
```

该数据位于 2024-06-01 至 2024-06-16 objective test 之前；objective 的 16
个时刻为每天 `00:00 UTC`，不是连续 16 小时。直接从 NetCDF 随机读取的实测
速度约 `50 s/update`，不适合正式训练。

pressure-level 最初请求 CDS 服务端转换后的 NetCDF。实测完整一天的服务端
阶段约 28 分钟，随后还要传输约 3.1 GB，成为整个实验的关键瓶颈。现已切换为
CDS 原生 GRIB：同一天、同区域、同 7 个变量和 37 个压力层的服务端阶段为
6.2 分钟，完整请求加传输为 27.0 分钟。GRIB 文件约 3.27 GB。

格式切换不是近似替代。2024-04-04 的 GRIB 和 NetCDF 先在 189 个跨变量、
时间、压力层和空间坐标的原始值上比较，`max_abs_error=0`；随后两条路径分别
完成 268 通道 CRA5 z-score shard 转换，对全部 `1,692,703,008` 个 float32
值逐值比较，`unequal_values=0`、`max_abs_error=0`。GRIB 直接转换 shard
约 40 秒，NetCDF 路径约 427 秒。

pressure-level 下载使用跨进程 submission lock，保证 CDS 同时只有一个
`accepted/running` 请求；结果一旦生成便释放该锁，使下一天约 6 分钟的服务端
生成可与前一天的对象存储下载重叠。对象下载最多使用 4 个 lane，并通过 HTTP
`Range` 断点续传；网络中断不会再删除已有 `.tmp` 文件。下载器会把已有
NetCDF 和 GRIB 视为同一日期，避免切换格式后重复下载。single-level 保持
NetCDF 串行 worker；转换器接受 GRIB+single-NetCDF 或旧的
pressure-NetCDF+single-NetCDF，两者写出相同 shard。

2026-07-23 的实际队列中，单日 pressure 请求曾在 `accepted` 状态等待超过
50 分钟，明显长于约 6 分钟的生成时间。CDS QoS 同时报告 MARS 全局
`5744 queued / 460 running`，且每个用户最多只有 1 个 MARS 请求运行；
所以增加本地 worker 无法缩短服务端队列。下载器因此新增同月多日原生 GRIB
请求：正式配置单次 9 天，共 55,944 items，并强制保持在 60,000 items
以下。返回的 GRIB 按
`dataDate` 拆回原有逐日文件，每天必须严格包含
`7 variables * 37 levels * 24 hours = 6216` 条消息后才原子发布。本地用
正式 3.27 GB 日文件验证，拆分前后均为 `3,272,388,336` bytes，逐字节一致，
耗时约 7.5 秒。首个真实 9 日 CDS 请求用于验证服务端结果大小、拆分和逐日
shard 转换。

因此下载完成的日文件会自动转换为 CRA5 z-score 后的 float32 mmap shard：

```text
/workspace/Data/ERA5/hourly_center512_shards_20240301_90d
```

每个 shard 形状为 `[268,24,513,513]`，约 6.8 GB；90 天约 610 GB。真实
数据的逐值对比覆盖 pressure 首末层、single-level 和 `tp`，NetCDF 与 shard
读取路径的 `max_abs=0`。shard 后端 batch 32 实测在 worker 启动后约
`0.4-0.8 s/update`。

后台下载与转换：

```bash
scripts/run_era5_hourly_download.sh
```

90 个 shard 完整后，8-GPU、W&B 强制绑定的 10k pilot 使用：

```bash
scripts/run_caesar_era5_hourly_pilot.sh
```

### 6.6 逐小时真实数据早期验证

2026-07-23 使用首日和前两日真实 shard 分别完成了 V/D Stage 1 的
500-update 参数诊断。训练前先记录 update 0，并且只有验证目标真正优于
update 0 时才覆盖 best checkpoint；这修复了旧脚本把第一个退化 checkpoint
误称为 best 的问题。

共同结论如下：

- 发布代码默认的 `mean_range` 最适合原始权重；
- 论文逐帧字面形式 `mean_range_hw` 可以训练，但初始 MSE/BPP 均更差；
- `min_max` 与原始权重分布明显不匹配，V/D 初始 MSE 分别约为
  `mean_range` 的 4.5 倍；
- `lr=1e-4` 配合 100-update warmup 在前 500 updates 没有发生旧实验的
  早期崩溃。

V 的代表性结果：

| 配置 | update 0 MSE/BPP | update 500 MSE/BPP |
| --- | --- | --- |
| `lr=1e-4, lambda=1e-4, mean_range` | `2.400e-4 / 0.2222` | `2.294e-4 / 0.1949` |

D Stage 1 的代表性结果：

| 配置 | update 0 MSE/BPP | update 500 MSE/BPP |
| --- | --- | --- |
| `lr=1e-4, lambda=1e-4, mean_range` | `6.2e-5 / 0.2037` | `6.2e-5 / 0.1638` |

这些是内部 VAE 指标，只证明训练路径稳定，不等于最终压缩改善。完整流程会
继续使用真实 PCA/residual codec 的 held-out 三点 RD 曲线选 checkpoint，
再在独立的 June objective-v1 ERA5 块上跑七点曲线。

随后使用第二天连续 16 小时、30 个代表通道执行了完整 CAESAR codec 三点
诊断。相对原始权重的 piecewise log-rate BD-rate 为：

| 模型 | update 100 | update 500 |
| --- | ---: | ---: |
| CAESAR-V | `-1.14%` | `-7.33%` |
| CAESAR-D Stage 1 | `-4.62%` | `-11.89%` |

三点曲线均单调。V 的 probe 完全位于其训练/验证之后，因此 V 的结果证明当前
逐小时数据路径上的早期内部 rate 改善已经传递到未见数据的 PCA/Zstd 最终
码率。D smoke 使用前 32 小时训练、后 16 小时验证，当前 probe 的第 24–39
小时与其 train/validation 各重叠 8 小时；D 结果只能证明完整 codec 路径和
数值稳定性，不能单独作为泛化证据。

第 3 天 shard 到达后，又在小时 `48–63` 上完成了独立 D 三点验证。该时间段
与前两天 smoke 的训练和验证均无重叠：

| 模型 | update 100 | update 500 |
| --- | ---: | ---: |
| CAESAR-D Stage 1 | `-5.05%` | `-12.25%` |

第 3 天三条曲线同样单调，证明当前 D 的早期改善可以传递到真正未见的连续
小时数据，而不是第 2 天窗口重叠导致的假象。输出位于
`unified_results/caesar_era5_hourly_day3_d_rd_diagnostic/diagnostic.json`。
这些早期诊断仍不替代正式 90 天训练和独立 June objective-v1 结论。

诊断还触发了发布 PCA 代码在 loose EB 下对秩亏 float32 协方差执行
`torch.linalg.eigh` 的稳定性问题。现在单残差向量使用等价的单位正交基，
float32 分解失败时从原始残差重算 float64 协方差；修复后失败点成功，原先
正常的另外两个 EB 点逐值不变。

NetCDF 与 shard 的真实逐元素检查覆盖 8 个关键通道，包括 pressure 变量和
层边界、single-level 起点、`tp` 及末通道，结果全部为 `max_abs=0`。正式
转换器还强制检查每日日文件为 `24x513x513` 且所有归一化值有限。
训练前审计进一步要求 90 个日期连续、`.npy` header 与 JSON metadata 一致、
源 NetCDF 存在且抽样值有限。正式时间划分为：

- train：2024-03-01 00:00 至 2024-05-19 23:00，共 1920 小时；
- validation：2024-05-20 00:00 至 2024-05-29 23:00，共 240 小时；
- objective test：2024-06-01 至 2024-06-16，每天 `00:00 UTC`，共 16 个
  独立时刻，与 train/validation 不重叠。

正式审计还会从 objective 原始 NetCDF 独立重建 CRA5 z-score 值，并与
`era5_test.npy` 跨 pressure 变量边界、层边界、全部 single-level 变量和三组
空间坐标进行内容级比对。当前共检查 621 个值，`max_abs=0`，因此测试数组的
日期来源和归一化链路都已得到实值验证，而不是仅依赖文件命名推断。

完整自动流程：

```bash
scripts/run_caesar_era5_hourly_full_pipeline.sh
```

流程顺序为 Stage 1 V/D 网格、held-out RD 筛选、D Stage 2 网格、再次 RD
筛选，以及独立 objective-v1 七点复测。最终阶段会用当前同一代码、同一输入
和同一组 EB 同时重跑 original V/D，不复用端点不同且早于 PCA 数值修复的历史
baseline。所有训练 run 写入 W&B project `caesar-era5-hourly-tuning`。在
1920 个训练时刻下，时间窗 batch sampler
每个窗口使用 `floor(268/32)=8` 个 batch，因此 V 和 D Stage 1 分别为
`1920` 和 `960` updates/epoch。正式 pilot 每 1000 updates 验证，并在 W&B
中显式记录 `train/epoch` 与 `val/epoch`，对应 V 的 0.52 epoch 和 D 的
1.04 epoch，可以直接检查首 epoch 是否再次出现旧实验的骤降。

D Stage 2 使用 micro-batch 32 和两次梯度累积，有效 batch 为论文中的 64；
5k pilot 同时覆盖 `3e-7/1e-6/3e-6/1e-5` 稳定性搜索及论文给出的 `1e-4`
学习率，所有候选均保留早期 milestone 并由最终 codec RD 选择。

### 6.7 正式 90 天训练与独立七点结论

2026-07-25 正式流程全部完成。90 天数据审计通过，共 2160 个逐小时时刻；
训练集为前 1920 小时，验证集为随后 240 小时，最终测试使用与二者不重叠的
June objective-v1 16 个时刻。16 组 Stage 1 V/D 配置和 5 组 D Stage 2
配置均正常结束并写入 W&B，未再出现首 epoch PSNR 骤降。

held-out 三点真实 codec 筛选结果如下：

| 模型 | 选中 checkpoint | held-out BD-rate |
| --- | --- | ---: |
| CAESAR-V | `v_lr1e5_lam3e4_mr_update10000` | `-21.79%` |
| CAESAR-D Stage 1 | `d_lr1e5_lam3e4_mr_update10000` | `-26.32%` |
| CAESAR-D Stage 2 | `d_s2_mr_lr3e7_update50` | `-26.32%` |

这里的 Stage 2 选择依据是包含 PCA 和残差编码的完整 codec RD 曲线，而不是
diffusion 内部 validation loss。后者在更晚 update 上可能更低，但不保证最终
码率更好。

在独立 objective-v1 上使用同一输入、同一 EB 集合重测 original 和 fine-tuned
权重，最终七点结果为：

| 模型 | 归一化 | EB 点数 | objective-v1 BD-rate |
| --- | --- | ---: | ---: |
| CAESAR-V | `mean_range` | 7 | `-10.26%` |
| CAESAR-D | `mean_range` | 7 | `-12.25%` |

两种模型的 BPP 和 PSNR 均随 EB 严格单调，数值全部有限，
`fine_tuning_success_gate` 已通过。最终权重和结果位于：

- `checkpoints/caesar_era5_hourly_selected/caesar_v.pt`
- `checkpoints/caesar_era5_hourly_selected/caesar_d.pt`
- `checkpoints/caesar_era5_hourly_selected/selection.json`
- `unified_results/caesar_era5_hourly_final/summary.json`
- `unified_results/caesar_era5_hourly_final/caesar_era5_stable_rd.png`

最终排查表明，旧失败不能归因于“CRA5 z-score 与 CAESAR 归一化冲突”这一项。
对每个 patch 使用 `mean_range` 时，正仿射的 CRA5 z-score 会在数学上抵消；
真正影响结论的是旧实验仅使用稀疏的 daily-00 数据、训练/验证路径不一致、
第一个退化 checkpoint 被错误覆盖为 best，以及只保存粗粒度 10k 节点导致
看不到早期转折。正式流程通过逐小时 90 天数据、严格按时间划分、update 0
基线、1000-update warmup、密集早期 milestone 和真实 codec RD 选择解决了
这些问题。

## 2. 已完成的 CAESAR ERA5 权重测试

### 2.1 输出位置

完整权重对比图：

- `results/caesar_era5_checkpoint_sweep/V/caesar_v_era5_checkpoint_comparison.png`
- `results/caesar_era5_checkpoint_sweep/D/caesar_d_era5_checkpoint_comparison.png`
- `results/caesar_era5_checkpoint_sweep/V/sweep_results.json`
- `results/caesar_era5_checkpoint_sweep/D/sweep_results.json`

早期拐点对比图：

- `results/caesar_era5_early_transition/V/caesar_v_era5_early_transition.png`
- `results/caesar_era5_early_transition/D/caesar_d_era5_early_transition.png`
- `results/caesar_era5_early_transition/V/sweep_results.json`
- `results/caesar_era5_early_transition/D/sweep_results.json`

每条曲线均使用 7 个 EB 点；V 与 D 分图绘制，并已加入 tune 前的原始权重作为 baseline。

### 2.2 早期拐点结果

以下给出 `EB = 1e-3` 的代表性结果；同样的相对趋势在图中的 RD 曲线上可见。

#### CAESAR-V

| 权重 | BPP | PSNR (dB) | 判断 |
| --- | ---: | ---: | --- |
| pre-tune `caesar_v.pt` | 0.486375 | 62.3025 | baseline |
| update 10k | 1.248262 | 60.8002 | 已明显退化 |
| update 20k | 1.163547 | 60.9410 | 仍明显差于 baseline |

结论：V 的性能恶化发生在 `0 -> 10k` 之间。此前仅按每 10k 保存 checkpoint，已经错过真正的初始变化区间。

#### CAESAR-D

| 权重 | BPP | PSNR (dB) | 判断 |
| --- | ---: | ---: | --- |
| pre-tune `caesar_d.pt` | 0.692245 | 62.3141 | baseline |
| Stage1 VAE 10k | 1.301470 | 60.9225 | 最大幅度退化已发生 |
| Stage1 VAE 40k | 0.916097 | 61.5569 | 较 10k 恢复，但未回 baseline |
| Stage1 VAE 100k | 0.854961 | 61.7878 | 仍差于 baseline |
| Stage1 VAE best | 0.864212 | 61.7654 | 内部 best 不等于最终 best |
| Stage2 diffusion 10k | 0.820709 | 61.9129 | 部分恢复 |
| Stage2 diffusion 40k | 0.825118 | 61.9015 | 变化很小 |

结论：

- D 的主要退化在 Stage1 VAE 的 `0 -> 10k` 已经发生。
- Stage1 后续训练只是在退化后部分恢复。
- Stage2 diffusion 在使用已退化的 VAE 基础上只能有限改善，`10k -> 40k` 变化很小是合理现象。
- Stage2 的 interrupted/resume 不是首次性能下降的原因；下降发生在 resume 之前。

## 3. 图中指标含义

### 3.1 Trend

图中的 `trend` 是为快速辨认训练顺序而连接 checkpoint 点的辅助线，表示权重随训练步数推进时曲线/代表点的移动方向。它不是新的压缩指标，也不参与模型优劣判定。

真正比较模型时，应在相同 EB 或相近 BPP 下比较 PSNR，或使用全条 RD 曲线判断支配关系。

### 3.2 Global RD 与 Mean RD

ERA5 测试由大量变量、层和时刻组成。

- `Global RD`：先汇总所有数据的总压缩字节与总平方误差，再计算整体 BPP/PSNR。数据规模或动态范围贡献较大的部分会更影响该曲线。
- `Mean RD`：先对每个测试对象计算 BPP/PSNR，再对对象做平均。每个对象权重更接近一致。

如果两者趋势一致，结论较稳；若两者不同，说明不同变量或层级上的收益/退化不均匀，应进一步查看 per-variable/per-level 结果。

## 4. 你当前 CAESAR 在 ERA5 上如何训练

### 4.1 ERA5 原始变量组织

预处理脚本为 `utils/prepare_era5_finetune_data.py`。

使用变量：

- pressure variables：`z, q, u, v, t, r, w`
- pressure levels：37 层，从 `1000` 到 `1` hPa
- single-level variables：`v10, u10, v100, u100, t2m, tcc, sp, tp, msl`

总通道数：

```text
7 * 37 + 9 = 268
```

其中 `tp` 会先从 meter 转换到 millimeter，即乘以 `1000`。

### 4.2 预处理和划分

脚本读取每个时间点的 `pressure.nc` 和 `single.nc`，构造单时刻数据：

```text
[268, 721, 1440]
```

随后使用 CRA5 的 `mean_std.json` 与 `mean_std_single.json`，按每个物理通道执行 z-score：

```text
x_z = (x - channel_mean) / channel_std
```

当前实际生成的数据为：

| 文件 | 形状 | 说明 |
| --- | --- | --- |
| `era5_train.npy` | `[268, 61, 721, 1152]` | 经度左侧 80% |
| `era5_val.npy` | `[268, 61, 721, 288]` | 经度右侧 20% |
| `era5_test.npy` | `[268, 16, 721, 1440]` | 独立测试文件 |

注意：train/val 是同一批时间点按经度切分，而不是按时间切分。这可以测试空间泛化，但不能独立检验新时间段上的泛化。

### 4.3 CAESAR 输入格式

训练入口为 `scripts/finetune_caesar_era5.py`。其 `ERA5MmapDataset` 读取 `[C, T, H, W]` 后，不把 `C=268` 作为模型输入通道，而是把每个通道单独变成一个样本：

```text
model input: [batch, 1, temporal_frames, 256, 256]
```

具体为：

| 模型 | 时间长度 | 默认 stride | 单个训练样本 |
| --- | ---: | ---: | --- |
| CAESAR-V | 8 | 8 | `[1, 8, 256, 256]` |
| CAESAR-D | 16 | 16 | `[1, 16, 256, 256]` |

训练时每个 `(channel, temporal window)` 抽一个随机空间 crop。验证时将空间场反射 padding 后枚举全部 `256 x 256` block。

对于当前 `T=61` 数据：

| 模型 | 时间窗口数 | train items/遍历 | val items/遍历 |
| --- | ---: | ---: | ---: |
| V | 8 | `268 * 8 = 2144` | `268 * 8 * 6 = 12864` |
| D | 4 | `268 * 4 = 1072` | `268 * 4 * 6 = 6432` |

### 4.4 CAESAR 的第二次归一化

输入 `.npy` 已经过 CRA5 channel z-score。进入 CAESAR dataset 后，默认 `norm_type=mean_range`，又对每个单通道时空 patch 做：

```text
offset = patch.mean()
scale  = patch.max() - patch.min()
x_model = (x_z - offset) / scale
```

因此模型实际看见的是：

```text
CRA5 channel z-score -> 单变量/单窗口/单 patch mean-range normalization
```

这与 CRA5 的整场 channel z-score 输入并不相同。它会去除每个 patch 的均值和振幅差异，使模型更难保留气象变量的绝对尺度信息；同时模型训练优化的重构误差是在 patch-normalized 空间中定义的。

### 4.5 实际训练参数

从已完成的 W&B 运行日志中恢复出的设置如下。

#### CAESAR-V

| 参数 | 设置 |
| --- | --- |
| stage | 1 |
| pretrained checkpoint | `checkpoints/caesar/caesar_v.pt` |
| updates | `100000` |
| batch size | `32` |
| learning rate | `1e-4` |
| `lambda_rate` | `1e-5` |
| rate mode | `bits` |
| normalization | `mean_range` |
| save interval | `10000` updates |
| validation interval | `2000` updates |
| best internal val | update `94000`, `val_loss=0.008291` |

#### CAESAR-D Stage 1: VAE

| 参数 | 设置 |
| --- | --- |
| stage | 1 |
| pretrained checkpoint | `checkpoints/caesar/caesar_d.pt` |
| updates | `100000` |
| batch size | `32` |
| learning rate | `1e-4` |
| input use | keyframes only, index `0,3,6,9,12,15` |
| save interval | `10000` updates |
| best internal val | update `94000`, `val_loss=0.003898` |

#### CAESAR-D Stage 2: Diffusion

| 参数 | 设置 |
| --- | --- |
| stage | 2 |
| base D checkpoint | `checkpoints/caesar/caesar_d.pt` |
| VAE checkpoint | `checkpoints/caesar/caesar_d_tuning_era5_vae.pt` |
| planned updates | `200000` |
| batch size | `32` |
| gradient accumulation | `2`, effective batch `64` |
| learning rate | `1e-4` |
| first run | interrupted after snapshots through `140000` |
| best logged pre-interruption | update `74000`, `val_diff_loss=0.000851` |
| resume | from `caesar_d_tuning_era5_update140000.pt`, additional `60000` updates |
| resume best logged | resume update `34000`, `val_diff_loss=0.000870` |

## 5. CRA5 如何处理 ERA5

### 5.1 输入变量和标准化

CRA5 配置 `models/CRA5/config/vaeformer_era5_268v_1h.py` 定义的 ERA5 变量与当前 CAESAR 预处理保持一致：

- `7` 个 pressure variables，`37` 个 pressure levels
- `9` 个 single-level variables
- 共 `268` 通道
- `tp` 同样转换为 millimeter

`models/CRA5/cra5/dataset/era5_base_npy.py` 中：

- `get_mean_std()` 按每个变量/压力层读取 mean/std。
- 配置使用 `norm_type='channel'`。
- 配置设置 `is_norm=False`。
- `get_data()` 在 `not self.is_norm` 时执行 `normalization(data_tmp)`。

因此，在仓库内这套数据读取路径下，CRA5 对原始 ERA5 场执行一次 per-channel z-score，然后交给模型。

CRA5 API 的推理流程也一致：读取单时刻原始 ERA5 数据，执行 `(data - mean) / std`，再传给预训练模型；解码后可以使用同一组 mean/std 反归一化。

### 5.2 CRA5 的实际输入张量

CRA5 不是把每个气象通道拆开训练。README 的预训练示例明确给出：

```python
input_data_norm = torch.rand(1, 268, 721, 1440)
```

VAEformer 的模型定义同样固定：

```text
in_chans=268
out_chans=268
img_size=(721, 1440)
```

也就是说，一次压缩的是一个时间点上的完整全球气象场：

```text
[B, 268, 721, 1440]
```

这会让模型同时学习：

- 同一变量不同气压层之间的竖直相关性；
- 不同物理变量之间的相关性，例如风、温度、湿度、位势高度；
- 完整空间场中的大尺度结构和全球位置模式。

### 5.3 空间结构与模型结构

CRA5 使用 VAEformer，而非 CAESAR 的单通道视频编码器。对于 `model_version=268`：

| 项目 | CRA5 VAEformer |
| --- | --- |
| input/output channels | `268 / 268` |
| input image size | `(721, 1440)` |
| main patch size | `(11, 10)` |
| main patch stride | `(10, 10)` |
| latent/embed dimension | `256` |
| entropy model | hyperprior + GaussianConditional |

因此 CRA5 的效果不能简单归因于“ERA5 统计量用得更好”；更关键的是它的输入组织和模型结构正面利用了气象场的多变量相关性。

### 5.4 CRA5 训练目标能够确认到什么程度

仓库包含 `RateDistortionLoss` 与 VAEformer 的 `training_step()`：

- `bpp_loss` 从模型的 likelihoods 计算，即对 VAEformer 自身熵模型码率的可微估计。
- `mse_loss` 由重建场与输入场计算。
- VAEformer `forward()` 输出 `x_hat` 和 `likelihoods={"y", "z"}`。

这证明 CRA5 的模型实现支持在自身实际编码机制上做 rate-distortion 优化。

但本仓库未包含一份完整、可直接执行且记录了原始预训练 run 超参数的 CRA5 训练入口或训练日志。因此，以下内容不能仅凭当前仓库严谨确认：

- 公开预训练权重最终使用的准确 learning rate、训练轮数和 optimizer；
- 最终选择 checkpoint 时使用的具体主指标与超参权重；
- 是否存在未纳入仓库的额外训练阶段。

## 6. CRA5 与当前 CAESAR 的关键差异

| 方面 | CRA5 | 当前 CAESAR ERA5 微调 | 影响 |
| --- | --- | --- | --- |
| 样本含义 | 单时刻完整气象场 | 单变量时序 patch | CAESAR 丢失跨变量/跨层联合编码能力 |
| 输入形状 | `[B,268,721,1440]` | `[B,1,T,256,256]` | 优化对象不同 |
| 主要依赖关系 | 变量/层/空间联合 | 单变量的空间/时间 | CRA5 更匹配多变量 ERA5 |
| 标准化 | channel z-score 一次 | channel z-score 后再 patch `mean_range` | CAESAR 的训练空间被二次变换 |
| 模型 | VAEformer 2D multichannel | V/D 3D temporal/single channel | 不能仅换 dataloader 完全复用 CRA5 方法 |
| 码率目标 | likelihood-based model bitstream | VAE internal rate，最终另加 residual 后处理 | CAESAR internal best 可能不是 final RD best |
| 划分规模 | 配置为 1998-2017 train, 2018 val/test | 当前训练仅 61 个时刻，且 val 为同时间空间切片 | CAESAR 更容易快速适配/泛化不足 |

## 7. 为什么 tune 后反而更差

### 7.1 首要原因：训练目标与最终测试目标不一致

CAESAR Stage1 训练保存 best checkpoint 的标准是：

```text
MSE(model_output, normalized_patch) + lambda_rate * internal_rate
```

而最终 `CAESAR/compressor.py` 在 VAE/Diffusion 重构后，还调用：

```text
postprocessing_encoding(original_data, recons_data, eb)
```

该步骤用 `PCACompressor` 和 `Zstd` 对残差补偿，使输出满足 EB，并把 residual bytes 加入最终字节数。

因此存在直接错配：

- 训练改善 normalized patch MSE 或 internal latent bits；
- 但最终测试看的是 `latent bytes + residual bytes`；
- 一个让 VAE 内部 loss 降低的权重，可能产生更难由 residual coder 高效修正的误差结构，从而让总 BPP 变差。

这与观察完全一致：W&B 中内部验证 loss 持续改善，而最终 EB-RD 结果从很早开始就劣于 pre-tune。

### 7.2 数据量相对于训练步数过小

当前只有 `61` 个时间点，且每个通道独立训练。

粗略换算：

| 模型阶段 | 每遍历 items | effective batch | 10k updates 约相当于遍历次数 |
| --- | ---: | ---: | ---: |
| V Stage1 | 2144 | 32 | 约 149 次 |
| D Stage1 | 1072 | 32 | 约 299 次 |
| D Stage2 | 1072 | 64 | 约 597 次 |

所以第一个保存出来的 `10k` checkpoint 已经不是“非常早”的模型，而是已经在少量 ERA5 样本上反复适配很多轮。这解释了为何拐点落在 `0 -> 10k` 内。

### 7.2.1 V/D VAE 在 10k 前退化的直接机制

这里说的“效果下降”同时指两个方向：同一 EB 下总 BPP 上升，PSNR 下降。以 `EB=1e-3` 为例：

| 模型 | checkpoint | BPP | PSNR |
| --- | --- | ---: | ---: |
| V | pre-tune | 0.486375 | 62.3025 |
| V | update10k | 1.248262 | 60.8002 |
| D | pre-tune | 0.692245 | 62.3141 |
| D | Stage1 VAE update10k | 1.301470 | 60.9225 |

这说明退化不是单纯“码率变高但质量也变高”，而是最终 RD 曲线整体向坏方向移动。

从损失函数看，Stage1 VAE 优化的是：

```text
MSE(model_output, normalized_patch) + lambda_rate * internal_rate
```

这个目标与最终测试指标有三层错配：

1. MSE 在 `mean_range` patch-normalized 空间里计算，不是在最终 z-score ERA5 block 或物理量空间里直接计算。
2. `internal_rate` 是模型内部估计/统计的 latent bits；最终 BPP 是 `latent bytes + residual bytes`，残差由 EB 后处理、PCA 和 Zstd 决定。
3. 当前 `lambda_rate=1e-5` 且 `rate_mode=bits`，rate 项的尺度与最终 total BPP 并不等价；内部 loss 降低不保证最终 EB sweep 上的 BPP/PSNR 变好。

因此，VAE tune 可能让 normalized patch MSE 变好，但改变误差频谱、残差分布或 latent 分布，使 residual coder 更难压，最后出现 BPP 增加且 PSNR 降低。

从训练数据处理看，ERA5 已经先做了 CRA5 风格的逐通道 z-score，CAESAR 训练时又对每个 patch 做 `mean_range`：

```text
x_model = (x_zscore - patch_mean) / (patch_max - patch_min)
```

这会移除每个 patch 的绝对均值和幅度信息。对自然图像/视频预训练模型来说，这种局部归一化可能还能工作；但 ERA5 的变量、层次和空间位置带有强物理尺度，二次归一化会让训练目标更偏向局部纹理拟合，而不是最终压缩时真正关心的全局场误差和残差可压缩性。

另外，当前 CAESAR 把 `268` 个 ERA5 通道拆成独立的 `[1,T,256,256]` 单通道样本训练。训练集只有 `61` 个时间点，V 的 `10k` 已经约等于 149 遍 item，D Stage1 的 `10k` 约等于 299 遍 item。因此“10k 前”并不是真正的早期；模型已经有足够多步去快速偏离 pre-tune 权重。

从测试输入处理看，checkpoint sweep 使用预生成的 ERA5 test blocks，经 CAESAR 完整 `compress/decompress` 流程评估。这个流程包含 EB residual postprocessing，所以测试输入不是只看 VAE forward 的 normalized MSE。pre-tune 权重虽然没有 ERA5 专门适配，但它的 latent/reconstruction error 形态可能更适合现有 residual coder；tune 后 VAE 产生的误差虽然对训练 loss 友好，却可能让残差更碎、更难被 PCA/Zstd 压缩。

D 还有一个额外问题：Stage1 VAE 只调 VAE，而 early-transition 测试里这些 VAE checkpoint 是和原始 diffusion 搭配评估的。VAE tune 后 latent 分布和 keyframe reconstruction 分布发生偏移，原始 diffusion 看到的是已经变过的条件分布，所以 D 在 `0 -> 10k` 的掉点会比“只看 VAE 自身 MSE”更明显。后续 Stage2 diffusion 能补一部分，但不能修复 Stage1 已经造成的最终 RD 错配。

### 7.2.2 结合 CAESAR 论文后的修正判断

参考论文：`papers/unified.pdf`，即 Applied Sciences 2025 正式版
“CAESAR: A Unified Framework for Foundation and Generative Models for
Efficient Compression of Scientific Data”。

这次 ERA5 tune 的若干超参数确实来自论文：

| 项目 | 论文设置 | 当前 ERA5 tune |
| --- | --- | --- |
| VAE patch | `256 x 256` random crop | `256 x 256` random crop |
| VAE batch | `32` | `32` |
| VAE fine-tune lr | `1e-4` | `1e-4` |
| VAE fine-tune steps | V `100k`，D keyframe compressor `100k` | V `100k`，D Stage1 `100k` |
| lambda | `1e-5` 起步 | `1e-5` |
| D training | 先 keyframe VAE，再 diffusion | 同样两阶段 |

但论文同时隐含了几个当前 ERA5 不满足的前提：

1. 论文里的 ERA5 不是当前这套 268 通道数据。论文表述的 ERA5 形状是 `6 x 3 x 2160 x 512 x 512`，即 6 个变量、3 个垂直层、2160 个时间步。当前 CAESAR ERA5 使用的是 `268 x 61 x 721 x 1152` 训练输入，变量/层更多，但时间长度少了约 35 倍。
2. 论文说测试主要在 E3SM、S3D、JHTDB 上做，其他数据包括 ERA5 保留作训练；因此论文并没有直接证明这套 fine-tune recipe 在当前 ERA5 test split 上一定提升。
3. 论文写的是每帧归一化到 zero mean 和 unit range。当前默认 `mean_range` 是对整个 `[T,H,W]` patch 做一个 mean/range；代码里 `mean_range_hw` 才更接近“每帧”归一化。
4. 论文中的 fine-tune 数据时间覆盖远大于当前数据。当前 `10k` step 对 V/D Stage1 已经分别约等于 149/299 次遍历 item；论文的 `100k` 经验不能直接按 step 数迁移。
5. 论文明确说明 tight error bound 下 PCA-based postprocessing 会引入额外 bit cost，使压缩优势变小。这与当前观察到的 residual bytes/total BPP 对 VAE 误差形态敏感是同一机制。

因此，“按论文设置”本身没有错，但这些设置在论文里依赖更大的时序数据、更接近原始 CAESAR 数据组织的输入，以及论文报告过的 benchmark 分布。当前 ERA5 更像是把 CRA5 的多变量全球场拆成 CAESAR 单通道时序 patch 来微调，数据规模和归一化语义都变了，所以同样的 `lr=1e-4, lambda=1e-5, 100k steps` 会过强。

更准确的判断是：

- `loss = MSE + lambda * rate` 是论文公式，不是脚本写错；
- 出问题的是这个论文训练目标没有被用作最终 EB-RD 的 checkpoint selection；
- 当前 ERA5 数据量太小，导致论文 fine-tune step 数在这里变成强过拟合/强分布迁移；
- 当前默认 `mean_range` 比论文“each frame”归一化更粗，会把一个时序块的所有帧绑定到同一个 scale，可能放大 ERA5 时间/区域幅度差异；
- D Stage1 只训练 keyframe compressor 符合论文思路，但如果只替换 VAE 而 diffusion 仍是 pre-tune，就会产生 latent distribution mismatch；这解释了 D VAE 10k 处大幅下降。

### 7.3 输入组织没有利用 ERA5 最有价值的冗余

CRA5 通过 268 通道联合建模获得很强效果；当前 CAESAR 将各变量独立编码，无法利用同一时间点上：

- `u/v`、温度、湿度、位势高度之间的依赖；
- 相邻压力层之间的冗余；
- 单层变量和压力变量之间的关联。

CAESAR 的 temporal 设计仍可能利用单变量时间相关性，但在仅 61 个时刻的微调数据上，它舍弃的跨变量信息可能比新增的适配收益更重要。

### 7.4 D Stage2 变化小的具体解释

D Stage2 只训练 diffusion；VAE 已冻结。最终 reconstruction 和 residual correction 的基础误差形态很大程度由 Stage1 VAE 决定。

当前 Stage1 在 10k 时已经明显破坏最终 RD，后续 Stage2 只是从该基础上补插帧生成部分。因此：

- Stage2 可以让结果比 Stage1 稍好；
- 但无法恢复 pre-tune VAE 的原始压缩特征；
- 不同 Stage2 step 最终曲线接近，不表示 diffusion 没有学习，而是最终指标更受固定 VAE 与 residual postprocessing 支配。

## 8. 后续调整建议

### 8.1 立即执行：不改模型结构的低成本实验

#### A. 用最终 EB-RD 而不是内部 val loss 选择 checkpoint

训练每次保存权重后，在小型、固定的 ERA5 validation subset 上运行最终 compressor 的 EB sweep，记录：

- `Global RD`
- `Mean RD`
- 固定 EB 点的 total BPP / PSNR
- latent bytes 与 postprocessing bytes 分解

以最终 total BPP 或曲线指标选择 checkpoint，避免再次选到内部 loss 更好但最终码流更差的模型。

#### B. 将保存/评测间隔提前到 10k 之前

建议首次诊断范围：

```text
0, 100, 250, 500, 1000, 2000, 4000, 6000, 8000, 10000
```

V 与 D Stage1 都应这样做。当前实验已证明 `10k` 保存粒度无法定位退化起点。

#### C. 大幅降低适配强度

第一轮可比较：

| 实验 | LR | 最大 updates |
| --- | ---: | ---: |
| conservative-1 | `1e-5` | `10000` |
| conservative-2 | `3e-6` | `10000` |
| freeze/partial | `1e-5` | `10000` |

在最终 EB-RD 指标未超过 pre-tune 之前，不建议继续 `1e-4`、`100k/200k` 的长训练。

#### D. 增加时间独立验证

当前 longitude split 应保留为一种空间验证，但应另加 held-out timestamps。至少保证用于 checkpoint 选择的时间段与训练时间不重叠，否则容易把对同一时期的适配误判为泛化改善。

### 8.2 优先消融：归一化策略

应系统检查下面三种设置对最终 RD 的影响：

| 设置 | 输入给 CAESAR 的方式 | 目的 |
| --- | --- | --- |
| current | CRA5 channel z-score + patch `mean_range` | 当前 baseline |
| raw + CAESAR norm | 不预做 z-score，仅 patch `mean_range` | 验证重复标准化影响 |
| zscore-compatible model input | 仅 channel z-score，调整/绕过 instance norm | 接近 CRA5 表达，但需确保模型数值稳定 |

第三种不能只通过传一个现有参数完成，因为当前 CAESAR dataset 只支持 patch 归一化类型，且解压/后处理使用 offset/scale 约定。实施时需要同时校验训练、压缩、反归一化和 EB 误差计算路径。

### 8.3 进一步：分析最终字节来源

下一轮 sweep 应将总 BPP 分拆为：

```text
total_bpp = latent_bpp + postprocess_residual_bpp
```

若微调后 `latent_bpp` 下降但 `postprocess_residual_bpp` 大幅上升，就可以直接证明训练目标错配是 RD 变差的主要机制；若两者都变差，则还存在明显的表示退化或过拟合。

### 8.4 结构级改进：学习 CRA5 的核心优势

若目标是取得类似 CRA5 的 ERA5 专用表现，仅优化单通道 CAESAR 微调脚本不够。可选路线为：

1. 将输入从单变量改为多通道联合块，例如先按相关变量/垂直层分组，再逐步扩展到全 268 通道。
2. 设计同时保留时间维和变量维的模型输入，如 `[B, C_group, T, H, W]`，而不是把变量当独立样本。
3. 为 ERA5 定义与最终 EB/residual correction 更一致的目标函数或 checkpoint 选择策略。

这是模型结构和训练协议的修改，不能复用当前 pretrained CAESAR 权重而不处理首尾层、entropy model 和后处理协议的兼容问题。

## 9. 建议的下一轮实验顺序

按照成本和信息增益排序：

1. 扩展评测脚本输出 `latent_bpp` 与 `postprocess_bpp`，对已测的 pre-tune、V 10k、D Stage1 10k、D Stage2 10k 做字节来源核查。
2. 以 `1e-5` 和 `3e-6` 分别重新短训 V 与 D Stage1，保存 `100/250/500/1k/2k/4k/6k/8k/10k`，每个点跑固定的小型 EB sweep。
3. 采用时间独立 validation subset 选 checkpoint，并与当前 longitude validation 结论对照。
4. 做预处理/归一化消融，确认 z-score 后再次 patch `mean_range` 是否是主要不利因素。
5. 若短训和选择策略仍不能超过 pre-tune，再投入多通道/分组通道的结构性方案。

## 10. 代码依据索引

### 当前 CAESAR ERA5 路径

- `utils/prepare_era5_finetune_data.py`：变量、压力层、`tp * 1000`、CRA5 mean/std z-score、longitude split、mmap 形状。
- `utils/download_era5.py`：逐小时/日批 ERA5 下载、区域裁剪和 CDS 退避重试。
- `utils/prepare_era5_hourly_shards.py`：日批 NetCDF 到 `[268,T,H,W]` float32 mmap shard 的等价转换。
- `utils/era5_netcdf_dataset.py`：NetCDF/shard 后端、时间窗分组 sampler 和边缘 reflect padding。
- `utils/build_era5_hourly_validation_probe.py`：从 held-out 240 小时构建 30 通道真实 codec 选择 probe。
- `scripts/finetune_caesar_era5.py`：`ERA5MmapDataset`、单通道时序 patch 输入、patch `mean_range`、V/D 两阶段训练、内部 checkpoint selection。
- `scripts/run_caesar_era5_hourly_pilot.sh`：8-GPU、W&B 强制绑定的 10k V/D pilot。
- `scripts/run_caesar_era5_hourly_pilot_eval.sh`：早期 checkpoint 的 held-out 真实 3-EB 筛选。
- `models/CAESAR/CAESAR/compressor.py`：最终压缩流程和 `postprocessing_encoding()` residual bytes。

### CRA5 路径

- `models/CRA5/config/vaeformer_era5_268v_1h.py`：268 variables、`crop_size=(721,1440)`、`norm_type='channel'`、训练/验证年份配置。
- `models/CRA5/cra5/dataset/era5_base_npy.py`：原始 ERA5 整场加载、`tp * 1000`、per-channel z-score、整场 `input` 构建。
- `models/CRA5/cra5/api/cra5_api.py`：推理时读取、标准化、反标准化和 binary encode/decode 路径。
- `models/CRA5/cra5/models/vaeformer/vaeformer.py`：`in_chans=268`、`out_chans=268`、`img_size=(721,1440)`、hyperprior/likelihood 输出。
- `models/CRA5/cra5/models/compressai/losses/rate_distortion.py`：likelihood-based `bpp_loss` 与 reconstruction loss。
- `models/CRA5/Readme.md`：预训练模型输入样例 `[1,268,721,1440]`。

## 11. 最终判断

CRA5 效果好，最值得借鉴的不是单个超参数，而是三件事：

1. 将 ERA5 作为多变量联合气象场建模，而不是拆成独立变量；
2. 使用稳定的 per-channel 物理量标准化，并在同一输入语义下训练和评测；
3. 让 rate-distortion 优化和最终产生的实际码流尽量一致。

当前 CAESAR ERA5 微调首先需要修正 checkpoint 选择和训练强度，再验证归一化影响。只有在这些低成本问题排除后，才值得推进接近 CRA5 输入形式的多通道结构修改。
