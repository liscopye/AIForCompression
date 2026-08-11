# Objective-v1 全模型全数据集基准观察

## 结论摘要

最终主结果位于 `unified_results/objective_all_to_all_v1`。严格审计为 1642/1642 条记录合规、10/10 个数据集完整。每个点使用 2 次预热和 5 次正式重复；所有 CAESAR-V、CAESAR-D 和 cuSZ-Hi 曲线均保留 7 个完整、有效、单调的 corpus 点。Turb-Rot 另外包含 CAESAR-V/D 的数据集微调权重，各 7 个点。

本轮学习式图像模型只包含 DCAE 和 LIC-HPCM（base/large），不包含 LIC-TCM。其他主方法为 CAESAR-V/D、cuSZ-Hi、nvJPEG/nvJPEG2000、DCMVC 和 DCVC-RT。非视频数据只用视频模型的 I-frame；UVG 只用完整 I+P 序列结果。

## Turb-Rot 微调权重

Turb-Rot 专用权重 `caesar_v_tuning_Turb-Rot.pt` 和 `caesar_d_tuning_Turb-Rot.pt` 使用与原始权重完全相同的两个 256-plane section、外部归一化、CAESAR 内部 ScientificDataset/PCA 路径、batch 64 和端到端计时边界。它们作为 Turb-Rot 数据集内消融曲线，不替代原始 CAESAR，也不进入其他数据集。

两种微调权重都使用 `EB=[0.3, 1e-3, 3e-4, 1e-4, 3e-5, 1e-6, 3e-9]`。聚合后 CAESAR-V-tuned 覆盖 0.254-29.78 BPP，CAESAR-D-tuned 覆盖 0.298-28.22 BPP，7 点均单调。相同 EB 下，微调权重通常以更低 BPP 达到接近的 PSNR；例如 `EB=1e-4` 时 V 从 7.53 降到 5.45 BPP，D 从 6.09 降到 4.29 BPP，而 PSNR 都约为 81.7 dB。低端 V-tuned 也从原始 V 的 47.87 dB 提升到 53.09 dB，同时 BPP 从 0.264 略降到 0.254。

后续吞吐量与 BPP 汇总表中的“CAESAR”仍指原始权重，避免把原始模型和数据集专用微调消融混成一个统计分布；微调曲线的逐点吞吐量和质量可在 HTML 的 Turb-Rot 页面查看。

主要观察如下：

- DCAE、LIC-HPCM、DCMVC/DCVC-RT 主要覆盖低 BPP；CAESAR 主要覆盖中高 BPP。
- CAESAR 在多数数据集把高端扩展到约 29-34 BPP；Lysozyme 因 batch=64 下低 EB 显存上限，只覆盖到约 1.35 BPP。
- cuSZ-Hi 的最高有效 BPP 强烈依赖数据和 3D 形状，约为 0.005-3.40 BPP；达到内部错误、误差界失败或非单调区后不再继续。
- nvJPEG/nvJPEG2000 和 cuSZ-Hi 在不少重叠区有很强的 RD 与吞吐量，但这不等价于学习模型失去意义；不同方法覆盖的 BPP、误差保证、跨数据泛化和解码语义不同。
- 只应在同一数据集、实测 BPP 重叠区内比较 RD，不能跨曲线空白区外推。

## 公平性边界

benchmark 只控制 codec 外部输入：样本、crop、mask、canonical float32 数值、数据集级外部归一化和 checksum。同一数据集的所有 codec 从相同 normalized canonical tensor 开始。

模型内部行为保持原实现：CAESAR 的 `ScientificDataset`、实例归一化、8/16 帧分区、latent、diffusion 和 PCA；cuSZ-Hi 的 predictor、量化和 Huffman；学习图像/视频模型的 padding、latent 和熵模型；nvJPEG(2000) 的颜色或小波变换。benchmark 不替模型修正内部行为，只隔离失败结果。

整块输入策略如下：

- S2C：4 个 tile x 4 个 band，形成 16-plane corpus stack。
- Kodak：24 张图 x RGB，方向可逆对齐后形成 72-plane stack。
- UVG 1080p：30 个连续 RGB frame，以时间为 3D 深度。
- Hurricane、NYX、Tomo 等使用完整科学 3D 体；ERA5 使用全部 268 个变量；Lysozyme 使用约 500-depth chunk。
- CAESAR 只在输入深度不满足 8/16 时重复最后一层 padding，评价前严格裁掉。

Lysozyme 的 `4294967295` 是共享 benchmark mask 元数据：不参与 range、EB、MSE 和 PSNR，所有 codec 看到同一有效区域，mask 本身计为零码率元数据。

## 计时定义

端到端 wall time 从 host canonical tensor 开始，到内存 bitstream，再解码并重组为 host canonical tensor。它包含接口转换、H2D/D2H、模型或 predictor、熵编码、PCA、反变换和重组；不包含磁盘 I/O、模型构造/权重加载、PSNR、LPIPS 和绘图。

跨 codec 排名优先使用 `wall_throughput_MBps`。编码和解码吞吐量是诊断项，不能用单个 CUDA kernel 时间替代端到端值。CAESAR 使用作者评估路径对应的推理 batch 64。

## 吞吐量实测

下表使用各方法 Pareto 点的端到端吞吐量。AI 图像/视频包括 DCAE、LIC-HPCM、DCMVC/DCVC-RT；传统/显式控制包括 cuSZ-Hi 和 nvJPEG(2000)。数值为均值（最小-最大），单位 MB/s。

| 数据集 | AI 图像/视频 | CAESAR | 传统/显式控制 |
|---|---:|---:|---:|
| E3SM | 13.6 (4.63-29.4) | 9.16 (0.41-25.5) | 3.79 (2.39-5.27) |
| ERA5 | 11.7 (3.88-34.6) | 26.8 (4.50-45.8) | 2.53 (1.67-3.72) |
| Hurricane | 38.8 (19.1-78.0) | 15.5 (1.96-35.2) | 21.5 (13.6-43.1) |
| NYX | 39.1 (16.5-86.7) | 16.9 (2.63-51.1) | 60.7 (39.0-79.7) |
| Turb-Rot | 18.3 (5.36-35.4) | 18.6 (1.48-37.1) | 20.1 (6.61-37.9) |
| Tomo | 33.0 (14.3-80.9) | 24.2 (2.75-47.7) | 46.8 (19.6-75.8) |
| Lysozyme | 55.0 (20.5-242) | 36.9 (21.9-53.7) | 42.1 (34.5-47.0) |
| S2C | 39.0 (16.3-103) | 1.39 (1.27-1.62) | 21.0 (5.76-42.1) |
| Kodak | 38.1 (16.3-136) | 17.4 (2.09-32.9) | 24.7 (3.78-52.7) |
| UVG 1080p | 60.5 (21.1-332) | 12.7 (1.98-29.4) | 35.9 (14.5-64.3) |

CAESAR 在 ERA5 上能较好摊薄固定开销，但 S2C 16-plane、UVG 30-frame 和高 BPP PCA 残差端明显变慢。高 BPP 端的码流、CPU 重组和 D 版本 diffusion 都是真实端到端成本，因此加入约 32 BPP 端点后，CAESAR 平均吞吐量低于旧的中段曲线统计。

## BPP 覆盖

| 数据集 | AI 图像/视频 | CAESAR | cuSZ/nvJPEG(2000) |
|---|---:|---:|---:|
| E3SM | 0.0072-0.109 | 0.263-32.76 | 0.0179-4.66 |
| ERA5 | 0.0092-0.141 | 0.243-30.35 | 0.0184-3.49 |
| Hurricane | 0.0055-0.141 | 0.203-30.94 | 0.0044-4.24 |
| NYX | 0.0016-0.0053 | 0.0366-33.16 | 0.0038-0.0204 |
| Turb-Rot | 0.0052-0.091 | 0.264-32.40 | 0.0155-3.88 |
| Tomo | 0.0017-0.0064 | 0.269-30.68 | 0.0041-2.70 |
| Lysozyme | 0.0015-0.0053 | 0.218-1.35 | 0.0009-0.678 |
| S2C | 0.0043-0.166 | 0.197-30.39 | 0.0009-4.77 |
| Kodak | 0.0109-0.330 | 0.342-33.56 | 0.0452-3.40 |
| UVG 1080p | 0.0003-0.0582 | 0.301-28.96 | 0.0347-2.60 |

这张表解释了为什么很多图上 DCAE/HPCM 与 CAESAR 几乎不相接。CAESAR 的基础 latent 形成码率下限，EB 主要控制 PCA 残差；DCAE/HPCM 的点来自不同权重。cuSZ/nvJPEG(2000) 能填补部分中间区域，但 cuSZ 的稳定高端通常在远低于 32 BPP 处结束。

## 单调性与失败处理

最终曲线先要求完整 sample 覆盖，再要求有限指标、cuSZ 误差界成立，并删除被支配点。probe 结果永久不进入正式主结果。

本轮发现并处理了三类边界问题：

- cuSZ 在过小 EB 时可能出现误差界失败、负 PSNR、Huffman `exceeding max len` 或 CUDA 非法访问；曲线止于最后完整有效点。
- Tomo 的单 sample probe 曾错误影响 EB 选择；调度器已修复为每个候选控制值必须覆盖完整 objective sample 清单。
- 精确无损重建的 `MSE=0` 原先写成非标准 JSON `Infinity`；现在按固定尺度公式 `max(MSE,1e-30)` 统一封顶为 300 dB。

最终主结果中没有 `NaN/Infinity`，没有错误行或 sample 不完整点。NYX、Tomo、Lysozyme 中部分固定权重方法的被支配点只从 Pareto 主图隐藏，原始合规点仍保留在 JSON。

## 指标解释

论文主质量指标是固定单位范围上的 `normalized_mse/normalized_psnr`，其中 `normalized_psnr=-10 log10(max(normalized_mse,1e-30))`。legacy `psnr` 只作样本动态范围诊断。

LPIPS 在 Kodak 和 UVG 的每个正式点均已统计。科学标量场没有统一、领域认可的 RGB 渲染，逐平面自适应渲染会破坏物理幅值，因此不把科学 LPIPS 作为主排名指标。

## 可复现产物

- 合并结果：`unified_results/objective_all_to_all_v1/combined_summary.json`
- EB 调度：`unified_results/objective_all_to_all_v1/eb_schedule.json`
- 严格审计：`unified_results/objective_all_to_all_v1/objective_protocol_audit.json`
- 分析 JSON：`unified_results/objective_all_to_all_v1/analysis/objective_analysis.json`
- 单文件中文结果索引：`unified_results/objective_all_to_all_v1/index.html`
- 协议：`benchmark_protocols/objective_v1.json`
- 协议说明：`docs/objective_benchmark_protocol.md`

当前结果支持同一数据集、实测重叠 BPP 区间内的公平比较；不支持跨空白区外推，也不把 codec 内部 kernel time 当作统一端到端结论。
