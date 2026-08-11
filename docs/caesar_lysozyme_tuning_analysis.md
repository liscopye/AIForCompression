# CAESAR Lysozyme Fine-tune 与 Pre-tune 对比分析

## 1. 结论摘要

Lysozyme 上的 fine-tune 与 ERA5 的现象不同：现有有效结果显示，CAESAR-V 在相近高保真失真水平下显著降低码率；CAESAR-D 的旧结果也表现出同样趋势，但旧 D sweep 混用了不同大小的测试子集，因此不能直接作为严格曲线使用，本文采用统一测试子集的重测结果替代。

Lysozyme 更适合当前 CAESAR 输入假设：它是单通道连续衍射帧序列，一个模型样本对应同一物理信号随时间的变化。ERA5 则将大量异质变量/层分别当作单通道样本，丢失了变量之间的相关性，训练样本量也很有限。这是 Lysozyme tune 后 RD 表现更容易提升、而 ERA5 tune 后容易退化的核心结构性原因。

需要特别指出两个评估风险：

1. 早期成功训练的 V 和 D Stage 1 运行将 `lysozyme_test_nf16.npz` 作为 validation 数据选择最优 checkpoint，因此当前测试集参与过模型选择。现有结果适合定位效果和比较方案，不应当作完全无偏的最终报告数字。
2. 早期训练脚本对空间裁剪和归一化的顺序与正式测试不同；后来的 `finetune_caesar_fixed.py` 已将 D Stage 2 修正为与测试一致的路径。

## 2. 数据来源与当前文件

预处理脚本为 `scripts/prepare_lysozyme_data.py`。它从原始 HDF5 文件的 `entry/data/data` 中取每个文件的第 0 帧，转换为 `float32`，再做中心 `1024 x 1024` 裁剪。原始默认目录为：

```text
/workspace/Data/nfs/chess/raw/2018-1/g3/finke-707-2/20180305/lysozyme_chip3
```

当前 mmap 文件的实际形状如下：

| Split | 文件 | 形状 `[V,S,T,H,W]` | 含义 |
| --- | --- | ---: | --- |
| Train | `mmap/lysozyme_train_nf16.npy` | `[1,720,16,1024,1024]` | 11,520 帧 |
| Validation | `mmap/lysozyme_val_nf16.npy` | `[1,80,16,1024,1024]` | 1,280 帧 |
| Test | `mmap/lysozyme_test_nf16.npy` | `[1,200,16,1024,1024]` | 3,200 帧 |
| V Test | `lysozyme_test_nf8.npz` | `[1,400,8,1024,1024]` | 同一 Test 的 3,200 帧，改成长度 8 |

这里存在一个记录问题：`/workspace/Data/lysozyme_processed/metadata.json` 仍记录旧的 `train=800, test=200`，没有记录当前 `val=80`，与实际 mmap 文件不一致。后续重做数据时应让 metadata 与最终产物同时生成并保留 split index。

### 2.1 划分方式与数据泄漏核验

脚本注释写的是按时间顺序进行 `80/20` 切分，但代码实际实现为：

```python
np.random.seed(42)
indices = np.random.permutation(n_chunks)
```

也就是说，数据先按连续 16 帧组成 chunk，再随机按 chunk 分为 train/val/test；单个 chunk 内的时间连续性保留，但测试集不是未来时间段 holdout。

我对当前 train/val/test 文件逐帧做了完整内容哈希比对，结果如下：

| 比较 | 重合帧数 |
| --- | ---: |
| Train16 vs Val16 | 0 |
| Train16 vs Test16 | 0 |
| Val16 vs Test16 | 0 |
| Train16 vs Test8 | 0 |
| Val16 vs Test8 | 0 |
| Test16 vs Test8 | 3,200 |

`test_nf8` 与 `test_nf16` 是同一批测试帧，并且顺序一致，仅按 `8` 帧重新分组。因此 V 与 D 的测试数据在原始帧层面是可比的；当前文件没有 train/test 原始帧泄漏。

## 3. Train 输入如何处理

### 3.1 CAESAR-V 的已用训练路径

W&B 中可追踪到的成功 V 训练运行使用 `scripts/finetune_caesar.py`：

| 参数 | 设置 |
| --- | --- |
| 数据 | `lysozyme_train_nf16.npz` |
| validation | `lysozyme_test_nf16.npz` |
| 模型 | `V` |
| `n_frame` | `8` |
| crop | 随机 `256 x 256` |
| batch size | `32` |
| iterations | `100000` |
| learning rate | `1e-4` |
| `lambda_rate` | `1e-5` |
| 最佳内部 val loss | 约 `0.0020`，首次记录于 step `42000` |

输入变化为：

```text
[1, S, 16, 1024, 1024]
  -> 每个 16 帧 chunk 切成两个 8 帧样本
  -> [1, 8, 1024, 1024]
  -> mean_range 归一化
  -> 随机裁剪为 [1, 8, 256, 256]
```

早期脚本的归一化为一个样本块上的：

```text
x_norm = (x - mean(x)) / (max(x) - min(x))
```

其训练损失是 VAE 重建 MSE 与 frame bit rate 的线性组合：

```text
loss = MSE(reconstruction, x) + lambda_rate * mean(frame_bit)
```

注意：W&B 成功 V 运行记录的默认保存名为 `caesar_v_tuning_lysozyme_vae.pt`，而正式测试使用的是 `caesar_v_tuning_lysozyme.pt`。后者存在并已用于评估，但从已有运行记录中无法完整追溯它是如何由前者复制或生成的，后续实验应固定产物命名并保存 checkpoint provenance。

### 3.2 CAESAR-D 两阶段训练

CAESAR-D 的训练由 VAE keyframe 编码器和 diffusion 插帧/重建两阶段构成。

早期成功的 Stage 1 运行使用 `scripts/finetune_caesar.py`：

| 参数 | 设置 |
| --- | --- |
| 数据 | `lysozyme_train_nf16.npz` |
| validation | `lysozyme_test_nf16.npz` |
| 模型/阶段 | `D`, VAE Stage 1 |
| `n_frame` | `16` |
| batch size | `16` |
| iterations | `100000` |
| learning rate | `1e-4` |
| `lambda_rate` | `1e-5` |
| 最佳内部 val loss | 约 `0.0007`，记录于 step `86000` |

这份旧脚本在 Stage 1 中将完整 `16` 帧样本送入 VAE 损失路径。新版 `finetune_caesar_fixed.py` 默认只训练 D 实际会压缩的 keyframes，即帧索引 `0, 3, 6, 9, 12, 15`，后者与推理机制更一致。

D 的后期 Stage 2 使用了修正脚本 `scripts/finetune_caesar_fixed.py`，可追踪运行设置为：

| 参数 | 设置 |
| --- | --- |
| Train | `mmap/lysozyme_train_nf16.npy`, `[1,720,16,1024,1024]` |
| Validation | `mmap/lysozyme_val_nf16.npy`, `[1,80,16,1024,1024]` |
| 模型/阶段 | `D`, diffusion Stage 2 |
| VAE | 固定 `caesar_d_tuning_lysozyme_vae.pt` |
| 初始化 full checkpoint | `caesar_d_tuning_lysozyme_step270000.pt` |
| batch size / accumulation | `32 / 2` |
| iterations 计划值 | `130000` |
| learning rate | `1e-4` |
| 记录到的 best | step `68000` 附近，val loss 约 `0.000011` |

修正脚本在空间裁剪之后做 `mean_range` 归一化，并使用推理路径一致的量化 latent (`inference_qlatent`) 训练 diffusion。这个顺序和 latent 语义均比旧脚本更可靠。

### 3.3 训练与测试之间的输入差异

早期 V/D Stage 1 使用的旧脚本顺序是：

```text
完整 1024 x 1024 时序块 -> mean_range 归一化 -> 随机/中心裁剪为 256 x 256
```

正式测试以及新版固定训练脚本的顺序是：

```text
完整 1024 x 1024 时序块 -> 切/裁为 256 x 256 -> 每个小块单独 mean_range 归一化
```

衍射图像存在局部峰值和大动态范围，这两种顺序得到的输入数值分布不等价。Lysozyme 已经取得改善，说明模型仍学习到了有用结构；但进一步实验应统一使用新版顺序，否则 train/test distribution shift 会影响 checkpoint 比较。

## 4. Test 输入与最终指标如何生成

测试入口是 `scripts/eval_caesar_lysozyme.py`，内部使用 `models/CAESAR/dataset.py` 的 `ScientificDataset`。正式 sweep 的路径如下：

| 模型 | 测试文件 | 时间长度 | 输入通道 |
| --- | --- | ---: | ---: |
| CAESAR-V | `lysozyme_test_nf8.npz` | 8 | 1 |
| CAESAR-D | `lysozyme_test_nf16.npz` | 16 | 1 |

对于当前 sweep 的 `--max_blocks 10`，D 测试先选取 10 个 `[16,1024,1024]` 时序块，再将每块分成 16 个 `256 x 256` tile，因此模型看到的测试张量是：

```text
[1, 10, 16, 1024, 1024]
  -> [1, 160, 16, 256, 256]
  -> 每个 [16,256,256] tile 单独 mean_range 归一化
```

模型输出之后，还会执行基于指定 error bound (`EB`) 的残差后处理：`PCACompressor` 与 Zstd 对误差进行编码。最终 BPP/CR 使用的字节数包含 latent bytes 与这部分 residual metadata/data bytes。因此这里比较的是完整 compressor 的 Rate-Distortion 结果，不只是网络重建 MSE。

指标含义：

| 指标 | 含义 |
| --- | --- |
| BPP | 总压缩字节数乘 8，再除以原始像素/数值个数；越小越省空间 |
| CR | 原始字节数 / 压缩字节数；越大越省空间 |
| Global PSNR / NRMSE | 把全部测试值合在一起累计误差后计算，受高幅值区域影响较大 |
| Mean PSNR / NRMSE | 先逐 tile/样本算指标再取平均，使每个测试块权重更接近相等 |

## 5. CAESAR-V: 有效的 Pretune vs Fine-tune 结果

现有 V 七点 sweep 的 original 与 finetuned 在每个 EB 上均使用同样大小的测试数据，`original_size_bytes=1677721600`，可以直接比较：

| EB | Pretune BPP | Tune BPP | Pretune PSNR | Tune PSNR | BPP 变化 | PSNR 变化 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `1e-5` | 9.400133 | 9.258426 | 101.321 | 101.087 | -0.141707 | -0.233 dB |
| `5e-5` | 5.428954 | 3.598093 | 86.799 | 86.697 | -1.830861 | -0.102 dB |
| `1e-4` | 4.068389 | 2.462285 | 80.679 | 80.617 | -1.606105 | -0.062 dB |
| `5e-4` | 1.150178 | 0.456805 | 67.628 | 67.343 | -0.693374 | -0.285 dB |
| `1e-3` | 0.596118 | 0.196876 | 64.160 | 62.382 | -0.399242 | -1.778 dB |
| `5e-3` | 0.207083 | 0.021372 | 56.279 | 53.089 | -0.185711 | -3.191 dB |
| `1e-2` | 0.112301 | 0.005730 | 51.776 | 51.210 | -0.106571 | -0.566 dB |

最值得采用的是高保真区域：在 `EB=1e-4` 时，fine-tune 将 BPP 减少约 `39.5%`，PSNR 仅下降 `0.062 dB`；在 `EB=5e-5` 时，BPP 减少约 `33.7%`，PSNR 下降 `0.102 dB`。说明 fine-tune 主要把同等误差容限下的码率压低，而不是在固定 EB 上把 PSNR 显著抬高。

V 的已生成对比图位于：

```text
results/eb_sweep_V/caesar_v_lysozyme_comparison.png
```

## 6. CAESAR-D: 旧结果问题与修正评估

原 `results/eb_sweep_D` 目录不能直接用于结论：多数点的 original 和 finetuned 使用了 `671088640` 或 `3355443200` 字节的同一测试规模，但 `EB=1e-3` 这一对分别使用了 `3355443200` 与 `2013265920` 原始字节，明显不是相同测试子集；整条曲线还混接了不同测试规模的点。

因此 D 结果应以新的统一测试集重跑为准。修正评估统一设置为：

```text
model_type = D
ckpt = both (pretune 与 finetuned)
EB = [1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2]
max_blocks = 10
测试张量形状 = [1,160,16,256,256]
```

修正后的七点 JSON 汇总与图保存在同一目录中：

```text
results/eb_sweep_D_corrected/
```

七点重测已经完成。所有 original/tuned 文件的 `original_size_bytes` 均为 `671088640`，因此下表是严格配对比较：

| EB | Pretune BPP | Tune BPP | Pretune PSNR | Tune PSNR | BPP 变化 | PSNR 变化 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `1e-5` | 8.608904 | 5.572732 | 101.418 | 101.512 | -3.036171 | +0.095 dB |
| `5e-5` | 5.165330 | 1.940567 | 86.741 | 86.759 | -3.224764 | +0.018 dB |
| `1e-4` | 4.033166 | 1.370535 | 80.594 | 80.588 | -2.662630 | -0.006 dB |
| `5e-4` | 1.381227 | 0.374027 | 67.353 | 67.323 | -1.007201 | -0.031 dB |
| `1e-3` | 0.722797 | 0.162756 | 62.748 | 62.441 | -0.560041 | -0.307 dB |
| `5e-3` | 0.183108 | 0.025480 | 55.059 | 54.902 | -0.157629 | -0.157 dB |
| `1e-2` | 0.102849 | 0.009438 | 51.582 | 48.544 | -0.093411 | -3.039 dB |

D 的收益比 V 更明显。在 `EB=1e-4` 时，BPP 从 `4.033166` 降到 `1.370535`，降低约 `66.0%`，PSNR 仅下降 `0.006 dB`；在 `EB=5e-4` 时，BPP 降低约 `72.9%`，PSNR 仅下降 `0.031 dB`。这说明 D 两阶段 fine-tune 在高保真工作区有效降低了需要编码的数据量。`EB=1e-2` 时 PSNR 下降约 `3.04 dB`，因此极低 BPP 端不适合作为质量保持的结论点。

D 的修正图位于：

```text
results/eb_sweep_D_corrected/caesar_d_lysozyme_corrected_comparison.png
```

### 6.1 为什么 D Tune 的吞吐量曲线不稳定

当前图中的 throughput 不能理解为单纯的神经网络推理速度。测试脚本用 `原始数据 MB / 完整 compress 或 decompress 墙钟时间` 计算吞吐量；而完整流程包含：

```text
Encode = keyframe latent 编码 + diffusion 重建一次 + PCA/Zstd 残差编码
Decode = keyframe latent 解码 + diffusion 重建一次 + 可选 PCA/Zstd 残差解码
```

特别是 D 的 `compress()` 为了生成满足 EB 的残差，会先运行一次 `decompress_caesar_d()`；因此 encode 本身已经包含一次 diffusion 重建。`decompress()` 中只有当 `meta_data["data_bytes"] > 0` 时才运行 PCA 残差恢复，其耗时会随 EB 触发的 residual mask/coeff 结构改变，而不是只随网络大小改变。

修正 sweep 中 D tuned 的总 decode 时间如下：

| EB | Decode 时间 | Decode 吞吐量 |
| ---: | ---: | ---: |
| `1e-5` | 331.56 s | 1.93 MB/s |
| `5e-5` | 100.34 s | 6.38 MB/s |
| `1e-4` | 29.97 s | 21.35 MB/s |
| `5e-4` | 29.88 s | 21.42 MB/s |
| `1e-3` | 114.03 s | 5.61 MB/s |
| `5e-3` | 183.33 s | 3.49 MB/s |
| `1e-2` | 29.75 s | 21.51 MB/s |

此外，最初为尽快得到 RD 结论，七个 EB 点使用七张 GPU 并行评估。GPU 模型计算彼此独立，但 Zstd、CPU 数据搬运和内存带宽仍由同一主机共享；这种运行方式适合快速获得配对 RD 点，不适合作为精确吞吐量 benchmark。

这不是 diffusion 随机性导致的：`decompress_caesar_d()` 每次执行都重置 `torch`/CUDA seed 为 `2025`，并且 compressor 初始化启用了 cuDNN deterministic，当前重建基底是可复现的。

为排除资源竞争，我随后使用 GPU 0 按 EB 顺序串行重跑全部七点，每个 EB 内仍为 pretune 后 tuned。串行测试的输入规模与 RD 数字均与配对评估完全一致，仅计时发生变化：

| EB | Tune Encode 并行 | Tune Encode 串行 | Tune Decode 并行 | Tune Decode 串行 |
| ---: | ---: | ---: | ---: | ---: |
| `1e-5` | 566.82 s | 28.05 s | 331.56 s | 22.09 s |
| `5e-5` | 506.55 s | 25.45 s | 100.34 s | 19.41 s |
| `1e-4` | 505.30 s | 25.47 s | 29.97 s | 20.99 s |
| `5e-4` | 524.27 s | 26.21 s | 29.88 s | 21.16 s |
| `1e-3` | 499.56 s | 24.08 s | 114.03 s | 22.31 s |
| `5e-3` | 436.39 s | 26.79 s | 183.33 s | 19.65 s |
| `1e-2` | 504.43 s | 22.39 s | 29.75 s | 19.28 s |

串行后 tuned encode 时间集中在 `22.39-28.05 s`，decode 时间集中在 `19.28-22.31 s`；原先从 `1.93` 到 `21.51 MB/s` 的 decode 吞吐跳变消失。由此可以确认，并发版本的吞吐量异常主要来自七任务同时争用 CPU/内存/后处理资源，而不是 fine-tune 权重在不同 EB 上导致运行速度突变。

串行吞吐量对比图位于：

```text
results/eb_sweep_D_serial_gpu0/caesar_d_lysozyme_serial_gpu0_comparison.png
```

目前可以采用串行图作为单次速度观测；若要正式报告稳定 benchmark，仍应在单卡条件下预热并重复至少三次，同时分别记录 `latent model`、`diffusion`、`PCA/Zstd residual` 三部分耗时和 residual bytes。

## 7. 为什么 Lysozyme Tune 表现比 ERA5 好

| 项目 | Lysozyme | ERA5 |
| --- | --- | --- |
| 数据语义 | 单通道衍射图像随时间变化 | 多变量、多高度层气象场 |
| CAESAR 实际输入 | `[B,1,T,H,W]` 正好表示同一信号的时间演化 | 将不同变量/层拆成独立单通道样本 |
| 丢失的信息 | 较少，主要是 tile 边界与归一化尺度 | 变量间、垂直层间相关性基本不进入模型 |
| 数据量/多样性 | 大量空间 tile 与连续帧，域特征稳定 | 可用时次很少，变量统计分布差异大 |
| Fine-tune 收益来源 | 学习衍射峰结构和局部时序模式，降低 latent/残差码率 | 容易对有限时间片和单变量统计过拟合 |
| 已见现象 | 高保真区明显降低 BPP | tune 后整体 RD 退化或不稳定 |

Lysozyme 的提升并不意味着现有训练目标已完全等价于最终压缩目标。训练 loss 主要约束 VAE/diffusion 的重建与内部码率，而评估还包含 EB 残差编码。Lysozyme 上两者方向恰好较一致；ERA5 的退化说明仍需要用最终完整 codec 的 validation RD 来选 checkpoint。

## 8. 后续调整建议

1. 用独立 final-test 重做正式报告。当前 V 与 D Stage 1 曾使用测试集做 checkpoint 选择，应从原始数据重新留出一个从未参与选择的最终测试集合。
2. 固定为 crop/tile 后再归一化。训练统一迁移到 `finetune_caesar_fixed.py` 的输入处理，避免旧脚本与测试路径不一致。
3. 修复数据记录。更新预处理脚本中“按时间切分”的错误说明，生成包含 train/val/test 索引、形状与文件哈希的 metadata。
4. 对 D 分阶段选择 checkpoint。Stage 1 仅训练实际被编码的 keyframes；Stage 2 用冻结的最佳 VAE 并以完整 codec RD 或至少多个 EB 的验证曲线选择 diffusion 权重。
5. 保持 sweep 可复现。每个 EB 的 original/tuned 必须检查 `original_size_bytes` 和输入 tile 数一致后才纳入同一张图。
6. 保存权重来源链路。为 V 和 D 记录初始化 checkpoint、best step、脚本版本、训练数据 split 和最终导出文件名，避免 `.pt` 与 `_vae.pt` 之间不可追溯。

## 9. 代码与产物索引

| 内容 | 路径 |
| --- | --- |
| 数据预处理 | `scripts/prepare_lysozyme_data.py` |
| 早期 V / D Stage 1 训练 | `scripts/finetune_caesar.py` |
| 修正后的 mmap / D Stage 2 训练 | `scripts/finetune_caesar_fixed.py` |
| Lysozyme 测试入口 | `scripts/eval_caesar_lysozyme.py` |
| 正式测试数据处理 | `models/CAESAR/dataset.py` |
| CAESAR 编解码与残差后处理 | `models/CAESAR/CAESAR/compressor.py` |
| V 已有结果与图 | `results/eb_sweep_V/` |
| D 修正重测输出 | `results/eb_sweep_D_corrected/` |
| D 单卡串行吞吐量复核 | `results/eb_sweep_D_serial_gpu0/` |
