# 压缩基准复现与迁移清单

本文档记录当前十个数据集的正式测试口径、模型集合、控制点、输入组织和复现入口。目标是迁移环境后仅依赖本文档、协议文件和脚本即可重新生成结果，不再从历史目录名推断实验设置。

## 1. 唯一可信入口

正式论文结果使用 `aifc-objective-v1`：

- 协议：`benchmark_protocols/objective_v1.json`
- 数据加载与样本选择：`compression_pipeline/objective_data.py`
- 3D/corpus 堆叠：`compression_pipeline/objective_stacking.py`
- 输入 manifest：`scripts/prepare_objective_inputs.py`
- 图像、科学模型和传统 codec：`scripts/run_objective_benchmark.py`
- UVG P-frame：`scripts/run_objective_video.py`
- 合并与严格筛选：`scripts/build_objective_all_to_all.py`
- 审计：`scripts/audit_objective_benchmark.py`
- 画图和网页：`scripts/analyze_objective_benchmark.py`
- 当前完整结果：`unified_results/objective_all_to_all_v1`

以下目录是历史结果或专项消融，不能直接与 objective-v1 主结果混合：

- 旧 `unified_results/final/all_models_*`：部分数据只用了 3 channel 或单 sample，已在 2026-08-11 清理；历史模型去向见正式 `index.html`。
- `unified_results/caesar_era5_daily_*`：CAESAR 训练诊断和直接 reconstruction 结果。
- `unified_results/caesar_era5_daily_v_100k_eb_compare`：268 变量、16 天的初始/微调专项对比，但没有使用 objective-v1 的统一外部归一化。
- `DCAE/HPCM+CAESAR-PCA`、`CAESAR no-PCA`：消融结果，除非按 objective-v1 重新运行，否则不进入主模型排名。

## 2. 公平性边界和指标

同一数据集的所有 codec 必须接收相同的 canonical float32 数值、相同 crop、相同 mask 和相同数据集级固定归一化。benchmark 不改写模型内部行为：

- CAESAR 自己的 instance normalization 和 PCA 保持原实现。
- cuSZ-Hi 的 predictor、EB 解释和内部分块保持原实现。
- nvJPEG/nvJPEG2000 的颜色变换、整数转换和 codec 内部实现保持原样。
- DCAE/HPCM/DCMVC/DCVC-RT 的 padding、latent、熵模型和内部颜色变换保持原样。

主码率：

```text
BPP = 8 * (payload_bytes + side_info_bytes) / canonical_symbol_count
```

科学数据主质量指标：

```text
normalized_mse = mean_v(MSE_v / frozen_scale_v^2)
normalized_psnr = -10 * log10(normalized_mse)
```

`frozen_scale_v` 来自数据集完整 objective corpus，不按 sample 或模型重新计算。Kodak/UVG 使用 RGB PSNR、MS-SSIM 和 LPIPS；科学数据的 LPIPS 只作诊断，不作为主排名指标。

正式时间边界是 host memory 中的 canonical tensor 到内存 bitstream，再到 host memory 中完整重建 tensor。包括 H2D/D2H、模型内部归一化、padding、PCA、熵编码和重组；不包括磁盘读取、模型初始化、权重加载、指标计算和绘图。正式吞吐量必须使用 `2` 次预热和 `5` 次测量。

## 3. 模型和控制参数

| 模型族 | 正式控制点 | 输入路径 |
|---|---|---|
| DCAE | 6 个 MSE 权重：`0.0018, 0.0035, 0.0067, 0.013, 0.025, 0.05` | 2D 三通道组 |
| LIC-HPCM-base | 6 个 MSE 权重：`0.0018, 0.0035, 0.0067, 0.013, 0.025, 0.0483` | 2D 三通道组 |
| LIC-HPCM-large | 同上 6 个权重 | 2D 三通道组 |
| CAESAR-V | 每个数据集 7 个 EB，8-plane 窗口 | `[V,S,T,H,W]`，`S=1,T=8` |
| CAESAR-D | 每个数据集 7 个 EB，16-plane 窗口 | `[V,S,T,H,W]`，`S=1,T=16` |
| cuSZ-Hi | 每个数据集 7 个 EB | 真正 3D volume |
| nvJPEG2000 | target PSNR `20,30,40,50,60,70,80` | 科学数据逐变量 3D stack |
| nvJPEG | quality `1,5,10,25,50,75,95` | Kodak/UVG RGB frame |
| DCMVC-I | `q_index=0,1,2,3` | 非视频数据的 2D I-frame |
| DCVC-RT-I | `qp=0,21,42,63` | 非视频数据的 2D I-frame |
| DCMVC-IP | `q_index=0,1,2,3` | UVG 30 帧 GOP，含 I-frame |
| DCVC-RT-IP | `qp=0,21,42,63` | UVG 30 帧 GOP，含 I-frame |

当前正式集合不包含 LIC-TCM、Visemz 和 GraphComp。DCAE/HPCM+CAESAR-PCA 与 CAESAR no-PCA 是单独消融，不是 objective-v1 主曲线。

主要权重路径：

```text
checkpoints/dcae/*.pth.tar
checkpoints/lic-hpcm/hpcm-base/mse/*
checkpoints/lic-hpcm/hpcm-large/mse/*
checkpoints/caesar/caesar_v.pt
checkpoints/caesar/caesar_d.pt
checkpoints/dcmvc/cvpr2023_image_psnr.pth.tar
checkpoints/dcvc-rt/cvpr2025_image.pth.tar
```

ERA5 当前低码率和图像 codec 过渡区的最佳 CAESAR-V 微调权重：

```text
checkpoints/caesar_era5_v_decoder_quality_100k/from_lowrate_lr3em4.pt
```

它是在低码率 100k 权重上冻结 encoder/entropy-rate 路径、只微调 decoder 的验证集最佳权重。训练和真实 codec 结果见第 9.4 节。此前的 `caesar_era5_daily_v_full_100k` 权重仍保留，适合复现第 9.1 节的历史实验。

## 4. 数据集和 canonical 输入

统一 canonical layout 是 `[V,T,H,W]`。`T` 可以是真实时间、Z、投影角度或可逆 corpus 深度。

| 类别 | 数据集 | 原始路径 | objective 输入和组织 |
|---|---|---|---|
| 通用图像 | Kodak | `/workspace/Data/Kodac/kodim*.png` | 全部 24 张原图；每张 `[3,1,H,W]`。CAESAR/cuSZ 将全部 RGB 按 image-major、RGB-minor 堆成 72 planes；竖图可逆旋转后统一空间 shape。 |
| 通用视频 | UVG Twilight 1080p | `/workspace/Data/UVG_Twilight_1080p` | 连续 30 帧 `[3,30,1080,1920]`。CAESAR 以 RGB 为 V、真实时间为 T；cuSZ 每个 RGB channel 一个 time-3D volume；视频模型使用完整 GOP。 |
| 科学场 | E3SM | `/workspace/Data/E3SM/caesar_processed/e3sm_5vars_paperlike240_first800_caesar.npz` | 5 变量、section 0、`t=0:16` 和 `t=400:416` 两块，均为 `[5,16,240,240]`。 |
| 科学场 | ERA5 | `/workspace/Data/ERA5/finetune_processed/era5_test.npy` | 全部 268 个 CRA5-normalized 变量、前 16 个测试时刻、中心裁切 240x240，`[268,16,240,240]`。禁止退回 3-variable 诊断输入。 |
| 科学场 | Hurricane | `/workspace/Data/SDRBENCH-Hurricane-ISABEL-100x500x500/100x500x500/PRECIPf48.log10.bin.f32` | 一个变量的前 96 帧，`[1,96,500,500]`；96 同时整除 CAESAR-V/D 深度。 |
| 科学场 | NYX | `/workspace/Data/SDRBENCH-EXASKY-NYX-512x512x512/SDRBENCH-EXASKY-NYX-512x512x512/baryon_density.f32` | 完整 baryon-density volume，`[1,512,512,512]`，Z 作为 3D 深度。 |
| 科学场 | Turb_Rot | `/workspace/Turb_Rot_testset.npz` | variable 0，section 0 和 8，各 `[1,256,256,256]`；256 planes 全部参与。 |
| 科学图像 | Tomo | `/workspace/Data/tomo_00001.h5` | projection `0:512` 与 `989:1501` 两块；每块中心裁切 512x512，shape `[1,512,512,512]`。 |
| 科学图像 | Lysozyme | `/workspace/Data/lysozyme_processed/mmap/lysozyme_test_nf16.npy` | 两个不重叠 stack，每个 31 个 16-frame chunk，即 `[1,496,1024,1024]`。 |
| 科学图像 | S2C | Sentinel-2 SAFE 目录，见 `objective_data.py` | B02/B03/B04/B08，前四个确定性非恒定 1024 tile。每 tile `[4,1,1024,1024]`；CAESAR/cuSZ 按 tile-major、band-minor 可逆堆成 16 planes。 |

### 4.1 不同 codec 如何消费 canonical tensor

- DCAE/HPCM/DCMVC-I/DCVC-RT-I：
  - 通用媒体逐 RGB frame 输入。
  - S2C 每个 tile 的 4 bands 按三个一组，最后不足三通道时只做 shape padding，评价和 BPP 丢弃 padding channel。
  - 其他科学数据逐变量处理 `[T,H,W]`，由 image view 按相邻 planes 三个一组输入。
- CAESAR：
  - V 模型按 8 planes，D 模型按 16 planes reshape 为 `[V,S,T,H,W]`。
  - `DataLoader(batch_size=64)` 对齐作者 `eval_caesar.ipynb`。
  - 使用 `ScientificDataset(inst_norm=True,norm_type="mean_range")`，这是 codec 内部行为。
  - Kodak/S2C/UVG corpus 深度不足倍数时 repeat-last padding，padding 计入码率和时间，评价前裁回真实深度。
- cuSZ-Hi：
  - 不能把 3D 降成逐 2D slice。
  - ERA5/E3SM 等多变量数据是“每个变量一个完整 `[T,H,W]` volume”。
  - S2C 使用完整 16-plane spectral/corpus volume。
  - Kodak 使用完整 72-plane corpus stack。
  - UVG 使用三个 `[30,H,W]` RGB time volumes。
- nvJPEG2000：
  - 科学数据逐变量压缩完整 `[T,H,W]` stack，固定单位范围转换为 uint16。
  - target PSNR 是 codec 控制参数，不是 CAESAR/cuSZ 的 EB。
- nvJPEG：
  - 只用于 Kodak 和 UVG；逐 RGB frame 压缩，不支持科学 float 3D 主轨。

## 5. 固定归一化和 mask

| 数据集 | objective 外部归一化 |
|---|---|
| Kodak | 固定 `/255` |
| UVG | adapter 已输出 `[0,1]`，identity |
| Tomo | 固定 `/65535` |
| ERA5 | 输入已有 CRA5 z-score；再使用完整 objective tensor 冻结的逐变量 affine min/max，不 clipping |
| E3SM | 两个 objective block 联合得到逐变量 min/max |
| Hurricane | 完整 96 帧联合得到一个 min/max |
| NYX | 完整 512³ volume 得到一个 min/max |
| Turb_Rot | section 0 和 8 联合得到一个 min/max |
| S2C | 四个 tile 联合得到逐 band min/max；不能使用 `/10000` clipping |
| Lysozyme | 两个 stack 的有效值联合得到 min/max |

Lysozyme 中 `raw >= 4294967000` 为 invalid。mask 在压缩前形成，invalid 输入统一替换为 0；range、MSE、PSNR 和 max error 只统计有效位置。该 frozen mask 是所有 codec 共享的 benchmark metadata，当前不计入任一方法的码率。

每次迁移后必须重新运行输入准备，并核对已有 checksum：

```bash
source /workspace/ai4cp/bin/activate

for dataset in e3sm_npz era5_npy hurricane nyx turb_rot_npz tomo lysozyme s2c kodak uvg_twilight_1080p; do
  python scripts/prepare_objective_inputs.py \
    --dataset "$dataset" \
    --output-root unified_results/objective_v1
done
```

如果同一数据文件得到不同的 `canonical_sha256` 或 `normalized_canonical_sha256`，不得与现有结果合并。

## 6. 每个数据集的 EB

这些值来自 `unified_results/objective_all_to_all_v1/eb_schedule.json`，目标是覆盖最低有效 BPP 到接近 32 BPP。它们是当前正式复现值，不应替换成统一的全数据集默认列表。

| 数据集 | CAESAR-D EB | CAESAR-V EB | cuSZ-Hi EB |
|---|---|---|---|
| E3SM | `0.3,0.01,0.003,0.001,1e-4,3e-6,1e-8` | 同 D | `0.8,0.5,0.2,0.1,0.05,0.02,0.005` |
| ERA5 | `0.1,0.01,0.003,0.001,1e-4,3e-6,1e-9` | 同 D | `0.5,0.2,0.1,0.05,0.02,0.01,0.005` |
| Hurricane | `0.3,0.03,0.01,0.003,3e-4,1e-5,1e-8` | `0.3,0.03,0.01,0.003,3e-4,3e-5,1e-8` | `0.5,0.2,0.1,0.05,0.02,0.015,0.012` |
| Kodak | `0.3,0.03,0.01,0.003,0.001,1e-4,3e-8` | `0.3,0.03,0.01,0.003,0.001,3e-5,3e-8` | `0.8,0.5,0.2,0.05,0.02,0.01,0.006` |
| Lysozyme | `0.1,0.03,0.01,0.003,0.001,3e-4,1e-4` | 同 D | `0.5,0.2,0.1,0.05,0.02,0.01,0.005` |
| NYX | `0.1,3e-4,1e-4,1e-6,1e-8,1e-10,1e-12` | `0.2,1e-4,1e-6,1e-8,1e-10,1e-11,1e-12` | `0.5,0.2,0.1,0.05,0.02,0.01,0.005` |
| S2C | `0.3,0.01,0.003,0.001,3e-4,3e-5,1e-8` | `0.3,0.01,0.003,0.001,3e-4,1e-5,1e-8` | `0.8,0.4,0.3,0.2,0.05,0.02,0.005` |
| Tomo | `0.1,0.01,0.003,0.001,3e-4,3e-6,3e-9` | `0.1,0.01,0.003,0.001,3e-4,1e-5,3e-9` | `0.5,0.15,0.1,0.05,0.02,0.015,0.012` |
| Turb_Rot | `0.1,0.003,0.001,3e-4,1e-4,1e-6,3e-9` | `0.1,0.003,0.001,3e-4,1e-4,3e-6,3e-9` | `0.5,0.3,0.1,0.05,0.02,0.015,0.01` |
| UVG 1080p | `0.3,0.01,0.003,0.001,3e-4,3e-5,1e-8` | 同 D | `0.8,0.5,0.1,0.05,0.02,0.01,0.005` |

Turb_Rot 专用微调权重的 V/D EB 均为：

```text
0.3, 0.001, 0.0003, 0.0001, 3e-5, 1e-6, 3e-9
```

ERA5 100k CAESAR-V 微调权重使用 objective-v1 ERA5 的同一组 V EB：

```text
0.1, 0.01, 0.003, 0.001, 0.0001, 3e-6, 1e-9
```

## 7. 正式复现命令

单个非视频数据集的完整主模型测试：

```bash
source /workspace/ai4cp/bin/activate

python scripts/run_objective_benchmark.py \
  --dataset era5_npy \
  --gpu 2 \
  --output-root unified_results/objective_v1 \
  --input-root unified_results/objective_v1 \
  --models DCAE HPCM CAESAR-V CAESAR-D cuSZ-Hi nvJPEG2000 DCMVC-I DCVC-RT-I \
  --caesar-eb 0.1 0.01 0.003 0.001 0.0001 3e-6 1e-9 \
  --cusz-eb 0.5 0.2 0.1 0.05 0.02 0.01 0.005 \
  --warmups 2 \
  --repeats 5
```

UVG P-frame：

```bash
python scripts/run_objective_video.py \
  --gpu 2 \
  --root unified_results/objective_v1 \
  --models dcvc dcmvc \
  --warmups 2 \
  --repeats 5
```

ERA5 100k 微调权重的快速 RD 补测和合图：

```bash
bash scripts/run_caesar_era5_v_100k_objective_rd.sh
```

该脚本使用 GPU 2–7、`0` 次预热和 `1` 次正式调用。其 BPP/PSNR 是真实完整编解码结果，可进入 RD 图；其吞吐量不能与正式 `2+5` 结果比较。如需正式吞吐量，将脚本中的 `--warmups 0 --repeats 1` 改成 `--warmups 2 --repeats 5` 后重跑。

合并、审计和画图：

```bash
python scripts/build_objective_all_to_all.py \
  --baseline unified_results/objective_v1 \
  --sources unified_results/objective_v1 \
  --schedule unified_results/objective_all_to_all_v1/eb_schedule.json \
  --output unified_results/objective_all_to_all_v1

python scripts/audit_objective_benchmark.py \
  unified_results/objective_all_to_all_v1 \
  --strict

python scripts/analyze_objective_benchmark.py \
  --root unified_results/objective_all_to_all_v1
```

## 8. 迁移环境检查表

1. 保留仓库、`models/`、`checkpoints/`、`benchmark_protocols/objective_v1.json` 和本文档。
2. 数据放到第 4 节约定路径，或统一修改 `compression_pipeline/objective_data.py`，不要只修改某个模型脚本。
3. 激活 `/workspace/ai4cp`，验证 PyTorch/CUDA、CompressAI 扩展、DCVC CUDA 扩展和 nvJPEG/nvJPEG2000。
4. 编译并确认 `models/cuSZ-Hi/build/cuszhi` 可执行。
5. 对十个数据集运行 `prepare_objective_inputs.py`。
6. 对照旧 `samples.json` 检查 canonical 和 normalized checksum。
7. 按第 6 节逐数据集传入 EB，不使用 runner 默认值代替已选 schedule。
8. 每张物理 GPU 只运行一个正式 benchmark 进程，关闭 MPS 共享和其他负载。
9. 正式吞吐量使用 `2+5`；快速 RD 可以 `0+1`，但必须在结果 manifest 中标注。
10. 合并前运行 strict audit；失败点保留在原始 JSON，主图只使用通过 gate 的有效点。

## 9. ERA5 专项说明

ERA5 正式输入固定为：

```text
source: /workspace/Data/ERA5/finetune_processed/era5_test.npy
selection: variables 0:268, times 0:16, center crop 240x240
canonical shape: [268,16,240,240]
CAESAR-V view: [268,2,8,240,240]
CAESAR-D view: [268,1,16,240,240]
```

旧网页中出现的 `[3,1,8,240,240]` 是三变量诊断协议，不能用于评价 100k 微调是否优于原始权重，也不能与 objective-v1 的 DCAE/HPCM/cuSZ/nvJPEG2000 曲线叠加。

当前 ERA5 统一主图应包含：

- DCAE
- LIC-HPCM-base / large
- CAESAR-D original
- CAESAR-V original
- CAESAR-V fine-tuned 100k
- cuSZ-Hi
- nvJPEG2000
- DCMVC-I
- DCVC-RT-I

100k 曲线使用与原始 CAESAR-V 相同的 objective tensor、固定外部归一化、CAESAR `mean_range` 内部归一化、PCA 路径和七个 EB。只有 checkpoint 不同。

### 9.1 ERA5 微调结论边界

该实验只能支持“100k 微调权重优于 CAESAR-V 初始权重”，不能支持“微调后的 CAESAR-V 超过 DCAE 或 LIC-HPCM”。

“100k”只表示训练 update 数对齐论文量级，不表示训练 recipe 和论文数据逐项相同。当前最佳权重的实际配置来自 `scripts/run_caesar_era5_daily_v_full_100k.sh`：

| 项目 | 论文参考配置 | 当前最佳 100k |
|---|---:|---:|
| updates | `100000` | `100000` |
| patch / batch | `256x256 / 32` | `256x256 / 32` |
| learning rate | `1e-4` | `3e-5` |
| lambda | `1e-5` 起步 | `3e-5` |
| rate 表达 | 论文 rate-distortion 目标 | `rate_mode=bpp` |
| warmup | 未作为当前脚本同一设置发布 | `500 updates` |
| CAESAR normalization | 论文描述逐帧 zero-mean/unit-range | 当前 `mean_range` 作用于时序 patch |
| ERA5 组织 | 论文描述 `6 variables x 3 levels x 2160 times` | 当前 `268 variables x 2160 hours`，变量独立成为训练 item，`frame_step=24` |

论文原始 `lr=1e-4,lambda=1e-5` 路径此前已经测试过，在当前 ERA5 组织上效果较差；当前权重是降低学习率并以真实 codec RD 验证后保留的结果。因此可称为“论文规模的 100k fine-tune”，不能称为“论文训练实验的完全同条件复现”。

实测覆盖范围：

| 曲线 | BPP 范围 | normalized PSNR 范围 |
|---|---:|---:|
| DCAE | `0.02675–0.13262` | `32.04–40.73 dB` |
| LIC-HPCM-base | `0.02490–0.12866` | `32.20–40.96 dB` |
| LIC-HPCM-large | `0.02289–0.12033` | `32.06–41.02 dB` |
| CAESAR-V original | `0.28730–30.13025` | `36.98–168.11 dB` |
| CAESAR-V fine-tuned 100k | `0.24850–29.41828` | `39.01–168.11 dB` |

DCAE/HPCM 的最高实测 BPP 仍低于 fine-tuned CAESAR-V 的最低 BPP，因此不存在可做同码率插值的 BPP 重叠区。按 PSNR 轴在共同质量 `39.01 dB` 处做 log-BPP 线性插值：

| 方法 | 达到约 `39.01 dB` 的 BPP |
|---|---:|
| DCAE | `0.09712` |
| LIC-HPCM-base | `0.08886` |
| LIC-HPCM-large | `0.08589` |
| CAESAR-V fine-tuned 100k | `0.24850` |

在这一共同质量点，CAESAR-V 使用约 `2.6–2.9×` 的码率。因此当前结果显示 DCAE/HPCM 在其覆盖的低码率、低至中等质量区域具有更好的 RD；CAESAR 的作用是把曲线延伸到这些固定权重没有覆盖的高 PSNR、高 BPP 区域。覆盖更高质量不等于在重叠质量区超过图像模型。

当前合并输出：

```text
unified_results/objective_era5_caesar_v_finetuned_100k_rd/summary.json
unified_results/objective_era5_caesar_v_finetuned_100k_rd/manifest.json
unified_results/objective_era5_caesar_v_finetuned_100k_rd/era5_objective_all_models_with_caesar_v_finetuned_100k.png
```

### 9.2 V/D 低码率原始权重重启实验

为判断 CAESAR 的约 `0.25 BPP` 下限能否通过更强 rate penalty 降低，使用以下脚本启动 10k 筛选：

```bash
bash scripts/run_caesar_era5_vd_lowrate_10k.sh
```

本轮不继承任何 ERA5 微调权重。来源固定为：

| 模型 | checkpoint | SHA-256 |
|---|---|---|
| CAESAR-V | `checkpoints/caesar/caesar_v.pt` | `4acebeb189ab0f8b99de167326cc32b5390bc9c5025a85d9502641b73a7ad355` |
| CAESAR-D | `checkpoints/caesar/caesar_d.pt` | `3cb2bbadbd9756275504500801f2b28f32ce6320fd084d7d63c4d8178cfbdbbe` |

共同配置为 90 天 ERA5 shard、时间顺序切分 `1776 train + 384 validation`、`256x256` patch、batch 32、`frame_step=24`、`mean_range`、normalized-domain distortion、BPP rate term、`lr=1e-5` 和 500-update warmup。V 使用 8 帧，D 使用 16 帧。

V 和 D Stage 1 分别测试 `lambda_rate={1e-4,3e-4,1e-3}`，checkpoint 保存于：

```text
checkpoints/caesar_era5_vd_lowrate_10k/
```

训练起点验证值必须在同一模型内一致：

| 模型 | initial normalized MSE | initial estimated BPP |
|---|---:|---:|
| CAESAR-V | `0.000329` | `0.254066` |
| CAESAR-D Stage 1 | `0.000041` | `0.189467` |

10k 后不能按内部 validation loss 直接决定最佳权重。必须先在固定 ERA5 validation probe 上对 500、2k、5k、10k checkpoint 跑真实 codec EB-RD，比较 total BPP、PSNR 以及 latent/residual bytes。D Stage 1 的候选 VAE 通过筛选后，还必须从原始 D diffusion 启动对应的 Stage 2 训练；仅把新 VAE 与原始 diffusion 组合不作为最终 D 结果。

上述三组 rate penalty 均继续进行独立的完整 100k 训练。由于 10k checkpoint 不包含 Adam 状态，完整实验不采用“加载 10k 模型后再训练 90k”的近似方式，而是重新从原始 checkpoint 初始化模型和优化器：

```bash
bash scripts/run_caesar_era5_vd_lowrate_100k.sh
```

输出位置和 W&B group：

```text
checkpoints/caesar_era5_vd_lowrate_100k/
logs/caesar_era5_vd_lowrate_100k/
wandb group: vd-lowrate-from-original-100k
```

保存步数为 10k、25k、50k、75k、100k。V 的 100k checkpoint 可以直接进入真实 codec RD 筛选；D 的 100k 仅代表 Stage 1 keyframe VAE 完成，不能替代 Stage 2 diffusion 训练。

### 9.3 V 低码率 100k 结果

三组 V checkpoint 均在 objective-v1 ERA5 的完整 `268x16x240x240` 输入上完成七个 EB 的真实 codec 测试。由于不同 lambda 的曲线会交叉，不存在所有质量范围内唯一支配其他权重的 checkpoint。按本轮“进入 DCAE/HPCM 低 BPP 区”的目标，选择：

```text
checkpoints/caesar_era5_vd_lowrate_100k/v_lr1em5_lam1em3_full100k_update100000.pt
lr=1e-5
lambda_rate=1e-3
```

该权重的最左点为 `0.10663 BPP / 37.93 dB`。在旧 100k 最左点对应的共同质量 `39.01 dB` 上做 log-BPP 插值：

| 曲线 | BPP |
|---|---:|
| CAESAR-V original | `0.32044` |
| 旧 100k，`lr=3e-5, lambda=3e-5` | `0.24850` |
| 新低码率 100k，`lr=1e-5, lambda=1e-3` | `0.12464` |

新低码率权重相对旧 100k 节省约 `49.84%` 码率，相对 original 节省约 `61.11%`。该结论限定在低码率共同质量区；约 50 dB 以上旧 100k 曲线仍更好。

合并 JSON、manifest 和对比图位于：

```text
unified_results/objective_era5_caesar_v_100k_best_compare/summary.json
unified_results/objective_era5_caesar_v_100k_best_compare/manifest.json
unified_results/objective_era5_caesar_v_100k_best_compare/caesar_v_era5_original_previous_new_100k.png
unified_results/objective_era5_caesar_v_100k_best_compare/caesar_v_era5_original_previous_new_100k.pdf
```

### 9.4 冻结编码器的 decoder 质量微调

低码率 100k 权重已经把 CAESAR-V 左端从约 `0.25 BPP` 降到 `0.1066 BPP`，但在 DCAE/HPCM 过渡区仍有质量差距。若继续训练整个模型，验证质量会提高，但 entropy model 的 BPP 增长更快，真实 RD 曲线整体右移。最终采用冻结码率路径的方案：

- 起点：`checkpoints/caesar_era5_vd_lowrate_100k/v_lr1em5_lam1em3_full100k_update100000.pt`
- 起点 SHA-256：`d590a954e7cfd31047b07852c6cd7cec7fee55423d26e3c41ab76ed78ca2980c`
- 最终权重：`checkpoints/caesar_era5_v_decoder_quality_100k/from_lowrate_lr3em4.pt`
- 最终 SHA-256：`e44f2951844d6e873b024b4c288da315e50d015f9d323262e24b5d1e5e7dae57`
- `trainable_scope=decoder`：只训练 `entropy_model.dec.*` 和 `sr_model.*`，共 `473036 / 1501356` 个参数。
- encoder、hyperprior、prior 和其他产生码率的模块全部冻结；训练前后估计 BPP 均为 `0.096693`。
- `lambda_rate=0` 不是取消码率约束后训练整个模型，而是因为所有码率相关参数已经冻结，只优化重建误差。

训练输入与正式测试保持同一外部数据语义：90 天 ERA5 hourly shard，按时间顺序使用前 `1776 h` 训练、后 `384 h` 验证；`n_frame=8`、`frame_step=24`、`temporal_stride=8`、`256x256` patch、batch 32、`mean_range`，normalized-domain MSE。最终路径使用 `lr=3e-4`、250-update warmup、100k updates，脚本为：

```bash
bash scripts/run_caesar_era5_v_decoder_quality_100k.sh
```

三条并行路径均跑满 100k，并由验证集自动保留最佳 checkpoint：

| 路径 | 起点 | lr | 最佳 normalized MSE | 对应 source MSE | 估计 BPP |
|---|---|---:|---:|---:|---:|
| `from_lowrate_lr3em4` | 低码率 100k | `3e-4` | `0.000224` | `0.028239` | `0.096693` |
| `from_decoder10k_lr1em4` | decoder 10k best | `1e-4` | `0.000226` | `0.028512` | `0.096693` |
| `from_decoder10k_lr3em5` | decoder 10k best | `3e-5` | `0.000229` | `0.028801` | `0.096693` |

W&B runs：

```text
from_lowrate_lr3em4:       https://wandb.ai/1796248596-university-of-chinese-academy-of-sciences/caesar-era5-hourly-tuning/runs/sjmv425f
from_decoder10k_lr1em4:   https://wandb.ai/1796248596-university-of-chinese-academy-of-sciences/caesar-era5-hourly-tuning/runs/no3e6ob3
from_decoder10k_lr3em5:   https://wandb.ai/1796248596-university-of-chinese-academy-of-sciences/caesar-era5-hourly-tuning/runs/2uv6k5vl
```

最终权重在 objective-v1 的完整 `[268,16,240,240]` 输入上测试 13 个 EB：

```text
0.3, 0.1, 0.05, 0.03, 0.025, 0.02, 0.015,
0.01, 0.003, 0.001, 1e-4, 3e-6, 1e-9
```

运行命令：

```bash
bash scripts/run_caesar_era5_v_decoder_final_rd.sh
```

过渡区的真实 codec 结果如下；BPP 包含必要 side information：

| EB | BPP | normalized PSNR |
|---:|---:|---:|
| `0.3` | `0.10557` | `37.899 dB` |
| `0.1` | `0.10659` | `38.170 dB` |
| `0.05` | `0.11064` | `38.914 dB` |
| `0.03` | `0.12260` | `39.962 dB` |
| `0.025` | `0.13086` | `40.466 dB` |
| `0.02` | `0.14528` | `41.164 dB` |
| `0.015` | `0.17329` | `42.165 dB` |
| `0.01` | `0.23922` | `43.772 dB` |

13 点整条曲线严格单调，完整范围为 `0.10557–30.04012 BPP` 和 `37.899–168.112 dB`。在与低码率起点共同且 `BPP <= 0.3` 的 EB 上，平均提高 `0.189 dB`，BPP 不增并通常略降。高保真极端点不是全部受益：`EB=3e-6` 的 PSNR 比起点低约 `0.215 dB`，但曲线仍保持单调；因此该权重应定义为低码率和过渡区优化权重，不应宣称全 RD 范围支配起点。

DCAE 的最高点为 `0.13262 BPP / 40.730 dB`。最终 CAESAR-V 在相同 BPP 的 log-BPP 插值结果为 `40.555 dB`，仍低 `0.175 dB`，所以不能宣称同码率超过 DCAE。另一方面，最终曲线的 `0.14528 BPP / 41.164 dB` 已高于 DCAE、HPCM-base 和 HPCM-large 的最高实测 PSNR，构成从图像 codec 曲线末端向右上方的高质量延伸。这正是本轮目标，但“右上延伸”不等于“同码率支配”。

与 CAESAR-V original 做完整的 matched-quality 比较时，最终权重在共同质量范围内始终使用更低 BPP。代表性结果为：`38 dB: -65.09%`、`40 dB: -63.54%`、`45 dB: -34.38%`、`50 dB: -14.77%`、`60 dB: -11.00%`、`80 dB: -11.57%`、`110 dB: -2.89%`、`150 dB: -1.25%`、`168 dB: -0.31%`。逐个共同 EB 的原始 BPP/PSNR、最终 BPP/PSNR 和差值均保存在最终 manifest 的 `common_eb_original_to_final`；matched-quality 插值结果保存在 `matched_quality_original_comparison`。

按相同 BPP 比较时，在两条曲线的共同范围 `0.28730–30.04012 BPP` 内对 `log(BPP)` 做分段线性插值，最终权重的区间平均 PSNR 增益为 `2.157 dB`，并且在全部插值节点上均为正值，最小为 `0.259 dB`，最大为 `7.863 dB`。代表点为：`0.3 BPP: +7.313 dB`、`0.5: +2.195 dB`、`1.0: +0.991 dB`、`2.0: +1.275 dB`、`5.0: +2.161 dB`、`10.0: +2.736 dB`、`15.0: +0.948 dB`、`30.0: +0.264 dB`。这些结果保存在 manifest 的 `matched_bpp_original_comparison` 和 `matched_bpp_summary`。

最终 JSON、manifest 和论文图：

```text
unified_results/objective_era5_caesar_v_decoder_final_rd/era5_npy/summary.json
unified_results/objective_era5_caesar_v_decoder_final_compare/summary.json
unified_results/objective_era5_caesar_v_decoder_final_compare/manifest.json
unified_results/objective_era5_caesar_v_decoder_final_compare/caesar_v_decoder_finetune_vs_image_codecs.png
unified_results/objective_era5_caesar_v_decoder_final_compare/caesar_v_decoder_finetune_vs_image_codecs.pdf
unified_results/objective_era5_caesar_v_decoder_final_compare/caesar_v_decoder_finetune_vs_original.png
unified_results/objective_era5_caesar_v_decoder_final_compare/caesar_v_decoder_finetune_vs_original.pdf
```

### 9.5 CAESAR-D 冻结码率 decoder 与匹配 Stage2

CAESAR-D 不能只替换 Stage1 VAE 后继续使用原始 diffusion；Stage2 通过 fine-tuned VAE 的 `inference_qlatent` 工作，因此完整候选必须保存匹配的 `vae` 和 `diffusion`。当前采用两步路径：

- Stage1 起点：`checkpoints/caesar_era5_vd_lowrate_100k/d_s1_lr1em5_lam3em4_full100k_update100000.pt`
- Stage1 最终 decoder：`checkpoints/caesar_era5_d_decoder_quality_100k/lam3em4_from_lowrate_lr3em4.pt`
- Stage1 最终 SHA-256：`35c3cc707b3ece6adc12ceae263e79c92c4ee4315479186d819aba18c680c883`
- `trainable_scope=decoder`：只训练 D VAE 的 `dec.*`，共 `146529 / 861505` 个参数；encoder、hyperprior 和其他码率模块冻结。
- held-out 估计 BPP 固定为 `0.096434`；source MSE 从 `0.007098` 降到保留 checkpoint 的约 `0.006809`。
- Stage1 训练：`100k` updates，`lr=3e-4`，250-update warmup，batch 32，`n_frame=16`、`frame_step=24`，其余数据切分和归一化与 V 路径一致。

Stage1 W&B：

```text
https://wandb.ai/1796248596-university-of-chinese-academy-of-sciences/caesar-era5-hourly-tuning/runs/prn4axus
```

当前真实 codec 筛选最好的 Stage2 是独立 5k 运行：

- Stage2：`checkpoints/caesar_era5_d_stage2_overlap_5k/lam3em4_stage2_lr1em4_update5000.pt`
- Stage2 SHA-256：`efc199c702b8c4d57d700584656c215f1d6a660fcb94233684fcad7fd6e116df`
- 完整配对权重：`checkpoints/caesar_era5_d_complete_candidates/lam3em4_decoder100k_stage2_overlap5000.pt`
- 完整权重 SHA-256：`658ee9c282c9df62786153f33c619a1345952f0db840686166c04666b0ee712e`
- Stage2 使用原始 diffusion 起点、fine-tuned 低码率 VAE latent、`lr=1e-4`、32 diffusion steps、batch 32 和 2 次梯度累积，即有效 batch 64。
- Stage2 5k W&B：`https://wandb.ai/1796248596-university-of-chinese-academy-of-sciences/caesar-era5-hourly-tuning/runs/rob8pfw9`

完整 `[268,16,240,240]` objective-v1 13 点曲线严格单调。过渡区结果为：

| EB | BPP | normalized PSNR |
|---:|---:|---:|
| `0.3` | `0.11402` | `23.334 dB` |
| `0.1` | `0.13310` | `28.517 dB` |
| `0.05` | `0.16580` | `33.038 dB` |
| `0.03` | `0.22435` | `36.493 dB` |
| `0.025` | `0.24693` | `37.711 dB` |
| `0.02` | `0.28028` | `39.191 dB` |
| `0.015` | `0.33768` | `41.105 dB` |
| `0.01` | `0.42714` | `43.805 dB` |

与 D-original 按相同 BPP 比较，在共同范围 `0.22539–30.24979 BPP` 内对 `log(BPP)` 做分段线性插值，平均 PSNR 增益为 `1.674 dB`；全部插值节点均为正，最小 `0.161 dB`、最大 `12.959 dB`。代表点为：`0.3 BPP: +5.435 dB`、`0.5: +2.709 dB`、`1.0: +0.463 dB`、`2.0: +0.309 dB`、`5.0: +0.913 dB`、`10.0: +2.005 dB`、`15.0: +0.492 dB`、`20.0: +0.652 dB`、`30.0: +0.171 dB`。

这组权重尚未达到图像 codec 的低码率质量：在 DCAE 最高实测码率 `0.13262 BPP` 处，CAESAR-D 插值为 `28.395 dB`，比 DCAE 的 `40.730 dB` 低 `12.335 dB`。达到 DCAE endpoint 的 `40.730 dB` 时，当前 D 需要约 `0.32558 BPP`。因此当前结论只能是完整 D 相对 D-original 显著改善，不能宣称超过 DCAE/HPCM。

Stage2 训练命令以 `200000` updates 为上限启动，W&B 为：

```text
https://wandb.ai/1796248596-university-of-chinese-academy-of-sciences/caesar-era5-hourly-tuning/runs/y19kbo0d
```

该命令按用户要求至少跑满 `100000` updates，并在 `update100000.pt` 原子落盘后停止。内部 noise-validation 的最佳值出现在 35k，之后到 100k 未再刷新。为避免 surrogate loss 误导，使用 CPU 在同一完整 `268x16x240x240` 输入、同一 seed、同一 decoder-100k 和相同 condition latent 上复核真实 32-step sampling：

| Stage2 checkpoint | condition latent BPP | 全帧 PSNR | predicted-frame PSNR | 相对 5k |
|---|---:|---:|---:|---:|
| 5k | `0.111297` | `21.591 dB` | `19.556 dB` | baseline |
| validation-best 35k | `0.111297` | `21.439 dB` | `19.404 dB` | `-0.152 dB` |
| 75k | `0.111297` | `21.495 dB` | `19.460 dB` | `-0.096 dB` |
| 100k | `0.111297` | `21.516 dB` | `19.481 dB` | `-0.075 dB` |

四者 keyframe PSNR 均为 `45.746 dB`，所以差异完全来自 Stage2 diffusion。结论是 100k 训练完成了论文量级排查，但没有超过 5k 的真实生成质量；不能因为 35k validation loss 最低就把它选为 codec 权重。结果与来源哈希位于：

```text
unified_results/diagnostic_caesar_d_stage2_cpu_full268/
scripts/run_caesar_era5_d_cpu_sampling_audit.sh
```

历史 50k 里程碑曾完成 13 点真实 codec 审计，但复核 tensor hash 后发现，当时 watcher 使用的是同一路径较早保存的 decoder snapshot，不是当前 5k 正式候选中的 decoder-100k。该曲线本身仍是可解码的完整 D 结果，但不能作为“固定 decoder、只比较 Stage2”的严格对照，也不再进入正式前沿。旧结果保留于：

```text
unified_results/objective_era5_caesar_d_stage2_50000_rd/era5_npy/summary.json
unified_results/objective_era5_caesar_d_stage2_50000_compare/comparison.json
```

`5k/10k/25k/.../200k` 里程碑使用固定 decoder 做 4 点 objective 筛选；只有真实 codec 曲线优于当前 5k 候选时才替换最终权重。相关入口和结果为：

```text
scripts/run_caesar_era5_d_decoder_quality_100k.sh
scripts/run_caesar_era5_d_stage2_full_200k.sh
scripts/watch_caesar_era5_d_stage2_objective_probes.sh
scripts/run_caesar_era5_d_decoder100k_stage2_5k_13pt.sh
unified_results/objective_era5_caesar_d_decoder100k_stage2_overlap5k_rd/era5_npy/summary.json
unified_results/objective_era5_caesar_d_decoder100k_stage2_overlap5k_compare/comparison.json
unified_results/objective_era5_caesar_d_decoder100k_stage2_overlap5k_compare/caesar_d_stage2_5k_vs_original_and_image_codecs.png
unified_results/objective_era5_caesar_d_decoder100k_stage2_overlap5k_compare/caesar_d_stage2_5k_vs_original_and_image_codecs.pdf
```

### 9.6 V/D 完整曲线与同 BPP 对比

V/D 的 original、当前最佳完整曲线、图像 codec 低码率参考以及 matched-BPP 增益已合并为同一张图。左列显示完整 RD 范围，中列放大低码率过渡区，右列按 `log(BPP)` 分段线性插值后显示相同 BPP 的 PSNR 差值。图和机器可读 manifest 位于：

```text
unified_results/objective_era5_caesar_vd_complete_compare/caesar_vd_complete_vs_original.png
unified_results/objective_era5_caesar_vd_complete_compare/caesar_vd_complete_vs_original.pdf
unified_results/objective_era5_caesar_vd_complete_compare/manifest.json
```

复现命令：

```bash
python scripts/build_caesar_era5_vd_complete_compare.py \
  --baseline unified_results/objective_all_to_all_v1/combined_summary.json \
  --v-final unified_results/objective_era5_caesar_v_decoder_final_rd/era5_npy/summary.json \
  --v-variant decoder_quality_100k_lr3em4 \
  --d-original unified_results/objective_era5_caesar_d_original_13pt_rd/era5_npy/summary.json \
  --d-final unified_results/objective_era5_caesar_d_decoder100k_stage2_overlap5k_rd/era5_npy/summary.json \
  --d-variant d_lam3em4_decoder100k_stage2_overlap5000 \
  --d-keyframe unified_results/objective_era5_caesar_d_lam1em3_keyframe_only_rd/era5_npy/summary.json \
  --d-keyframe-variant d_lam1em3_keyframe_only_ablation \
  --output unified_results/objective_era5_caesar_vd_complete_compare
```

最终 matched-BPP 结论：

| 模型 | 共同 BPP 范围 | log-BPP 区间平均增益 | 最小增益 | 最大增益 |
|---|---:|---:|---:|---:|
| CAESAR-V tuned vs V-original | `0.28730–30.04012` | `+2.157 dB` | `+0.259 dB` | `+7.863 dB` |
| CAESAR-D Stage2 5k vs D-original | `0.22539–30.24979` | `+1.674 dB` | `+0.161 dB` | `+12.959 dB` |
| CAESAR-D keyframe-only ablation vs D-original | `0.22539–29.34041` | `+4.687 dB` | `+1.053 dB` | `+21.600 dB` |

另行测试了更强码率惩罚的 D Stage1（`lambda_rate=1e-3`）配合冻结码率 decoder 和 original Stage2。其 13 点曲线严格单调，范围扩展至 `0.07465–31.22213 BPP`，但在与 D-original 的共同范围内平均为 `-1.164 dB`，最差为 `-4.911 dB`；在 DCAE endpoint `0.13262 BPP` 处仅约 `32.632 dB`。因此它只作为低码率补充实验保留，不替代上表中的 D 当前最佳权重。原始结果和比较结果为：

```text
checkpoints/caesar_era5_d_complete_candidates/lam1em3_decoder_best_original_stage2.pt
unified_results/objective_era5_caesar_d_lam1em3_decoder_best_original_stage2_rd/era5_npy/summary.json
unified_results/objective_era5_caesar_d_lam1em3_decoder_best_original_stage2_compare/comparison.json
```

### 9.7 CAESAR-D 时间生成瓶颈诊断

为避免把 Stage1 质量、EB 后处理和 Stage2 时间生成混为一谈，使用真实 range-coded keyframe latent 分别统计 keyframe 与 predicted-frame PSNR。诊断脚本为：

```text
scripts/diagnose_caesar_d_temporal_reconstruction.py
```

在完整 `268 x 16 x 240 x 240` ERA5 测试输入上，更低码率的 `lambda_rate=1e-3` D Stage1 得到：

| 模式 | condition frames | latent BPP | 全帧 PSNR | keyframe PSNR | predicted-frame PSNR |
|---|---:|---:|---:|---:|---:|
| 正式 rate-2 + original diffusion | `8/16` | `0.09644` | `23.397 dB` | `43.636 dB` | `20.408 dB` |
| 全部帧 range-coded Stage1 上界 | `16/16` | `0.19251` | `43.620 dB` | `43.620 dB` | N/A |

全帧上界已经位于 DCAE endpoint（`0.13262 BPP / 40.730 dB`）的右上方，证明 Stage1 本身有足够的重建能力；正式 rate-3/rate-2 曲线的主要质量损失来自 Stage2 对缺失日期的生成。全帧模式不使用 diffusion，因此只能标成 keyframe-only ablation，不能标成正式 CAESAR-D 结果。机器可读诊断为：

```text
unified_results/diagnostics/caesar_d_lam1em3_all_frame_range_coded_full268.json
unified_results/diagnostics/caesar_d_lam1em3_rate2_original_diffusion_full268.json
```

该上界也已接入完整 objective-v1 EB 后处理路径。13 点曲线严格单调，范围为 `0.19251–29.34041 BPP / 43.620–168.112 dB`；相对 D-original 的 matched-BPP 平均增益为 `+4.687 dB`。这条曲线用于说明移除时间生成后的可达上界，不用于替换正式 D 排名：

```text
unified_results/objective_era5_caesar_d_lam1em3_keyframe_only_rd/era5_npy/summary.json
unified_results/objective_era5_caesar_d_lam1em3_keyframe_only_compare/comparison.json
unified_results/objective_era5_caesar_d_lam1em3_keyframe_only_compare/caesar_d_stage2_5k_vs_original_and_image_codecs.png
```

另行筛选了直接预测 reconstructed latent 的 `x0` 目标以及 `noise + weight * x0` hybrid 目标。外部输入、按日时间组织、`mean_range` 归一化和 Stage1 均保持不变；四组各训练 500 updates。虽然内部验证 objective 均下降，但 first-32-variable 真实 32-step sampling 全部低于未微调 original diffusion 的 `29.812 dB`：

| Stage2 目标 | 最佳 checkpoint PSNR |
|---|---:|
| hybrid, `x0_weight=0.01` | `28.303 dB` |
| hybrid, `x0_weight=0.1` | `28.919 dB` |
| hybrid, `x0_weight=1.0` | `28.917 dB` |
| x0 only | `28.202 dB` |

`x0_weight=0.1` 的早期 checkpoint 也从 update 50 的 `29.585 dB` 下降到 update 250 的 `28.680 dB`，因此未扩展成长训练。该结果再次说明训练 surrogate loss 下降不能替代最终 codec sampling 审计。入口、checkpoint 和 W&B：

```text
scripts/run_caesar_era5_d_x0_objective_pilot.sh
checkpoints/caesar_era5_d_x0_objective_pilot/
hybrid_w001: https://wandb.ai/1796248596-university-of-chinese-academy-of-sciences/caesar-era5-hourly-tuning/runs/go7p0nde
hybrid_w01:  https://wandb.ai/1796248596-university-of-chinese-academy-of-sciences/caesar-era5-hourly-tuning/runs/x0mhdwx9
hybrid_w1:   https://wandb.ai/1796248596-university-of-chinese-academy-of-sciences/caesar-era5-hourly-tuning/runs/rkvdcxgh
x0_only:     https://wandb.ai/1796248596-university-of-chinese-academy-of-sciences/caesar-era5-hourly-tuning/runs/f0udpunj
```

### 9.8 D hard-channel 与 diffusion ensemble 诊断

为确认 D 的时间生成误差是否主要由难变量训练不足导致，分别在 specific humidity（变量 `37:74`）、relative humidity（`185:222`）和 single-level variables（`259:268`）上继续训练 original noise objective。训练时的 diffusion validation loss 明显下降，但固定 seed、32-step 的真实 sampling PSNR 从最早 checkpoint 起就下降；因此不能用 surrogate validation loss 选择正式 codec 权重，也不再延长这些 specialist：

| 变量组 | original sampling PSNR | specialist sampling PSNR |
|---|---:|---:|
| specific humidity | `19.827 dB` | `19.186 dB`（`lr=1e-4`, 1k） |
| relative humidity | `17.762 dB` | `17.108 dB`（`lr=1e-4`, 1k） |
| single-level | `19.375 dB` | `18.221 dB`（`lr=1e-4`, 500） |

对应 checkpoint、日志和机器可读结果位于：

```text
checkpoints/caesar_era5_d_hard_channel_specialists/
logs/caesar_era5_d_hard_channel_specialists/
unified_results/diagnostic_caesar_d_hard_channel_specialists/
scripts/run_caesar_era5_d_hard_channel_specialists.sh
scripts/evaluate_caesar_era5_d_hard_channel_specialists.sh
```

另一条有效诊断是 diffusion ensemble。`diffusion_ensemble_size=N` 对同一已编码 keyframe latent 顺序生成 `N` 组 predicted frames，分别经 Stage1 decoder 还原到像素域后取平均。它不改变 latent 或 PCA residual 的码率定义，也不传输额外随机变量，因为编码端和解码端都从固定 seed 开始；但 diffusion 部分解码计算量近似增至 `N` 倍。默认值仍为 `1`，避免改变既有 CAESAR-D 行为。

在完整 `268x16x240x240` 输入、`interpo_rate=3` 上，当前 5k 完整权重的无 PCA base reconstruction 从 ensemble-1 的 `21.632 dB` 提高到 ensemble-4 的 `22.961 dB`，即同一 latent BPP `0.11130` 下提高 `1.328 dB`。该数字是时间生成诊断，不是 13 点最终 RD 增益。正式 EB 曲线使用以下入口；结果必须在完成后再进入第 9.6 节的 matched-BPP 前沿：

CPU 上的完整 268 变量同设备复核进一步比较了 5k 与 100k：5k 从 `21.591` 提高到 `22.961 dB`（`+1.371 dB`），100k 从 `21.516` 提高到 `22.813 dB`（`+1.297 dB`）。ensemble-4 下 5k 仍比 100k 高 `0.149 dB`，因此最终 ensemble 候选固定为 5k，不再延长 Stage2。CPU/GPU 对 5k ensemble-4 的差异仅约 `0.0007 dB`；CPU 结果只用于 sampling 筛选，正式 BPP 仍必须使用 GPU `nvCOMP Zstd` 路径。

```text
scripts/run_caesar_era5_d_ensemble4_full_curves.sh
unified_results/objective_era5_caesar_d_decoder100k_stage2_overlap5k_ensemble4_rd/
unified_results/objective_era5_caesar_d_lam1em3_original_stage2_ensemble4_rd/
unified_results/diagnostic_caesar_d_stage2_cpu_ensemble4_full268/
```
