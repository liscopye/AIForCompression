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

`frozen_scale_v` 来自数据集完整 objective corpus，不按 sample 或模型重新计算。Kodak/UVG 使用全部原生 RGB 图像/帧计算 LPIPS；每个科学 canonical sample 使用冻结 normalization 后按展平顺序等距抽取的 32 个固定平面，将灰度复制为 RGB，只作诊断，不作为主排名指标。

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

它是在低码率 100k 权重上冻结 encoder/entropy-rate 路径、只微调 decoder 的验证集最佳权重。旧 `caesar_era5_daily_v_full_100k` 是一条有效且在部分高质量区非支配的历史曲线，但其 checkpoint 已于 2026-08-11 清理；结果 JSON 和图仍保存在 `unified_results_backup_20260811/`。

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
| 科学场 | Turb_Rot | `/workspace/Data/Turb_Rot_testset.npz` | variable 0，section 0 和 8，各 `[1,256,256,256]`；256 planes 全部参与。 |
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

## 9. ERA5 CAESAR-V 微调模型

ERA5 只保留经过筛选的 CAESAR-V 微调权重，不保留搜索网格、中间里程碑和失败候选。CAESAR-D 没有完成可作为正式结论的微调评测，因此只保留 original checkpoint 和通用 benchmark 支持。

| 用途 | 权重 | SHA-256 |
|---|---|---|
| CAESAR-V original | `checkpoints/caesar/caesar_v.pt` | `4acebeb189ab0f8b99de167326cc32b5390bc9c5025a85d9502641b73a7ad355` |
| CAESAR-D original | `checkpoints/caesar/caesar_d.pt` | `3cb2bbadbd9756275504500801f2b28f32ce6320fd084d7d63c4d8178cfbdbbe` |
| ERA5 V 最佳 | `checkpoints/caesar_era5_v_decoder_quality_100k/from_lowrate_lr3em4.pt` | `e44f2951844d6e873b024b4c288da315e50d015f9d323262e24b5d1e5e7dae57` |

V 微调链只保留以下两个入口，必须按顺序运行：

```bash
bash scripts/run_caesar_era5_v_lowrate_100k.sh
bash scripts/run_caesar_era5_v_decoder_quality_100k.sh
```

两个脚本只生成最终 V 所需路径；共同训练实现为 `scripts/finetune_caesar_era5.py`。训练链会重新生成中间 Stage1 权重，仓库只长期保留最终入选权重。

最佳 V 是低码率 Stage1 后冻结编码器和码率路径、只优化 decoder 的 100k 路径。

既有正式结果仍位于：

```text
unified_results/objective_runs/era5_npy/objective_era5_caesar_v_decoder_final_rd/
unified_results/objective_all_to_all_v1/
```
