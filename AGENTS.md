# AGENTS.md

本文为在此仓库中工作的编码 agent 提供仓库级说明。开始工作前应先阅读本文；涉及正式评测时，还应阅读本文指向的协议和复现文档。

# AIForCompression 项目

目标是搭建一个可复用的学习型/传统压缩 benchmark 框架，用统一的数据中间层评测多种数据集和多种模型，而不是为每个“数据集 x 模型”写一次性脚本。

当前主框架是：

```text
Dataset Adapter -> CanonicalSample -> Model View -> Codec Runner -> Metrics -> summary.json
```

主入口：

```bash
python scripts/run_dataset_compression.py \
  --dataset <dataset> \
  --data_root <path> \
  --output_dir unified_results/<run_name> \
  --models <model...> \
  --max_samples <n>
```

## 当前项目结构

```text
AIForCompression/
├── compression_pipeline/           # 共享评测框架
│   ├── adapters/                   # 数据集 -> CanonicalSample
│   ├── canonical.py                # CanonicalSample / DatasetManifest
│   ├── views.py                    # 图像分组、CAESAR 视图和逆变换
│   ├── runner.py                   # 图像模型分组运行器
│   ├── torch_codecs.py             # 模型 API 封装
│   ├── model_registry.py           # 模型 checkpoint 发现与加载
│   ├── cra5_runner.py              # CRA5 原生 ERA5 运行器
│   └── caesar_runner.py            # CAESAR 序列运行器
├── scripts/                        # 评测和 Slurm 入口
├── utils/                          # 绘图、聚合、下载和统计工具
├── tests/                          # 框架行为回归测试
├── docs/                           # 设计说明和使用文档
├── models/                         # 上游模型源码树
├── checkpoints/                    # 按模型组织的权重
├── normalization/                  # adapter 使用的 ERA5 逐日 mean/std
└── unified_results/                # 评测输出和图表
```

## 支持的数据集

`scripts/run_dataset_compression.py --dataset` 当前支持：

- `kodak`：RGB 图像目录，canonical 格式为 `[C,H,W] uint8`。
- `uvg`：YUV420 视频帧，canonical 格式为 RGB 帧。
- `era5`：配对的 `{timestamp}_pressure.nc` 和 `{timestamp}_single.nc`，canonical 格式为 268 通道浮点场。
- `tomo`：重建的断层扫描 HDF5 volume，按切片或切片组读取。
- `hurricane`：`.bin.f32` 时序场。
- `s2c`：Sentinel-2 SAFE/SAFE.zip 中的 JP2 波段 tile。
- `nyx`：SDRBench NYX `.f32` 三维 volume。
- `shanghai_xray`：同步辐射 X-ray TIFF 图像。
- `isot1024`：各向同性湍流 HDF5 速度数据。
- `lysozyme`：CHESS lysozyme HDF5 衍射帧。

Pipeline 的整体职责和扩展方法见 `docs/compression_pipeline概览.md`。正式数据集和模型的处理规则见 `docs/benchmark_reproduction_manifest.md`，机器可读的评测契约见 `benchmark_protocols/objective_v1.json`。通道分组、归一化、CAESAR 序列形状、BPP、PSNR、LPIPS、显存和吞吐量以这两者为准。

面向图像模型的 adapter 必须输出 `CanonicalSample(layout="channel_height_width")`。支持序列的 adapter 还可以实现 `load_sequence()`，为 CAESAR 返回 `[V,T,H,W]` 和时间戳。

## 支持的模型系列

### 通用图像/帧内模型

以下模型通过 `build_image_groups()` 和 `run_image_grouped_sample()` 运行：

- `DCAE`
- `LIC_TCM`
- `LIC-HPCM`
- `RwkvCompress`
- `WeConvene`
- `DCVC-RT` intra wrapper
- `DCMVC` intra wrapper

规则：

- 图像/uint8 数据使用 `/255 -> [0,1]`。
- 浮点科学数据使用逐通道 min/max 归一化，并在计算指标前执行逆变换。
- 多通道样本拆分为三通道组；最后一组不足三通道时，重复最后一个真实通道进行 padding。
- 指标在重建后的原始数据空间计算，而不是在归一化后的模型输入空间计算。

### CRA5

CRA5 是 ERA5 原生模型。在主评测路径中，它只接受 ERA5 风格的 268 通道科学数据；运行时不应传入 `--max_channels`：

```bash
python scripts/run_dataset_compression.py \
  --dataset era5 \
  --data_root /data/run01/scxj523/zsh/project/Data/ERA5/2024 \
  --output_dir unified_results/era5_cra5 \
  --models CRA5 \
  --max_samples 1
```

在非 ERA5 数据上运行 CRA5 必须显式传入 `--allow_cra5_adapted`。该路径会调整样本尺寸并复制通道，以适配 CRA5 的 268 通道输入；它仅用于消融或诊断基线，不是公平的默认评测。

### CAESAR

CAESAR 使用原生序列视图，不经过图像分组运行器：

- `caesar_v` 需要连续 8 帧。
- `caesar_d` 需要连续 16 帧。
- CAESAR 运行器的输入为 `[V,T,H,W]`；`build_caesar_view()` 将其转换为 `[V,S,T,H,W]`，其中 `S=1`。
- `--caesar_start_index` 用于选择连续窗口。
- `--caesar_eb` 控制 error-bound 扫描。

示例：

```bash
python scripts/run_dataset_compression.py \
  --dataset era5 \
  --data_root /data/run01/scxj523/zsh/project/Data/ERA5/2024 \
  --output_dir unified_results/era5_caesar_v \
  --models caesar_v \
  --max_samples 8 \
  --max_channels 6 \
  --caesar_eb 1e-4 1e-3 1e-2
```

## 指标契约

每个成功结果都应包含：

- `mse`, `rmse`, `psnr`
- `bpp`, `bitstream_bytes`, `original_bytes`, `compression_ratio`
- `encode_time_total`, `decode_time_total`
- `encode_time_per_group_avg`, `decode_time_per_group_avg`
- `encode_throughput_MBps`, `decode_throughput_MBps`
- 兼容旧格式的别名：`encode_time_avg`、`decode_time_avg`、`encode_throughput`、`decode_throughput`
- 适用时包含 `model_name`、`model_id`、`sample_id`、`dataset_id`

PSNR 在原始数据空间中按 `10 * log10(data_range^2 / mse)` 计算。每个样本的 `data_range` 为 `original.max() - original.min()`；常量样本使用 `1.0`。

## HPC / Slurm 规则

当前 5090 主机环境：

```bash
source /workspace/ai4cp/bin/activate
cd /workspace/AIForCompression
```

本机 Turb_Rot 评测数据使用 `/workspace/Data/Turb_Rot_testset.npz`；正式的 Turb_Rot tuned CAESAR checkpoint 目录使用 `/workspace/AIForCompression/checkpoints/caesar_tuned`。

旧集群环境：

使用 Slurm 分配的 GPU。除非在 Slurm 外部调试，否则作业脚本不得取消、导出或覆盖 `CUDA_VISIBLE_DEVICES`。

标准环境：

```bash
eval "$(/data/home/scxj523/run/miniconda3/bin/conda shell.bash hook)"
conda activate /data/run01/scxj523/zsh/envs/zsh
cd /data/run01/scxj523/zsh/project/AIForCompression
```

Smoke test 入口：

```bash
sbatch scripts/run_framework_smoke_model.sh <case> [max_model_jobs|all] [max_samples|all]
```

示例：

```bash
sbatch scripts/run_framework_smoke_model.sh kodak_dcae all all
sbatch scripts/run_framework_smoke_model.sh era5_caesar_v 1 8
```

## 开发规则

- 除非必须修复具体模型，否则将 `models/` 视为上游/vendor 模型代码，不随意修改。
- 优先在 `compression_pipeline/` 下增加可复用的 adapter、view 或 codec 代码，不要创建新的单次使用脚本。
- 添加数据集时，在 `compression_pipeline/adapters/` 中实现 adapter，在 `scripts/run_dataset_compression.py` 中注册，并记录其 canonical layout。
- 添加模型时，通过 `compression_pipeline/model_registry.py` 注册；如果模型不提供 CompressAI 风格的 `compress()`/`decompress()`，还需增加 codec wrapper。
- 测试应聚焦可复用的框架行为，至少运行：

```bash
pytest -q tests/test_compression_pipeline.py
```

`tests/test_model_registry.py` 依赖部分历史 checkpoint；仅在这些本地权重齐全时运行。当前精简后的 checkpoint 集合缺少 LIC-TCM `mse_lambda_0.05.pth.tar`，因此该测试会有一项资源缺失失败。

## 关键路径

- 项目根目录：`/workspace/AIForCompression`
- Turb_Rot 数据：`/workspace/Data/Turb_Rot_testset.npz`
- Checkpoint：`/workspace/AIForCompression/checkpoints`
- CAESAR tuned checkpoint：`/workspace/AIForCompression/checkpoints/caesar_tuned`
- 模型源码：`/workspace/AIForCompression/models`
- ERA5 normalization：`/workspace/AIForCompression/normalization`
