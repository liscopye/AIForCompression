# AIForCompression

仓库每个一级目录的用途和迁移必要性见 [`目录说明.md`](目录说明.md)。

统一评测 AI、视频和科学数据压缩方法的实验仓库。当前正式协议为 `aifc-objective-v1`，覆盖 10 个数据集；主结果已经完成严格审计。

## 快速入口

| 想做什么 | 入口 |
|---|---|
| 浏览文档导航 | `docs/README.md` |
| 激活环境并启动测试 | `docs/环境激活与测试启动.md` |
| 理解 Pipeline 架构 | `docs/compression_pipeline概览.md` |
| 查看全部正式结果 | `unified_results/objective_all_to_all_v1/index.html` |
| 查看机器可读汇总 | `unified_results/objective_all_to_all_v1/combined_summary.json` |
| 检查结果是否合规 | `unified_results/objective_all_to_all_v1/objective_protocol_audit.md` |
| 查看正式协议 | `benchmark_protocols/objective_v1.json` |
| 了解完整复现参数 | `docs/benchmark_reproduction_manifest.md` |
| 查看 ERA5 CAESAR-V 微调记录 | `docs/ERA5_CAESAR-V微调记录.md` |
| 查看 ERA5/CAESAR 微调结论 | `docs/项目交接总览.md` |
| 迁移到其他机器 | `docs/迁移到新机器指南.md` |

最简单的阅读方式是用浏览器打开：

```text
unified_results/objective_all_to_all_v1/index.html
```

该页面是单文件报告，包含全模型汇总、各数据集的 RD/吞吐量/显存图和结果表，无需启动服务。

## 安装与基本使用

```bash
git clone --recurse-submodules https://github.com/liscopye/AIForCompression.git
cd AIForCompression

python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

如果仓库已经下载但缺少外部模型：

```bash
git submodule update --init --recursive
```

部分模型还需要 CUDA、模型作者提供的依赖和对应 checkpoint。运行前可先查看具体命令的帮助：

```bash
python scripts/run_objective_benchmark.py --help
```

## 数据在哪里

数据不提交到 Git。当前机器的数据主要位于 `/workspace/Data`，正式输入由协议固定：

| 数据集 | 当前输入 |
|---|---|
| Kodak | `/workspace/Data/Kodac/kodim*.png`，24 张图像 |
| UVG | `/workspace/Data/UVG_Twilight_1080p`，连续 30 帧 |
| E3SM | `/workspace/Data/E3SM/caesar_processed/e3sm_5vars_paperlike240_first800_caesar.npz` |
| ERA5 | `/workspace/Data/ERA5/finetune_processed/era5_test.npy`，268 变量 × 16 时刻 |
| Hurricane | SDRBench Hurricane，前 96 帧 |
| NYX | 完整 `512³` baryon-density volume |
| Turb-Rot | `/workspace/Data/Turb_Rot_testset.npz`，section 0 和 8 |
| Tomo | `/workspace/Data/tomo_00001.h5`，两个 512-depth block |
| Lysozyme | `/workspace/Data/lysozyme_processed/mmap/lysozyme_test_nf16.npy` |
| S2C | Sentinel-2，4 个 tile × 4 个 band |

精确样本 ID、裁剪范围和 checksum 以 `benchmark_protocols/objective_v1.json` 与 `docs/benchmark_reproduction_manifest.md` 为准。

## 数据如何处理

所有方法先接收同一份 canonical 数据，再进入各模型自身的预处理：

```text
原始数据
  → dataset adapter 读取和裁剪
  → canonical float32 [V,T,H,W]
  → 固定的数据集级外部归一化
  → 模型自身的 padding / PCA / predictor / 内部归一化
  → 压缩与解压
  → 还原到 canonical 空间
  → 统一计算 BPP、PSNR、吞吐量和显存
```

主要原则：

- 同一数据集的所有模型使用相同样本、crop、mask 和外部归一化。
- BPP 包含解码所需的 side information。
- 主质量指标是固定尺度上的 normalized PSNR；Kodak 和 UVG 另外报告 LPIPS。
- 吞吐量不包含磁盘 I/O、模型加载和指标计算。
- 正式计时使用 2 次预热和 5 次测量。

数据读取位于 `compression_pipeline/adapters/`，统一堆叠和协议处理位于 `compression_pipeline/objective_data.py` 与 `compression_pipeline/objective_stacking.py`。

## 模型范围

正式主结果包含：

- DCAE
- LIC-HPCM base / large
- CAESAR-V / CAESAR-D
- DCMVC-I / DCMVC-IP
- DCVC-RT-I / DCVC-RT-IP
- cuSZ-Hi
- nvJPEG / nvJPEG2000

`models/` 保存模型源码或 submodule；`checkpoints/` 保存本地权重。GraphComp、Visemz、LIC-TCM 和部分 PCA/no-PCA 路径属于外部复现、历史实验或消融，不进入 objective-v1 正式排名。

## 常用命令

准备某个数据集的固定输入和 normalization manifest：

```bash
python scripts/prepare_objective_inputs.py \
  --dataset era5_npy \
  --output-root unified_results/objective_v1
```

运行非视频正式模型；模型和 EB 应使用复现清单中的数据集专用配置：

```bash
python scripts/run_objective_benchmark.py \
  --dataset era5_npy \
  --gpu 0 \
  --input-root unified_results/objective_v1 \
  --output-root unified_results/objective_v1 \
  --models DCAE LIC-HPCM CAESAR-V CAESAR-D
```

重新生成图表和单文件报告：

```bash
python scripts/analyze_objective_benchmark.py \
  --root unified_results/objective_all_to_all_v1
```

严格审计正式结果：

```bash
python scripts/audit_objective_benchmark.py \
  unified_results/objective_all_to_all_v1 \
  --strict
```

## 结果目录

```text
unified_results/
├── objective_all_to_all_v1/                         # 正式全模型主结果；直接查看 index.html
├── objective_runs/                                  # 补测、微调和探针来源，按数据集归档
│   ├── era5_npy/<run_name>/
│   ├── uvg_twilight_1080p/<run_name>/
│   └── <dataset>/<run_name>/
└── diagnostics/                                     # 失败实验机器记录
```

`objective_all_to_all_v1` 是跨模型主排名，页面及正式 JSON 保持自包含；`objective_runs` 保存生成正式结果时的来源记录，避免补测目录散落在 `unified_results/` 根目录。ERA5 CAESAR-V 微调专项位于 `objective_runs/era5_npy/`。可用 `python scripts/organize_objective_results.py` 预览归档计划，确认后加 `--execute` 执行。

## 仓库结构

| 路径 | 内容 |
|---|---|
| `compression_pipeline/` | 统一数据、模型 runner、指标和 codec 接口 |
| `scripts/` | 数据准备、评测、聚合、审计和画图入口 |
| `models/` | 模型源码与 submodule |
| `checkpoints/` | 本地模型权重 |
| `benchmark_protocols/` | 机器可读正式协议 |
| `normalization/` | ERA5 NetCDF adapter 使用的按日 mean/std；CRA5 全局统计在模型目录内 |
| `unified_results/` | 正式结果和必要诊断证据 |
| `docs/` | 复现、交接、结果解释和失败记录 |
| `tests/` | 数据处理、审计和 runner 测试 |

## 深入阅读

- `docs/结果目录索引.md`：哪些结果应看、哪些已废弃。
- `docs/benchmark_reproduction_manifest.md`：数据、EB、权重和完整命令。
- `docs/objective_benchmark_protocol.md`：公平性边界与指标定义。
- `docs/项目交接总览.md`：ERA5/CAESAR 权重、结论和项目状态。
- `docs/迁移到新机器指南.md`：新机器的软件安装、数据和权重复制及迁移验收。
