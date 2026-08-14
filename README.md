# AIForCompression

AIForCompression 是一个统一评测学习型压缩、传统图像/视频压缩和科学数据压缩方法的实验框架。项目通过共享的数据中间层和指标协议，让不同模型从一致的输入、裁剪、mask 和外部归一化开始，并输出可比较的率失真、吞吐量、显存和感知质量结果。

## 核心流程

```text
Dataset Adapter
  -> CanonicalSample
  -> Model View
  -> Codec Runner
  -> Metrics
  -> summary.json
```

- Dataset Adapter 将不同格式的数据转换成统一 canonical 表示。
- Model View 完成模型需要的分组、序列组织和可逆变换。
- Codec Runner 封装不同模型的压缩和解压接口。
- Metrics 在还原后的统一数据空间计算 BPP、PSNR、LPIPS、吞吐量和显存。

正式评测协议为 `aifc-objective-v1`，机器可读定义位于 `benchmark_protocols/objective_v1.json`。

## 支持范围

当前数据集包括 Kodak、UVG、ERA5、E3SM、Hurricane、NYX、Turb-Rot、Tomo、Lysozyme、S2C、Shanghai X-ray 和 ISOT1024 等图像、视频与科学数据。

当前模型和 codec 包括：

- DCAE、LIC-TCM、LIC-HPCM
- DCVC-RT、DCMVC
- CAESAR-V、CAESAR-D、CRA5
- cuSZ-Hi、nvJPEG、nvJPEG2000
- 部分 PCA hybrid 消融路径

并非所有模型都适用于所有数据集；正式组合、权重和控制参数以复现清单为准。

## 快速开始

```bash
git clone --recurse-submodules https://github.com/liscopye/AIForCompression.git
cd AIForCompression

python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

当前工作站环境：

```bash
source /workspace/ai4cp/bin/activate
cd /workspace/AIForCompression
```

通用评测入口：

```bash
python scripts/run_dataset_compression.py \
  --dataset <dataset> \
  --data_root <path> \
  --output_dir unified_results/<run_name> \
  --models <model...> \
  --max_samples <n>
```

运行正式 objective 评测前，请按照复现清单准备冻结输入和数据集级 normalization。

## 结果

正式全模型结果位于：

```text
unified_results/objective_all_to_all_v1/
```

用浏览器打开 `unified_results/objective_all_to_all_v1/index.html`，可以查看各数据集的率失真曲线、LPIPS、吞吐量、峰值显存和结果表。

`unified_results/era5_caesar_v/` 和 `unified_results/lysozyme_caesar_tuned/` 保存确认有效的 CAESAR 微调专项结果。结果目录内每个文件的含义见 `unified_results/README.md`。

## 仓库结构

| 路径 | 作用 |
|---|---|
| `compression_pipeline/` | 统一数据表示、adapter、model view、runner 和 codec 接口 |
| `scripts/` | 数据准备、训练、评测、审计和结果生成入口 |
| `utils/` | 数据统计、下载、检查和绘图工具 |
| `models/` | 上游模型源码与 submodule |
| `checkpoints/` | 模型权重 |
| `benchmark_protocols/` | 机器可读评测协议 |
| `normalization/` | 旧 ERA5 NetCDF 流程的逐日统计，不是 objective 的通用 normalization |
| `unified_results/` | 正式结果、图表和必要的微调证据 |
| `tests/` | Pipeline、数据处理和结果审计测试 |
| `docs/` | 使用、复现和设计文档 |

更完整的目录说明见 `目录说明.md`。

## 文档入口

| 内容 | 文档 |
|---|---|
| 文档总导航 | `docs/README.md` |
| 环境激活与启动测试 | `docs/环境激活与测试启动.md` |
| Pipeline 设计 | `docs/compression_pipeline概览.md` |
| 正式处理规则与复现参数 | `docs/benchmark_reproduction_manifest.md` |
| 评测协议说明 | `docs/objective_benchmark_protocol.md` |
| ERA5/Lysozyme CAESAR 微调复现 | `docs/ERA5与Lysozyme_CAESAR微调测试复现.md` |
| 结果文件说明 | `unified_results/README.md` |
| 数据集说明 | `docs/Data数据集.md` |

## 开发与检查

新增数据集时优先实现可复用 adapter；新增模型时通过模型注册和 codec wrapper 接入，避免为每个“数据集 × 模型”创建一次性脚本。

基础回归测试：

```bash
pytest -q tests/test_compression_pipeline.py
```

正式结果审计：

```bash
python scripts/audit_objective_benchmark.py \
  unified_results/objective_all_to_all_v1 \
  --strict
```
