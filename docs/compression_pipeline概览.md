# `compression_pipeline` 概览

`compression_pipeline/` 是本仓库的数据和模型适配层。它把不同格式的数据转换成统一表示，再交给不同 codec，最后用统一方式计算指标。目标是避免为每个“数据集 × 模型”组合重复写预处理、重建和计时代码。

## 1. 两条使用路径

仓库中有两条入口，不能混用：

| 路径 | 入口 | 用途 |
|---|---|---|
| 通用 Pipeline | `scripts/run_dataset_compression.py` | 开发 adapter、接入模型、smoke test 和探索性实验 |
| objective-v1 | `scripts/prepare_objective_inputs.py`、`scripts/run_objective_benchmark.py` | 使用冻结样本、归一化和计时规则生成正式结果 |

通用 Pipeline 适合开发，但其默认样本选择或逐样本归一化不自动等于 objective-v1 正式协议。需要复现正式排名时，以 `benchmark_protocols/objective_v1.json` 和 `docs/benchmark_reproduction_manifest.md` 为准。

## 2. 主数据流

```text
原始文件
  → Dataset Adapter
  → CanonicalSample / ObjectiveSample
  → Model View
  → Codec Runner
  → 逆变换到原始数据空间
  → Metrics
  → summary.json
```

各层职责：

- Adapter 只负责读取、选择样本和转换轴布局，不应偷偷改变评测定义。
- Canonical 层描述模型无关的数据和元信息。
- View 层执行模型需要的分组、padding 和可逆归一化。
- Codec 层封装模型的 encode/decode 或 compress/decompress。
- Runner 组织 round trip、计时、码流大小和指标计算。
- 正式 objective 路径额外冻结样本、normalization、mask、checksum 和重复计时规则。

## 3. 目录与模块

| 文件/目录 | 作用 |
|---|---|
| `adapters/` | Kodak、UVG、ERA5、E3SM、Hurricane、NYX、Turb-Rot、Tomo、Lysozyme、S2C 等数据读取器 |
| `canonical.py` | 通用 `CanonicalSample` 和 `DatasetManifest` 数据结构 |
| `views.py` | 三通道图像分组、padding、归一化、逆变换和 CAESAR 序列视图 |
| `model_registry.py` | 模型/权重发现和加载，生成 `ModelJob` |
| `torch_codecs.py` | PyTorch/CompressAI 风格 codec 封装 |
| `nvjpeg_codecs.py` | nvJPEG 与 nvJPEG2000 调用封装 |
| `runner.py` | 通用图像分组 round trip 和结果汇总 |
| `caesar_runner.py` | CAESAR-V/D 原生序列运行器 |
| `cra5_runner.py` | CRA5 原生 ERA5 运行器 |
| `metrics.py` | MSE、PSNR、BPP、压缩率、LPIPS、吞吐量和显存指标 |
| `objective_data.py` | objective-v1 固定样本、dataset-level normalization 和 checksum |
| `objective_stacking.py` | 正式科学数据 corpus 的 pack、padding、crop 和 unpack |
| `objective_caesar.py` | objective-v1 中 CAESAR 所需的处理逻辑 |
| `era5_constants.py` | ERA5 变量相关常量 |

## 4. 通用数据表示

`CanonicalSample` 位于 `canonical.py`，主要字段为：

```python
CanonicalSample(
    dataset_id="...",
    sample_id="...",
    kind="image" or "scientific",
    array=array,
    layout="channel_height_width",
    metadata={...},
)
```

面向图像 codec 的样本统一为 `[C,H,W]`。需要时序的 adapter 可通过 `load_sequence()` 提供 `[V,T,H,W]`；`build_caesar_view()` 再转换为 CAESAR 使用的 `[V,S,T,H,W]`，其中通常 `S=1`。

正式路径使用 `ObjectiveSample`，固定为 `[V,T,H,W]`，并可附带同形状 mask。不要直接改变该布局，否则 checksum、堆叠和审计都会失效。

## 5. 图像模型如何处理多通道数据

`build_image_groups()` 将 `[C,H,W]` 按三个通道拆分：

```text
[C,H,W]
  → 每 3 通道一组
  → 最后一组不足 3 通道时复制最后一个真实通道
  → 转成 [1,3,H,W]
  → codec round trip
  → 丢弃 padding 通道并拼回原顺序
```

图像 `uint8` 数据使用 `/255`。科学浮点数据可使用逐通道 min/max、预先提供的 z-score 信息，或正式协议已经完成的 dataset-level normalization。所有可逆变换必须在计算指标前还原；解码所需的逐样本 normalization 参数要计入 side information。

## 6. 模型运行和结果

`model_registry.image_model_jobs()` 根据模型名称和 checkpoint 生成任务。通用图像模型由 `run_image_grouped_sample()` 运行；CAESAR、CRA5、nvJPEG 等使用各自原生 runner。

一个成功结果至少应包含：

- `mse`、`rmse`、`psnr`
- `bpp`、`bitstream_bytes`、`original_bytes`、`compression_ratio`
- encode/decode 总时间、平均时间和吞吐量
- `model_id`、`dataset_id`、`sample_id`
- 适用时的 LPIPS、显存和 side-information 字段

质量指标在重建后的原始数据空间计算。吞吐量的正式定义、warmup 和 repeat 次数以 objective-v1 协议为准。

## 7. 常用入口

查看通用入口参数：

```bash
python scripts/run_dataset_compression.py --help
```

通用 smoke 示例：

```bash
python scripts/run_dataset_compression.py \
  --dataset kodak \
  --data_root /workspace/Data/Kodac \
  --output_dir unified_results/smoke_kodak \
  --models DCAE \
  --max_samples 1 \
  --max_model_jobs 1
```

正式 objective-v1 的基本顺序：

```bash
python scripts/prepare_objective_inputs.py \
  --dataset era5_npy \
  --output-root unified_results/objective_v1

python scripts/run_objective_benchmark.py \
  --dataset era5_npy \
  --gpu 0 \
  --input-root unified_results/objective_v1 \
  --output-root unified_results/objective_v1 \
  --models DCAE LIC-HPCM CAESAR-V CAESAR-D
```

UVG 视频正式入口会在 `frames/` 不存在时自动调用 `scripts/export_objective_uvg_frames.py`，导出并校验 30 张 canonical PNG 后再运行 DCVC-RT/DCMVC P-frame。

模型、EB、checkpoint 和样本参数必须使用复现清单中的数据集专用配置，不能只照抄上面的结构示例。

## 8. 添加新数据集

1. 在 `compression_pipeline/adapters/` 新建 adapter。
2. 为图像路径输出 `CanonicalSample(layout="channel_height_width")`；序列模型需要时实现 `load_sequence()`。
3. 给出稳定且可追溯的 `dataset_id`、`sample_id` 和 source metadata。
4. 在 `scripts/run_dataset_compression.py` 注册参数和 adapter。
5. 增加读取、shape、dtype、crop 和逆变换测试。
6. 如果进入正式评测，再更新 objective 数据加载、协议 JSON、复现清单和审计规则。

不要只在运行脚本中临时读取新数据；数据语义应集中在 adapter 或 objective data 层。

## 9. 添加新模型

1. 优先把模型源码保留在 `models/`，共享适配代码放在 `compression_pipeline/`。
2. 在 `model_registry.py` 中定义 checkpoint 发现、模型 ID 和加载方式。
3. 若提供 CompressAI 风格 API，使用或扩展 `torch_codecs.py`；否则实现具有 `roundtrip()` 的 codec wrapper。
4. 确保真实码流字节数、encode/decode 时间和重建张量均由 wrapper 返回。
5. 在主入口注册模型，并先用一个样本和一个 checkpoint 做 smoke test。
6. 若进入 objective-v1，补充固定配置、重复计时、失败记录和审计覆盖。

不要把模型加载时间、磁盘 I/O 或指标计算混入正式 encode/decode 时间。

## 10. 修改后的最低验证

```bash
pytest -q \
  tests/test_compression_pipeline.py \
  tests/test_objective_data.py \
  tests/test_objective_stacking.py \
  tests/test_audit_objective_benchmark.py

python scripts/audit_objective_benchmark.py \
  unified_results/objective_all_to_all_v1 \
  --strict
```

`tests/test_model_registry.py` 会检查部分历史模型的本地 checkpoint；只有对应权重齐全时才运行它。当前精简后的 checkpoint 集合不含 LIC-TCM `mse_lambda_0.05.pth.tar`，直接运行该测试会出现一项资源缺失失败，并不表示 Pipeline 逻辑损坏。

若只修改某个 adapter 或 codec，还应为对应数据集运行一次最小 smoke test。不要覆盖正式结果目录；开发输出使用新的 `unified_results/<run_name>`。

## 11. 进一步阅读

- `docs/Data数据集.md`：本机数据集内容、格式与路径。
- `docs/objective_benchmark_protocol.md`：正式公平性和指标定义。
- `docs/benchmark_reproduction_manifest.md`：正式样本、参数、权重与命令。
- `unified_results/README.md`：结果目录、文件和命名规则。
