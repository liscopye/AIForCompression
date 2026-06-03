# 压缩 Pipeline 使用说明

这一层把数据集处理和模型执行拆开：

```text
Dataset Adapter -> CanonicalSample -> Model View -> Codec Runner -> Metrics
```

当前支持的数据集：

- `kodak`：RGB 图像目录，输出 canonical `[C, H, W] uint8`。
- `era5`：配对的 `{timestamp}_pressure.nc` 和 `{timestamp}_single.nc`，输出 canonical `[C, H, W] float32`。

## 准备 Canonical 数据

Kodak：

```bash
python scripts/run_compression_pipeline.py \
  --dataset kodak \
  --data_root /data/run01/scxj523/zsh/project/Data/Kodac \
  --output_dir prepared_datasets/kodak \
  --max_samples 24
```

ERA5：

```bash
python scripts/run_compression_pipeline.py \
  --dataset era5 \
  --data_root /data/run01/scxj523/zsh/project/Data/ERA5/2024 \
  --output_dir prepared_datasets/era5_2024 \
  --max_samples 1
```

输出内容包括：

- `dataset_manifest.json`
- `samples.jsonl`
- 每个 canonical sample 对应一个 `.npy` 数组

## 运行图像类压缩

对于暴露 CompressAI-like API 的模型：

```python
compressed = model.compress(x)
reconstructed = model.decompress(compressed["strings"], compressed["shape"])
```

运行：

```bash
python scripts/run_dataset_compression.py \
  --dataset kodak \
  --data_root /data/run01/scxj523/zsh/project/Data/Kodac \
  --output_dir unified_results/kodak_image_models \
  --models DCAE LIC_TCM \
  --max_samples 2
```

Slurm 提交全权重、全 Kodak sample 的 framework smoke 任务时使用：

```bash
sbatch scripts/run_framework_smoke_model.sh kodak_dcae all all
```

第三个参数 `all` 会传 `--max_samples -1`，遍历 Kodak 目录下全部 sample。集群作业脚本不要手动 `unset`、`export` 或覆盖 `CUDA_VISIBLE_DEVICES`；该变量由 Slurm 按分配到的 GPU 设置，脚本改动会破坏 GPU 绑定。

ERA5 使用同一个 runner。View builder 会按 3 个通道一组分组；最后一组不足 3 个通道时，重复最后一个真实通道进行补齐；每个通道归一化到 `[0, 1]`；计算指标前再执行反变换：

```bash
python scripts/run_dataset_compression.py \
  --dataset era5 \
  --data_root /data/run01/scxj523/zsh/project/Data/ERA5/2024 \
  --output_dir unified_results/era5_image_models \
  --models DCAE LIC_TCM \
  --max_samples 1
```

## 运行 CRA5

CRA5 已通过其原生 268 通道 ERA5 路径接入。它只兼容保留完整 268 个通道的 ERA5 canonical sample：

```bash
python scripts/run_dataset_compression.py \
  --dataset era5 \
  --data_root /data/run01/scxj523/zsh/project/Data/ERA5/2024 \
  --output_dir unified_results/era5_cra5 \
  --models CRA5 \
  --max_samples 1
```

使用 CRA5 时不要传 `--max_channels`。Runner 会以 `x.shape == [1, 268, H, W]` 调用 `model.compress(x)`，然后调用 `model.decompress(strings, z_shape)`。

默认情况下，CRA5 会拒绝非 ERA5 / 非 268 通道样本，避免把 resized/channel-replicated 的非公平 baseline 混进主 benchmark。若确实要做 Kodak/Tomo 等诊断性 ablation，必须显式传：

```bash
--allow_cra5_adapted
```

该模式会把输入 resize 到 CRA5 空间尺寸并复制/截取到 268 通道，只应在结果名和说明中标注为 adapted CRA5。

## 运行 CAESAR

CAESAR 通过自己的原生 model view 接入，不走 image-group runner。它需要 ERA5 的连续时间段 stack：

```text
[V, T, H, W] -> [V, S, T, H, W]
```

其中 `S=1`，`T` 是连续时间帧数：

- `caesar_v`：需要至少 8 个连续 timestamp。
- `caesar_d`：需要至少 16 个连续 timestamp。

示例：

```bash
python scripts/run_dataset_compression.py \
  --dataset era5 \
  --data_root /data/run01/scxj523/zsh/project/Data/ERA5/2024 \
  --output_dir unified_results/era5_caesar_v \
  --models caesar_v \
  --max_samples 8
```

如果要跑 `caesar_d`：

```bash
python scripts/run_dataset_compression.py \
  --dataset era5 \
  --data_root /data/run01/scxj523/zsh/project/Data/ERA5/2024 \
  --output_dir unified_results/era5_caesar_d \
  --models caesar_d \
  --max_samples 16
```

可以用 `--caesar_start_index` 选择连续窗口的起点。Runner 会检查窗口内 timestamp 是否按固定时间间隔递增，避免把不连续时间片送进 CAESAR。

## 运行 DCVC-FM

ERA5 上的视频 intra smoke test 目前走单独脚本，不走 `run_dataset_compression.py`。该脚本会把 268 个通道按 3 通道分组、逐组做 min-max 归一化后送入 `DCVC-FM` 的 image/intra model：

```bash
python scripts/test_video_intra_era5.py \
  --data_root /data/run01/scxj523/zsh/project/Data/ERA5/2024 \
  --output_dir unified_results/video_intra_era5 \
  --model DCVC_FM \
  --dcvc_checkpoint /data/run01/scxj523/zsh/project/AIForCompression/checkpoints/dcvc-fm/cvpr2024_image.pth.tar \
  --max_samples 1
```

输出写入 `summary.json`，模型名会标记为 `DCVC_FM_Intra_q{0,21,42,63}`。

如果要在集群上做 smoke test：

```bash
sbatch scripts/run_video_models_268.sh
```

`DCVC-FM` 需要单独准备权重目录 `checkpoints/dcvc-fm/`，至少包含：

- `cvpr2024_image.pth.tar`
- `cvpr2024_video.pth.tar`（如果后续要跑 video/P-frame 路径）

## 运行 DCVC-RT / DCMVC（Kodak 图像压缩）

DCVC-RT 和 DCMVC 都已通过 compression pipeline 接入，使用其 intra（图像）模型，可用于 Kodak 数据集。两个模型均不支持 CompressAI API，使用自定义 codec 包装器。

DCVC-RT 需要 GPU（依赖 CUDA 推理扩展）。DCMVC 支持 CPU 和 GPU。

```bash
python scripts/run_dataset_compression.py \
  --dataset kodak \
  --data_root /data/run01/scxj523/zsh/project/Data/Kodac \
  --output_dir unified_results/kodak_dcvc_dcmvc \
  --models DCVC-RT DCMVC \
  --max_samples 24
```

两个模型各有多个质量等级：
- **DCVC-RT** (`DCVC_RT_Intra_q{0,21,42,63}`)：4 个 qp 等级，权重 `checkpoints/dcvc-rt/cvpr2025_image.pth.tar`
- **DCMVC** (`DCMVC_Intra_q{0,1,2,3}`)：4 个 q_index 等级，权重 `checkpoints/dcmvc/cvpr2023_image_psnr.pth.tar`

只测试单个模型：

```bash
# 仅 DCVC-RT
python scripts/run_dataset_compression.py \
  --dataset kodak \
  --data_root /data/run01/scxj523/zsh/project/Data/Kodac \
  --output_dir unified_results/kodak_dcvc_rt \
  --models DCVC-RT \
  --max_samples 24

# 仅 DCMVC
python scripts/run_dataset_compression.py \
  --dataset kodak \
  --data_root /data/run01/scxj523/zsh/project/Data/Kodac \
  --output_dir unified_results/kodak_dcmvc \
  --models DCMVC \
  --max_samples 24
```
