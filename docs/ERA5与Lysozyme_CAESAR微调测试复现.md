# ERA5 与 Lysozyme CAESAR 微调、测试复现

本文只说明当前保留的两条 CAESAR 微调链：ERA5 上的 CAESAR-V，以及 Lysozyme 上的 CAESAR-V/CAESAR-D。命令默认在当前 5090 主机执行。

## 1. 公共环境

```bash
source /workspace/ai4cp/bin/activate
cd /workspace/AIForCompression
```

开始前确认 GPU、基础权重和数据可见：

```bash
python -c 'import torch; print(torch.cuda.device_count(), torch.cuda.get_device_name(0))'
test -f checkpoints/caesar/caesar_v.pt
test -f checkpoints/caesar/caesar_d.pt
```

训练脚本会记录 W&B 日志，运行前需要已经完成 `wandb login`。测试不依赖 W&B。

## 2. ERA5：CAESAR-V

### 2.1 数据与训练范围

训练读取：

```text
/workspace/Data/ERA5/hourly_center512_shards_20240301_90d
```

这里应有 90 个 `*_hourly.npy` 日分片，每个分片包含 268 个变量、24 个小时和 513×513 空间网格。训练随机裁切为 256×256；`frame_step=24` 表示序列取每天同一 UTC 时刻，与正式 daily ERA5 测试的时间间隔保持一致。

检查数据：

```bash
find /workspace/Data/ERA5/hourly_center512_shards_20240301_90d \
  -maxdepth 1 -name '*_hourly.npy' -type f | wc -l
```

### 2.2 微调

ERA5 训练必须按下面顺序执行。

当前两个入口分别固定使用物理 GPU 4 和 GPU 5；迁移到 GPU 数量较少的机器时，应先修改脚本末尾 `launch` 的 GPU 编号。

第一阶段从 original CAESAR-V 开始，训练 100k updates，学习低码率表示：

```bash
bash scripts/run_caesar_era5_v_lowrate_100k.sh
```

第一阶段所需输出：

```text
checkpoints/caesar_era5_vd_lowrate_100k/v_lr1em5_lam1em3_full100k_update100000.pt
```

第二阶段读取上述权重，冻结 encoder 和码率路径，只训练 decoder 100k updates，以恢复重建质量：

```bash
bash scripts/run_caesar_era5_v_decoder_quality_100k.sh
```

最终入选权重：

```text
checkpoints/caesar_era5_v_decoder_quality_100k/from_lowrate_lr3em4.pt
```

日志分别位于：

```text
logs/caesar_era5_v_lowrate_100k/
logs/caesar_era5_v_decoder_quality_100k/
```

两个 shell 入口共同调用 `scripts/finetune_caesar_era5.py`。正常复现应运行 shell 入口，不需要手工重写长参数。

### 2.3 测试最终权重

Objective runner 要求 checkpoint 目录中的文件名为 `caesar_v.pt`。为了不改动正式权重，先建立临时软链接，再把新结果写到临时目录：

```bash
mkdir -p /workspace/tmp/era5_caesar_v_checkpoint
ln -sfn \
  /workspace/AIForCompression/checkpoints/caesar_era5_v_decoder_quality_100k/from_lowrate_lr3em4.pt \
  /workspace/tmp/era5_caesar_v_checkpoint/caesar_v.pt

python scripts/run_objective_benchmark.py \
  --dataset era5_npy \
  --gpu 0 \
  --input-root unified_results/objective_all_to_all_v1 \
  --output-root /workspace/tmp/era5_caesar_v_reproduction \
  --models CAESAR-V \
  --caesar-checkpoint-root /workspace/tmp/era5_caesar_v_checkpoint \
  --caesar-variant decoder_quality_100k_lr3em4 \
  --caesar-eb 0.1 0.01 0.003 0.001 0.0001 3e-6 1e-9 \
  --warmups 2 \
  --repeats 5
```

快速检查可在命令末尾加 `--smoke --warmups 0 --repeats 1`。正式复现不要加 `--smoke`。

当前保留的参考结果：

```text
unified_results/era5_caesar_v/decoder_final_rd/summary.json
```

其中 `daily_real_codec/` 是早期单点有效性验证，`daily_v_100k_eb_compare/` 是早期 full-100k 与 original 的多 EB 对比；最终权重以 `decoder_final_rd/` 为主。

## 3. Lysozyme：CAESAR-V 与 CAESAR-D

### 3.1 数据

训练和独立验证读取 mmap 数组：

```text
/workspace/Data/lysozyme_processed/mmap/lysozyme_train_nf16.npy
/workspace/Data/lysozyme_processed/mmap/lysozyme_val_nf16.npy
```

专项测试脚本读取：

```text
/workspace/Data/lysozyme_processed/lysozyme_test_nf8.npz
/workspace/Data/lysozyme_processed/lysozyme_test_nf16.npz
```

检查文件：

```bash
test -f /workspace/Data/lysozyme_processed/mmap/lysozyme_train_nf16.npy
test -f /workspace/Data/lysozyme_processed/mmap/lysozyme_val_nf16.npy
test -f /workspace/Data/lysozyme_processed/lysozyme_test_nf8.npz
test -f /workspace/Data/lysozyme_processed/lysozyme_test_nf16.npz
```

### 3.2 双卡微调

```bash
bash scripts/run_caesar_lysozyme_retrain_2gpu.sh
```

该入口同时启动两条任务：

- GPU 0：CAESAR-V Stage 1，100k updates。
- GPU 1：CAESAR-D Stage 1，100k updates；成功后自动开始 Stage 2，200k updates。

最终权重：

```text
checkpoints/caesar_lysozyme/caesar_v_tuning_lysozyme.pt
checkpoints/caesar_lysozyme/caesar_d_tuning_lysozyme_vae.pt
checkpoints/caesar_lysozyme/caesar_d_tuning_lysozyme.pt
```

其中 `caesar_d_tuning_lysozyme_vae.pt` 是 D 的 Stage 1 中间依赖，复现 Stage 2 时必须保留；最终推理使用 `caesar_d_tuning_lysozyme.pt`。日志位于 `logs/caesar_lysozyme_retrain/`。

### 3.3 测试与画图

先用单个 EB 检查 V 和 D 是否能正常加载、压缩和解压：

```bash
python scripts/sweep_caesar_eb_lysozyme.py \
  --model_type V --gpu 0 --test_eb 1e-3 \
  --output_root /workspace/tmp/lysozyme_caesar_smoke

python scripts/sweep_caesar_eb_lysozyme.py \
  --model_type D --gpu 1 --test_eb 1e-3 \
  --output_root /workspace/tmp/lysozyme_caesar_smoke
```

完整运行 original 与 finetuned 的 EB 曲线：

```bash
python scripts/sweep_caesar_eb_lysozyme.py --model_type V --gpu 0
python scripts/sweep_caesar_eb_lysozyme.py --model_type D --gpu 1
```

两个命令可以在两个终端并行执行。输出位于：

```text
unified_results/lysozyme_caesar_tuned/v/
unified_results/lysozyme_caesar_tuned/d/
```

每个目录的 `sweep_results.json` 保存曲线数据。测试完成后重新画图：

```bash
python utils/plot_caesar_lysozyme_eb_sweep.py --model_type both
```

生成的 `psnr_vs_bpp.png` 和 `time_vs_eb.png` 位于各模型目录，合并图位于：

```text
unified_results/lysozyme_caesar_tuned/psnr_vs_bpp_combined.png
```

底层单次评测入口是 `scripts/eval_caesar_lysozyme.py`；通常直接使用 sweep 入口即可，因为它会统一 checkpoint、EB、输出命名和 original/finetuned 对比。

## 4. 复现完成后的检查

```bash
python -m json.tool \
  unified_results/era5_caesar_v/decoder_final_rd/summary.json >/dev/null

python -m json.tool \
  unified_results/lysozyme_caesar_tuned/v/sweep_results.json >/dev/null

python -m json.tool \
  unified_results/lysozyme_caesar_tuned/d/sweep_results.json >/dev/null
```

复现时不要用 Lysozyme 的训练脚本训练 ERA5，也不要把 ERA5 daily 测试误换成连续小时测试。两者的数据布局、时间采样、归一化和 checkpoint 选择不同。
