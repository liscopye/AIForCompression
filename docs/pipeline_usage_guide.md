# 压缩模型评测 Pipeline 使用指南

## 环境

```bash
conda activate /data/run01/scxj523/zsh/envs/zsh
cd /data/run01/scxj523/zsh/project/AIForCompression
```

如有 `hdf5plugin` 导入错误（Lysozyme 数据需要）：
```bash
pip install hdf5plugin
```

---

## 1. 基本用法

所有测试通过 `scripts/run_dataset_compression.py` 运行：

```bash
python scripts/run_dataset_compression.py \
  --dataset <数据集名> \
  --data_root <数据路径> \
  --output_dir unified_results/<输出目录> \
  --models <模型列表> \
  --max_samples <样本数>
```

## 2. 数据集速查

| 数据集 | `--dataset` | `--data_root` | 额外参数 |
|--------|------------|---------------|---------|
| Kodak | `kodak` | `Data/Kodac` | `--resolution 512 512` |
| UVG | `uvg` | `Data/UVG` | — |
| Hurricane | `hurricane` | `Data/hurricane/100x500x500` | — |
| S2C | `s2c` | `Data/s2c/S2C_MSIL2A_*...SAFE` | `--tile_size 1024` |
| NYX | `nyx` | `Data/nyx/SDRBENCH-EXASKY-NYX-512x512x512` | — |
| ERA5 | `era5` | `Data/ERA5/2024` | `--max_channels 6` (全部用-1) |
| isot1024 | `isot1024` | `Data/isot1024/isotropic1024-coarse-velocity.h5` | — |
| Tomo | `tomo` | `Data/tomo/tomo_00001.h5` | `--tomo_group_frames 3` |
| Lysozyme | `lysozyme` | `Data/lysozyme/.../lysozyme_chip3/` | — |

### 2.1 提交示例

```bash
# Hurricane — 图像模型 + DCMVC
cat > run_hurricane.sh << 'EOF'
#!/bin/bash
#SBATCH --job-name=hurricane_test
#SBATCH --partition=gpu_5090
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/hurricane_%j.log
#SBATCH --error=logs/hurricane_%j.log
eval "$(/data/home/scxj523/run/miniconda3/bin/conda shell.bash hook)"
conda activate /data/run01/scxj523/zsh/envs/zsh
cd /data/run01/scxj523/zsh/project/AIForCompression

python scripts/run_dataset_compression.py \
  --dataset hurricane \
  --data_root /data/run01/scxj523/zsh/project/Data/hurricane/100x500x500 \
  --output_dir unified_results/hurricane \
  --models DCAE LIC-HPCM DCMVC \
  --max_samples 10
EOF
sbatch run_hurricane.sh
```

**注意**：DCVC-RT 和 DCAE 有 RansEncoder 冲突，不能同进程运行，需分开提交：

```bash
# 分开提交 DCVC-RT
python scripts/run_dataset_compression.py ... --models DCVC-RT --output_dir unified_results/<ds>_dcvc
```

## 3. 模型列表

所有模型通过 `--models` 参数指定，可多选。checkpoints 在 `checkpoints/` 下。

### 图像模型（Intra 编码）

| 模型 | `--models` | 论文/来源 | 质量等级 | 归一化 |
|------|-----------|----------|---------|--------|
| **DCAE** | `DCAE` | auto-regressive | 6 ckpt（lambda=0.0018~0.05） | pipeline |
| **LIC-HPCM** | `LIC-HPCM` | base + large | 各 6 ckpt（0.0018~0.0483） | pipeline |
| **LIC_TCM** | `LIC_TCM` | TCM | 7 ckpt | pipeline |
| **WeConvene** | `WeConvene` | ECCV 2024 | 6 ckpt | pipeline |
| **RwkvCompress** | `RwkvCompress` | LALIC | 6 ckpt | pipeline |
| **CRA5** | `CRA5` | VAEformer, 268ch | 1 质量（Q268） | 自身 (x-μ)/σ |

### 视频模型

| 模型 | `--models` | 模式 | QP/q_index | 归一化 |
|------|-----------|------|-----------|--------|
| **DCVC-RT** | `DCVC-RT` | Intra（I 帧） | QP=0,21,42,63 | pipeline + YCbCr |
| **DCVC-RT** | P-frame | P 帧（仅 UVG） | QP=0,21,42,63 | RGB→YCbCr |
| **DCMVC** | `DCMVC` | Intra（I 帧） | q=0,1,2,3 | pipeline（RGB） |
| **DCMVC** | P-frame | P 帧（仅 UVG） | q=0,1,2,3 | RGB |

### 序列模型

| 模型 | `--models` | 最少帧数 | 控制 | 归一化 |
|------|-----------|---------|------|--------|
| **CAESAR-V** | `caesar_v` | 8 | eb（误差界） | **无** |
| **CAESAR-D** | `caesar_d` | 16 | eb（误差界） | **无** |

### 模型冲突

- **DCVC-RT 不能和 DCAE 同进程**（RansEncoder 冲突）——分开提交
- 其他图像模型可混合使用
- DCVC P-frame 和 DCMVC P-frame 用独立脚本 `scripts/test_uvg_pframe.py` / `scripts/run_dcmvc_pframe.py`
- CRA5 默认只跑 ERA5 268 通道样本；非 ERA5 诊断实验必须显式传 `--allow_cra5_adapted`，且不要作为主公平 benchmark。

## 4. CAESAR 测试

CAESAR 需要序列数据，通过 `--caesar_eb` 扫误差界：

```bash
python scripts/run_dataset_compression.py \
  --dataset nyx \
  --data_root .../SDRBENCH-EXASKY-NYX-512x512x512 \
  --output_dir unified_results/nyx_caesar \
  --models caesar_v caesar_d \
  --max_samples 16 \
  --caesar_eb 1e-4 5e-4 1e-3 5e-3 1e-2 5e-2 1e-1 5e-1
```

CAESAR-D 需要 ≥16 帧，不够的用 `load_sequence` 重复末尾帧补齐。

## 5. DCVC-RT P-frame（仅 UVG）

```bash
python scripts/test_uvg_pframe.py \
  --model dcvc \
  --data_dir /data/run01/scxj523/zsh/project/Data/UVG_png/Twilight \
  --output_dir unified_results/uvg_dcvc_pframe \
  --max_frames 30
```

## 6. 画图

```bash
# 图像+视频模型
python utils/plot_all_datasets.py --mode image_video

# CAESAR 单独
python utils/plot_all_datasets.py --mode caesar

# 全模型合并
python utils/plot_all_datasets.py --mode all
```

输出到 `unified_results/<ds>_overview/`（image_video）、`unified_results/<ds>_overview_caesar/`（CAESAR）、`unified_results/<ds>_overview_all/`（合并）。

## 7. 归一化

### 通用图像/视频模型
- **uint8 数据**（Kodak, UVG）：`/255 → [0,1]`
- **float32 科学数据**：per_channel_minmax → [0,1]（每通道独立）

### ERA5 特殊处理
使用 `AIForCompression/normalization/` 下的每日 mean/std 做 **两级归一化**：
1. `(x-μ)/σ` → z-score
2. 分 3 通道组 → per_channel_minmax → [0,1]

ERA5 适配器自动加载当天归一化参数。

### CAESAR
Pipeline 不做 image-model 的 per-channel minmax；原始 float32 会进入 CAESAR view。CAESAR 上游 `ScientificDataset` 内部使用 `inst_norm=True, norm_type="mean_range"`，解压后通过 `recons_data()` 回到原始数据空间再算指标。

### CRA5
默认只接受 ERA5 原生 268 通道 canonical sample。`--allow_cra5_adapted` 会把非 ERA5 输入 resize 并复制到 268 通道，这是 ablation/diagnostic 路径，不应和主结果直接比较。

### 反归一化
- uint8：`round(clip(x̂,0,1)×255)`
- minmax：`x̂×scale + cmin`
- z-score：`x̂×z_scale + z_min` → `z×zscore_std + zscore_mean`

## 8. PSNR 计算

所有模型统一：`10·log₁₀(data_range²/mse)`，在**原始数据空间**计算。

`data_range = original.max() - original.min()` —— 每个 sample 自己的范围。

## 9. 添加新数据集

1. 在 `compression_pipeline/adapters/` 创建适配器，继承 `CanonicalSample` 模式
2. 适配器必须实现 `iter_samples()` → `Iterator[CanonicalSample]`
3. CAESAR 支持需额外实现 `load_sequence()` → `[V,T,H,W] + timestamps`
4. 在 `scripts/run_dataset_compression.py` 注册：
   - 第 14-23 行：import adapter
   - 第 33 行：`--dataset` choices
   - 第 51-86 行：`iter_dataset_samples` 分支
   - 第 108-130 行：CAESAR `load_sequence` 分支
5. 在 `utils/plot_all_datasets.py` 的 `DATASETS` 字典添加配置

## 10. 目录结构

```
unified_results/
├── kodak/summary.json              # 图像模型
├── kodak_dcvc/summary.json         # DCVC-RT
├── kodak_caesar/summary.json       # CAESAR
├── kodak_overview/                 # 图像+视频+CAESAR 图表
├── kodak_overview_caesar/          # CAESAR 单独图表
├── uvg/
├── hurricane/
├── ...
```

## 11. 快速检查

```bash
# 查看某个结果
python3 -c "
import json
d=json.load(open('unified_results/hurricane/summary.json'))
ok=[x for x in d if 'error' not in x]
for m in set(x['model_id'] for x in ok):
    items=[x for x in ok if x['model_id']==m]
    bpp=sum(i['bpp'] for i in items)/len(items)
    psnr=sum(i['psnr'] for i in items if i['psnr']!=float('inf'))/len(items)
    print(f'{m}: bpp={bpp:.4f} psnr={psnr:.1f}dB [{len(items)}]')
"
```
