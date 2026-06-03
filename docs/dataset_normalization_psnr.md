# 九大数据集：归一化、PSNR 计算与数据格式总览

---

## 一、数据集分类与全称

| 缩写 | 全称 | 类别 | 说明 |
|------|------|------|------|
| **Kodak** | Kodak Lossless True Color Image Suite | 通用图像 | 24 张 768×512 自然照片，压缩领域标准测试集 |
| **UVG** | Ultra Video Group | 通用视频 | 4K/1080p YUV420 自然视频，7 个序列 |
| **Tomo** | Tomography Projection (ALS 8.3.2) | 科学图像 | 同步辐射 X 射线断层投影，1501 角×1792×2048 uint16 |
| **S2C** | Sentinel-2C MSI L2A | 科学图像 | 欧空局 Sentinel-2C 卫星多光谱影像，10980×10980 |
| **Lysozyme** | Lysozyme Serial Crystallography (CHESS) | 科学图像 | 溶菌酶串行晶体学衍射，12960帧×1065×1030 uint32 |
| **ERA5** | ECMWF Reanalysis v5 | 科学数据 | 全球气象再分析，268 通道（7 变量×37 气压层+9 地面变量） |
| **Hurricane** | Hurricane Simulation (CM1) | 科学数据 | 飓风数值模拟输出，100 时间步×500×500，21 个物理变量 |
| **NYX** | NYX Cosmological Simulation | 科学数据 | 宇宙学 N 体+流体模拟，512³ 网格，6 个物理场 |
| **isot1024** | Isotropic Turbulence 1024 (JHU) | 科学数据 | 约翰霍普金斯大学各向同性湍流 DNS，10 时间步×256³，3 速度分量 (u,v,w) |

---

## 二、归一化方法

两种归一化逻辑（`compression_pipeline/views.py:93-110`）：

| 条件 | 方法 | 正变换 | 逆变换 |
|------|------|--------|--------|
| `kind="image"` + `uint8` | `/255` | `x / 255 → [0,1]` | `round(clip(x̂,0,1)×255) → uint8` |
| 其他 | 逐通道 minmax | `(x - cmin) / (cmax - cmin) → [0,1]` | `x̂ × scale + cmin → float32` |

### 逐数据集归一化详表

| # | 数据集 | 原始格式 | 样本形状 | `kind` | `dtype` | 归一化 | 通道含义 | 单样本 data_range |
|---|--------|---------|---------|--------|---------|--------|---------|-------------------|
| 1 | **Kodak** | PNG 目录 | [3,768,512] | `image` | uint8 | `/255` | R,G,B | ≈255 |
| 2 | **UVG** | YUV420→RGB | [3,2160,3840] | `image` | uint8 | `/255` | R,G,B | ≈255 |
| 3 | **Tomo** | HDF5 (.h5) | [3,1792,2048] | `scientific_field` | float32 | minmax | 投影角₀,₁,₂ (3角度堆叠) | ≈uint16范围 |
| 4 | **S2C** | SAFE.zip JP2→tile | [3,1024,1024] | `s2c` | float32 | minmax | B02,B02,B02 (三通道相同) | ≈0~45554/tile |
| 5 | **Lysozyme** | HDF5 (.h5) | [3,1065,1030] | `lysozyme` | float32 | minmax | I,I,I (单通道复制) | ≈0~4294967295 |
| 6 | **ERA5** | NetCDF (.nc) | [≤3,H,W] | `era5` | float32 | minmax | 气象变量（按 3 通道分组） | ≈变量值范围 |
| 7 | **Hurricane** | .bin.f32 | [3,500,500] | `hurricane` | float32 | minmax | P_z₀,P_z₁,P_z₂ | ≈0~0.002 |
| 8 | **NYX** | .f32 | [3,512,512] | `nyx` | float32 | minmax | ρ_z₀,ρ_z₁,ρ_z₂ | ≈0~115860/slice |
| 9 | **isot1024** | HDF5 (.h5) | [3,256,256] | `isotropic1024` | float32 | minmax | u,v,w (速度分量) | ≈5 |

---

## 三、PSNR 计算

```
mse = mean((原始 - 重建)²)                    ← 反归一化后，**原始数据空间**
data_range = 原始.max() - 原始.min()           ← 每个 sample 实际的 max-min
psnr = 10 · log₁₀(data_range² / mse)
```

### 两条计算路径

**路径 A：uint8 `/255`（Kodak, UVG）**
```
原始 uint8 [0,255]  →  /255 → [0,1]  →  模型  →  round(clip(x̂×255))  →  uint8
PSNR: data_range≈255, mse 在整数空间

例: Kodak DCAE mse=14.5 → 10·log₁₀(255²/14.5) = 36.5 dB
```

**路径 B：per_channel_minmax（其余 7 个）**
```
原始 float32  →  (x-cmin)/scale → [0,1]  →  保存{cmin,scale}  →  模型  →  x̂·scale+cmin  →  float32
PSNR: data_range = 重建后 array.max()-min(), mse 在原始 float32 空间

例: NYX data_range≈50000, mse=56.3 → 10·log₁₀(50000²/56.3) = 55.2 dB
```

### 边缘处理

| 情况 | 处理 |
|------|------|
| MSE < 1e-30 | PSNR = ∞ |
| MSE < 1e-12 | 绘图时过滤（无数据区/均匀 tile） |
| data_range < 1e-8 | 设为 1.0 防止除零 |

---

## 四、数据集详细信息

### 通用图像/视频

#### 1. Kodak（Kodak Lossless True Color Image Suite）
| 属性 | 值 |
|------|-----|
| 路径 | `Data/Kodac/` |
| 格式 | PNG 文件 |
| 数量 | 24 张 |
| 分辨率 | 768×512 |
| 通道 | 3 (RGB) |
| 位深度 | 8-bit |
| 适配器 | `compression_pipeline/adapters/kodak.py` → `KodakAdapter` |
| 典型用途 | 图像压缩标准评测 |

#### 2. UVG（Ultra Video Group）
| 属性 | 值 |
|------|-----|
| 路径 | `Data/UVG/` |
| 格式 | YUV420 原始视频 (`*.yuv`) |
| 数量 | 1 个序列（Twilight），600 帧 |
| 分辨率 | 3840×2160（测试用 1920×1080） |
| 帧率 | 50 fps |
| 通道 | 3 (YUV→RGB 转换后) |
| 适配器 | `compression_pipeline/adapters/uvg.py` → `UVGAdapter` |
| 典型用途 | 视频压缩标准评测 |

---

### 科学图像


#### 3. Tomo（Tomography Projection, ALS 8.3.2）
| 属性 | 值 |
|------|-----|
| 路径 | `Data/tomo/tomo_00001.h5` |
| 格式 | HDF5 (`exchange/data`, `exchange/theta`) |
| 维度 | [1501, 1792, 2048] uint16 |
| 内容 | 同步辐射 X 射线断层扫描投影，1501 个投影角度 |
| 处理 | 3 个相邻角度堆叠为 [3,1792,2048] 伪 RGB |
| 适配器 | `compression_pipeline/adapters/tomo_h5.py` → `TomoH5Adapter` |
| 典型用途 | 科学仪器数据压缩 |

#### 4. S2C（Sentinel-2C MSI L2A）
| 属性 | 值 |
|------|-----|
| 路径 | `Data/S2C_MSIL2A_*.SAFE.zip` |
| 格式 | SAFE 格式 ZIP 包，内含 JP2 压缩的多光谱影像 |
| 覆盖 | 10980×10980 像素 |
| 波段数 | 13 个光谱波段 (B01-B12, B8A)，分辨率 10/20/60m |
| 测试波段 | B02（蓝光，10m 分辨率） |
| 采集时间 | 2026-05-09 |
| 地理范围 | Tile 51RUQ |
| 处理 | 中心裁切 10240² → 切为 100 个 1024² tile，单通道复制为 3 通道 |
| 适配器 | `compression_pipeline/adapters/s2c.py` → `S2CAdapter` |
| 典型用途 | 遥感影像压缩 |


---

#### 5. Lysozyme（Lysozyme Serial Crystallography, CHESS）
| 属性 | 值 |
|------|-----|
| 路径 | `Data/lysozyme/nfs/chess/raw/2018-1/g3/finke-707-2/20180305/lysozyme_chip3/` |
| 格式 | HDF5 (`.h5`)，每文件一帧，LZ4 压缩 |
| 维度 | 12960 帧 × [1, 1065, 1030] uint32 |
| 内容 | 溶菌酶蛋白质晶体的串行飞秒晶体学（SFX）衍射图样 |
| 来源 | CHESS 同步辐射光源，Finke 实验线站 |
| 处理 | 单帧读取 → 单通道复制为 3 通道 → [3,1065,1030] |
| 依赖 | `hdf5plugin`（LZ4 滤镜） |
| 适配器 | `compression_pipeline/adapters/lysozyme.py` → `LysozymeAdapter` |
| 典型用途 | 串行晶体学衍射数据压缩 |


---

### 科学数据（3D/4D 数值场）

#### 6. ERA5（ECMWF Reanalysis v5）
| 属性 | 值 |
|------|-----|
| 路径 | `Data/ERA5/2024/` |
| 格式 | NetCDF (`{timestamp}_pressure.nc` + `{timestamp}_single.nc`) |
| 通道数 | 268（7 变量×37 气压层 + 9 地面变量） |
| 变量 | 位势高度(z)、温度(t)、比湿(q)、U/V 风分量、垂直速度(w)、相对湿度(r) |
| 地面变量 | 2m 温度、10m U/V 风、海平面气压、总降水等 |
| 处理 | 按 3 通道分组（最后一组不足 3 则补足），每组独立 minmax |
| 适配器 | `compression_pipeline/adapters/era5.py` → `ERA5Adapter` |
| 典型用途 | 气象科学数据压缩 |

#### 7. Hurricane（Hurricane Simulation, CM1 Model）
| 属性 | 值 |
|------|-----|
| 路径 | `Data/hurricane/100x500x500/` |
| 格式 | 原始 float32 二进制 (`.bin.f32`) |
| 维度 | 21 变量 × [100 时间步, 500, 500] |
| 测试变量 | P（气压） |
| 变量列表 | P, TC, U, V, W, QVAPOR, QCLOUD, QRAIN, QICE, QSNOW, QGRAUP, PRECIP, CLOUD 等 |
| 处理 | 每 3 时间步堆叠为 [3,500,500] |
| 适配器 | `compression_pipeline/adapters/hurricane.py` → `HurricaneAdapter` |
| 典型用途 | 数值天气预报输出压缩 |

#### 8. NYX（NYX Cosmological Simulation）
| 属性 | 值 |
|------|-----|
| 路径 | `Data/nyx/SDRBENCH-EXASKY-NYX-512x512x512/` |
| 格式 | 原始 float32 二进制 (`.f32`) |
| 维度 | 6 变量 × [512, 512, 512] |
| 测试变量 | baryon_density（重子密度） |
| 变量列表 | baryon_density, dark_matter_density, temperature, velocity_x/y/z |
| 物理含义 | 宇宙大尺度结构形成模拟：暗物质晕、星系际介质、激波加热 |
| 处理 | Z 轴每 3 切片堆叠为 [3,512,512] |
| 适配器 | `compression_pipeline/adapters/nyx.py` → `NYXAdapter` |
| 典型用途 | 宇宙学模拟数据压缩 |

#### 9. isot1024（Isotropic Turbulence 1024, JHU）
| 属性 | 值 |
|------|-----|
| 路径 | `Data/isot1024/isotropic1024-coarse-velocity.h5` |
| 格式 | HDF5 |
| 维度 | 10 时间步 × [256, 256, 256, 3] float32 |
| 变量 | Velocity (u,v,w) |
| 物理含义 | 各向同性湍流直接数值模拟（DNS），Re≈1000 |
| 处理 | 每 Z 切片直接为 [u,v,w] 3 通道为 [3,256,256] |
| 适配器 | `compression_pipeline/adapters/isotropic1024.py` → `Isotropic1024Adapter` |
| 典型用途 | 湍流科学数据压缩 |

---

## 五、DCVC-RT 色彩空间转换

DCVC-RT 期望 BT.709 YCbCr 输入，`DCVCRTCodec` 在压缩前后做 RGB↔YCbCr：

```
RGB [0,1]³  →  YCbCr BT.709  →  model  →  YCbCr  →  RGB [0,1]³

Y  = 0.2126R + 0.7152G + 0.0722B
Cb = (B-Y)/(2·(1-0.0722)) + 0.5
Cr = (R-Y)/(2·(1-0.2126)) + 0.5

R = Y + 2·(1-0.2126)·(Cr-0.5)
G = Y - 0.1873(Cb-0.5) - 0.4681(Cr-0.5)
B = Y + 2·(1-0.0722)·(Cb-0.5)
```

对非图像 3 通道数据，等价于线性去相关+方差集中——使 latent 分布回到模型训练分布附近。

---

## 六、CAESAR 序列压缩

CAESAR 是专门针对科学数据的序列压缩模型，通过误差界（error bound, eb）控制有损压缩精度。

### 模型变体

| 变体 | 最少连续帧 | 说明 |
|------|----------|------|
| **CAESAR-V** | 8 帧 | 单变量序列压缩 |
| **CAESAR-D** | 16 帧 | 多变量/更高压缩率 |

### 数据处理

CAESAR 不走 image-model 的 per_channel_minmax 归一化。Pipeline 传入原始 float32 科学数据，但 CAESAR 上游 `ScientificDataset` 会在模型内部使用 `inst_norm=True, norm_type="mean_range"` 做 instance normalization，并在评估前通过 `recons_data()` 反变换回原始数据空间：

```
适配器.load_sequence()
  → [V, T, H, W] float32
  → ScientificDataset mean_range instance norm
  → CAESAR 压缩/解压
  → recons_data() 回到原始数据空间
```

各数据集 `load_sequence` 的序列构造方式：

| 数据集 | 序列格式 | V | T | 备注 |
|--------|---------|---|---|------|
| Kodak | 24 张图像序列 | 3 (RGB) | 24 | 需 --resolution 统一尺寸 |
| UVG | 视频帧序列 | 3 (RGB) | 30 | YUV420→RGB float32 |
| Hurricane | 气压场时间序列 | 1 | 100 | P 通道 [T,H,W] |
| NYX | Z 切片序列 | 1 | 512 | 重子密度 [Z,H,W] |
| isot1024 | 时间步序列 | 1 | 10 | 中 Z 切片 [T,H,W] |
| ERA5 | 气象时间序列 | 268 | N | OOM（268 通道内存不足） |

### PSNR 计算

当前 pipeline 的 CAESAR 指标和其他模型一致，在重建后的原始数据空间计算。`data_range` 使用当前 sample 的 `original.max() - original.min()`，常量样本回退为 `1.0`。

### 使用方式

```bash
python scripts/run_dataset_compression.py \
  --dataset nyx \
  --data_root ... \
  --output_dir unified_results/nyx_caesar \
  --models caesar_v caesar_d \
  --max_samples 16
```
