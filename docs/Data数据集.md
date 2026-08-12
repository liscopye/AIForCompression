# `/workspace/Data` 数据资产盘点

盘点时间：2026-08-12。统计对象为当前机器上的 `/workspace/Data`，总磁盘占用约 **1.5 TB**。

## 一、统计口径

“下载量”按当前仍保留的原始下载载荷统计：有原始压缩包时采用压缩包字节数；没有压缩包时采用原始数据目录或原始单文件的现存字节数。该数字可以从磁盘复核，但不包含历史下载重试、失败请求和已经删除的文件，因此不是网络服务端累计流量。

“本地占用”使用 `du`，包括解压副本、格式转换、训练/验证切分、PNG、NPY/NPZ shard 和其他派生数据。同一份科学数值可能保存多次，所以本地总占用明显大于下载量。

当前可识别的原始下载载荷合计约 **522.63 GB（486.74 GiB）**。

## 二、下载量总表

| 数据集 | 原始下载载荷 | 当前本地相关占用 | 下载/来源文件 |
|---|---:|---:|---|
| ERA5 | 345.258 GB | 约 1.1 TB | 多批 NetCDF/GRIB，分布在 `ERA5/finetune`、`test` 和 hourly 目录 |
| Lysozyme/CHESS | 151.998 GB | 原始 142 GiB + 派生 138 GiB | `nfs/chess/raw/.../lysozyme_chip3` 下 60,604 个 HDF5 |
| Tomo | 11.032 GB | 约 11 GB | `tomo_00001.h5` |
| E3SM | 3.686 GB | 约 17 GB | `E3SM/day_5vars/` 下 5 个 NetCDF |
| UVG Twilight | 3.564 GB | 压缩包 3.4 GiB + 展开/派生约 7.4 GiB | `Twilight_3840x2160_50fps_420_8bit_YUV_RAW.7z` |
| NYX | 2.884 GB | 压缩包 2.7 GiB + 解压 3.1 GiB | `SDRBENCH-EXASKY-NYX-512x512x512.tar.gz` |
| Turb_Rot | 2.147 GB | 约 2.1 GB | `Turb_Rot_testset.npz` |
| Hurricane Isabel | 1.254 GB | 压缩包 1.2 GiB + 解压 1.9 GiB | `SDRBENCH-Hurricane-ISABEL-100x500x500.tar.gz` |
| Sentinel-2 | 0.793 GB | 压缩包 756 MB + SAFE 757 MB + tiles 10 MB | `.SAFE.zip` |
| Kodak | 0.015 GB | 约 15 MB | `Kodac/kodim01.png` 至 `kodim24.png` |

表中的 GB 为十进制 `10^9` bytes；`du` 输出通常接近二进制 GiB，两者显示值会略有差异。

## 三、逐数据集说明

### 3.1 ERA5

根目录：`/workspace/Data/ERA5`，当前约 **1.1 TB**，是占用最大的单个数据集。

原始/下载相关部分：

| 子目录 | 内容 | 文件与日期 | 逻辑字节 |
|---|---|---|---:|
| `finetune/` | daily pressure + single-level NetCDF | 61 天，2024-08-01 至 2024-09-30，122 文件 | 30.666 GB |
| `test/` | daily pressure + single-level NetCDF | 16 天，2024-06-01 至 2024-06-16，32 文件 | 7.938 GB |
| `hourly_20240301_90d/` | 早期 hourly 下载样本 | 2024-03-01 一天，2 个 NetCDF | 0.511 GB |
| `hourly_center512_20240301_90d/` | 90 天 hourly 中心区域原始请求及转换文件 | 2024-03-01 至 2024-05-29；84 GRIB、97 NetCDF、锁文件等 | 306.143 GB |

以上现存原始获取目录合计 **345.258 GB**。其中 hourly 目录混合保留 GRIB、NetCDF 和请求状态文件，因此它表示“原始获取阶段的现存磁盘载荷”，不等同于精确网络传输量。

主要派生数据：

| 子目录/文件 | 作用 | 大小 |
|---|---|---:|
| `hourly_center512_shards_20240301_90d/` | 90 个 daily NPY shard，每个约 6.77 GB，供最佳 CAESAR 微调 | 约 568 GiB / 609.37 GB logical |
| `finetune_processed/` | 按 channel 整理的 NPZ | 约 88 GiB |
| `finetune_processed_time_split/` | `era5_train.npy` + `era5_val.npy` | 约 64 GiB |
| `finetune_processed_time_split_t45_v16/` | 另一组时间切分 | 约 64 GiB |
| validation probe 与 `grib_conversion_probe/` | 固定验证、格式一致性检查 | 约 8–9 GiB |
| `hourly_shard_smoke/` | 单日 shard smoke 副本 | 约 6.4 GiB |

正式 objective-v1 使用固定 268 通道、16 时间步、中心 `240x240` 视图；最佳 CAESAR 微调使用 90 天 hourly shard。不要在不确认训练是否还要复现时删除 hourly shard。

### 3.2 Lysozyme / CHESS

原始数据：

```text
/workspace/Data/nfs/chess/raw/2018-1/g3/finke-707-2/20180305/lysozyme_chip3
```

- 60,604 个 `.h5` 文件。
- 总计 **151.998 GB（141.559 GiB）**。
- 单文件从约 3 KB 到 25.5 MB，平均约 2.51 MB。

派生目录 `/workspace/Data/lysozyme_processed` 约 **138 GiB**：

| 文件 | 内容 | 大小 |
|---|---|---:|
| `mmap/` | train/val/test NPY mmap | 约 63 GiB |
| `lysozyme_train_nf16.npz` | `[1,800,16,1024,1024]` 对应训练块 | 约 46 GiB |
| `lysozyme_val_nf16.npz` | validation | 约 5.1 GiB |
| `lysozyme_test_nf16.npz` | 16-frame 测试 | 约 13 GiB |
| `lysozyme_test_nf8.npz` | 8-frame 测试 | 约 13 GiB |

元数据记录完整数组为 `[1,1000,16,1024,1024] float32`，按时间切成 800 个训练块和 200 个测试块。原始 HDF5 用于重新生成；正式评测通常只需对应 test mmap/NPY。

### 3.3 E3SM

`E3SM/day_5vars/` 保存 5 个原始 NetCDF，合计 **3.686 GB**：`huss`、`pr`、`tas`、`tasmax`、`tasmin`，每个来自一个十年 historical 时间段。

`E3SM/caesar_processed/` 约 **13 GB**，包含多组 CAESAR NPZ：first16/160/800、不同变量组合以及 paper-like `240x240` 版本。当前推荐输入是：

```text
E3SM/caesar_processed/e3sm_5vars_paperlike240_first160_caesar.npz
```

其 canonical 布局为 `[V,S,T,H,W]`；正式常用前三个物理变量、section 0。

### 3.4 Tomography

文件 `tomo_00001.h5` 为 **11.032 GB**。它是原始投影数据，不是最终重建 volume：

- `exchange/data`: `[1501,1792,2048] uint16`
- dark/white reference：各 `[1,1792,2048] uint16`
- `exchange/theta`: 1501 个角度

当前 `TomoH5Adapter` 需要按协议确认读取的是投影还是已经重建的数据语义。仓库已删除那个写死旧路径且依赖缺失 `tomopy` 的一次性重建脚本，因此换机器时若需重建，应单独安装 TomoPy 并建立可复用处理流程。

### 3.5 Turb_Rot

`Turb_Rot_testset.npz` 为 **2,147,484,300 bytes（约 2.000 GiB）**。

- `data`: `[1,16,256,256,256] float64`
- 当前含 1 个实际变量、16 个 section、256 个时间步。
- `variable_name` 元数据写有 `vx/vy/vz`，但与数据首维 `V=1` 不一致；当前 adapter 按本地实际布局使用 variable 0 的相邻 section。
- 图像模型使用 sections `0/1/2` 组成三通道；CAESAR 使用 `[3,T,256,256]`。

这是单个 NPZ，没有额外解压副本。

### 3.6 NYX

下载包：`SDRBENCH-EXASKY-NYX-512x512x512.tar.gz`，**2.884 GB**。解压目录约 **3.1 GiB**，包含 6 个等大的 `512^3 float32` volume，每个 536,870,912 bytes：

- baryon density
- dark matter density
- temperature
- velocity x/y/z

当前 adapter 默认以 `baryon_density.f32` 为主要测试 volume，Z slices 作为图像组或 CAESAR 伪时间。

### 3.7 Hurricane Isabel

下载包：`SDRBENCH-Hurricane-ISABEL-100x500x500.tar.gz`，**1.254 GB**。解压后约 **1.9 GiB**，包含 20 个 `100x500x500 float32` 文件，每个 100,000,000 bytes。

内容包括 pressure、temperature、三维速度、降水和多种水汽/云变量；部分变量同时保存 log10 版本。adapter 按 `[T,H,W]` 时序场读取。

### 3.8 Sentinel-2 / S2C

下载包：

```text
S2C_MSIL2A_20260509T022531_N0512_R046_T51RUQ_20260509T055911.SAFE.zip
```

- 下载包 **792,632,924 bytes（0.793 GB）**。
- 解压 SAFE 约 757 MB，含 68 个 JP2 及 XML/预览文件。
- `s2c_tci_tiles_512_n16/` 是额外生成的 16 张 `512x512 RGB` TCI PNG，约 10 MB。

正式 scientific S2C 使用 SAFE 中的 B02/B03/B04/B08 JP2 波段，而不是仅使用 TCI PNG。

### 3.9 UVG Twilight

下载包 `Twilight_3840x2160_50fps_420_8bit_YUV_RAW.7z` 为 **3.564 GB**。

本地表示：

| 路径 | 内容 | 大小 |
|---|---|---:|
| `UVG_Twilight/Twilight_3840x2160_50fps_8bit.yuv` | 4K YUV420，600 帧 | 7.465 GB |
| `UVG_Twilight_1080p/Twilight_1920x1080_50fps_8bit.yuv` | objective 使用的 1080p 前 30 帧 | 93.312 MB |
| `UVG_png/Twilight/` | 30 张 4K RGB PNG | PNG 总目录的一部分 |
| `UVG_png/Twilight_1080p/` | 30 张 1080p RGB PNG | PNG 总目录的一部分 |
| `UVG_png/` 合计 | 60 张 PNG | 约 366 MB |

正式 objective-v1 使用连续 30 帧 1080p 序列；`run_objective_video.py` 也可从 canonical tensor 自动导出所需 PNG。

### 3.10 Kodak

`Kodac/` 包含标准 `kodim01.png` 至 `kodim24.png`，共 24 张 RGB 图像，总计 **15,394,305 bytes（约 15 MB）**。目录名保留为现有的 `Kodac` 拼写，脚本路径不要擅自改成 `Kodak`。

### 3.11 当前未下载的数据集

框架 adapter 还支持以下数据，但 `/workspace/Data` 当前没有对应正式原始目录：

- Shanghai synchrotron X-ray TIFF
- 独立的 Isotropic1024/JHTDB HDF5 数据集

`download_jhtdb_local.py` 是 JHTDB 获取工具，但本次盘点未发现下载完成的数据目录。因此“代码支持”不等于“本机已有数据”。

## 四、空间重复与清理建议

以下项目存在明确重复，但是否删除取决于是否还要重新预处理：

1. Sentinel-2、NYX、Hurricane 和 UVG 同时保留压缩包与解压目录。确认解压文件完整且无需再次分发时，可删除压缩包，约释放 **8.49 GB**。
2. UVG 正式评测只使用 1080p 30 帧；若不再需要 4K 实验，可另行清理 4K YUV 和 4K PNG，但应先核对结果复现需求。
3. ERA5 同时保留原始 GRIB/NetCDF、daily shard、channel NPZ 和两套 time split，是最大的重复来源。当前最佳微调依赖 hourly shard，不建议直接按大小删除。
4. Lysozyme 同时保留 152 GB 原始 HDF5、mmap 和多个 NPZ。只做正式测试时不需要全部训练/验证副本，但删除前应确定不再训练且 test mmap 可独立读取。
5. `s2c_tci_tiles_512_n16` 是预览/图像实验派生物，不能替代正式四波段 SAFE 输入。

本文件只记录和解释，没有删除 `/workspace/Data` 中任何数据。
