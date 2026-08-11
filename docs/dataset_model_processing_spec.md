# 数据集与模型处理规范

本文档定义 `scripts/run_dataset_compression.py` 当前使用的 benchmark 数据处理规范。后续评测、画图、结果解释都应以这里的口径为准，尤其是科学数据如何组织三通道、如何归一化、CAESAR 输入 shape、bpp/PSNR/LPIPS/吞吐量如何计算。

## 一、核心约定

所有图像/视频类压缩模型都通过 `CanonicalSample` 接收 `[C,H,W]` 数据。共享 runner 会把通道按 3 个一组切分，然后以 `[1,3,H,W]` 输入模型。

科学数据给图像/视频压缩模型时，原则是：**优先使用物理上真实不同的 3 个 channel**。不要在有自然替代维度时把单通道重复三次。自然替代维度包括：

- 不同物理变量
- 不同速度分量
- 相邻时间步
- 相邻体数据切片
- 相邻 section
- 不同光谱 band

只有当数据本身确实是单通道图像，并且没有合理的相邻维度可作为同一 sample 的 channel 时，才允许复制成 3 通道。

CAESAR 不走图像模型的三通道分组路径。adapter 提供 `[V,T,H,W]` 序列，`build_caesar_view()` 再转换成 `[V,1,T,H,W]`，其中 `S=1`。`caesar_v` 使用连续 8 帧，`caesar_d` 使用连续 16 帧。

## 二、归一化规范

### 2.1 图像/视频压缩模型

图像/视频压缩模型的归一化在 `compression_pipeline/views.py` 中完成。

对 `kind="image"` 且 `dtype=uint8` 的自然图像/视频：

```text
uint8 [0,255] -> x / 255.0 -> model -> round(clip(x_hat,0,1)*255) -> uint8
```

对没有 z-score metadata 的 float 科学数据：

```text
per sample, per 3-channel group, per channel:
  min_c = min(x_c)
  scale_c = max(max_c - min_c, 1e-8)
  y_c = (x_c - min_c) / scale_c
  x_hat_c = y_hat_c * scale_c + min_c
```

也就是说，这是 **per-channel minmax**，不是全局 minmax。`min` 和 `scale` 都是在当前 sample/tile/frame group 内、每个 channel 独立计算。

归一化参数会计入 `scientific_bpp_with_side_info`：

- minmax：`actual_channels * 2 * 4` bytes
- z-score + minmax：`actual_channels * 4 * 4` bytes

对 ERA5，adapter 会提供 daily z-score 的 mean/std metadata。因此图像 runner 实际执行两级归一化：

```text
per channel:
  z = (x - daily_mean_c) / max(daily_std_c, 1e-8)
  y = (z - z_min_c) / max(z_max_c - z_min_c, 1e-8)
```

重建后会反变换回原始物理单位再计算指标。

### 2.2 CAESAR

CAESAR 接收 adapter 给出的原始 float 序列。pipeline 不对 CAESAR 输入做图像模型那套 per-channel minmax。

CAESAR 内部使用：

```text
ScientificDataset(inst_norm=True, norm_type="mean_range")
```

其 instance normalization 是：

```text
offset = mean(data)
scale = max(data) - min(data)
normalized = (data - offset) / scale
```

CAESAR 解压后通过 `dataset.recons_data()` 回到原始数据空间。CAESAR 的 bpp 为：

```text
compressed_bits / original.size
```

当前 pipeline 的 CAESAR 结果没有额外计入 normalization side info。

## 三、padding、crop 与 channel grouping

图像/视频模型在归一化之后再按各自模型约束 padding：

| 模型 | codec 输入 | pad divisor | 说明 |
|---|---:|---:|---|
| DCAE | `[1,3,H,W]` | 128 | wrapper 按模型约束 pad/crop |
| LIC-HPCM | `[1,3,H,W]` | 256 | wrapper 按模型约束 pad/crop |
| DCVC-RT intra | `[1,3,H,W]` | 64 | replicate pad；内部执行 BT.709 RGB/YCbCr 转换 |
| DCMVC intra | `[1,3,H,W]` | 64 | replicate pad；不做 YCbCr 转换 |

如果 sample 有 `C > 3`，图像 runner 会按 `0:3`、`3:6`、... 分组。最后一组如果不足 3 个 channel，只会为了满足模型输入 shape 重复最后一个真实 channel。解码后会丢弃 padding channel，指标和 bpp 只按原始真实 channel 计算。

`--resolution H W` 表示 adapter 侧中心裁切。`S2C` 使用 `--tile_size` 做 tile benchmark，不走普通 `--resolution` 逻辑。

## 四、指标计算规范

所有 distortion 指标都在反归一化后的原始数据空间计算。

```text
mse = mean((original - reconstructed)^2)
data_range = original.max() - original.min()
psnr = 10 * log10(data_range^2 / mse)
```

如果 `data_range < 1e-8`，fallback 为 `1.0`。

当前科学序列实验推荐使用 `average_frame_psnr`：

- 对 `[C,H,W]` 图像/视频模型 sample，等价于该 sample 自身 range 下的 PSNR。
- 对 CAESAR `[V,S,T,H,W]`，沿时间轴 `T` 逐帧计算 PSNR，每一帧使用自己的 range，然后平均。

RD 曲线的 x 轴推荐使用：

```text
scientific_bpp_with_side_info
```

fallback 顺序为：

```text
scientific_bpp_with_side_info -> scientific_bpp -> bpp
```

对图像/视频模型处理科学数据：

```text
image_bpp = bitstream_bits / (H * W)
scientific_bpp = bitstream_bits / (C * H * W)
scientific_bpp_with_side_info = (bitstream_bits + side_info_bits) / (C * H * W)
```

因此当 `C=3` 时，科学 bpp 大约等于 `image_bpp / 3`，再加上归一化 side info 的开销。这是三通道科学输入必须遵守的 bpp 口径。

LPIPS 越低越好。但 LPIPS 网络是在自然图像上训练的，对科学变量图只作为辅助参考，不应比 PSNR/MSE 更权威。

吞吐量计算为：

```text
encode_throughput_MBps = original_bytes / encode_time / 1e6
decode_throughput_MBps = original_bytes / decode_time / 1e6
```

CAESAR 原始记录里保留的是 legacy 字段：

```text
encode_throughput
decode_throughput
```

单位是 bytes/s。绘图脚本会换算成 MB/s。

## 五、逐数据集处理规范

### 5.1 Kodak

| 项目 | 规范 |
|---|---|
| Adapter | `KodakAdapter` |
| 原始数据 | RGB 图像文件 |
| 图像模型 sample | `[3,H,W]`，`uint8`，RGB |
| 图像模型归一化 | `/255` |
| channel 规则 | 真实 RGB channel |
| crop | `load_sequence()` 支持可选中心裁切；`iter_samples()` 当前保持原始尺寸 |
| CAESAR sequence | `[3,T,H,W]`，把多张图像当作伪时间序列 |

### 5.2 UVG

| 项目 | 规范 |
|---|---|
| Adapter | `UVGAdapter` |
| 原始数据 | YUV420 视频 |
| 图像模型 sample | `[3,2160,3840]`，`uint8`，转成 RGB |
| 图像模型归一化 | `/255` |
| channel 规则 | YUV420 转 RGB 后的真实 RGB channel |
| crop | CAESAR sequence 支持可选中心裁切；frame sample 使用配置的原始 H/W |
| CAESAR sequence | `[3,T,H,W]`，真实视频帧 |

### 5.3 Tomo

| 项目 | 规范 |
|---|---|
| Adapter | `TomoH5Adapter` |
| 原始数据 | HDF5 `data`，`[Z,H,W]`，float32 |
| 图像模型 sample | `[group_frames,H,W]`；公平评测建议 `group_frames=3` |
| 图像模型归一化 | per-sample、per-group、per-channel minmax |
| channel 规则 | 相邻 reconstructed slices；`group_frames=3` 时不是重复通道 |
| crop | 可选中心裁切 |
| CAESAR sequence | `[1,T,H,W]`，把 reconstructed slices 当作伪时间序列 |

### 5.4 S2C

| 项目 | 规范 |
|---|---|
| Adapter | `S2CAdapter` |
| 原始数据 | Sentinel-2 SAFE JP2 band，默认 `B02` at `10m` |
| 图像模型 sample | `[3,tile_size,tile_size]` 或完整 `[3,H,W]` |
| 图像模型归一化 | per-sample/tile、per-channel minmax |
| 当前 channel 规则 | 当前 adapter 只读取一个 band，因此会把同一 band 重复三次 |
| 推荐改进 | 为公平科学三通道输入，应扩展 adapter 读取三个真实 band，例如 `B02/B03/B04` 或其它指定多光谱 band |
| tiling | 可选 `--tile_size`；先裁成 tile size 的整数倍，再切 tile，并跳过近似常量 tile |
| CAESAR sequence | `[1,T,H,W]`，tile 作为伪时间序列，使用重复通道中的第一个 channel |

### 5.5 Lysozyme

| 项目 | 规范 |
|---|---|
| Adapter | `LysozymeAdapter` |
| 原始数据 | HDF5 diffraction frame `/entry/data/data`，`[1,H,W]`，uint32/float32 |
| 图像模型 sample | `[3,H,W]` |
| 图像模型归一化 | per-sample、per-channel minmax |
| channel 规则 | 单张 diffraction 图重复三次；这是允许的，因为该数据项本身是单通道 |
| crop | CAESAR sequence 支持可选中心裁切 |
| CAESAR sequence | `[1,T,H,W]`，多帧 diffraction 图作为时间序列 |

### 5.6 ERA5

| 项目 | 规范 |
|---|---|
| Adapter | `ERA5Adapter` |
| 原始数据 | paired pressure/single NetCDF files |
| 图像模型 sample | `[C,H,W]`，最多 268 channels；可用 `--max_channels` 限制 |
| 图像模型归一化 | daily z-score，然后在 z-score 空间做 per-sample、per-channel minmax |
| channel 规则 | 真实气象变量/气压层按 `0:3`、`3:6` 等分组；只有最后不足 3 个 channel 时才 padding |
| crop | 可选中心裁切 |
| CAESAR sequence | `[V,T,H,W]`；V 是选中的变量/层 channel，T 是时间戳 |

ERA5 的 z-score metadata 由 adapter 加入，并由 `build_image_groups()` 使用。所有指标仍然在原始气象物理单位中计算。

### 5.7 Hurricane

| 项目 | 规范 |
|---|---|
| Adapter | `HurricaneAdapter` |
| 原始数据 | 一个 `.bin.f32` 物理场，默认 `P`，shape `[T,H,W]` |
| 图像模型 sample | `[3,H,W]` |
| 图像模型归一化 | per-sample、per-channel minmax |
| channel 规则 | 同一个变量的三个相邻时间步 |
| crop | CAESAR sequence 支持可选中心裁切 |
| CAESAR sequence | `[1,T,H,W]`，一个物理变量随时间变化 |

### 5.8 NYX

| 项目 | 规范 |
|---|---|
| Adapter | `NYXAdapter` |
| 原始数据 | 一个 `.f32` 体数据，默认 `baryon_density`，shape `[512,512,512]` |
| 图像模型 sample | `[3,512,512]` |
| 图像模型归一化 | per-sample、per-channel minmax |
| channel 规则 | 三个相邻 Z slices，不重复通道 |
| crop | CAESAR sequence 支持可选中心裁切 |
| CAESAR sequence | `[1,T,H,W]`，Z slices 作为伪时间序列 |

### 5.9 Isotropic1024

| 项目 | 规范 |
|---|---|
| Adapter | `Isotropic1024Adapter` |
| 原始数据 | HDF5 velocity datasets，`[X,Y,Z,3]` |
| 图像模型 sample | `[3,H,W]` |
| 图像模型归一化 | per-sample、per-channel minmax |
| channel 规则 | 每个 Z slice 的真实速度分量 `(u,v,w)` |
| crop | CAESAR sequence 支持可选中心裁切 |
| CAESAR sequence | `[3,T,H,W]`，使用 mid-Z velocity slice 随时间变化 |

### 5.10 Turb_Rot NPZ

| 项目 | 规范 |
|---|---|
| Adapter | `TurbRotNPZAdapter` |
| 原始数据 | NPZ `data`，layout `[V,S,T,H,W]` |
| 当前本地数据 | `/workspace/Data/Turb_Rot_testset.npz`，shape `[1,16,256,256,256]` |
| 当前图像模型 sample | `[3,256,256]`，使用 `image_group_mode=sections`、`section_start=0` |
| channel 规则 | variable `0` 的 sections `0/1/2`；这是三个真实 section，不是重复通道 |
| 图像模型归一化 | per-sample、per-channel minmax |
| crop | 可选中心裁切 |
| 当前 CAESAR sequence | `[3,T,256,256]`，使用 sections `0/1/2`；进入 CAESAR 后为 `[3,1,T,256,256]` |
| bpp 口径 | 图像/视频模型必须除以 `3*H*W` |

当前推荐命令参数：

```bash
--dataset turb_rot_npz \
--data_root /workspace/Data/Turb_Rot_testset.npz \
--turb_rot_image_group_mode sections \
--turb_rot_section_start 0 \
--turb_rot_section_index 0
```

### 5.11 E3SM NPZ

| 项目 | 规范 |
|---|---|
| Adapter | `E3SMNPZAdapter` |
| 原始数据 | NPZ `data`，layout `[V,S,T,H,W]` |
| 当前本地数据 | `/workspace/Data/E3SM/caesar_processed/e3sm_5vars_paperlike240_first160_caesar.npz`，shape `[5,6,160,240,240]` |
| 当前图像模型 sample | `[3,240,240]`，使用 section `0` 的 variables `0/1/2` |
| channel 规则 | 三个真实物理变量；与 3-channel codec 公平比较时，不应默认使用全部 5 个变量，除非实验目标是评估 multi-group 行为 |
| 图像模型归一化 | per-sample、per-channel minmax |
| crop | 可选中心裁切 |
| 当前 CAESAR sequence | `[3,T,240,240]`，使用 section `0` 的 variables `0/1/2`；进入 CAESAR 后为 `[3,1,T,240,240]` |
| bpp 口径 | 图像/视频模型必须除以 `3*H*W` |

当前推荐命令参数：

```bash
--dataset e3sm_npz \
--data_root /workspace/Data/E3SM/caesar_processed/e3sm_5vars_paperlike240_first160_caesar.npz \
--turb_rot_image_group_mode variables \
--npz_image_channels 3 \
--turb_rot_section_start 0 \
--turb_rot_section_index 0
```

E3SM 图中不要默认加入 Turb_Rot-tuned CAESAR checkpoint，除非实验目标明确是跨数据集 transfer。

## 六、当前 Turb_Rot/E3SM n=64 实验口径

当前 n=64 图使用以下 contract：

- 图像/视频模型：DCAE、LIC-HPCM、DCMVC、DCVC-RT
- Turb_Rot 图像/视频输入：`[3,256,256]`，sections `0/1/2`
- E3SM 图像/视频输入：`[3,240,240]`，variables `0/1/2`
- CAESAR Turb_Rot：original checkpoint + Turb_Rot-tuned checkpoint
- CAESAR E3SM：original checkpoint only
- PSNR：`average_frame_psnr`
- bpp：`scientific_bpp_with_side_info`
- 图中指标：PSNR、LPIPS、compression ratio、memory、encode throughput、decode throughput

结果目录：

```text
unified_results/turb_rot_npz_all_models_sections3_frame_psnr_n64
unified_results/e3sm_npz_all_models_vars3_frame_psnr_n64
```
