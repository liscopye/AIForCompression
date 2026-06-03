# 各模型数据处理流程

## 模型分类

| 模型 | 类别 | 输入形状 | 期望色彩空间 | 归一化 | 熵编码 | 特殊处理 |
|------|------|---------|-------------|--------|--------|---------|
| **DCAE** | 图像 | [1,3,H,W] | RGB | 由 pipeline 决定 | 自回归 (CPU RANS) | cudnn 关闭 |
| **LIC-HPCM** | 图像 | [1,3,H,W] | RGB | 由 pipeline 决定 | 非自回归 | — |
| **DCVC-RT** | 视频/图像 | [1,3,H,W] | **YCbCr BT.709** | 由 pipeline 决定 | 自回归 (GPU RANS) | float16 推理 |
| **DCMVC** | 视频/图像 | [1,3,H,W] | RGB | 由 pipeline 决定 | 非自回归 | — |
| **CAESAR** | 序列 | [V,T,H,W] | — | 上游 `ScientificDataset` 内部 `mean_range` instance norm | 基于误差界 (eb) | pipeline 不做 image-style minmax |

---

## 一、图像模型 (DCAE, LIC-HPCM)

### 数据处理链

```
CanonicalSample [C,H,W]
  → build_image_groups:
      kind="image"+uint8 → chunk/255.0 → [0,1]
      其他 → per_channel_minmax → [0,1]
  → CompressAILikeCodec.roundtrip:
      tensor [1,3,H,W] → center-pad(divisor) → model.compress(x_pad)
      → model.decompress(strings, shape)
      → unpad, clamp(0,1) → [1,3,H,W]
  → reconstruct_from_groups:
      uint8_255 → round(clip×255) → uint8
      minmax → x̂×scale + cmin → float32
```

### DCAE 特殊处理
- `torch.backends.cudnn.enabled = False`：自回归 slice 编码需要确定性
- `divisor=128`：pad 到 128 的倍数

### LIC-HPCM 特殊处理
- `divisor=256`：pad 到 256 的倍数
- 60 个质量等级：`torch.exp(torch.linspace(log(0.12), log(64.0), 60))`

---

## 二、DCVC-RT (Intra + P-frame)

### Intra 模式数据处理链

```
CanonicalSample [C,H,W]
  → build_image_groups (同上)
  → DCVCRTCodec.roundtrip:
      tensor [1,3,H,W]
      → RGB→YCbCr BT.709
         Y  = 0.2126R + 0.7152G + 0.0722B
         Cb = -0.1146R - 0.3854G + 0.5000B + 0.5
         Cr = 0.5000R - 0.4542G - 0.0458B + 0.5
      → replicate-pad(64) → model.compress(x_pad, qp)
      → model.decompress(bitstream, sps, qp)
      → unpad → YCbCr→RGB
         R = Y + 1.5748(Cr-0.5)
         G = Y - 0.1873(Cb-0.5) - 0.4681(Cr-0.5)
         B = Y + 1.8556(Cb-0.5)
      → clamp(0,1) → [1,3,H,W]
```

**关键**：即使输入不是真正的 RGB 颜色（如科学数据的三通道），YCbCr 转换仍会执行——它起到"通道去相关+方差集中"的域自适应作用。

### P-frame 模式 (仅 UVG)
- 首帧为 I 帧（DMCI 压缩），后续为 P 帧（DMC 压缩）
- P 帧使用运动估计网络 (ME_Spynet) 参考前一帧
- YCbCr 转换同 Intra 模式
- float16 推理（模型 `.half()` + 输入 `.half()`）

### QP 映射
| QP | 效果 |
|----|------|
| 0 | 最高质量，最低压缩 |
| 21 | 中等 |
| 42 | 中等 |
| 63 | 最低质量，最高压缩 |

---

## 三、DCMVC (Intra + P-frame)

### Intra 模式数据处理链

```
CanonicalSample [C,H,W]
  → build_image_groups (同上)
  → DCMVCCodec.roundtrip:
      tensor [1,3,H,W]  (保持 RGB，不做 YCbCr 转换)
      → replicate-pad(64)
      → model.compress(x_pad, q_in_ckpt=True, q_index)
      → model.decompress(bitstream, H, W, q_in_ckpt=True, q_index)
      → unpad, clamp(0,1) → [1,3,H,W]
```

**关键**：DCMVC 在 RGB 空间工作，不做 YCbCr 转换。这是它与 DCVC-RT 的核心差异。

### P-frame 模式 (仅 UVG)
- 使用 `DCMVC_model.DMC` 进行帧间压缩
- GOP=32，首帧 I 帧，后续 P 帧
- 无色彩空间转换

### Q 索引
| q_index | 效果 |
|---------|------|
| 0 | 最高质量 |
| 1 | 中高质量 |
| 2 | 中低质量 |
| 3 | 最低质量 |

---

## 四、CAESAR

### 数据处理链

```
适配器.load_sequence()
  → [V, T, H, W] float32 (pipeline 保留原始数据)
  → CAESAR ScientificDataset(inst_norm=True, norm_type="mean_range")
  → CAESAR 压缩/解压 (误差界控制)
  → dataset.recons_data() 反变换回原始数据空间
  → PSNR/MSE: 原始数据空间
```

### 各数据集序列构造

| 数据集 | V | T | H×W | 序列含义 |
|--------|---|---|-----|---------|
| Kodak | 3 (RGB) | 16 | 512×512 | 16 张图像伪序列 |
| UVG | 3 (RGB) | 16 | 2160×3840 | 16 帧视频 |
| Hurricane | 1 | 16 | 500×500 | 16 时间步气压场 |
| NYX | 1 | 16 | 512×512 | 16 层 Z 切片密度场 |
| isot1024 | 3 (u,v,w) | 10 | 256×256 | 10 时间步速度场切片 |
| ERA5 | 6 | 16 | 721×1440 | 6 气象变量伪序列 |
| Tomo | 1 | 16 | 1792×2048 | 16 投影角伪序列 |
| S2C | 1 | 16 | 1024×1024 | 16 tile 伪序列（空间相邻） |
| Lysozyme | 1 | 16 | 1065×1030 | 16 帧衍射图 |

### 误差界 (eb) 效果

| eb | 效果 |
|----|------|
| 1e-4 | 最高质量，最低压缩 |
| 5e-4~5e-3 | 中高质量 |
| 1e-2~5e-2 | 中等 |
| 1e-1~5e-1 | 低质量，最高压缩 |

---

## 五、模型间处理差异汇总

| 处理步骤 | DCAE | HPCM | DCVC-RT | DCMVC | CAESAR |
|---------|------|------|---------|-------|--------|
| 色彩空间 | RGB | RGB | YCbCr | RGB | 无 |
| 归一化来源 | pipeline | pipeline | pipeline | pipeline | 无 |
| pad 方式 | center-0 | center-0 | replicate | replicate | N/A |
| divisor | 128 | 256 | 64 | 64 | N/A |
| 精度 | float32 | float32 | float16 | float32 | float32 |
| 熵编码 | 自回归CPU | 非自回归 | 自回归GPU | 非自回归 | 误差界 |
| 序列支持 | ✗ | ✗ | ✓(P帧) | ✓(P帧) | ✓ |

---

## 六、DCVC-RT YCbCr 转换的域自适应效应

对于非图像数据（如 NYX 密度场、Hurricane 气压场），BT.709 YCbCr 转换并无颜色语义，但产生三个有益效果：

1. **方差集中**：Y 通道汇集大部分方差，Cb/Cr 仅编码通道间细微差异——匹配自然图像 YCbCr 的统计
2. **通道去相关**：RGB 空间三通道高度相关 → 模型编码器产生异常 latent。YCbCr 部分去相关后，latent 分布回到熵模型训练分布
3. **数值域归一化**：YCbCr 各通道固定在 [0,1] 范围，与模型内部 clamp(0,1) 操作一致

**例外**：单通道复制×3 的数据（S2C、Lysozyme）——Cb=Cr=0.5 常数，DCVC 熵模型对恒定通道适应性较差。Lysozyme 实际效果尚可（PSNR 43-61dB），S2C 需过滤常量 tile 避免失真。
