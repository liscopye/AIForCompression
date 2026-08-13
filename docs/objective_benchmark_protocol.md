# 十数据集客观压缩对比协议

## 目的

旧结果 `combined_summary.json` 汇集了大量模型和码率点，但不同模型使用了不同裁剪、plane 数量、归一化、BPP 分母和计时边界。它可以展示已有实验，不能直接用于跨 codec 排名。

2026-07-21 的 matched validation 已经证明“相同输入”会改变若干结论，但它仍是单 sample、单次计时的诊断实验。正式结果应遵守 `benchmark_protocols/objective_v1.json`，继续使用原来的十个数据集，不更换测试领域。

## All-to-all 主矩阵

客观比较必须拆成四条主轨道：

| 轨道 | 数据集 | 主方法 | 回答的问题 |
|---|---|---|---|
| Scientific numeric | E3SM、ERA5、Hurricane、NYX、Turb_Rot、Tomo、Lysozyme、S2C | DCAE、LIC-HPCM、CAESAR-V/D、cuSZ-Hi、nvJPEG2000、DCMVC-I、DCVC-RT-I | 相同科学数值的率失真 |
| RGB intra | Kodak | DCAE、LIC-HPCM、CAESAR-V/D、cuSZ-Hi、DCMVC-I、DCVC-RT-I、nvJPEG | 全部 24 张 RGB 图像的帧内/3D corpus 压缩 |
| Video temporal | UVG | DCAE、LIC-HPCM、CAESAR-V/D、cuSZ-Hi、nvJPEG、DCMVC-IP、DCVC-RT-IP | 相同 30 帧；视频 codec 包含 I-frame 成本的真实 P-frame 压缩 |
| Error bounded | 八个科学数据集 | cuSZ-Hi | 请求误差界是否真的成立，以及有效曲线范围 |

本轮模型集合不包含 LIC-TCM。学习式图像模型只包含 DCAE 和 LIC-HPCM（base/large）；CAESAR no-PCA、DCAE/HPCM+PCA 仍属于消融，不进入主矩阵。除 UVG 外的数据都没有合法的时间预测关系，因此 DCMVC/DCVC-RT 只运行 I-frame；UVG 运行完整 P-frame 路径。

S2C、Kodak 和 UVG 的 3D 组织方式固定如下：S2C 按 tile-major、band-minor 堆叠为 16 planes；Kodak 使用全部 24 张图和全部 RGB 通道得到 72 planes，竖图先做可逆旋转并在评价前转回；UVG 保持 RGB 为变量，30 帧为真实时间轴，cuSZ 对每个 RGB 通道压缩一个时间 3D volume。CAESAR 为 8/16 帧窗口增加的 repeat-last padding 必须计入码率和时间，评价时裁回原深度。

## 消除影响因素的具体规则

### 公平控制的边界

benchmark 只控制 codec 外部，不统一模型内部算法。

外部必须一致的是：原始样本和索引、crop/分辨率、有效 mask，以及数据集级颜色转换和归一化参数。**归一化按数据集定义，不按模型定义。** 每个数据集先生成唯一的 normalized canonical tensor，DCAE、HPCM、CAESAR、cuSZ-Hi 和 nvJPEG2000 都从相同的 float32 数值开始；解码后通过同一个数据集级 inverse transform 回到原始数值域。

CAESAR 的实例归一化、cuSZ-Hi 的 predictor、nvJPEG2000 的内部变换、模型 padding、latent 和熵模型都属于 codec 自身行为。即使模型在 normalized canonical tensor 上再次执行内部归一化，也保持原样。benchmark 只完整计时和记录它们，不修改、不替换，也不要求彼此相同。模型内部处理得更好，本来就应体现在最终 RD 和速度中。

CAESAR 推理采用作者 `eval_caesar.ipynb` 的 `DataLoader(batch_size=64)`。batch 只控制模型推理调度；完整场重建后仍由 CAESAR 自身执行一次 PCA 后处理。结果记录 `caesar_inference_batch_size`，防止不同 batch 的吞吐量被混合。

数据集级归一化必须冻结、可逆且禁止按 sample 调参。当前协议采用：Kodak 固定 `/255`；UVGAdapter 已完成固定 YUV420→RGB 并输出 `[0,1]`，因此再做 identity；Tomo 固定 `/65535`；ERA5 在已有 CRA5 z-score 上使用全 objective corpus 的逐变量固定 min/max；S2C 使用四个 objective tiles 联合得到的逐 band 固定 min/max（实际有效值超过 10000，不能直接 `/10000` 后 clipping）；其他科学 float 数据也使用完整 objective corpus 的逐变量或单变量 affine min/max。NYX 的 `log1p` 只能作为所有 codec 共同使用的单独消融，不能只给某个模型使用。

### 1. 输入身份

每个 codec 的每条记录必须保存：

- `canonical_sample_id`
- `canonical_sha256`
- `canonical_shape`
- `canonical_symbol_count`
- `canonical_valid_symbol_count`

所有方法必须覆盖相同 canonical symbols。允许 CAESAR 按 8/16 帧、图像模型按 2D group、cuSZ 按 3D block 内部分块，但不能少算、重复算或换 crop。

正式 sample 清单已经写入协议。例如 ERA5 主实验使用全部 268 变量和 16 个测试时刻；E3SM 使用全部 5 变量的两个 16-time block；NYX 使用完整 512³ volume；Kodak 使用全部 24 张图；UVG 使用提供文件中的全部 30 帧。

### 2. 外部转换和辅助码率

每条结果必须用 `external_input_manifest` 记录数据集级归一化参数、normalized canonical checksum，以及 codec API 必需的 dtype/layout conversion。科学数据最终都在逆转换后的原始物理域评价。

协议禁止按样本计算 min/max。由 calibration corpus 冻结、对整个数据集和所有 codec 相同的 min/max 或 mean/std 属于 benchmark 数据定义，不按样本收费。codec 内部 deterministic normalization、padding 和 latent 处理不由 benchmark 重新计费；但 codec 实际输出中的 PCA 系数、shape、mask、GOP header 和 I-frame 字节仍必须计入码流。

cuSZ-Hi 的 EB 也必须按同一 transform 换算。例如 `x_norm=(x-min_v)/scale_v` 时，物理绝对误差界 `EB_raw` 对应 `EB_norm=EB_raw/scale_v`。最终仍在反归一化后的原始域逐值验证 `max_abs_error <= EB_raw`。

统一 BPP：

```text
8 * (payload_bytes + side_info_bytes) / canonical_symbol_count
```

模型权重和通用 decoder 程序不计入单个样本码率。

Lysozyme 的 frozen canonical invalid mask 明确声明为所有方法共享的 benchmark metadata：所有 codec 都免费获得同一 mask，压缩前按同一规则替换 sentinel，评价时只统计有效位置。它不计入任何方法的码率；不能只让部分方法免费获得 mask。

### 3. 科学质量指标

全局 `max-min` PSNR 会严重抬高 E3SM、ERA5、NYX 和 Lysozyme。主指标改为固定尺度归一化 MSE：

```text
NMSE = mean_v(MSE_v / scale_v^2)
normalized PSNR = -10 log10(NMSE)
```

`scale_v` 每个物理变量只有一个，由声明的 calibration corpus 计算并冻结，不能按 sample 或 codec 重算。它只定义评价指标，不会拿去替换模型内部归一化。并列报告逐变量 PSNR 中位数、NRMSE、raw MSE、最大绝对误差。cuSZ-Hi 必须逐值验证请求 EB，违反 EB 的码流判为失败点。

所有数据集均报告 LPIPS。Kodak/UVG 使用全部原生 RGB 图像/帧；每个科学 canonical sample 在冻结数据集 normalization 后按展平顺序等距固定抽取 32 个二维平面，将灰度复制为三通道输入 LPIPS。该固定视图及索引对所有 codec 完全一致，只作诊断，不能替代科学数值主指标。

### 4. 端到端时间

输入边界是已加载到 host memory 的 canonical tensor；输出边界是 host memory 中完整重建的 canonical tensor。

计时包含外部 adapter、完整 codec 调用（包括它自己的归一化、分块、padding、模型/预测器和熵编码）、H2D/D2H、外部逆 adapter 和重组；不包含首次磁盘读取、模型初始化、权重加载、指标和绘图。编码和解码分开记录，同时记录 roundtrip。

每个 shape/控制点：

1. 独占一张物理 GPU，关闭 MPS 共享和并发负载。
2. 预热 2 次。
3. 正式测量 5 次。
4. 报告 median、p10、p90，不用单次最快值冒充峰值。

### 5. 曲线比较

- CAESAR 和 cuSZ-Hi 每条 EB 曲线至少 7 个有效且尽量单调的点；DCAE/HPCM 使用发布的 6 个权重，DCMVC/DCVC-RT 使用发布的 4 个质量档。
- EB 必须按数据集探测。左端逼近仍能有效评价的最低 BPP，右端尽量接近 32 BPP；如果 codec 在到达 32 BPP 前违反误差界、损坏重建或失败，则停在最后一个有效点并保留失败探测证据。
- 原始失败/震荡点必须保留并标记；绘图可以显示单调前沿，但不能静默删除证据。
- 只在两个方法实际覆盖的 BPP 或质量交集内插值。
- 不跨 DCAE/HPCM 与 CAESAR 之间的空白 BPP 外推。
- 先聚合整个 canonical corpus 的总字节和总 SSE，再计算 BPP/PSNR；不能把每张图的 PSNR 简单平均作为主结果。
- 对 sample/变量 bootstrap，报告 95% CI；吞吐量范围来自 5 次正式测量。

## 结果定位

| 结果集 | 可以做什么 | 不能做什么 |
|---|---|---|
| `objective-v1` 正式重测 | 论文主 RD、吞吐量和数据集分轨结论 | 跨轨道宣称统一优胜者 |
| `era5_caesar_v` | ERA5 上 CAESAR-V 原始与有效微调权重的专项比较 | 代替全模型正式排名 |
| `lysozyme_caesar_tuned` | Lysozyme 上 CAESAR-V/D 原始与微调权重的专项比较 | 代替完整 Objective-v1 corpus |

## 自动审计

旧结果、中间来源和 matched validation 已清理；正式模型去向汇总在单文件 `index.html` 中。

正式流水线应在发布结果前使用 `--strict`。只有所有 objective sample、主方法、checksum、固定指标、side information 和重复计时均通过时，命令才返回成功。

## 推荐执行顺序

1. 先完成 E3SM、Turb_Rot、Tomo 和 UVG。这四个数据集分别覆盖异构变量、平滑科学场、连续投影和真实视频，最容易发现协议错误。
2. 再完成 ERA5 全 268 变量、NYX 全 volume、Hurricane 和 S2C。
3. 最后运行成本最高的两个 496-frame Lysozyme stack，并补全 mask 码率。
4. 主轨通过后再运行 no-PCA 和 image+PCA 消融，避免消融结果污染主榜。

不能通过审计的历史点可以继续保留在 JSON 中，但正式绘图脚本必须默认过滤，或者明确标成 `legacy/non-comparable`。
