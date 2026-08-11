# CAESAR Original vs Tuned 对比评测指南

## 前置条件

```bash
source /workspace/ai4cp/bin/activate
```

---

## 1. Lysozyme 数据集

### 1.1 单点评测

```bash
# CAESAR-V (8 帧，无扩散，快)
CUDA_VISIBLE_DEVICES=<GPU_ID> python scripts/eval_caesar_lysozyme.py \
  --model_type V --ckpt both \
  --device cuda:0 \
  --output_dir results_lysozyme \
  --eb 1e-3 --batch_size 32 --max_blocks 50

# CAESAR-D (16 帧，有扩散，慢)
CUDA_VISIBLE_DEVICES=<GPU_ID> PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python scripts/eval_caesar_lysozyme.py \
  --model_type D --ckpt both \
  --device cuda:0 \
  --output_dir results_lysozyme \
  --eb 1e-3 --batch_size 8 --max_blocks 30
```

| 参数 | 含义 |
|---|---|
| `--model_type` | `V` 或 `D` |
| `--ckpt` | `original` / `finetuned` / `both` |
| `--tuned_suffix` | fine-tuned checkpoint 后缀，lysozyme 默认 `_tuning_lysozyme` |
| `--eb` | 误差界（error bound），控制压缩精度 |
| `--max_blocks` | 评测块数上限，`None`=全部 |
| `--test_data` | lysozyme 默认自动选，也可手动指定 |

### 1.2 EB Sweep（多误差界遍历）

```bash
CUDA_VISIBLE_DEVICES=<GPU_ID> python scripts/sweep_caesar_eb_lysozyme.py \
  --model_type D --gpu <GPU_ID>
```

- `--model_type V/D/both`
- `--test_eb 1e-3` 可只跑单点（测试用）
- 7 个 EB：`1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2`
- 结果目录：`results/eb_sweep_D/` 或 `results/eb_sweep/`

### 1.3 画图

```bash
# 单模型
python utils/plot_caesar_compare.py \
  --orig results/eb_sweep_D/summary_original.json \
  --tuned results/eb_sweep_D/summary_finetuned.json \
  --output results/eb_sweep_D/caesar_d_comparison.png \
  --title "CAESAR-D Lysozyme: Finetuned vs Original"

# V+D 合并图
python utils/plot_caesar_compare.py \
  --orig results/eb_sweep_V/summary_combined_original.json \
  --tuned results/eb_sweep_V/summary_combined_finetuned.json \
  --output results/eb_sweep_V/caesar_combined_comparison.png \
  --title "CAESAR Lysozyme: Finetuned vs Original (V & D)"
```

`summary_*.json` 转格式脚本见下一节的示例。

---

## 2. ERA5 数据集

### 2.1 预处理测试数据

```bash
# 解压原始 nc 文件
mkdir -p Data/ERA5/test
tar xzf ERA5.tar.gz -C Data/ERA5/test --strip-components=2

# 转成 z-score .npy（mmap）
python utils/prepare_era5_finetune_data.py \
  --input_dir Data/ERA5/test \
  --output_dir Data/ERA5/finetune_processed/era5_test_tmp

# 或者用自定义脚本处理（输出 era5_test.npy）
```

### 2.2 创建 CAESAR 兼容格式的 .npz 测试块

```bash
python3 -c "
import numpy as np, os
PAD = 256
test_npy = 'Data/ERA5/finetune_processed/era5_test.npy'
out_dir = 'Data/ERA5/finetune_processed/test_blocks'
os.makedirs(out_dir, exist_ok=True)

data = np.load(test_npy, mmap_mode='r')
C_full, T, H, W = data.shape
pad_h = (PAD - H % PAD) % PAD
pad_w = (PAD - W % PAD) % PAD
h_blks = (H + pad_h) // PAD
w_blks = (W + pad_w) // PAD

ch_data = np.array(data[:10], dtype=np.float32)  # 用 10 个通道
ch_data = np.pad(ch_data, ((0,0),(0,0),(0,pad_h),(0,pad_w)), mode='reflect')

for n_frame, tag in [(8, 'V'), (16, 'D')]:
    t_wins = T - n_frame + 1
    blocks = []
    for c in range(ch_data.shape[0]):
        for t0 in range(t_wins):
            for bh in range(h_blks):
                for bw in range(w_blks):
                    h0, h1 = bh*PAD, (bh+1)*PAD
                    w0, w1 = bw*PAD, (bw+1)*PAD
                    blocks.append(ch_data[c, t0:t0+n_frame, h0:h1, w0:w1])
    arr = np.stack(blocks, axis=0)[:, np.newaxis, :, :, :]
    arr = np.squeeze(arr, 1)[np.newaxis, :]
    path = os.path.join(out_dir, f'era5_test_{tag}_nf{n_frame}.npz')
    np.savez_compressed(path, data=arr)
    print(f'{tag}: {arr.shape} -> {path}')
"
```

### 2.3 单点评测

```bash
CUDA_VISIBLE_DEVICES=<GPU_ID> python scripts/eval_caesar_lysozyme.py \
  --model_type V --ckpt both --tuned_suffix _tuning_era5 \
  --test_data Data/ERA5/finetune_processed/test_blocks/era5_test_V_nf8.npz \
  --device cuda:0 --output_dir results_era5 \
  --eb 1e-3 --batch_size 32 --max_blocks 50
```

### 2.4 EB Sweep + 画图

用内联 Python 脚本一键跑 V+D 的 7 个 EB 点 + 转格式 + 画图：

```bash
CUDA_VISIBLE_DEVICES=<GPU_ID> PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python3 << 'PYEOF'
import os, sys, json, time, subprocess

SCRIPT = "scripts/eval_caesar_lysozyme.py"
EB_VALUES = [1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2]
GPU = <GPU_ID>
env = os.environ.copy()
env["CUDA_VISIBLE_DEVICES"] = str(GPU)

for mt, nf, test_npz, max_blocks, bs in [
    ("V", 8, "Data/ERA5/finetune_processed/test_blocks/era5_test_V_nf8.npz", 50, 32),
    ("D", 16, "Data/ERA5/finetune_processed/test_blocks/era5_test_D_nf16.npz", 20, 8),
]:
    BASE = f"results/eb_sweep_ERA5_{mt}"
    os.makedirs(BASE, exist_ok=True)
    for eb in EB_VALUES:
        eb_str = f"{eb:.0e}".replace("e-0", "e-").replace("e-", "em")
        out_dir = os.path.join(BASE, f"eb_{eb_str}")
        os.makedirs(out_dir, exist_ok=True)
        result = subprocess.run([
            sys.executable, SCRIPT,
            "--model_type", mt, "--ckpt", "both", "--tuned_suffix", "_tuning_era5",
            "--test_data", test_npz, "--device", "cuda:0", "--output_dir", out_dir,
            "--eb", str(eb), "--batch_size", str(bs), "--max_blocks", str(max_blocks),
        ], env=env, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"ERR eb={eb} mt={mt}: {result.stderr[-200:]}")
    # 重建 sweep_results.json
    all_metrics = []
    for d in sorted(os.listdir(BASE)):
        if not os.path.isdir(os.path.join(BASE, d)): continue
        for variant in [f"CAESAR-{mt}_original", f"CAESAR-{mt}_finetuned"]:
            jp = os.path.join(BASE, d, f"{variant}.json")
            if os.path.exists(jp):
                with open(jp) as f: m = json.load(f)
                m["variant"] = variant
                m["eb"] = float(d.replace("eb_", "").replace("em", "e-"))
                all_metrics.append(m)
    with open(os.path.join(BASE, "sweep_results.json"), "w") as f:
        json.dump(all_metrics, f, indent=2)

# 转 summary 格式 + 画图
for mt in ["V", "D"]:
    base = f"results/eb_sweep_ERA5_{mt}"
    with open(f"{base}/sweep_results.json") as f: data = json.load(f)
    orig, tuned = [], []
    for r in data:
        ob = r.get("original_size_bytes", 0)
        enc_tp = ob / r["encode_time_total"] if r.get("encode_time_total", 0) > 0 else 0
        dec_tp = ob / r["decode_time_total"] if r.get("decode_time_total", 0) > 0 else 0
        entry = {"model_id": f"caesar_{mt.lower()}", "eb": r["eb"], "bpp": r["bpp"],
                 "psnr": r["psnr"], "mse": r["mse"], "encode_throughput": enc_tp,
                 "decode_throughput": dec_tp, "encode_time_avg": r["encode_time_total"],
                 "decode_time_avg": r["decode_time_total"]}
        (orig if "original" in r["variant"] else tuned).append(entry)
    orig.sort(key=lambda x: x["eb"]); tuned.sort(key=lambda x: x["eb"])
    with open(f"{base}/summary_original.json", "w") as f: json.dump(orig, f, indent=2)
    with open(f"{base}/summary_finetuned.json", "w") as f: json.dump(tuned, f, indent=2)

# 画图
subprocess.run(["python", "utils/plot_caesar_compare.py",
    "--orig", "results/eb_sweep_ERA5_V/summary_combined_original.json",
    "--tuned", "results/eb_sweep_ERA5_V/summary_combined_finetuned.json",
    "--output", "results/eb_sweep_ERA5_V/caesar_combined_era5_comparison.png",
    "--title", "CAESAR ERA5: Finetuned vs Original (V & D)"])
print("Done.")
PYEOF
```

---

## 3. Sweep 结果转 summary 格式

`plot_caesar_compare.py` 需要的 `summary_*.json` 格式：

```json
[
  {
    "model_id": "caesar_d",
    "eb": 0.0001,
    "bpp": 3.7068,
    "psnr": 80.26,
    "mse": 6174814208.0,
    "encode_throughput": 15623456.0,
    "decode_throughput": 20987654.0,
    "encode_time_avg": 5.8,
    "decode_time_avg": 3.2
  }
]
```

`sweep_results.json` → `summary_original.json` / `summary_finetuned.json` 的转换：

```python
import json

for base, tag in [("results/eb_sweep_D", "D"), ("results/eb_sweep_V", "V")]:
    with open(f"{base}/sweep_results.json") as f: data = json.load(f)
    orig, tuned = [], []
    for r in data:
        ob = r.get("original_size_bytes", 0)
        enc_tp = ob / r["encode_time_total"] if r["encode_time_total"] > 0 else 0
        dec_tp = ob / r["decode_time_total"] if r["decode_time_total"] > 0 else 0
        entry = {
            "model_id": f"caesar_{tag.lower()}",
            "eb": r["eb"], "bpp": r["bpp"], "psnr": r["psnr"], "mse": r["mse"],
            "encode_throughput": enc_tp, "decode_throughput": dec_tp,
            "encode_time_avg": r["encode_time_total"],
            "decode_time_avg": r["decode_time_total"],
        }
        (orig if "original" in r["variant"] else tuned).append(entry)
    orig.sort(key=lambda x: x["eb"]); tuned.sort(key=lambda x: x["eb"])
    with open(f"{base}/summary_original.json", "w") as f: json.dump(orig, f, indent=2)
    with open(f"{base}/summary_finetuned.json", "w") as f: json.dump(tuned, f, indent=2)
```

---

## 4. 输出文件说明

| 脚本 | 输出 | 说明 |
|---|---|---|
| `eval_caesar_lysozyme.py` | `{output_dir}/CAESAR-{V/D}_{ckpt}.json` | 单点评测结果 |
| `sweep_caesar_eb_lysozyme.py` | `results/eb_sweep_{D/V}/eb_{eb}/CAESAR-{V/D}_{original,finetuned}.json` | 每 EB 的评测 |
| | `results/eb_sweep_{D/V}/sweep_results.json` | 汇总所有 EB 点 |
| `plot_caesar_compare.py` | `.png` | PSNR-vs-BPP + Throughput 图 |

### 关键指标

| 指标 | JSON 字段 | 含义 |
|---|---|---|
| PSNR | `psnr` | 峰值信噪比（越高越好） |
| BPP | `bpp` | 每像素比特数（越低越好） |
| CR | `compression_ratio` | 原始大小/压缩后（越高越好） |
| 编码时间 | `encode_time_total` | 总编码耗时（秒） |
| 解码时间 | `decode_time_total` | 总解码耗时（秒） |
| MSE | `mse` | 均方误差 |
| 吞吐量 | `encode/decode_throughput` | summary 格式中计算，bytes/s |

---

## 5. 已知问题

- **CAESAR-D 小 EB 值 OOM**：diffusion 模型显存占用大，`eb <= 5e-4` 需要 `--max_blocks 10` 或更小
- **CAESAR-V 小 EB 值无问题**：无扩散，显存充足
- **encode/decode_time_total 为 0**：旧脚本有问题时会出现，重跑即可修复
- **tuned model 比 original 差**：ERA5 上 fine-tune 可能没收敛好，lysozyme 上效果明显
- **interpo_rate 必须为 3**：`eval_caesar_lysozyme.py` 已修复（原来硬编码为 4）
