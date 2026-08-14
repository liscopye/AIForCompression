# Objective benchmark 合规审计：aifc-objective-v1

- 记录：1650
- 完全合规记录：1650
- 完整数据集：10/10

## 全局检查

| 检查项 | 通过记录 | 总记录 |
|---|---:|---:|
| known_dataset | 1650 | 1650 |
| track_eligible | 1650 | 1650 |
| canonical_identity | 1650 | 1650 |
| rate_complete | 1650 | 1650 |
| metric_fixed_scale | 1650 | 1650 |
| external_input_declared | 1650 | 1650 |
| mask_policy_declared | 1650 | 1650 |
| timing_repeated | 1650 | 1650 |
| hardware_declared | 1650 | 1650 |
| codec_valid | 1650 | 1650 |
| protocol_tagged | 1650 | 1650 |
| codec_execution_declared | 1650 | 1650 |

## 数据集覆盖

| 数据集 | 记录 | objective samples | raw checksum | normalized checksum | 缺少主方法 | 点数不足 | 完整 |
|---|---:|---:|---:|---:|---|---|---:|
| e3sm_npz | 108 | 是 | 是 | 是 | - | - | 是 |
| era5_npy | 54 | 是 | 是 | 是 | - | - | 是 |
| hurricane | 54 | 是 | 是 | 是 | - | - | 是 |
| nyx | 54 | 是 | 是 | 是 | - | - | 是 |
| turb_rot_npz | 136 | 是 | 是 | 是 | - | - | 是 |
| tomo | 108 | 是 | 是 | 是 | - | - | 是 |
| lysozyme | 108 | 是 | 是 | 是 | - | - | 是 |
| s2c | 153 | 是 | 是 | 是 | - | - | 是 |
| kodak | 813 | 是 | 是 | 是 | - | - | 是 |
| uvg_twilight_1080p | 62 | 是 | 是 | 是 | - | - | 是 |

## 说明

旧结果未通过新协议不等于 codec 输出无效，而是不能支撑严格的跨 codec 客观排名。
