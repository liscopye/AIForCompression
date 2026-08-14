# 文档导航

日常使用先看仓库根目录的 `README.md`。本目录只保留当前有效的协议、复现、数据和环境资料。

| 文档 | 用途 |
|---|---|
| `环境激活与测试启动.md` | 当前机器激活环境、smoke、普通测试与 Objective-v1 正式测试命令 |
| `compression_pipeline概览.md` | Pipeline 架构、数据流、核心模块和扩展方法 |
| `objective_benchmark_protocol.md` | 正式评测的公平性边界和指标定义 |
| `benchmark_reproduction_manifest.md` | 数据、样本、EB、权重和完整复现参数 |
| `ERA5_CAESAR-V微调记录.md` | ERA5 CAESAR-V 的有效历史结果、失败方向和当前训练链 |
| `ERA5与Lysozyme_CAESAR微调测试复现.md` | ERA5 与 Lysozyme 的 CAESAR 微调、权重和测试复现命令 |
| `Data数据集.md` | `/workspace/Data` 每个数据集的下载量、本地占用、格式和派生副本 |

机器可读正式协议位于 `benchmark_protocols/objective_v1.json`；正式结果入口是 `unified_results/objective_all_to_all_v1/index.html`，结果文件说明见 `unified_results/README.md`。
