# 文档导航

日常使用先看仓库根目录的 `README.md`。本目录只保留当前有效的协议、复现、结果和交接资料。

| 文档 | 用途 |
|---|---|
| `环境激活与测试启动.md` | 当前机器激活环境、smoke、普通测试与 Objective-v1 正式测试命令 |
| `compression_pipeline概览.md` | Pipeline 架构、数据流、核心模块和扩展方法 |
| `objective_benchmark_protocol.md` | 正式评测的公平性边界和指标定义 |
| `benchmark_reproduction_manifest.md` | 数据、样本、EB、权重和完整复现参数 |
| `dataset_model_processing_spec.md` | 各数据集与模型的输入、归一化和指标处理规范 |
| `Data数据资产盘点.md` | `/workspace/Data` 每个数据集的下载量、本地占用、格式和派生副本 |
| `benchmark_observation_analysis.md` | objective-v1 主结果观察 |
| `结果目录索引.md` | 正式结果、专项结果和诊断证据的位置 |
| `项目交接总览.md` | 当前项目状态与 ERA5/CAESAR 最终交接 |
| `ERA5_CAESAR失败尝试记录.md` | ERA5 CAESAR 有效但未默认交付的曲线，以及已验证失败的训练方向 |
| `迁移到新机器指南.md` | 新机器安装、数据和权重复制及迁移验收 |

机器可读正式协议位于 `benchmark_protocols/objective_v1.json`；正式结果入口是 `unified_results/objective_all_to_all_v1/index.html`。

已删除的文档主要是早期设计计划、重复的 Pipeline 教程、过期的清理审计，以及已经合并进交接总览或失败记录的长篇阶段分析。需要追溯时可从 Git 历史恢复。
