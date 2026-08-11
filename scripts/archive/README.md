# 历史实验脚本

此目录保存 CAESAR ERA5 调参、探针、恢复训练、稳定性诊断和阶段性结果比较脚本。
它们用于解释既有实验记录和复现历史尝试，但不是当前 benchmark 的推荐入口。

新评测优先使用：

- `scripts/run_dataset_compression.py`：通用开发与 smoke test。
- `scripts/prepare_objective_inputs.py`：准备 objective-v1 输入。
- `scripts/run_objective_benchmark.py`：正式图像、科学数据和传统 codec 评测。
- `scripts/run_objective_video.py`：正式 UVG P-frame 评测。
- `scripts/build_objective_all_to_all.py`、`scripts/audit_objective_benchmark.py` 和
  `scripts/analyze_objective_benchmark.py`：合并、审计与展示正式结果。

归档脚本可能包含旧集群绝对路径，运行前需要根据当前机器修改环境变量或路径。
