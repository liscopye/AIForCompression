# Objective 来源结果归档

本目录保存正式 `objective_all_to_all_v1` 的补测、微调、LPIPS 回填和显存探针来源，按数据集组织：

```text
objective_runs/
└── <dataset_id>/
    └── <run_name>/
        ├── summary.json       # 该次运行的结果（适用时）
        ├── artifacts/         # codec 临时产物（适用时）
        └── _run_files/        # 原运行目录的图表、shard 或完成标记
```

跨数据集正式汇总、审计、图表和单文件页面仍位于相邻的
`objective_all_to_all_v1/`。移动这里的来源目录不会改变正式 `index.html`；重新合并时应显式选择所需的 `<dataset_id>/<run_name>` 作为来源。

整理新的散落目录前先预览：

```bash
python scripts/organize_objective_results.py
```

确认目标不存在冲突后执行：

```bash
python scripts/organize_objective_results.py --execute
```
