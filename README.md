# elthan-pose

YOLO11-pose 解剖先验改进 fork（论文 B 代码）。Base = ultralytics 8.4.128 vendored 快照，
在此之上叠加 B2' 可见性加权与 B1 解剖先验 loss 系列（全部 commit 自研）。

配套仓库：[pose-eval-harness](https://github.com/fanyifan090409-dev/pose-eval-harness)（论文 A 评测）、
[elthan-pose-research](https://github.com/fanyifan090409-dev/elthan-pose-research)（实验日志/数据/训练脚本）。

## 改动清单（commit 链，旧 → 新）

| Commit | 改动 |
|---|---|
| ec36986 | vendored ultralytics 8.4.128 纯净快照（round6 base） |
| 29437d6 | B2'：关键点可见性软加权（训练侧利用 v 通道，官方 issue #3409 确认训练忽略 v） |
| 9678bf5 | B1.2：骨长 + 方向一致性正则（BoneConsistencyLoss） |
| 394c66c | B1.3：左右对称软惩罚（SymmetryLoss）+ 解剖先验统一框架 |
| 0232c4c | B1.4：关节角度合理性铰链惩罚（AnglePlausibilityLoss） |
| 935cf8c | B1.5：解析式不确定性自动配权（uncertainty_weights） |
| 16e81ca | B1.6：手工规则 vs 可学习骨长先验对照（BoneStatsLoss / LearnableBonePrior） |
| c2ee539 | fix：anatomy loss 转 fp32 计算（AMP fp16 网格坐标平方 ≈1e5 超 65504 上限 → b14 NaN） |

## 结果（如实报告，负面结果）

内部 val：Box mAP50-95 全部变体落在 0.609–0.613 噪声级，与 baseline 无显著差。
外部评测（CrowdPose val / OCHuman test）：无一变体超过 baseline（0.53601 / 0.39911）。

结论：B1 解剖先验 loss 在该数据规模/模型容量下无增益，作为受控负面结果写入论文 B；
B2' 可见性加权无增益但方向与官方 issue 确认的「训练忽略 v」一致，属填补空白的负结果。
评测细节见 elthan-pose-research `docs/plans/round6-B1B2实施规划.md` 与 lab-notebook round6 日志。

## 使用

```bash
conda activate py310
cd ultralytics-dev
python -m pip install -e .
# 训练入口与配置见 elthan-pose-research scripts/round6_run.sh + datasets/pose-round5/data.yaml
```

## 许可

AGPL-3.0（继承 ultralytics 上游）。
