# 数据集说明

## 完整数据集

本项目的训练与评测使用竞赛主办方提供的离线轨迹数据集 `data.csv`（约 14 MB）。

**该数据集未随本仓库公开**，原因：

1. 数据集版权属于竞赛主办方，公开前需确认授权；
2. 评测环境为离线环境，数据集属于赛题资产。

如需复现训练流程，请自行获取数据集并放置为 `data/data.csv`（该路径已在 `.gitignore` 中排除，避免误提交）。

## 数据格式

单帧观测 5 维，动作 3 维（连续），历史长度 30 帧。

字段含义与示例见 [`data_fields.xlsx`](data_fields.xlsx)（由主办方提供的 `data_example.xlsx` 整理）。

CSV 主要列（示例）：

| 列 | 说明 |
|---|---|
| `state_0 ... state_4` | 5 维观测 |
| `action_0 ... action_2` | 3 维动作 |
| `reward` | 即时奖励 |
| `next_state_0 ... next_state_4` | 下一时刻观测 |
| `done` | 回合结束标志 |

## 评测记录（可公开）

`results/` 下的三份 `evaluation_results.csv` 是**我自己的评测输出**，不包含原始轨迹数据，可以安全公开：

| 文件 | 版本 | 行数 |
|---|---|---|
| `results/gen2_td/evaluation_results.csv` | v2 帧堆叠+时序差分特征 | 100 |
| `results/gen3_history/evaluation_results.csv` | v3 帧堆叠历史观测 | 100 |
| `results/gen4_tcn/evaluation_results.csv` | v4 TCN | 100 |

列为 `seed, score, total_reward, avg_state_error, safety_violations`。

> ⚠️ **提交前请自行确认竞赛协议是否允许公开数据集**。若允许，把 `.gitignore` 中的 `data/*.csv` 一行删掉即可。
