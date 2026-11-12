# 离线数据集说明

> 原始竞赛离线数据集 `data.csv`（约 14 MB，来自主办方工业控制基准环境 `IndustrialControlEnv`）属于评测方数据资产，本公开仓库不直接分发原始大文件。
>
> 为保证离线读取、历史窗口构造与训练链路可独立验证，本仓库提供了**与真实数据同构的合成轨迹生成器** [`../scripts/generate_demo_data.py`](../scripts/generate_demo_data.py)。

## `data.csv` 真实列契约（与代码一致）

`src/gen*_agent.py` 与 `training/train_cql_gru.py` 严格按以下列名读取离线轨迹数据（`data/data_fields.xlsx` 记录原始字段语义）：

| 列名 | 类型 | 说明 |
|---|---|---|
| `""` (`Unnamed: 0`) | `int` | 可选首列：导出 CSV 时的全局行序号 |
| `index` | `int` | 轨迹（episode）编号，用于识别相邻两条样本是否跨越轨迹边界并生成 `done` |
| `obs_1` ~ `obs_5` | `float` | 单帧 **5 维** 工业控制观测状态 |
| `action_1` ~ `action_3` | `float` | 连续 **3 维** 控制动作（策略输出经 `tanh` 映射至 `[-1, 1]`） |
| `reward` | `float` | 单步即时奖励信号 |

> 注：策略网络训练所需的时间序列历史窗（默认 30 帧）与相邻转移终止标志 `dones` 由各 Agent 在加载 CSV 后基于 `index` 边界动态构造，无需在 CSV 中预存 `next_state_*` 或 `done` 列。

## 生成与运行合成数据冒烟测试

```bash
# 生成 12 条模拟轨迹（约 540 步）到 data/demo_data.csv
python scripts/generate_demo_data.py --output data/demo_data.csv --episodes 12 --steps 45

# 运行全量测试（含合成轨迹加载 + TCN 单步反向传播冒烟测试）
pytest tests -v
```
