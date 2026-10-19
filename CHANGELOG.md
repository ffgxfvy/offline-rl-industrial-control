# Changelog

本仓库的代码来自工业控制策略评测竞赛的提交版本，此处记录为作品集整理时所做的**工程化改动**。算法逻辑与评测结果未作任何修改。

## [Portfolio Release] - 2026-10

### 修复

- **`src/gen4_tcn_agent.py` — `Chomp1d` 在 `chomp_size == 0` 时返回空张量**
  - 原实现：`return x[:, :, :-self.chomp_size]`
  - 问题：Python 中 `-0 == 0`，因此 `x[:, :, :-0]` 实际是 `x[:, :, :0]`，返回长度为 0 的张量，而不是预期的恒等映射。
  - 影响范围：**当时未触发**。`TemporalBlock` 的 `padding = (kernel_size - 1) * dilation_size`，在 `kernel_size = 3`、`dilation ≥ 1` 时恒大于 0。该缺陷只在把 `kernel_size` 调为 1（即去掉卷积本身的时间跨度）时才暴露。
  - 修复：`chomp_size == 0` 时直接短路返回输入。对 `kernel_size > 1` 的路径**行为完全不变**，已训练权重不受影响。
  - 由 `tests/test_tcn_architecture.py::test_chomp1d_is_identity_when_zero` 覆盖。

### 新增

- `tests/` —— 64 个 pytest 测试，覆盖三块：
  - `test_tcn_architecture.py`：TCN 结构正确性，重点是**因果性不变量**（扰动 t 时刻输入不得影响 t 之前的输出）、感受野公式、`Chomp1d` 边界。
  - `test_history_managers.py`：三代历史/序列管理器的维度与顺序语义。
  - `test_agent_interface.py`：`BaseAgent` 接口契约（动作维度、`[-1,1]` 裁剪、多种子可复现、极端观测鲁棒性、`reset`/`close` 幂等）。
  - `test_results.py`：`results/*/evaluation_results.csv` 的数据契约（100 seeds、列名、无缺失、三代得分单调提升）。
  - `test_plot_backend.py`：绘图脚本可运行且输出与 README 数据一致。
- `scripts/plot_evolution.py` 增加 `--outdir` 参数（便于测试与 CI，不污染 `assets/`）。
- `.github/workflows/ci.yml` —— 推送与 PR 时自动跑测试。
- `requirements-dev.txt`、`pyproject.toml`（pytest 配置）。

### 未改动

- 四个 agent 实现（`src/gen2_td_agent.py`、`src/gen3_history_agent.py`、`src/gen4_tcn_agent.py`、`src/variants/*`）的算法逻辑；
- `results/` 下的三份评测记录；
- `assets/` 下由原训练过程产出的损失曲线。
