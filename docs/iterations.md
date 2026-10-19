# 三代策略的迭代记录

本文记录同一个工业控制任务上，策略网络的完整迭代过程与每一步的实验动机。所有得分均为 **100 个随机种子（seed 0–99）的评测均值**，原始数据在 `results/*/evaluation_results.csv`，可运行 `python scripts/plot_evolution.py` 复现对比图。

---

## 总览

| 版本 | 提出时间 | 核心改动 | 得分均值 | 状态误差 | 标准差 | 结论 |
|---|---|---|---|---|---|---|
| v1 | 早期 | 单帧观测 + MLP | 未留存评测记录 | — | — | 缺少时序信息，性能受限 |
| v2 | 迭代 II | 帧堆叠 + 手工时序差分特征 | 6.224 | 1.276 | 1.63 | 有提升，但依赖人工特征设计 |
| v3 | 迭代 II | 改为纯帧堆叠、端到端 | 10.862 | 0.960 | 3.17 | 得分涨、**方差崩**（关键转折点） |
| v4 | 迭代 II | **TCN 时序卷积网络** | **30.442** | **0.394** | **2.07** | 质变：更强且更稳 |

---

## 问题背景：为什么时序建模是瓶颈

任务的观测是**单帧 5 维**，但控制系统本质上是部分可观测的（POMDP）——只看当前帧无法判断系统处于哪个阶段（加速、稳态、扰动恢复……），因此**必须把历史观测纳入策略输入**。

但"把历史给网络"有两种截然不同的实现路径：

- **路径 A（人工特征）**：手动设计统计量，把"历史"压缩成固定维度的特征向量；
- **路径 B（端到端 + 显式时序结构）**：把历史序列原样送入带时序归纳偏置的网络，让网络自己学。

v2 走路径 A，v3 走"路径 B 的一半"，v4 才真正走通路径 B。**这个实验最大的收获是：给网络加归纳偏置，比给它堆人工特征更有效。**

---

## v2：帧堆叠 + 手工时序差分特征

**实现**：`src/gen2_td_agent.py` 中的 `EnhancedHistoryManager`

```python
class EnhancedHistoryManager:
    def __init__(self, frame_stack=4, state_dim=5, use_temporal_diff=True):
        ...
    def get_enhanced_state(self):
        stacked = self.get_stacked_state()          # 帧堆叠
        if self.use_temporal_diff:
            mean_features = np.mean(recent_frames, axis=0)   # 历史均值
            std_features  = np.std(recent_frames, axis=0)    # 历史标准差
            temporal_features.extend([mean_features, std_features])
        return np.concatenate([stacked, temporal_features])
```

**动机**：既然网络自己不知道"变化趋势"，那就把趋势和波动显式算给它——一阶统计量（均值）代表近期水平，二阶统计量（标准差）代表波动强度。

**同时引入**：`ReplayBuffer(capacity=100000)` 做经验回放；训练用 `train_conservative_cql` 流程。

**结果**：得分 **6.224**，状态误差 1.276，标准差 1.63。

**诊断**：分数确实动了，但**手工统计量会丢失相位与顺序信息**——均值和标准差对"先升后降"和"先降后升"这两段历史给出完全相同的特征。这限制了性能上限。

---

## v3：改为纯帧堆叠、端到端学习

**实现**：`src/gen3_history_agent.py` 中的 `HistoryManager`

```python
class HistoryManager:
    def __init__(self, frame_stack=4, state_dim=5):
        self.history = deque(maxlen=frame_stack)
    def get_stacked_state(self):
        return np.concatenate(self.history)   # 4 帧 × 5 维 = 20 维
```

**动机**：去掉人工特征，把完整的历史序列交给网络，让它自己决定要提取什么。

**结果**：得分 **10.862**（↑ 74%），状态误差 0.960（↓ 25%）。

**但——标准差从 1.63 飙升到 3.17（接近翻倍）。**

这个现象是整条迭代路线上的**关键转折点**。解读：

| 观察 | 含义 |
|---|---|
| 均值涨、上限涨（最高 19.81） | 网络**有能力**学到比 v2 更好的策略 |
| 最小值也涨（7.34 vs 3.96） | 下限没有崩，不是"训练发散" |
| **标准差接近翻倍** | 网络学到的**表征不稳定**，不同种子下收敛到了差异很大的解 |

**诊断**：问题不在"数据给得够不够"，而在**网络结构缺少时序归纳偏置**。把 4 帧拼成一个 20 维向量喂给全连接网络，等价于告诉网络"这 20 个数字之间没有顺序关系"——网络只能靠数据量硬学顺序，而离线数据集规模有限，于是不同种子下学到的东西差异很大。

**这是一个"数据没变、只换结构就有机会质变"的信号。**

---

## v4：TCN 时序卷积网络

**实现**：`src/gen4_tcn_agent.py`

```python
class Chomp1d(nn.Module):
    """裁掉因果卷积右侧多出来的 padding，保证严格因果（只用历史预测未来）"""
    def __init__(self, chomp_size): ...
    def forward(self, x): return x[:, :, :-self.chomp_size].contiguous()

class TemporalBlock(nn.Module):
    """TCN 基础块：因果卷积 + 残差连接 + Dropout"""
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        ...
    def forward(self, x):
        out = self.net(x)          # 两层因果卷积 + 激活 + Dropout
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)  # 残差

class TCN(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=3, dropout=0.2):
        # num_channels = [64, 128, 64]，逐层膨胀
        ...
    def calculate_receptive_field(self, kernel_size, num_levels):
        """显式计算感受野：1 + 2*(k-1)*Σ d_i"""
        return 1 + 2 * (kernel_size - 1) * (2 ** num_levels - 1)
```

**为什么选 TCN 而不是 RNN/LSTM/GRU？**

| 维度 | TCN | RNN 系 |
|---|---|---|
| 感受野 | **可以精确计算并显式控制**（膨胀卷积） | 隐式，靠门控机制 |
| 并行性 | 卷积可并行，训练更快 | 必须按时间步串行 |
| 梯度 | 残差连接，深层也稳定 | 长序列易梯度消失 |
| 因果性 | `Chomp1d` 保证严格因果 | 天然因果 |

对离线 RL 这种"数据集固定、需要反复跑 100 个种子做评测"的场景，**训练效率与结果稳定性**比单纯表达能力更重要。

**结果（质变）**：得分 **30.442**（相对 v3 ↑ 180%），状态误差 **0.394**（↓ 59%），累积奖励 **−1602.6**（v3 为 −2689.8）。

**更重要的是标准差从 3.17 回落到 2.07**：不仅更强，而且**更稳**——印证了 v3 的诊断是对的，问题确实出在缺少时序归纳偏置。

---

## 结论与收获

1. **归纳偏置 > 人工特征 > 什么都不做。** 同样一份数据、同一个 CQL 算法，仅改变时序编码方式，得分相差 **4.9 倍**。
2. **"方差"是比"均值"更早的预警信号。** v3 的均值看起来是进步（10.86 > 6.22），如果只看均值就会以为方向对了；**是标准差翻倍暴露了"结构不对"这个根本问题**。这个经验直接适用于任何深度学习调参场景。
3. **离线设定下的评测必须用多种子。** 单次运行的分数完全可能是运气（v2 的最高分 16.55 是均值的 2.7 倍）。用 100 个种子把分布画出来，结论才站得住。
4. **保守性（CQL）不是万能的。** `cql_weight` 需要仔细调：太小则 Q 值过估计、策略动作越界；太大则策略过度保守、不敢动。`src/variants/` 里那几个变体就是这个权衡的探索记录。

## 复现

```bash
python scripts/plot_evolution.py     # 复现图与上表统计量
```

三份评测记录：`results/gen2_td/`、`results/gen3_history/`、`results/gen4_tcn/`，每份 100 行，列为 `seed, score, total_reward, avg_state_error, safety_violations`。
