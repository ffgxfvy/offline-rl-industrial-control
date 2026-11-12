"""离线数据读取、轨迹切分与 TCN-CQL 单步反向传播冒烟测试。"""
from __future__ import annotations

import math
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import generate_demo_data  # noqa: E402
import gen2_td_agent  # noqa: E402
import gen3_history_agent  # noqa: E402
import gen4_tcn_agent  # noqa: E402


def test_synthetic_csv_loads_across_all_three_generations(tmp_path):
    csv_path = generate_demo_data.generate_synthetic_csv(
        tmp_path / "demo_data.csv", num_episodes=4, steps_per_episode=25, seed=7
    )
    seq_len = 10

    s2, a2, r2, ns2, d2 = gen2_td_agent.load_data_with_enhanced_history(
        str(csv_path), frame_stack=4, use_temporal_diff=True
    )
    s3, a3, r3, ns3, d3 = gen3_history_agent.load_data_with_history(str(csv_path), frame_stack=4)
    s4, a4, r4, ns4, d4 = gen4_tcn_agent.load_data_with_sequences(str(csv_path), seq_len=seq_len)

    # 4 episodes x (25 - 4 + 1) = 88 valid stacked frames (v2 含差分/均值/标准差增强维度=45); 4 x (25 - 10 + 1) = 64 valid TCN windows
    assert s2.shape == (88, 45)
    assert s3.shape == (88, 4 * 5)
    assert s4.shape == (64, seq_len * 5)
    assert ns4.shape == (64, seq_len * 5)
    assert a2.shape == (88, 3) and a3.shape == (88, 3) and a4.shape == (64, 3)
    assert r4.shape == (64,)
    assert d4.shape == (64,)
    # 每一条 episode 的最后一个窗口 done == 1.0，总计 4 个终止边界
    assert int(d4.sum()) == 4


def test_tcncql_single_training_update_step_produces_finite_losses(tmp_path):
    csv_path = generate_demo_data.generate_synthetic_csv(
        tmp_path / "demo_data.csv", num_episodes=4, steps_per_episode=25, seed=11
    )
    seq_len = 10
    s, a, r, ns, d = gen4_tcn_agent.load_data_with_sequences(str(csv_path), seq_len=seq_len)

    trainer = gen4_tcn_agent.TCNCQL(seq_len=seq_len, state_dim=5, action_dim=3)
    batch = (s[:16], a[:16], r[:16], ns[:16], d[:16])
    losses = trainer.update(batch)

    assert set(losses.keys()) == {"total_loss", "td_loss", "cql_loss", "policy_loss"}
    for name, val in losses.items():
        assert isinstance(val, float)
        assert math.isfinite(val), f"{name} is not finite: {val}"

    act = trainer.get_action(s[0])
    assert act.shape == (3,)
    assert np.all(act >= -1.0) and np.all(act <= 1.0)
