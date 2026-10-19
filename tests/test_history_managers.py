"""历史/序列管理器的纯逻辑测试（不依赖 torch）。"""
import numpy as np
import pytest

from gen2_td_agent import EnhancedHistoryManager
from gen3_history_agent import HistoryManager
from gen4_tcn_agent import SequenceManager

STATE_DIM = 5


def _obs(v):
    return np.full(STATE_DIM, v, dtype=np.float32)


# ---------------- gen2: EnhancedHistoryManager ----------------

def test_enhanced_manager_feature_dim():
    """v2 的核心设计：帧堆叠之外额外拼接历史窗口的均值与标准差特征。"""
    m = EnhancedHistoryManager(frame_stack=4, state_dim=STATE_DIM, use_temporal_diff=True)
    m.reset()
    for i in range(6):
        m.add_observation(_obs(i))
    feat = m.get_enhanced_state()
    assert feat.ndim == 1
    assert feat.shape[0] == m.get_feature_dimension()
    # 4 帧堆叠 + 均值 + 标准差 ⇒ 至少是帧堆叠维度的 2 倍
    assert feat.shape[0] >= 4 * STATE_DIM * 2


def test_enhanced_manager_without_temporal_diff_is_pure_stacking():
    m = EnhancedHistoryManager(frame_stack=4, state_dim=STATE_DIM, use_temporal_diff=False)
    m.reset()
    for i in range(6):
        m.add_observation(_obs(i))
    assert m.get_enhanced_state().shape[0] == 4 * STATE_DIM


def test_enhanced_manager_reset_keeps_dimension():
    m = EnhancedHistoryManager(frame_stack=4, state_dim=STATE_DIM)
    m.reset()
    m.add_observation(_obs(1.0))
    before = m.get_enhanced_state().shape
    m.reset()
    assert m.get_enhanced_state().shape == before


# ---------------- gen3: HistoryManager ----------------

def test_history_manager_stacks_frames():
    """v3：纯帧堆叠，输出维度应恰为 frame_stack * state_dim。"""
    m = HistoryManager(frame_stack=4, state_dim=STATE_DIM)
    m.reset()
    for i in range(3):
        m.add_observation(_obs(i))
    assert m.get_stacked_state().shape == (4 * STATE_DIM,)


def test_history_manager_keeps_latest_frames_old_to_new():
    """deque(maxlen) 语义：只保留最近 frame_stack 帧，顺序为旧 → 新。"""
    m = HistoryManager(frame_stack=3, state_dim=STATE_DIM)
    m.reset()
    for i in range(10):
        m.add_observation(_obs(i))
    stacked = m.get_stacked_state().reshape(3, STATE_DIM)
    assert stacked[0][0] == pytest.approx(7.0)
    assert stacked[-1][0] == pytest.approx(9.0)


# ---------------- gen4: SequenceManager ----------------

def test_sequence_manager_returns_flattened_sequence():
    """注意：get_sequence() 返回的是**扁平**一维向量（seq_len * state_dim），
    TCNPolicyNetwork.forward 内部再 view 回 (batch, seq_len, state_dim)。"""
    m = SequenceManager(seq_len=10, state_dim=STATE_DIM)
    m.reset()
    for i in range(15):
        m.add_observation(_obs(i))
    seq = m.get_sequence()
    assert seq.shape == (10 * STATE_DIM,)
    assert m.get_sequence_dimension() == 10 * STATE_DIM


def test_sequence_manager_is_ordered_old_to_new():
    """TCN 依赖时间顺序：最后一帧必须是最近一次观测。"""
    m = SequenceManager(seq_len=10, state_dim=STATE_DIM)
    m.reset()
    for i in range(15):
        m.add_observation(_obs(i))
    seq = m.get_sequence().reshape(10, STATE_DIM)
    assert seq[0][0] == pytest.approx(5.0)
    assert seq[-1][0] == pytest.approx(14.0)
