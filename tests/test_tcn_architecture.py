"""TCN 结构正确性测试。

这一组测试守的是本项目最关键的技术点：**因果性**。
如果 Chomp1d 被误删或 padding 算错，t 时刻的输出就会"看到未来"，
在线部署时表现为不可解释的性能崩塌，而离线评测往往还能跑出不错的分数。
"""
import pytest
import torch

from gen4_tcn_agent import Chomp1d, TCN, TCNPolicyNetwork, TCNQNetwork, TemporalBlock

KERNEL_SIZE = 3
CHANNELS = [64, 128, 64]
SEQ_LEN = 10
STATE_DIM = 5
ACTION_DIM = 3


# ---------------- Chomp1d ----------------

def test_chomp1d_trims_only_the_right_side():
    x = torch.arange(5, dtype=torch.float32).view(1, 1, 5)
    assert torch.equal(Chomp1d(2)(x), x[:, :, :3])


def test_chomp1d_is_identity_when_zero():
    x = torch.randn(2, 3, 7)
    assert torch.equal(Chomp1d(0)(x), x)


# ---------------- TemporalBlock ----------------

def test_temporal_block_preserves_sequence_length():
    """因果卷积 + Chomp1d 的组合必须让输出长度等于输入长度。"""
    block = TemporalBlock(4, 4, KERNEL_SIZE, 1, dilation=2, padding=4, dropout=0.0)
    x = torch.randn(2, 4, 9)
    assert block(x).shape == (2, 4, 9)


def test_temporal_block_projects_channels_when_dimensions_differ():
    """输入/输出通道不同时应通过 1x1 卷积做残差投影。"""
    block = TemporalBlock(2, 6, KERNEL_SIZE, 1, dilation=1, padding=2, dropout=0.0)
    assert block(torch.randn(1, 2, 8)).shape == (1, 6, 8)


def test_temporal_block_is_causal():
    """核心不变量：扰动 t 时刻的输入，不得改变 t 之前任何时刻的输出。"""
    torch.manual_seed(0)
    block = TemporalBlock(2, 2, KERNEL_SIZE, 1, dilation=1, padding=2, dropout=0.0).eval()

    x = torch.randn(1, 2, 8)
    ref = block(x)

    t = 5
    x_perturbed = x.clone()
    x_perturbed[:, :, t] += 10.0
    out = block(x_perturbed)

    # t 之前完全不受影响（因果）
    assert torch.allclose(ref[:, :, :t], out[:, :, :t], atol=1e-6)
    # t 时刻开始受影响（否则说明扰动没生效，测试本身失效）
    assert not torch.allclose(ref[:, :, t], out[:, :, t], atol=1e-6)


# ---------------- TCN ----------------

def test_tcn_receptive_field_matches_formula():
    """感受野公式：1 + 2*(k-1)*(2^L - 1)，L 为层数。"""
    tcn = TCN(num_inputs=STATE_DIM, num_channels=CHANNELS, kernel_size=KERNEL_SIZE)
    expected = 1 + 2 * (KERNEL_SIZE - 1) * (2 ** len(CHANNELS) - 1)
    assert expected == 29                       # [3, 3, 3] 通道、kernel=3 时
    assert tcn.receptive_field == expected
    assert tcn.receptive_field == tcn.calculate_receptive_field(KERNEL_SIZE, len(CHANNELS))


def test_tcn_dilation_grows_exponentially():
    """通道数每加一层，膨胀系数翻倍 ⇒ 感受野快速增长。"""
    rf2 = TCN(num_inputs=STATE_DIM, num_channels=[64, 64], kernel_size=3).receptive_field
    rf3 = TCN(num_inputs=STATE_DIM, num_channels=[64, 64, 64], kernel_size=3).receptive_field
    assert rf2 == 1 + 2 * 2 * (2 ** 2 - 1) == 13
    assert rf3 == 29


def test_tcn_outputs_pooled_feature_vector():
    """TCN.forward 内部做全局平均池化，输出 [batch, channels[-1]]。"""
    tcn = TCN(num_inputs=STATE_DIM, num_channels=CHANNELS, kernel_size=KERNEL_SIZE).eval()
    out = tcn(torch.randn(4, SEQ_LEN, STATE_DIM))
    assert out.shape == (4, CHANNELS[-1])


# ---------------- 策略 / Q 网络 ----------------

def test_policy_network_accepts_flattened_sequence():
    """接口约定：输入是扁平向量 [batch, seq_len * state_dim]，输出 [batch, action_dim]。"""
    net = TCNPolicyNetwork(seq_len=SEQ_LEN, state_dim=STATE_DIM, action_dim=ACTION_DIM).eval()
    out = net(torch.randn(3, SEQ_LEN * STATE_DIM))
    assert out.shape == (3, ACTION_DIM)


def test_q_network_returns_scalar_per_sample():
    net = TCNQNetwork(seq_len=SEQ_LEN, state_dim=STATE_DIM, action_dim=ACTION_DIM).eval()
    q = net(torch.randn(3, SEQ_LEN * STATE_DIM), torch.randn(3, ACTION_DIM))
    assert q.shape == (3, 1)


def test_policy_network_rejects_wrong_input_width():
    net = TCNPolicyNetwork(seq_len=SEQ_LEN, state_dim=STATE_DIM, action_dim=ACTION_DIM).eval()
    with pytest.raises(RuntimeError):
        net(torch.randn(2, SEQ_LEN * STATE_DIM + 1))
