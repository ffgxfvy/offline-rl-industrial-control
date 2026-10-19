"""Agent 对外接口契约测试。

评测服务只依赖 BaseAgent 的三个方法（reset / act / close），
所以这些约束属于"接口契约"——一旦破坏，提交可能被评测服务拒绝。

注意：三个版本各自带了一份 BaseAgent 定义（比赛提交代码的固有形态），
因此"是否继承 BaseAgent"必须在本模块内比对，不能跨模块导入。
"""
import importlib

import numpy as np
import pytest

OBS_DIM = 5
ACT_DIM = 3

AGENTS = [
    ("gen2_td", "gen2_td_agent", "PolicyAgent"),
    ("gen3_history", "gen3_history_agent", "PolicyAgent"),
    ("gen4_tcn", "gen4_tcn_agent", "TCNPolicyAgent"),
]
IDS = [a[0] for a in AGENTS]
MODCLS = [(a[1], a[2]) for a in AGENTS]


def _make(module_name, class_name, seed=0):
    module = importlib.import_module(module_name)
    return module, getattr(module, class_name)(seed=seed)


def _obs(seed=0, scale=1.0):
    rng = np.random.RandomState(seed)
    return (rng.randn(OBS_DIM) * scale).astype(np.float32)


@pytest.mark.parametrize("module_name,class_name", MODCLS, ids=IDS)
def test_agent_subclasses_its_module_base_agent(module_name, class_name):
    module = importlib.import_module(module_name)
    assert issubclass(getattr(module, class_name), module.BaseAgent)


@pytest.mark.parametrize("module_name,class_name", MODCLS, ids=IDS)
def test_act_returns_flat_action_in_range(module_name, class_name):
    """评测框架期望：输入单帧观测 (5,)，输出动作 (3,) 且落在 [-1, 1]。"""
    _, agent = _make(module_name, class_name)
    action = np.asarray(agent.act(_obs())).reshape(-1)
    assert action.shape == (ACT_DIM,)
    assert np.all(np.isfinite(action))
    assert np.all(action >= -1.0) and np.all(action <= 1.0)


@pytest.mark.parametrize("module_name,class_name", MODCLS, ids=IDS)
def test_act_is_valid_over_a_long_episode(module_name, class_name):
    _, agent = _make(module_name, class_name, seed=1)
    for step in range(60):
        action = np.asarray(agent.act(_obs(seed=step, scale=3.0))).reshape(-1)
        assert action.shape == (ACT_DIM,)
        assert np.all(action >= -1.0) and np.all(action <= 1.0)


@pytest.mark.parametrize("module_name,class_name", MODCLS, ids=IDS)
def test_act_clips_extreme_observations(module_name, class_name):
    """观测放大 100 倍时动作仍须被裁剪到合法区间。"""
    _, agent = _make(module_name, class_name, seed=2)
    for _ in range(5):
        action = np.asarray(agent.act(np.full(OBS_DIM, 100.0, dtype=np.float32))).reshape(-1)
        assert np.all(action >= -1.0) and np.all(action <= 1.0)


@pytest.mark.parametrize("module_name,class_name", MODCLS, ids=IDS)
def test_reset_is_safe_and_repeatable(module_name, class_name):
    _, agent = _make(module_name, class_name, seed=3)
    for step in range(5):
        agent.act(_obs(seed=step))
    agent.reset()
    agent.reset()
    action = np.asarray(agent.act(_obs())).reshape(-1)
    assert action.shape == (ACT_DIM,)


@pytest.mark.parametrize("module_name,class_name", MODCLS, ids=IDS)
def test_same_seed_gives_same_action_sequence(module_name, class_name):
    """相同种子 + 相同观测序列 ⇒ 相同动作序列（评测结果可复现的前提）。"""
    def rollout():
        _, agent = _make(module_name, class_name, seed=42)
        agent.reset()
        return np.stack([np.asarray(agent.act(_obs(seed=i))).reshape(-1) for i in range(8)])

    np.testing.assert_allclose(rollout(), rollout(), atol=1e-6)


@pytest.mark.parametrize("module_name,class_name", MODCLS, ids=IDS)
def test_close_is_idempotent(module_name, class_name):
    _, agent = _make(module_name, class_name, seed=4)
    agent.act(_obs())
    agent.close()
    agent.close()


def test_tcn_receptive_field_covers_sequence_length():
    """TCN 感受野必须覆盖整个输入序列，否则靠前的帧等于被丢弃。"""
    _, agent = _make("gen4_tcn_agent", "TCNPolicyAgent", seed=0)
    rf = agent.policy_net.tcn.receptive_field
    assert rf >= agent.seq_len, f"感受野 {rf} 未覆盖序列长度 {agent.seq_len}"


def test_action_responds_to_observation_changes():
    """防止「策略退化成输出常量」这种静默失效。"""
    _, agent = _make("gen4_tcn_agent", "TCNPolicyAgent", seed=5)
    agent.reset()
    actions = [np.asarray(agent.act(_obs(seed=i, scale=2.0))).reshape(-1) for i in range(10)]
    assert not np.allclose(actions[0], actions[-1], atol=1e-8), "动作对观测不敏感"
