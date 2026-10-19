import os
import random
from abc import ABC, abstractmethod

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class BaseAgent(ABC):
    """
    基类，定义了所有 Agent 的统一接口／行为：
      - 随机种子管理
      - 设备（CPU/CUDA）选择
      - 观测与动作的历史缓存
      - 动作产生的统一流程（reshape → 前向推理 → clip → 缓存 → 返回）
    """

    def __init__(self, seed: int = None):
        if seed is not None:
            self.seed(seed)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.obs_history = []
        self.act_history = []

    def seed(self, seed: int = 123) -> None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

    def reset(self) -> None:
        self.obs_history.clear()
        self.act_history.clear()

    def act(self, obs: np.ndarray) -> np.ndarray:
        # NOTE: 这里的obs已经是5维的单帧观测
        # reshape(-1)是为了兼容旧代码，但实际上obs已经是(5,)
        obs = obs.reshape(-1).astype(np.float32)
        action = self.get_action(obs)
        action = np.clip(action, -1.0, 1.0).reshape(-1).astype(np.float32)
        
        self.obs_history.append(obs)
        self.act_history.append(action)

        return action

    @abstractmethod
    def get_action(self, obs: np.ndarray) -> np.ndarray:
        ...

    def close(self) -> None:
        pass

# ==============================================================================
# MODIFIED: 删除了原有的Mlp类，替换为新的Actor网络结构
# ==============================================================================
class Actor(nn.Module):
    """
    策略网络，使用GRU处理时序信息。
    结构与训练脚本中的Actor保持一致。
    """
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super(Actor, self).__init__()
        # GRU层用于处理序列信息
        # input_dim: 单帧观测的维度 (5)
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)
        # 两个全连接层
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, state_history: torch.Tensor) -> torch.Tensor:
        # state_history 的期望形状: (batch_size, seq_len, input_dim)
        gru_out, _ = self.gru(state_history)
        # 取序列的最后一个输出
        last_step_out = gru_out[:, -1, :]
        
        x = F.relu(self.fc1(last_step_out))
        # 使用tanh将动作限制在[-1, 1]
        action = torch.tanh(self.fc2(x))
        return action

# ==============================================================================
# MODIFIED: PolicyAgent类被修改以使用新的CQL+GRU模型
# ==============================================================================
class PolicyAgent(BaseAgent):
    """
    基于预训练的CQL+GRU模型的策略Agent。
    维护最近30帧的观测历史，将它们作为序列输入网络，
    输出当前动作。
    """
    HIST_LEN = 30  # 使用的历史观测帧数
    OBS_DIM = 5    # 单帧观测维度

    def __init__(self):
        super().__init__()

        # 定位模型文件（与当前脚本同目录）
        model_path = os.path.join(os.path.dirname(__file__), "model.pth")

        # 网络结构与保存的权重需保持一致
        self.policy = Actor(input_dim=self.OBS_DIM, hidden_dim=256, output_dim=3)
        
        # 加载模型权重，强制在CPU上加载以保证通用性
        state_dict = torch.load(model_path, map_location="cpu")
        self.policy.load_state_dict(state_dict)

        self.policy.to(self.device)
        self.policy.eval() # 切换到评估模式

    def get_action(self, obs: np.ndarray) -> np.ndarray:
        """
        根据当前观测和历史构造网络输入并进行前向推理：
          1. 将历史obs + 当前obs合并成列表
          2. 不足30帧时用最早那帧进行填充；超过30帧则取最新30帧
          3. 将帧序列构造成 (1, HIST_LEN, OBS_DIM) 的形状
          4. 转为Tensor，移动到device
          5. 无梯度前向推理，输出动作
          6. 转回Numpy数组并返回
        """
        seq = self.obs_history + [obs]
        
        if len(seq) < self.HIST_LEN:
            # 不足时用第一帧填充到指定长度
            padding = [seq[0]] * (self.HIST_LEN - len(seq))
            state_history = np.array(padding + seq, dtype=np.float32)
        else:
            # 超过时取最新的帧
            state_history = np.array(seq[-self.HIST_LEN:], dtype=np.float32)

        # 调整形状为 (1, seq_len, obs_dim) 以匹配GRU输入
        state_history = state_history.reshape(1, self.HIST_LEN, self.OBS_DIM)
        
        # 转换为Tensor并移动到设备
        x = torch.from_numpy(state_history).to(self.device)
        
        # 无梯度进行前向推理
        with torch.no_grad():
            action = self.policy(x)

        # 将结果从Tensor转回Numpy数组
        return action.squeeze(0).cpu().numpy()


if __name__ == "__main__":
    # 用于本地简单测试的示例代码
    agent = PolicyAgent()
    # 构造一个假的单帧观测，形状为(5,)
    dummy_obs = np.random.rand(5).astype(np.float32)

    # 模拟两次调用
    a1 = agent.act(dummy_obs)
    a2 = agent.act(dummy_obs * 2)

    print("Action at step 1:", a1)
    print("Action at step 2:", a2)
    print("obs_history length:", len(agent.obs_history))

    agent.reset()
    agent.close()
    print("After reset, obs_history length:", len(agent.obs_history))
