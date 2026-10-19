import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
import os
from collections import deque
import random
from abc import ABC, abstractmethod

# 设置随机种子
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# ==================== 基础Agent类 ====================
class BaseAgent(ABC):
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

# ==================== TCN时序卷积网络 ====================
class TemporalBlock(nn.Module):
    """TCN的基础块，包含因果卷积和残差连接"""
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = nn.Conv1d(n_inputs, n_outputs, kernel_size,
                               stride=stride, padding=padding, dilation=dilation)
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(n_outputs, n_outputs, kernel_size,
                               stride=stride, padding=padding, dilation=dilation)
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(
            self.conv1, self.chomp1, self.relu1, self.dropout1,
            self.conv2, self.chomp2, self.relu2, self.dropout2
        )

        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)

class Chomp1d(nn.Module):
    """确保因果性的填充截断"""
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        # 防御性修复（详见 CHANGELOG.md）
        # chomp_size == 0 时，Python 中 -0 == 0，x[:, :, :-0] 等价于 x[:, :, :0]，
        # 会返回空张量而不是恒等映射。当 padding = (kernel_size-1)*dilation > 0
        # （本项目 kernel_size=3）时不会触发；显式短路可避免将 kernel_size 调成 1 时踩坑。
        if self.chomp_size == 0:
            return x
        return x[:, :, :-self.chomp_size]

class TCN(nn.Module):
    """时序卷积网络主架构"""
    def __init__(self, num_inputs, num_channels, kernel_size=3, dropout=0.2):
        super(TCN, self).__init__()
        layers = []
        num_levels = len(num_channels)
        
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, 
                                   stride=1, dilation=dilation_size,
                                   padding=(kernel_size-1) * dilation_size, 
                                   dropout=dropout)]

        self.network = nn.Sequential(*layers)
        self.receptive_field = self.calculate_receptive_field(kernel_size, num_levels)
        
    def calculate_receptive_field(self, kernel_size, num_levels):
        """计算感受野大小"""
        return 1 + 2 * (kernel_size - 1) * (2 ** num_levels - 1)
    
    def forward(self, x):
        # x: [batch, seq_len, features] -> [batch, features, seq_len]
        x = x.transpose(1, 2)
        output = self.network(x)
        # 全局平均池化
        output = F.adaptive_avg_pool1d(output, 1).squeeze(-1)
        return output

class TCNPolicyNetwork(nn.Module):
    """基于TCN的策略网络"""
    def __init__(self, seq_len, state_dim, action_dim, tcn_channels=[64, 128, 64], hidden_dim=64):
        super(TCNPolicyNetwork, self).__init__()
        self.seq_len = seq_len
        self.state_dim = state_dim
        
        # TCN部分
        self.tcn = TCN(num_inputs=state_dim, num_channels=tcn_channels)
        
        # 计算TCN输出维度
        tcn_output_dim = tcn_channels[-1]
        
        # 策略头
        self.policy_head = nn.Sequential(
            nn.Linear(tcn_output_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
        
        # 打印网络信息
        print(f"TCN策略网络: 序列长度={seq_len}, 状态维度={state_dim}")
        print(f"TCN感受野: {self.tcn.receptive_field}")
        print(f"TCN输出维度: {tcn_output_dim}")
        
    def forward(self, x):
        # x: [batch, seq_len * state_dim] -> [batch, seq_len, state_dim]
        batch_size = x.shape[0]
        x = x.view(batch_size, self.seq_len, self.state_dim)
        
        # TCN处理
        tcn_features = self.tcn(x)
        
        # 策略输出
        action = self.policy_head(tcn_features)
        return action

class TCNQNetwork(nn.Module):
    """基于TCN的Q网络"""
    def __init__(self, seq_len, state_dim, action_dim, tcn_channels=[64, 128, 64], hidden_dim=64):
        super(TCNQNetwork, self).__init__()
        self.seq_len = seq_len
        self.state_dim = state_dim
        
        # TCN部分（只处理状态序列）
        self.tcn = TCN(num_inputs=state_dim, num_channels=tcn_channels)
        tcn_output_dim = tcn_channels[-1]
        
        # Q值头（状态特征 + 动作）
        self.q_head = nn.Sequential(
            nn.Linear(tcn_output_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, x, action):
        # x: [batch, seq_len * state_dim] -> [batch, seq_len, state_dim]
        batch_size = x.shape[0]
        x = x.view(batch_size, self.seq_len, self.state_dim)
        
        # TCN处理状态序列
        state_features = self.tcn(x)
        
        # 合并状态特征和动作
        q_input = torch.cat([state_features, action], dim=1)
        q_value = self.q_head(q_input)
        return q_value

# ==================== 历史序列管理器 ====================
class SequenceManager:
    """管理历史序列数据"""
    
    def __init__(self, seq_len=10, state_dim=5):
        self.seq_len = seq_len
        self.state_dim = state_dim
        self.history = deque(maxlen=seq_len)
        
    def reset(self):
        """重置历史序列"""
        self.history.clear()
        # 用零向量初始化序列
        for _ in range(self.seq_len):
            self.history.append(np.zeros(self.state_dim, dtype=np.float32))
    
    def add_observation(self, obs):
        """添加新的观测"""
        self.history.append(obs.copy())
    
    def get_sequence(self):
        """获取完整序列"""
        if len(self.history) < self.seq_len:
            # 如果序列不足，用最早的观测填充
            while len(self.history) < self.seq_len:
                self.history.appendleft(self.history[0] if self.history else np.zeros(self.state_dim))
        
        # 将序列堆叠成一维向量
        sequence = np.concatenate(list(self.history), axis=0)
        return sequence
    
    def get_sequence_dimension(self):
        """获取序列维度"""
        return self.seq_len * self.state_dim

# ==================== 经验回放缓冲区 ====================
class ReplayBuffer:
    def __init__(self, capacity=100000):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        """存储经验"""
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        """随机采样批次"""
        if len(self.buffer) < batch_size:
            return None
        
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        return (
            np.array(states, dtype=np.float32),
            np.array(actions, dtype=np.float32),
            np.array(rewards, dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(dones, dtype=np.float32)
        )
    
    def __len__(self):
        return len(self.buffer)

# ==================== 序列数据加载 ====================
def load_data_with_sequences(file_path, seq_len=10):
    """加载离线数据并生成序列数据"""
    print(f"加载数据并生成序列数据 (序列长度: {seq_len})...")
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"找不到数据文件: {file_path}")
    
    data = pd.read_csv(file_path)
    
    # 提取状态、动作、奖励
    states = data[['obs_1', 'obs_2', 'obs_3', 'obs_4', 'obs_5']].values.astype(np.float32)
    actions = data[['action_1', 'action_2', 'action_3']].values.astype(np.float32)
    rewards = data['reward'].values.astype(np.float32)
    
    # 处理下一个状态和完成标志
    next_states = np.zeros_like(states)
    dones = np.zeros(len(states), dtype=np.float32)
    
    for i in range(len(states)-1):
        next_states[i] = states[i+1]
        if data['index'].iloc[i] != data['index'].iloc[i+1]:
            dones[i] = 1.0
    next_states[-1] = states[-1]
    dones[-1] = 1.0
    
    # 生成序列数据
    sequences = []
    next_sequences = []
    valid_indices = []
    
    # 按episode分组处理
    episode_indices = data['index'].unique()
    
    for ep_idx in episode_indices:
        ep_data = data[data['index'] == ep_idx]
        ep_states = states[data['index'] == ep_idx]
        
        if len(ep_states) >= seq_len:
            # 为每个有效位置生成序列
            for i in range(seq_len - 1, len(ep_states)):
                # 当前序列
                current_seq = ep_states[i - seq_len + 1: i + 1].flatten()
                
                # 下一个序列
                if i + 1 < len(ep_states):
                    next_seq = ep_states[i - seq_len + 2: i + 2].flatten()
                else:
                    # 如果是episode末尾，用当前序列填充
                    next_seq = current_seq
                
                # 找到原始数据中的索引
                original_idx = ep_data.index[i]
                
                sequences.append(current_seq)
                next_sequences.append(next_seq)
                valid_indices.append(original_idx)
    
    # 筛选有效数据
    sequences_array = np.array(sequences, dtype=np.float32)
    next_sequences_array = np.array(next_sequences, dtype=np.float32)
    actions_array = actions[valid_indices]
    rewards_array = rewards[valid_indices]
    dones_array = dones[valid_indices]
    
    print(f"序列数据处理完成: {len(sequences_array)} 条有效数据")
    print(f"序列维度: {sequences_array.shape[1]} ({seq_len} × 5)")
    return sequences_array, actions_array, rewards_array, next_sequences_array, dones_array

# ==================== TCN-CQL算法 ====================
class TCNCQL:
    def __init__(self, seq_len, state_dim, action_dim, tcn_channels=[64, 128, 64]):
        self.seq_len = seq_len
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.sequence_dim = seq_len * state_dim
        
        print(f"TCN-CQL配置: 序列长度={seq_len}, 状态维度={state_dim}, 序列维度={self.sequence_dim}")
        
        # TCN网络
        self.q_net1 = TCNQNetwork(seq_len, state_dim, action_dim, tcn_channels)
        self.q_net2 = TCNQNetwork(seq_len, state_dim, action_dim, tcn_channels)
        self.policy_net = TCNPolicyNetwork(seq_len, state_dim, action_dim, tcn_channels)
        
        # 目标网络
        self.target_q_net1 = TCNQNetwork(seq_len, state_dim, action_dim, tcn_channels)
        self.target_q_net2 = TCNQNetwork(seq_len, state_dim, action_dim, tcn_channels)
        
        # 优化器
        self.q_optimizer = optim.Adam(list(self.q_net1.parameters()) + list(self.q_net2.parameters()), 
                                     lr=0.0003)
        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=0.0003)
        
        # 超参数
        self.gamma = 0.99
        self.cql_weight = 0.5
        self.tau = 0.005
        
        # 初始化目标网络
        self.update_target_networks(tau=1.0)
        
        # 验证网络
        self._validate_network_dims()
    
    def _validate_network_dims(self):
        """验证网络维度"""
        test_sequence = torch.randn(1, self.sequence_dim)
        test_action = torch.randn(1, self.action_dim)
        
        try:
            policy_out = self.policy_net(test_sequence)
            q_out = self.q_net1(test_sequence, test_action)
            
            print(f"✅ TCN网络维度验证通过:")
            print(f"  策略网络: {self.sequence_dim} -> {policy_out.shape[1]}")
            print(f"  Q网络: {self.sequence_dim} + {self.action_dim} -> {q_out.shape[1]}")
        except Exception as e:
            print(f"❌ TCN网络维度验证失败: {e}")
            raise
    
    def update_target_networks(self, tau=None):
        """软更新目标网络"""
        tau = tau or self.tau
        for target_param, param in zip(self.target_q_net1.parameters(), self.q_net1.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
        for target_param, param in zip(self.target_q_net2.parameters(), self.q_net2.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
    
    def get_action(self, sequence):
        """根据序列获取动作"""
        if len(sequence) != self.sequence_dim:
            raise ValueError(f"序列维度应为{self.sequence_dim}，但得到{len(sequence)}")
        
        sequence_tensor = torch.FloatTensor(sequence).unsqueeze(0)
        with torch.no_grad():
            raw_action = self.policy_net(sequence_tensor)
            action = torch.tanh(raw_action)
        return action.numpy()[0]
    
    def compute_cql_loss(self, sequences, current_q1, current_q2):
        """计算CQL损失"""
        batch_size = sequences.shape[0]
        
        with torch.no_grad():
            policy_actions = torch.tanh(self.policy_net(sequences))
        
        random_actions = torch.rand(batch_size, self.action_dim) * 2 - 1
        
        policy_q1 = self.q_net1(sequences, policy_actions)
        policy_q2 = self.q_net2(sequences, policy_actions)
        
        random_q1 = self.q_net1(sequences, random_actions)
        random_q2 = self.q_net2(sequences, random_actions)
        
        cql_loss1 = torch.logsumexp(random_q1, dim=0) - current_q1.mean()
        cql_loss2 = torch.logsumexp(random_q2, dim=0) - current_q2.mean()
        
        cql_loss = (cql_loss1 + cql_loss2) / 2
        return cql_loss
    
    def update(self, batch):
        """更新网络参数"""
        sequences, actions, rewards, next_sequences, dones = batch
        
        # 转换为tensor
        sequences = torch.FloatTensor(sequences)
        actions = torch.FloatTensor(actions)
        rewards = torch.FloatTensor(rewards).unsqueeze(1)
        next_sequences = torch.FloatTensor(next_sequences)
        dones = torch.FloatTensor(dones).unsqueeze(1)
        
        batch_size = sequences.shape[0]
        
        # 计算目标Q值
        with torch.no_grad():
            next_actions = torch.tanh(self.policy_net(next_sequences))
            next_q1 = self.target_q_net1(next_sequences, next_actions)
            next_q2 = self.target_q_net2(next_sequences, next_actions)
            next_q = torch.min(next_q1, next_q2)
            target_q = rewards + self.gamma * (1 - dones) * next_q
        
        # 当前Q值
        current_q1 = self.q_net1(sequences, actions)
        current_q2 = self.q_net2(sequences, actions)
        
        # TD损失
        td_loss1 = F.mse_loss(current_q1, target_q)
        td_loss2 = F.mse_loss(current_q2, target_q)
        td_loss = (td_loss1 + td_loss2) / 2
        
        # CQL正则化损失
        cql_loss = self.compute_cql_loss(sequences, current_q1, current_q2)
        
        # 总Q损失
        q_loss = td_loss + self.cql_weight * cql_loss
        
        # 更新Q网络
        self.q_optimizer.zero_grad()
        q_loss.backward()
        torch.nn.utils.clip_grad_norm_(list(self.q_net1.parameters()) + list(self.q_net2.parameters()), 
                                      max_norm=1.0)
        self.q_optimizer.step()
        
        # 更新策略网络
        policy_actions = torch.tanh(self.policy_net(sequences))
        policy_q1 = self.q_net1(sequences, policy_actions)
        policy_q2 = self.q_net2(sequences, policy_actions)
        policy_q = torch.min(policy_q1, policy_q2)
        policy_loss = -policy_q.mean()
        
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)
        self.policy_optimizer.step()
        
        # 更新目标网络
        self.update_target_networks()
        
        return {
            'total_loss': q_loss.item(),
            'td_loss': td_loss.item(),
            'cql_loss': cql_loss.item(),
            'policy_loss': policy_loss.item()
        }

# ==================== TCN策略Agent ====================
class TCNPolicyAgent(BaseAgent):
    def __init__(self, seed: int = None, seq_len=10):
        super().__init__(seed)
        
        self.raw_state_dim = 5
        self.action_dim = 3
        self.seq_len = seq_len
        self.sequence_dim = seq_len * self.raw_state_dim
        
        # 序列管理器
        self.sequence_manager = SequenceManager(seq_len=seq_len, state_dim=self.raw_state_dim)
        
        # TCN策略网络
        self.policy_net = TCNPolicyNetwork(seq_len, self.raw_state_dim, self.action_dim)
        
        # 自动查找模型文件
        current_dir = os.path.dirname(os.path.abspath(__file__))
        print(f"当前目录: {current_dir}")
        
        possible_paths = [
            os.path.join(current_dir, 'model.pth'),
            os.path.join(current_dir, 'agent', 'model.pth'),
            os.path.join(current_dir, '..', 'model.pth'),
            os.path.join(current_dir, '..', 'agent', 'model.pth'),
        ]
        
        model_loaded = False
        for model_path in possible_paths:
            if os.path.exists(model_path):
                try:
                    self.policy_net.load_state_dict(torch.load(model_path, map_location='cpu'))
                    print(f"✅ 已加载TCN预训练模型: {model_path}")
                    print(f"   序列长度: {seq_len}")
                    model_loaded = True
                    break
                except Exception as e:
                    print(f"❌ 加载TCN模型失败 {model_path}: {e}")
        
        if not model_loaded:
            print("⚠️  未找到TCN预训练模型，将使用随机初始化的策略")
        
        self.policy_net.to(self.device)
        self.policy_net.eval()
        
        print(f"TCN策略Agent初始化完成: 序列维度={self.sequence_dim}, 输出维度={self.action_dim}")
    
    def reset(self) -> None:
        """重置时同时重置序列管理器"""
        super().reset()
        self.sequence_manager.reset()
    
    def get_action(self, obs: np.ndarray) -> np.ndarray:
        """使用TCN处理序列选择动作"""
        if len(obs) != self.raw_state_dim:
            raise ValueError(f"观测维度应为{self.raw_state_dim}，但得到{len(obs)}")
        
        # 添加观测到序列
        self.sequence_manager.add_observation(obs)
        
        # 获取完整序列
        sequence = self.sequence_manager.get_sequence()
        
        # 验证序列维度
        if len(sequence) != self.sequence_dim:
            raise ValueError(f"序列维度应为{self.sequence_dim}，但得到{len(sequence)}")
        
        sequence_tensor = torch.FloatTensor(sequence).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            raw_action = self.policy_net(sequence_tensor)
            action = torch.tanh(raw_action)
        
        action_np = action.cpu().numpy()[0]
        return action_np

# ==================== 主训练函数 ====================
def main():
    print("开始TCN-CQL离线强化学习训练")
    
    # TCN配置参数
    SEQ_LEN = 10  # 序列长度
    TCN_CHANNELS = [64, 128, 64]  # TCN通道数
    
    # 获取当前脚本目录
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 数据文件路径
    data_path = os.path.join(current_dir, '..', 'data', 'data.csv')
    data_path = os.path.normpath(data_path)
    print(f"数据路径: {data_path}")
    
    # 加载序列数据
    sequences, actions, rewards, next_sequences, dones = load_data_with_sequences(
        data_path, seq_len=SEQ_LEN
    )
    
    # 初始化经验回放缓冲区
    replay_buffer = ReplayBuffer(capacity=100000)
    for i in range(len(sequences)):
        replay_buffer.push(sequences[i], actions[i], rewards[i], next_sequences[i], dones[i])
    print(f"经验回放缓冲区初始化完成: {len(replay_buffer)} 条经验")
    
    # 初始化TCN-CQL算法
    agent = TCNCQL(
        seq_len=SEQ_LEN,
        state_dim=5,
        action_dim=3,
        tcn_channels=TCN_CHANNELS
    )
    
    # 训练参数
    batch_size = 256
    num_epochs = 500
    
    # 记录损失
    losses = {
        'total_loss': [],
        'td_loss': [],
        'cql_loss': [],
        'policy_loss': []
    }
    
    print("TCN训练中...")
    for epoch in range(num_epochs):
        batch = replay_buffer.sample(batch_size)
        if batch is None:
            continue
        
        loss_dict = agent.update(batch)
        
        for key in losses.keys():
            losses[key].append(loss_dict[key])
        
        if epoch % 50 == 0:
            print(f"Epoch {epoch}: "
                  f"Total = {loss_dict['total_loss']:.3f}, "
                  f"TD = {loss_dict['td_loss']:.3f}, "
                  f"CQL = {loss_dict['cql_loss']:.3f}, "
                  f"Policy = {loss_dict['policy_loss']:.3f}")
            
            if epoch % 100 == 0:
                # 测试零序列
                test_sequence = np.zeros(agent.sequence_dim)
                test_action = agent.get_action(test_sequence)
                print(f"  测试动作（零序列）: {test_action}")
    
    print("TCN训练完成")
    
    # 绘制训练损失图
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 2, 1)
    plt.plot(losses['total_loss'])
    plt.title('TCN Total Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 2, 2)
    plt.plot(losses['td_loss'])
    plt.title('TCN TD Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 2, 3)
    plt.plot(losses['cql_loss'])
    plt.title('TCN CQL Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 2, 4)
    plt.plot(losses['policy_loss'])
    plt.title('TCN Policy Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 保存损失图
    plot_path = os.path.join(current_dir, f'tcn_training_losses_seq{SEQ_LEN}.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"TCN训练损失图已保存: {plot_path}")
    plt.show()
    
    # 保存模型
    model_save_path = os.path.join(current_dir, 'model.pth')
    torch.save(agent.policy_net.state_dict(), model_save_path)
    print(f"✅ TCN模型已保存: {model_save_path}")
    print(f"✅ 总训练轮数: {num_epochs}")
    print(f"✅ 序列长度: {SEQ_LEN}")
    print(f"✅ TCN通道: {TCN_CHANNELS}")

if __name__ == "__main__":
    main()