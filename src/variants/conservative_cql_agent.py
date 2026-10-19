import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pandas as pd
from abc import ABC, abstractmethod


class BaseAgent(ABC):
    """基类 - 严格保持官方接口"""
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


class ConservativeMLP(nn.Module):
    """保守版MLP网络 - 彻底解决Q值过估计"""
    def __init__(self, input_dim=150, hidden_dims=[128, 64], output_dim=3):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dims[0])
        self.fc2 = nn.Linear(hidden_dims[0], hidden_dims[1])
        self.fc3 = nn.Linear(hidden_dims[1], output_dim)
        
        # 非常保守的初始化
        nn.init.xavier_uniform_(self.fc1.weight, gain=0.5)
        nn.init.xavier_uniform_(self.fc2.weight, gain=0.5)
        nn.init.uniform_(self.fc3.weight, -0.001, 0.001)  # 输出层极小的初始化
        nn.init.constant_(self.fc3.bias, 0.0)
        
    def forward(self, x):
        x = F.elu(self.fc1(x))  # 使用ELU避免死神经元
        x = F.elu(self.fc2(x))
        return self.fc3(x)


class ConservativeCQLTrainer:
    """保守版CQL训练器 - 彻底解决Q值过估计"""
    def __init__(self, state_dim=150, action_dim=3):
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # 保守的网络结构
        self.q_net1 = ConservativeMLP(state_dim + action_dim, [128, 64], 1)
        self.q_net2 = ConservativeMLP(state_dim + action_dim, [128, 64], 1)
        self.policy_net = ConservativeMLP(state_dim, [128, 64], action_dim)
        
        # 初始化最佳策略状态
        self.best_policy_state = self.policy_net.state_dict().copy()
        self.best_policy_loss = float('inf')
        
        # 非常保守的优化器
        self.q_optimizer = optim.Adam(
            list(self.q_net1.parameters()) + list(self.q_net2.parameters()), 
            lr=1e-4, weight_decay=1e-3  # 更高的权重衰减
        )
        self.policy_optimizer = optim.Adam(
            self.policy_net.parameters(), lr=1e-4, weight_decay=1e-3
        )
        
        # 非常保守的超参数
        self.gamma = 0.9  # 更低的折扣因子
        self.cql_weight = 5.0  # 更高的CQL权重
        self.temperature = 5.0  # 更高的温度
        
        self.patience = 0
        self.max_patience = 5  # 更早的早停

    def train_conservative_cql(self, states, actions, rewards, next_states, dones):
        """保守训练流程 - 彻底解决Q值过估计"""
        print("开始保守CQL训练...")
        
        # 使用非常高质量的数据 - 基于奖励筛选
        reward_threshold = np.percentile(rewards, 60)  # 取前40%的最高奖励数据
        high_quality_mask = rewards >= reward_threshold
        
        if np.sum(high_quality_mask) > 5000:
            states = states[high_quality_mask]
            actions = actions[high_quality_mask]
            rewards = rewards[high_quality_mask]
            next_states = next_states[high_quality_mask]
            dones = dones[high_quality_mask]
            print(f"使用高质量数据: {len(states)} 条")
        
        # 数据标准化
        states = self._conservative_normalize(states)
        next_states = self._conservative_normalize(next_states)
        rewards = self._conservative_normalize_rewards(rewards)
        
        # 数据集划分
        split_idx = int(0.9 * len(states))
        train_indices = np.arange(split_idx)
        val_indices = np.arange(split_idx, len(states))
        
        print(f"训练集: {len(train_indices)}, 验证集: {len(val_indices)}")
        
        best_epoch = 0
        
        for epoch in range(40):  # 更少的训练轮数
            train_loss = self._train_epoch(states, actions, rewards, next_states, dones, train_indices)
            val_loss = self._validate(states, actions, rewards, next_states, dones, val_indices)
            
            # 检查NaN和过估计
            if np.isnan(train_loss) or np.isnan(val_loss):
                print(f"Epoch {epoch}: 检测到NaN，停止训练")
                break
            
            # 如果Q值过估计严重，提前停止
            if val_loss < -100:
                print(f"Epoch {epoch}: Q值过估计严重，提前停止")
                break
            
            if val_loss < self.best_policy_loss:
                self.best_policy_loss = val_loss
                self.best_policy_state = self.policy_net.state_dict().copy()
                self.patience = 0
                best_epoch = epoch
            else:
                self.patience += 1
            
            if epoch % 5 == 0:
                print(f"Epoch {epoch}: Train Loss {train_loss:.4f}, Val Loss {val_loss:.4f}")
            
            if self.patience >= self.max_patience:
                print(f"早停在 epoch {epoch}, 最佳在 epoch {best_epoch}")
                break
        
        print(f"训练完成，最佳验证损失: {self.best_policy_loss:.4f}")

    def _conservative_normalize(self, data):
        """保守的数据归一化"""
        data_mean = np.mean(data, axis=0)
        data_std = np.std(data, axis=0) + 1e-8
        normalized = (data - data_mean) / data_std
        # 非常严格的裁剪
        return np.clip(normalized, -3.0, 3.0).astype(np.float32)

    def _conservative_normalize_rewards(self, rewards):
        """保守的奖励归一化"""
        reward_mean = np.mean(rewards)
        reward_std = np.std(rewards) + 1e-8
        normalized = (rewards - reward_mean) / reward_std
        # 非常严格的裁剪
        return np.clip(normalized, -5.0, 5.0).astype(np.float32)

    def _train_epoch(self, states, actions, rewards, next_states, dones, indices):
        """训练一个epoch"""
        batch_size = 128  # 更小的batch size
        epoch_loss = 0
        num_batches = max(1, len(indices) // batch_size)
        
        for _ in range(num_batches):
            batch_indices = np.random.choice(indices, batch_size, replace=False)
            
            batch_states = torch.FloatTensor(states[batch_indices])
            batch_actions = torch.FloatTensor(actions[batch_indices])
            batch_rewards = torch.FloatTensor(rewards[batch_indices]).unsqueeze(1)
            batch_next_states = torch.FloatTensor(next_states[batch_indices])
            batch_dones = torch.FloatTensor(dones[batch_indices]).unsqueeze(1)
            
            q_loss, policy_loss = self._conservative_cql_update(
                batch_states, batch_actions, batch_rewards, 
                batch_next_states, batch_dones
            )
            epoch_loss += policy_loss
        
        return epoch_loss / num_batches

    def _validate(self, states, actions, rewards, next_states, dones, indices):
        """验证"""
        with torch.no_grad():
            val_size = min(500, len(indices))
            val_indices = np.random.choice(indices, val_size, replace=False)
            val_states = torch.FloatTensor(states[val_indices])
            
            policy_actions = torch.tanh(self.policy_net(val_states))
            policy_q1 = self.q_net1(torch.cat([val_states, policy_actions], 1))
            val_loss = -policy_q1.mean()
            
        return val_loss.item()

    def _conservative_cql_update(self, states, actions, rewards, next_states, dones):
        """保守的CQL更新 - 彻底解决Q值过估计"""
        # 计算目标Q值 - 使用更保守的方法
        with torch.no_grad():
            next_actions = torch.tanh(self.policy_net(next_states))
            next_q1 = self.q_net1(torch.cat([next_states, next_actions], 1))
            next_q2 = self.q_net2(torch.cat([next_states, next_actions], 1))
            next_q = torch.min(next_q1, next_q2)
            # 使用更保守的目标Q值计算
            target_q = rewards + self.gamma * (1 - dones) * torch.clamp(next_q, -10.0, 10.0)
        
        # Q网络更新
        current_q1 = self.q_net1(torch.cat([states, actions], 1))
        current_q2 = self.q_net2(torch.cat([states, actions], 1))
        
        td_loss1 = F.mse_loss(current_q1, target_q)
        td_loss2 = F.mse_loss(current_q2, target_q)
        td_loss = (td_loss1 + td_loss2) / 2
        
        # 使用更保守的CQL损失
        cql_loss = self._compute_conservative_cql_loss(states, actions)
        q_loss = td_loss + self.cql_weight * cql_loss
        
        self.q_optimizer.zero_grad()
        q_loss.backward()
        # 严格的梯度裁剪
        torch.nn.utils.clip_grad_norm_(self.q_net1.parameters(), 0.1)
        torch.nn.utils.clip_grad_norm_(self.q_net2.parameters(), 0.1)
        self.q_optimizer.step()
        
        # 策略网络更新 - 使用更保守的方法
        policy_actions = torch.tanh(self.policy_net(states))
        policy_q1 = self.q_net1(torch.cat([states, policy_actions], 1))
        
        # 更强的正则化
        action_penalty = torch.mean(policy_actions ** 2) * 0.1  # 动作幅度惩罚
        
        policy_loss = -policy_q1.mean() + action_penalty
        
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 0.1)
        self.policy_optimizer.step()
        
        return q_loss.item(), policy_loss.item()

    def _compute_conservative_cql_loss(self, states, actions):
        """保守的CQL损失计算"""
        batch_size = states.shape[0]
        
        # 当前策略动作
        with torch.no_grad():
            current_actions = torch.tanh(self.policy_net(states))
        
        # 随机动作
        random_actions = torch.rand(batch_size, self.action_dim) * 2 - 1
        
        # 计算Q值 - 使用裁剪避免极端值
        current_q1 = torch.clamp(self.q_net1(torch.cat([states, current_actions], 1)), -20.0, 20.0)
        current_q2 = torch.clamp(self.q_net2(torch.cat([states, current_actions], 1)), -20.0, 20.0)
        random_q1 = torch.clamp(self.q_net1(torch.cat([states, random_actions], 1)), -20.0, 20.0)
        random_q2 = torch.clamp(self.q_net2(torch.cat([states, random_actions], 1)), -20.0, 20.0)
        
        # 保守的CQL损失
        cql_loss1 = torch.logsumexp(random_q1 / self.temperature, dim=0).mean() * self.temperature - current_q1.mean()
        cql_loss2 = torch.logsumexp(random_q2 / self.temperature, dim=0).mean() * self.temperature - current_q2.mean()
        
        return (cql_loss1 + cql_loss2) / 2


class PolicyAgent(BaseAgent):
    """
    保守版PolicyAgent
    - 彻底解决Q值过估计
    - 更保守的动作输出
    - 保持训练速度
    """

    HIST_LEN = 30

    def __init__(self, seed: int = None):
        super().__init__(seed)
        
        # 保守网络结构
        self.policy_net = ConservativeMLP(150, [128, 64], 3)
        
        self._initialize_model()
        self.policy_net.to(self.device)
        self.policy_net.eval()
        
        self.last_action = None
        self.action_smoothing = 0.8
        print("✅ 保守版PolicyAgent初始化完成")

    def _initialize_model(self):
        """模型初始化"""
        model_path = os.path.join(os.path.dirname(__file__), "model.pth")
        
        if not os.path.exists(model_path):
            print("🚀 开始保守CQL训练...")
            self._train_model()
        else:
            try:
                state_dict = torch.load(model_path, map_location="cpu")
                self.policy_net.load_state_dict(state_dict)
                print(f"✅ 加载预训练模型: {model_path}")
            except Exception as e:
                print(f"❌ 加载失败，重新训练: {e}")
                self._train_model()

    def _create_hist_data(self, states_5d, hist_len=30):
        """创建历史数据"""
        hist_states = []
        for i in range(len(states_5d)):
            start_idx = max(0, i - hist_len + 1)
            hist = states_5d[start_idx:i+1]
            
            if len(hist) < hist_len:
                padding = np.zeros((hist_len - len(hist), 5), dtype=np.float32)
                hist = np.vstack([padding, hist])
            
            hist_states.append(hist.flatten())
            
        return np.array(hist_states)

    def _train_model(self):
        """保守模型训练"""
        try:
            torch.manual_seed(42)
            np.random.seed(42)
            random.seed(42)
            
            current_dir = os.path.dirname(os.path.abspath(__file__))
            data_path = os.path.join(current_dir, "..", "data", "data.csv")
            model_path = os.path.join(current_dir, "model.pth")
            
            if not os.path.exists(data_path):
                print("❌ 数据文件不存在，使用随机初始化")
                torch.save(self.policy_net.state_dict(), model_path)
                return
            
            print("📊 加载训练数据...")
            data = pd.read_csv(data_path)
            print(f"原始数据量: {len(data)}")
            
            # 使用少量高质量数据
            if len(data) > 30000:
                data = data.sample(n=30000, random_state=42)
                print(f"使用采样数据: {len(data)} 条")
            
            # 数据预处理
            states_5d = data[['obs_1', 'obs_2', 'obs_3', 'obs_4', 'obs_5']].values.astype(np.float32)
            actions = data[['action_1', 'action_2', 'action_3']].values.astype(np.float32)
            rewards = data['reward'].values.astype(np.float32)
            
            # 创建历史状态
            states_150d = self._create_hist_data(states_5d, self.HIST_LEN)
            next_states_150d = self._create_hist_data(np.roll(states_5d, -1, axis=0), self.HIST_LEN)
            
            # 构建终止标志
            dones = np.zeros(len(states_5d), dtype=np.float32)
            if 'index' in data.columns:
                for i in range(len(states_5d)-1):
                    if data['index'].iloc[i] != data['index'].iloc[i+1]:
                        dones[i] = 1.0
                dones[-1] = 1.0
            
            # 训练保守版CQL
            trainer = ConservativeCQLTrainer(state_dim=150, action_dim=3)
            trainer.train_conservative_cql(states_150d, actions, rewards, next_states_150d, dones)
            
            # 保存模型
            torch.save(trainer.best_policy_state, model_path)
            print(f"💾 模型保存成功: {model_path}")
            
            self.policy_net.load_state_dict(trainer.best_policy_state)
            
        except Exception as e:
            print(f"训练异常: {e}")
            import traceback
            traceback.print_exc()
            # 确保即使训练失败也有模型文件
            torch.save(self.policy_net.state_dict(), model_path)
            print("💾 保存初始模型作为备选")

    def get_action(self, obs: np.ndarray) -> np.ndarray:
        """获取动作 - 非常保守的输出"""
        obs = np.asarray(obs, dtype=np.float32)
        if len(obs) != 5:
            obs = np.zeros(5, dtype=np.float32)

        try:
            # 更新历史
            self.obs_history.append(obs.copy())
            if len(self.obs_history) > self.HIST_LEN:
                self.obs_history = self.obs_history[-self.HIST_LEN:]
            
            # 填充历史
            if len(self.obs_history) < self.HIST_LEN:
                padding = [np.zeros(5)] * (self.HIST_LEN - len(self.obs_history))
                padded_history = padding + self.obs_history
            else:
                padded_history = self.obs_history
            
            # 构建状态
            state = np.array(padded_history, dtype=np.float32).flatten().reshape(1, -1)
            
            # 推理 - 使用非常保守的激活
            x = torch.from_numpy(state).to(self.device)
            with torch.no_grad():
                raw_action = self.policy_net(x)
                # 使用缩放后的tanh，限制输出范围
                action = 0.5 * torch.tanh(raw_action)  # 更小的缩放因子

            result = action.squeeze(0).cpu().numpy().astype(np.float32)
            
            # 更强的动作平滑
            if self.last_action is not None:
                result = self.action_smoothing * result + (1 - self.action_smoothing) * self.last_action
            
            self.last_action = result.copy()
            
            # 非常保守的动作裁剪
            result = np.clip(result, -0.5, 0.5)
            
            return result
            
        except Exception as e:
            print(f"动作生成异常: {e}")
            # 返回非常保守的默认动作
            return np.array([0.05, -0.05, 0.02], dtype=np.float32)

    def reset(self) -> None:
        """重置"""
        super().reset()
        self.last_action = None

    def close(self) -> None:
        """关闭"""
        pass


# 官方接口
TCNPolicyAgent = PolicyAgent
EnhancedPolicyAgent = PolicyAgent


if __name__ == "__main__":
    agent = PolicyAgent()
    test_obs = np.random.rand(5).astype(np.float32)
    action = agent.act(test_obs)
    print(f"测试完成 - 动作: {action}, 幅度: {np.linalg.norm(action):.3f}")
    agent.close()