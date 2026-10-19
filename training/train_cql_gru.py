# train_cql_gru.py
# 请将此文件放置在 IndustrialControl-main/ 目录下

from traceback import print_tb
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt

# ==============================================================================
# MODIFIED: 整个文件为新增，用于CQL+GRU模型训练
# ==============================================================================

# --- 1. 参数配置 ---
class Config:
    DATA_PATH = "./data/data.csv"
    MODEL_SAVE_PATH = "./agent/model.pth"
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 模型参数
    HIST_LEN = 30         # 历史观测帧数
    OBS_DIM = 5           # 单帧观测维度
    ACTION_DIM = 3        # 动作维度
    HIDDEN_DIM = 256      # GRU和MLP的隐藏层维度
    
    # 训练参数
    BATCH_SIZE = 256
    EPOCHS = 100          # 训练轮次，可根据实际情况调整
    LEARNING_RATE = 1e-4  # 学习率
    GAMMA = 0.99          # 折扣因子
    TAU = 0.005           # 软更新系数
    
    # CQL 算法特定参数
    CQL_ALPHA = 0.5       # CQL损失的权重，这是关键超参数
    NUM_RANDOM_ACTIONS = 10 # 用于计算log-sum-exp时采样的随机动作数量

print(f"Using device: {Config.DEVICE}")

# --- 2. 定义网络结构 ---

# Actor网络：GRU + MLP，输出确定性动作
class Actor(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(Actor, self).__init__()
        # GRU层用于处理序列信息
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)
        # 两个全连接层
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, state_history):
        # state_history: (batch, seq_len, obs_dim)
        # GRU的输出包括所有时间步的输出和最后一个时间步的隐藏状态
        gru_out, _ = self.gru(state_history)
        # 我们只取最后一个时间步的输出，它聚合了整个序列的信息
        last_step_out = gru_out[:, -1, :]
        
        x = F.relu(self.fc1(last_step_out))
        # 使用tanh将动作输出限制在[-1, 1]之间
        action = torch.tanh(self.fc2(x))
        return action

# Critic网络：GRU + MLP，输入状态历史和动作，输出Q值
class Critic(nn.Module):
    def __init__(self, input_dim, hidden_dim, action_dim):
        super(Critic, self).__init__()
        # GRU层处理状态历史
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)
        # 全连接层处理GRU输出和动作的拼接
        self.fc1 = nn.Linear(hidden_dim + action_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)

    def forward(self, state_history, action):
        gru_out, _ = self.gru(state_history)
        last_step_out = gru_out[:, -1, :]
        
        # 将最后一个时间步的GRU输出与动作拼接
        concat = torch.cat([last_step_out, action], dim=1)
        
        q = F.relu(self.fc1(concat))
        q = F.relu(self.fc2(q))
        q = self.fc3(q)
        return q

# --- 3. 数据处理 ---

# 自定义数据集，用于从CSV加载和预处理数据
class OfflineRLDataset(Dataset):
    def __init__(self, data_path, hist_len):
        print("Loading and preprocessing data...")
        data = pd.read_csv(data_path)
        self.hist_len = hist_len
        
        self.states = []
        self.actions = []
        self.next_states = []
        self.rewards = []
        self.terminals = []

        obs_cols = [f'obs_{i+1}' for i in range(Config.OBS_DIM)]
        action_cols = [f'action_{i+1}' for i in range(Config.ACTION_DIM)]

        # 按轨迹处理数据
        for idx in data['index'].unique():
            traj_data = data[data['index'] == idx].reset_index(drop=True)
            
            obs = traj_data[obs_cols].values
            actions = traj_data[action_cols].values
            rewards = traj_data['reward'].values
            
            T = len(traj_data)
            
            for t in range(T - 1):
                # 构造历史观测 state
                state_hist = self._get_history(obs, t)
                
                # 构造下一个状态的历史观测 next_state
                next_state_hist = self._get_history(obs, t + 1)

                self.states.append(state_hist)
                self.actions.append(actions[t])
                self.next_states.append(next_state_hist)
                self.rewards.append(rewards[t])
                # 比赛环境是固定长度的，最后一个时间步为 done
                self.terminals.append(1.0 if t == T - 2 else 0.0)

        print(f"Data loading complete. Total transitions: {len(self.states)}")

    def _get_history(self, obs, t):
        """辅助函数，用于获取t时刻之前的hist_len帧观测"""
        if t < self.hist_len - 1:
            # 不足hist_len时，用第一帧填充
            padding = np.tile(obs[0], (self.hist_len - 1 - t, 1))
            history = np.concatenate([padding, obs[:t + 1]], axis=0)
        else:
            # 取最近的hist_len帧
            history = obs[t - self.hist_len + 1 : t + 1]
        return history.astype(np.float32)

    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx):
        return {
            'states': self.states[idx],
            'actions': self.actions[idx].astype(np.float32),
            'next_states': self.next_states[idx],
            'rewards': np.float32(self.rewards[idx]),
            'terminals': np.float32(self.terminals[idx])
        }


# --- 4. CQL Agent 实现 ---

class CQLAgent:
    def __init__(self, config):
        self.config = config
        
        # 初始化Actor和Critic网络
        self.actor = Actor(config.OBS_DIM, config.HIDDEN_DIM, config.ACTION_DIM).to(config.DEVICE)
        self.actor_target = Actor(config.OBS_DIM, config.HIDDEN_DIM, config.ACTION_DIM).to(config.DEVICE)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=config.LEARNING_RATE)

        # 初始化两个Critic网络
        self.critic1 = Critic(config.OBS_DIM, config.HIDDEN_DIM, config.ACTION_DIM).to(config.DEVICE)
        self.critic2 = Critic(config.OBS_DIM, config.HIDDEN_DIM, config.ACTION_DIM).to(config.DEVICE)
        self.critic1_target = Critic(config.OBS_DIM, config.HIDDEN_DIM, config.ACTION_DIM).to(config.DEVICE)
        self.critic2_target = Critic(config.OBS_DIM, config.HIDDEN_DIM, config.ACTION_DIM).to(config.DEVICE)
        self.critic1_target.load_state_dict(self.critic1.state_dict())
        self.critic2_target.load_state_dict(self.critic2.state_dict())
        self.critic_optimizer = optim.Adam(list(self.critic1.parameters()) + list(self.critic2.parameters()), lr=config.LEARNING_RATE)
        self.actor_scheduler = torch.optim.lr_scheduler.StepLR(self.actor_optimizer, step_size=30, gamma=0.5)
        self.critic_scheduler = torch.optim.lr_scheduler.StepLR(self.critic_optimizer, step_size=30, gamma=0.5)
        
    def train(self, batch):
        # 从batch中提取数据并移动到设备
        states = torch.tensor(batch['states'], dtype=torch.float32).to(self.config.DEVICE)
        actions = torch.tensor(batch['actions'], dtype=torch.float32).to(self.config.DEVICE)
        next_states = torch.tensor(batch['next_states'], dtype=torch.float32).to(self.config.DEVICE)
        rewards = torch.tensor(batch['rewards'], dtype=torch.float32).unsqueeze(1).to(self.config.DEVICE)
        terminals = torch.tensor(batch['terminals'], dtype=torch.float32).unsqueeze(1).to(self.config.DEVICE)
        
        # --- Critic Loss 计算 ---
        with torch.no_grad():
            next_actions = self.actor_target(next_states)
            target_q1 = self.critic1_target(next_states, next_actions)
            target_q2 = self.critic2_target(next_states, next_actions)
            # 取两个目标Q值中较小的一个，抑制高估
            target_q = torch.min(target_q1, target_q2)
            # 计算目标Q值
            target_q = rewards + self.config.GAMMA * (1 - terminals) * target_q

        # 当前状态的Q值预测
        current_q1 = self.critic1(states, actions)
        current_q2 = self.critic2(states, actions)

        # 贝尔曼误差 (TD Loss)
        critic_loss_td = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)
        
        # --- CQL Loss 计算 ---
        batch_size = states.shape[0]
        
        # 1. 对策略网络采样的动作计算Q值
        policy_actions = self.actor(states)
        q1_policy = self.critic1(states, policy_actions)
        q2_policy = self.critic2(states, policy_actions)
        
        # 2. 对随机采样的动作计算Q值
        random_actions = torch.rand(batch_size * self.config.NUM_RANDOM_ACTIONS, self.config.ACTION_DIM).to(self.config.DEVICE) * 2 - 1
        # 扩展states以匹配随机动作的数量
        states_repeated = states.unsqueeze(1).repeat(1, self.config.NUM_RANDOM_ACTIONS, 1, 1).view(
            batch_size * self.config.NUM_RANDOM_ACTIONS, self.config.HIST_LEN, self.config.OBS_DIM
        )
        q1_random = self.critic1(states_repeated, random_actions)
        q2_random = self.critic2(states_repeated, random_actions)
        
        # 3. 计算 log-sum-exp
        # 将policy动作和random动作的Q值拼接起来
        cat_q1 = torch.cat(
            [q1_random.view(batch_size, -1), q1_policy], 1
        )
        cat_q2 = torch.cat(
            [q2_random.view(batch_size, -1), q2_policy], 1
        )
        
        # logsumexp项：惩罚OOD动作
        cql_loss_q1 = torch.logsumexp(cat_q1, dim=1).mean()
        cql_loss_q2 = torch.logsumexp(cat_q2, dim=1).mean()

        # 数据集动作项：提升数据集中的动作Q值
        cql_loss_q1 -= current_q1.mean()
        cql_loss_q2 -= current_q2.mean()
        
        # 最终Critic Loss
        critic_loss = critic_loss_td + self.config.CQL_ALPHA * (cql_loss_q1 + cql_loss_q2)

        # 更新Critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        # --- Actor Loss 计算 ---
        # Actor的目标是最大化其动作的Q值
        actor_actions = self.actor(states)
        q_actor = self.critic1(states, actor_actions)
        actor_loss = -q_actor.mean()
        
        # 更新Actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        # --- 软更新目标网络 ---
        self._soft_update(self.critic1_target, self.critic1)
        self._soft_update(self.critic2_target, self.critic2)
        self._soft_update(self.actor_target, self.actor)
        
        return critic_loss.item(), actor_loss.item(), critic_loss_td.item(), (cql_loss_q1 + cql_loss_q2).item()
    
    def _soft_update(self, target, source):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(self.config.TAU * param.data + (1.0 - self.config.TAU) * target_param.data)
            
    def save(self, path):
        # 只保存Actor模型用于最终部署
        torch.save(self.actor.state_dict(), path)
        print(f"Actor model saved to {path}")

# --- 5. 训练主流程 ---

def main():
    torch.manual_seed(42)
    
    config = Config()
    
    # 初始化数据集和DataLoader
    dataset = OfflineRLDataset(config.DATA_PATH, config.HIST_LEN)
    dataloader = DataLoader(dataset, batch_size=config.BATCH_SIZE, shuffle=True,
                            collate_fn=lambda b: {k: np.array([d[k] for d in b]) for k in b[0]})
    
    # 初始化Agent
    agent = CQLAgent(config)
    
    critic_losses = []
    actor_losses = []
    td_losses = []
    cql_losses = []
    # 训练循环
    for epoch in range(1, config.EPOCHS + 1):
        total_critic_loss = 0
        total_actor_loss = 0
        total_td_loss = 0
        total_cql_loss = 0
        for batch in dataloader:
            critic_loss, actor_loss, td_loss, cql_loss = agent.train(batch)
            total_critic_loss += critic_loss
            total_actor_loss += actor_loss
            total_td_loss += td_loss
            total_cql_loss += cql_loss
        agent.actor_scheduler.step()
        agent.critic_scheduler.step()
        avg_critic_loss = total_critic_loss / len(dataloader)
        avg_actor_loss = total_actor_loss / len(dataloader)
        avg_td_loss = total_td_loss / len(dataloader)
        avg_cql_loss = total_cql_loss / len(dataloader)
        
        # 保存损失值用于可视化
        critic_losses.append(avg_critic_loss)
        actor_losses.append(avg_actor_loss)
        td_losses.append(avg_td_loss)
        cql_losses.append(avg_cql_loss)
        print(f"Epoch {epoch:03d}/{config.EPOCHS} | Critic Loss: {avg_critic_loss:.4f} | Actor Loss: {avg_actor_loss:.4f}")
        

    # 保存最终模型
    agent.save(config.MODEL_SAVE_PATH)

    # 可视化损失曲线
    epochs = range(1, config.EPOCHS + 1)
    plt.figure(figsize=(15, 10))
    
    # 绘制总体Critic Loss
    plt.subplot(2, 2, 1)
    plt.plot(epochs, critic_losses, 'b-', label='Critic Loss')
    plt.title('Critic Loss Over Time')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # 绘制Actor Loss
    plt.subplot(2, 2, 2)
    plt.plot(epochs, actor_losses, 'r-', label='Actor Loss')
    plt.title('Actor Loss Over Time')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # 绘制TD Loss
    plt.subplot(2, 2, 3)
    plt.plot(epochs, td_losses, 'g-', label='TD Loss')
    plt.title('TD Loss Over Time')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # 绘制CQL Loss
    plt.subplot(2, 2, 4)
    plt.plot(epochs, cql_losses, 'm-', label='CQL Loss')
    plt.title('CQL Loss Over Time')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('training_losses.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 绘制组合图
    plt.figure(figsize=(12, 8))
    plt.plot(epochs, critic_losses, 'b-', label='Critic Loss', linewidth=2)
    plt.plot(epochs, actor_losses, 'r-', label='Actor Loss', linewidth=2)
    plt.plot(epochs, td_losses, 'g--', label='TD Loss', linewidth=2)
    plt.plot(epochs, cql_losses, 'm--', label='CQL Loss', linewidth=2)
    plt.title('Training Losses Over Time', fontsize=16)
    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel('Loss', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('combined_losses.png', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    main()
