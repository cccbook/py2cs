# ChatGPT 從 VPGAgent 改過來的 https://chatgpt.com/c/672ec2ea-bf70-8012-b7a1-cc753f6f67ad

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributions as distributions
import numpy as np

# 定義 ActorCriticAgent 類別
class ActorCriticAgent:
    def __init__(self, env):
        self.action_n = env.action_space.n
        self.gamma = 0.99
        
        # 建立 actor 網路
        self.actor_net = self.build_net(
            input_size=env.observation_space.shape[0],
            hidden_sizes=[],
            output_size=self.action_n,
            output_activator=nn.Softmax(dim=1)
        )
        
        # 建立 critic 網路
        self.critic_net = self.build_net(
            input_size=env.observation_space.shape[0],
            hidden_sizes=[],
            output_size=1  # Critic 輸出單一值表示當前狀態的價值
        )
        
        # 定義優化器
        self.actor_optimizer = optim.Adam(self.actor_net.parameters(), lr=0.005)
        self.critic_optimizer = optim.Adam(self.critic_net.parameters(), lr=0.005)

    def build_net(self, input_size, hidden_sizes, output_size, output_activator=None, use_bias=False):
        layers = []
        for input_size, output_size in zip(
                [input_size] + hidden_sizes, hidden_sizes + [output_size]):
            layers.append(nn.Linear(input_size, output_size, bias=use_bias))
            layers.append(nn.ReLU())
        layers = layers[:-1]  # 移除最後的 ReLU
        if output_activator:
            layers.append(output_activator)
        model = nn.Sequential(*layers)
        return model

    def reset(self, mode=None):
        self.mode = mode
        if self.mode == 'train':
            self.trajectory = []

    def step(self, observation, reward, terminated):
        state_tensor = torch.as_tensor(observation, dtype=torch.float).unsqueeze(0)
        
        # 使用 actor 網路計算行動的機率分佈，並抽樣行動
        prob_tensor = self.actor_net(state_tensor)
        action_tensor = distributions.Categorical(prob_tensor).sample()
        action = action_tensor.item()
        
        if self.mode == 'train':
            # 使用 critic 網路計算當前狀態的價值
            value_tensor = self.critic_net(state_tensor)
            self.trajectory += [observation, reward, terminated, action, value_tensor]
        
        return action

    def close(self):
        if self.mode == 'train':
            self.learn()

    def learn(self):
        # 將 observation 的列表轉換為 numpy array，再轉換為 tensor
        state_array = np.array(self.trajectory[0::5])
        state_tensor = torch.as_tensor(state_array, dtype=torch.float)

        reward_tensor = torch.as_tensor(self.trajectory[1::5], dtype=torch.float)
        action_tensor = torch.as_tensor(self.trajectory[3::5], dtype=torch.long)
        value_tensor = torch.cat(self.trajectory[4::5]).squeeze(1)

        # 計算折扣回報
        arange_tensor = torch.arange(state_tensor.shape[0], dtype=torch.float)
        discount_tensor = self.gamma ** arange_tensor
        discounted_reward_tensor = discount_tensor * reward_tensor
        discounted_return_tensor = discounted_reward_tensor.flip(0).cumsum(0).flip(0)

        # 計算 advantage
        advantage_tensor = discounted_return_tensor - value_tensor

        # Actor 的 loss 計算
        all_pi_tensor = self.actor_net(state_tensor)
        pi_tensor = torch.gather(all_pi_tensor, 1, action_tensor.unsqueeze(1)).squeeze(1)
        log_pi_tensor = torch.log(torch.clamp(pi_tensor, 1e-6, 1.))
        actor_loss = -(advantage_tensor * log_pi_tensor).mean()

        # Critic 的 loss 計算
        critic_loss = ((discounted_return_tensor - value_tensor) ** 2).mean()

        # 更新 actor 網路
        self.actor_optimizer.zero_grad()
        actor_loss.backward(retain_graph=True)  # 保留計算圖，以避免 RuntimeError
        self.actor_optimizer.step()

        # 更新 critic 網路
        self.critic_optimizer.zero_grad()
        critic_loss.backward()  # 只需要在 Critic 網路中進行一次 backward
        self.critic_optimizer.step()
