import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
import pandas as pd
import copy
# from Replayer import Replayer

class Replayer:
    def __init__(self, capacity):
        self.memory = pd.DataFrame(index=range(capacity),
                                   columns=['state', 'action', 'reward', 'next_state', 'next_action', 'terminated'])
        self.i = 0
        self.count = 0
        self.capacity = capacity

    def store(self, *args):
        self.memory.loc[self.i] = np.asarray(args, dtype=object)
        self.i = (self.i + 1) % self.capacity
        self.count = min(self.count + 1, self.capacity)

    def sample(self, size):
        indices = np.random.choice(self.count, size=size)
        return (np.stack(self.memory.loc[indices, field]) for field in self.memory.columns)

class SARSAAgent:
    def __init__(self, env):
        self.action_n = env.action_space.n
        self.gamma = 0.99

        self.replayer = Replayer(10000)

        self.evaluate_net = self.build_net(
                input_size=env.observation_space.shape[0],
                hidden_sizes=[64, 64], output_size=self.action_n)
        self.optimizer = optim.Adam(self.evaluate_net.parameters(), lr=0.001)
        self.loss = nn.MSELoss()

    def build_net(self, input_size, hidden_sizes, output_size):
        layers = []
        for input_size, output_size in zip(
                [input_size,] + hidden_sizes, hidden_sizes + [output_size,]):
            layers.append(nn.Linear(input_size, output_size))
            layers.append(nn.ReLU())
        layers = layers[:-1]
        model = nn.Sequential(*layers)
        return model

    def reset(self, mode=None):
        self.mode = mode
        if self.mode == 'train':
            self.trajectory = []

    def step(self, observation, reward, terminated):
        if self.mode == 'train' and np.random.rand() < 0.001:
            action = np.random.randint(self.action_n)
        else:
            state_tensor = torch.as_tensor(observation, dtype=torch.float).squeeze(0)
            q_tensor = self.evaluate_net(state_tensor)
            action_tensor = torch.argmax(q_tensor)
            action = action_tensor.item()
        
        if self.mode == 'train':
            self.trajectory += [observation, reward, terminated, action]
            if len(self.trajectory) >= 8:
                state, _, _, act, next_state, reward, terminated, next_action = self.trajectory[-8:]
                self.replayer.store(state, act, reward, next_state, next_action, terminated)
            if self.replayer.count >= self.replayer.capacity * 0.95:
                self.learn()
        return action

    def close(self):
        pass

    def learn(self):
        # 從記憶庫中隨機抽取
        states, actions, rewards, next_states, next_actions, terminateds = self.replayer.sample(1024)
        
        state_tensor = torch.as_tensor(states, dtype=torch.float)
        action_tensor = torch.as_tensor(actions, dtype=torch.long)
        reward_tensor = torch.as_tensor(rewards, dtype=torch.float)
        next_state_tensor = torch.as_tensor(next_states, dtype=torch.float)
        next_action_tensor = torch.as_tensor(next_actions, dtype=torch.long)
        terminated_tensor = torch.as_tensor(terminateds, dtype=torch.float)
        
        # 根據 SARSA 演算法進行更新
        # 使用 next_action 來獲取下一狀態的 Q 值
        next_q_tensor = self.evaluate_net(next_state_tensor).gather(1, next_action_tensor.unsqueeze(1)).squeeze(1)
        # 以下公式為 q(s,a) = r + gamma * q(s',a')
        target_tensor = reward_tensor + self.gamma * (1. - terminated_tensor) * next_q_tensor
        
        pred_tensor = self.evaluate_net(state_tensor)
        q_tensor = pred_tensor.gather(1, action_tensor.unsqueeze(1)).squeeze(1)
        
        loss_tensor = self.loss(target_tensor, q_tensor) # 目標：縮小 r + gamma * q(s',a') 與 q(s,a) 之間的差距
        # 參考 https://chatgpt.com/c/672da875-6be4-8012-a637-d81121396dae
        self.optimizer.zero_grad()
        loss_tensor.backward()
        self.optimizer.step()
