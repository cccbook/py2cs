import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
import pandas as pd
import copy
from Replayer import Replayer

class SARSALambdaAgent:
    def __init__(self, env, lambda_=0.9):
        self.action_n = env.action_space.n
        self.gamma = 0.99
        self.lambda_ = lambda_  # SARSA(λ) 中的 λ 值

        self.replayer = Replayer(10000)
        self.evaluate_net = self.build_net(
            input_size=env.observation_space.shape[0],
            hidden_sizes=[64, 64], output_size=self.action_n)
        self.optimizer = optim.Adam(self.evaluate_net.parameters(), lr=0.001)
        self.loss = nn.MSELoss()

        # 初始化資格跡，對於每個 (state, action) 配對分別保存資格跡
        self.eligibility_trace = {}  # 資格跡字典

    def build_net(self, input_size, hidden_sizes, output_size):
        layers = []
        for input_size, output_size in zip([input_size,] + hidden_sizes, hidden_sizes + [output_size,]):
            layers.append(nn.Linear(input_size, output_size))
            layers.append(nn.ReLU())
        layers = layers[:-1]
        model = nn.Sequential(*layers)
        return model

    def reset(self, mode=None):
        self.mode = mode
        if self.mode == 'train':
            self.eligibility_trace.clear()  # 重置資格跡字典

    def step(self, observation, reward, terminated):
        if self.mode == 'train' and np.random.rand() < 0.1:
            action = np.random.randint(self.action_n)
        else:
            state_tensor = torch.as_tensor(observation, dtype=torch.float).unsqueeze(0)
            q_values = self.evaluate_net(state_tensor)
            action = torch.argmax(q_values).item()

        # 將每個步驟的狀態、動作、獎勵、下一狀態等記錄為一個完整的項目
        if self.mode == 'train':
            if hasattr(self, 'last_state') and hasattr(self, 'last_action'):
                # 儲存完整的 `(state, action, reward, next_state, next_action, terminated)` 組合
                self.trajectory = [self.last_state, self.last_action, reward, observation, action, terminated]
                self.replayer.store(*self.trajectory)

            # 更新上一步的狀態和動作
            self.last_state, self.last_action = observation, action

            # 如果已儲存的數量夠多，則開始學習
            if self.replayer.count >= self.replayer.capacity * 0.9:
                self.learn()
        return action

    def close(self):
        pass

    def learn(self):
        # 從記憶庫中抽取樣本
        states, actions, rewards, next_states, next_actions, terminateds = self.replayer.sample(256)
        
        # 將數據轉為 PyTorch 張量
        state_tensor = torch.as_tensor(states, dtype=torch.float)
        action_tensor = torch.as_tensor(actions, dtype=torch.long)
        reward_tensor = torch.as_tensor(rewards, dtype=torch.float)
        next_state_tensor = torch.as_tensor(next_states, dtype=torch.float)
        next_action_tensor = torch.as_tensor(next_actions, dtype=torch.long)
        terminated_tensor = torch.as_tensor(terminateds, dtype=torch.float)

        # 計算 TD 誤差
        q_values = self.evaluate_net(state_tensor).gather(1, action_tensor.unsqueeze(1)).squeeze(1)
        next_q_values = self.evaluate_net(next_state_tensor).gather(1, next_action_tensor.unsqueeze(1)).squeeze(1)
        td_error = reward_tensor + self.gamma * (1 - terminated_tensor) * next_q_values - q_values

        # 更新資格跡並應用更新
        for idx, param in enumerate(self.evaluate_net.parameters()):
            param.grad = None  # 重置梯度
            
            # 計算每個步驟的資格跡
            for i in range(len(td_error)):
                state_action = (tuple(state_tensor[i].tolist()), action_tensor[i].item())
                self.eligibility_trace[state_action] = self.eligibility_trace.get(state_action, 0) + 1

                # 計算並累加梯度
                for j, p in enumerate(self.evaluate_net.parameters()):
                    if p.grad is None:
                        p.grad = td_error[i] * self.eligibility_trace[state_action] * torch.autograd.grad(q_values[i], p, retain_graph=True)[0]
                    else:
                        p.grad += td_error[i] * self.eligibility_trace[state_action] * torch.autograd.grad(q_values[i], p, retain_graph=True)[0]

                # 依據 SARSA(λ) 更新資格跡表
                for key in list(self.eligibility_trace.keys()):
                    self.eligibility_trace[key] *= self.gamma * self.lambda_
                    if self.eligibility_trace[key] < 1e-5:  # 清除微小的跡值
                        del self.eligibility_trace[key]
        
        # 使用 optimizer 更新參數
        self.optimizer.step()
