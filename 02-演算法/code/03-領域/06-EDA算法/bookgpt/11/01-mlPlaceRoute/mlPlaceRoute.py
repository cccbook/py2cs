import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random

# 定義環境：簡化的 Placement 問題
class PlacementEnv:
    def __init__(self, grid_size, num_components):
        self.grid_size = grid_size
        self.num_components = num_components
        self.grid = np.zeros((grid_size, grid_size))
        self.components = [(i, i) for i in range(num_components)]  # 初始位置
        self.step_count = 0

    def reset(self):
        self.grid = np.zeros((self.grid_size, self.grid_size))
        self.components = [(0, 0) for _ in range(self.num_components)]
        self.step_count = 0
        return self._get_state()

    def _get_state(self):
        state = np.zeros((self.grid_size, self.grid_size))
        for x, y in self.components:
            state[x][y] = 1
        return state.flatten()

    def step(self, action):
        comp_idx, direction = divmod(action, 4)
        x, y = self.components[comp_idx]

        # 移動：上下左右
        if direction == 0 and x > 0: x -= 1  # 上
        if direction == 1 and x < self.grid_size - 1: x += 1  # 下
        if direction == 2 and y > 0: y -= 1  # 左
        if direction == 3 and y < self.grid_size - 1: y += 1  # 右

        self.components[comp_idx] = (x, y)
        self.step_count += 1

        # 獎勵函數：減少元件距離中心的距離
        reward = -sum(abs(x - self.grid_size // 2) + abs(y - self.grid_size // 2) for x, y in self.components)
        done = self.step_count >= self.num_components * 10
        return self._get_state(), reward, done, {}

# 定義 DQN 模型
class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_size, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_size)
        )

    def forward(self, x):
        return self.fc(x)

# DQN 主程序
def train_placement():
    grid_size = 5
    num_components = 3
    env = PlacementEnv(grid_size, num_components)

    state_size = grid_size * grid_size
    action_size = num_components * 4  # 每個元件 4 個方向
    model = DQN(state_size, action_size)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.MSELoss()

    # 記憶庫與參數
    memory = deque(maxlen=2000)
    gamma = 0.95
    epsilon = 1.0
    epsilon_decay = 0.995
    epsilon_min = 0.1
    batch_size = 64
    episodes = 500

    # 訓練過程
    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        done = False

        while not done:
            # ε-貪婪策略
            if random.random() < epsilon:
                action = random.randint(0, action_size - 1)
            else:
                with torch.no_grad():
                    q_values = model(torch.tensor(state, dtype=torch.float32))
                    action = torch.argmax(q_values).item()

            next_state, reward, done, _ = env.step(action)
            memory.append((state, action, reward, next_state, done))
            state = next_state
            total_reward += reward

            # 訓練 DQN
            if len(memory) >= batch_size:
                batch = random.sample(memory, batch_size)
                states, actions, rewards, next_states, dones = zip(*batch)

                states = torch.tensor(states, dtype=torch.float32)
                actions = torch.tensor(actions, dtype=torch.int64).unsqueeze(-1)
                rewards = torch.tensor(rewards, dtype=torch.float32)
                next_states = torch.tensor(next_states, dtype=torch.float32)
                dones = torch.tensor(dones, dtype=torch.float32)

                q_values = model(states).gather(1, actions).squeeze()
                next_q_values = model(next_states).max(1)[0].detach()
                targets = rewards + gamma * next_q_values * (1 - dones)

                loss = loss_fn(q_values, targets)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        # ε 衰減
        if epsilon > epsilon_min:
            epsilon *= epsilon_decay

        print(f"Episode {episode + 1}/{episodes}, Total Reward: {total_reward}")

    return model

# 開始訓練
if __name__ == "__main__":
    trained_model = train_placement()
