import numpy as np
import random

# 定義環境
class Maze:
    def __init__(self):
        self.grid = np.array([[0, 0, 0, 0, 0],
                               [1, 1, 1, 1, 0],
                               [0, 0, 0, 1, 0],
                               [0, 1, 0, 1, 0],
                               [0, 0, 0, 0, 2]])
        self.start = (0, 0)
        self.end = (4, 4)
        self.state = self.start

    def reset(self):
        self.state = self.start
        return self.state

    def step(self, action):
        x, y = self.state
        if action == 0:   # 上
            x = max(x - 1, 0)
        elif action == 1: # 右
            y = min(y + 1, self.grid.shape[1] - 1)
        elif action == 2: # 下
            x = min(x + 1, self.grid.shape[0] - 1)
        elif action == 3: # 左
            y = max(y - 1, 0)

        if self.grid[x, y] == 1:  # 撞牆
            x, y = self.state  # 不變
        self.state = (x, y)
        
        if self.state == self.end:
            return self.state, 1, True  # 到達終點
        else:
            return self.state, -0.01, False  # 未到達終點，給予小的懲罰

# Q-learning 實作
class QLearningAgent:
    def __init__(self, learning_rate=0.1, discount_factor=0.9, exploration_prob=1.0, exploration_decay=0.99):
        self.q_table = np.zeros((5, 5, 4))  # 5x5的狀態，4個行動
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_prob = exploration_prob
        self.exploration_decay = exploration_decay

    def choose_action(self, state):
        if random.random() < self.exploration_prob:
            return random.randint(0, 3)  # 隨機選擇行動
        else:
            return np.argmax(self.q_table[state])  # 選擇最大 Q 值的行動

    def update_q_value(self, state, action, reward, next_state):
        best_next_action = np.argmax(self.q_table[next_state])
        td_target = reward + self.discount_factor * self.q_table[next_state][best_next_action]
        td_error = td_target - self.q_table[state][action]
        self.q_table[state][action] += self.learning_rate * td_error

    def decay_exploration(self):
        self.exploration_prob *= self.exploration_decay

# 訓練 Q-learning 代理
def train(episodes):
    maze = Maze()
    agent = QLearningAgent()
    
    for episode in range(episodes):
        state = maze.reset()
        done = False
        
        while not done:
            action = agent.choose_action(state)
            next_state, reward, done = maze.step(action)
            agent.update_q_value(state, action, reward, next_state)
            state = next_state
        
        agent.decay_exploration()
        
    return agent.q_table

# 訓練代理
q_table = train(1000)

# 顯示 Q 表
print("Q-Table:")
print(q_table)
