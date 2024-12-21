import numpy as np
import random

# 環境設定：簡單的格子世界
class GridWorld:
    def __init__(self):
        self.state_space = 5  # 總共5個狀態
        self.action_space = 2  # 0: 左, 1: 右
        self.current_state = 0  # 初始狀態
    
    def reset(self):
        self.current_state = 0  # 重置狀態
        return self.current_state
    
    def step(self, action):
        if action == 0:  # 向左移動
            self.current_state = max(0, self.current_state - 1)
        elif action == 1:  # 向右移動
            self.current_state = min(self.state_space - 1, self.current_state + 1)
        
        reward = 1 if self.current_state == self.state_space - 1 else 0  # 最後一個狀態獲得獎勵
        done = self.current_state == self.state_space - 1  # 是否達到終止狀態
        
        return self.current_state, reward, done

# Q-learning 算法實作
def q_learning(env, episodes, alpha, gamma, epsilon):
    q_table = np.zeros((env.state_space, env.action_space))  # Q表
    for episode in range(episodes):
        state = env.reset()
        done = False
        while not done:
            # ε-greedy 策略
            if random.uniform(0, 1) < epsilon:
                action = random.choice(range(env.action_space))  # 隨機選擇行動
            else:
                action = np.argmax(q_table[state])  # 選擇最大 Q 值的行動
            
            next_state, reward, done = env.step(action)  # 獲取下一狀態和獎勵
            
            # 更新 Q 值
            q_table[state, action] += alpha * (reward + gamma * np.max(q_table[next_state]) - q_table[state, action])
            state = next_state  # 移動到下一狀態
            
    return q_table

# 模擬設定
env = GridWorld()
episodes = 1000  # 總共1000個回合
alpha = 0.1  # 學習率
gamma = 0.9  # 折扣因子
epsilon = 0.1  # 探索概率

# 執行 Q-learning
q_table = q_learning(env, episodes, alpha, gamma, epsilon)

# 顯示學習結果
print("學習到的 Q 表：")
print(q_table)
