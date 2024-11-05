import numpy as np
import gym

# 環境設定
env = gym.make("FrozenLake-v1", is_slippery=True)
# env.seed(0)  # 為了結果一致性
np.random.seed(0)

# 參數設定
gamma = 0.99      # 折扣因子
alpha = 0.1       # 學習率
epsilon = 0.1     # ε-貪婪策略的探索參數
num_episodes = 10000  # 訓練回合數
num_states = env.observation_space.n
num_actions = env.action_space.n

# 初始化 Q 表
Q = np.zeros((num_states, num_actions))

def epsilon_greedy_action(state, Q, epsilon):
    """ε-貪婪策略選擇行動"""
    if np.random.rand() < epsilon:
        return np.random.choice(num_actions)  # 探索
    else:
        return np.argmax(Q[state])  # 利用

# SARSA 演算法
for episode in range(num_episodes):
    state,_ = env.reset()  # 重置環境
    action = epsilon_greedy_action(state, Q, epsilon)

    while True:
        # 執行動作，觀察新狀態、獎勵、和完成標誌
        # next_state, reward, done, _ = env.step(action)
        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        next_action = epsilon_greedy_action(next_state, Q, epsilon)

        # SARSA 更新 Q 值
        td_target = reward + gamma * Q[next_state][next_action] * (not done)
        td_error = td_target - Q[state][action]
        Q[state][action] += alpha * td_error

        # 更新狀態和行動
        state, action = next_state, next_action

        if done:
            break

# 結果顯示
print("Optimal Policy (from SARSA):")
print(np.argmax(Q, axis=1).reshape((4, 4)))  # 以 4x4 格式顯示 Frozen Lake 策略
print("\nQ-Table:")
print(Q.reshape((4, 4, num_actions)))
