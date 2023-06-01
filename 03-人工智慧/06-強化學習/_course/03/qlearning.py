#import gym
import gymnasium as gym
import numpy as np

# 建立FrozenLake環境
env = gym.make('FrozenLake-v1')

# 初始化Q-table
Q = np.zeros([env.observation_space.n, env.action_space.n])

# 定義參數
learning_rate = 0.8
discount_factor = 0.95
num_episodes = 2000

# 執行Q-learning算法
for episode in range(num_episodes):
    state, info = env.reset()
    done = False

    while not done:
        # 根據Q-table選擇動作
        action = np.argmax(Q[state, :] + np.random.randn(1, env.action_space.n) * (1.0 / (episode + 1)))

        # 執行動作，觀察下一個狀態和獎勵
        next_state, reward, done, _, info = env.step(action)

        # 更新Q-table
        Q[state, action] += learning_rate * (reward + discount_factor * np.max(Q[next_state, :]) - Q[state, action])

        # 更新狀態
        state = next_state

print('Q=', Q)

# env = gym.make('FrozenLake-v1', render_mode="human")
env = gym.make('FrozenLake-v1', render_mode="rgb_array")
# 測試Q-learning的表現
num_test_episodes = 100
num_successful_episodes = 0

for episode in range(num_test_episodes):
    state, info = env.reset()
    done = False

    while not done:
        # 根據Q-table選擇動作
        action = np.argmax(Q[state, :])

        # 執行動作，觀察下一個狀態和獎勵
        next_state, reward, done, _, info = env.step(action)

        # 更新狀態
        state = next_state

        # 判斷是否成功到達目標狀態
        if done and reward == 1:
            num_successful_episodes += 1

# 計算成功率
success_rate = num_successful_episodes / num_test_episodes
print("成功率: {:.2f}%".format(success_rate * 100))
