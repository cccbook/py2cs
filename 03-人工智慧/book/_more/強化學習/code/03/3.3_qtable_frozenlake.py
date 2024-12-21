# import gym
import gymnasium as gym
import numpy as np

# env = gym.make('FrozenLake-v1', render_mode="rgb_array")
env = gym.make('FrozenLake-v1', render_mode="rgb_array")
print('env=', env)
print('env.observation_space=', env.observation_space)
print('env.action_space=', env.action_space)
print('env.reward_range=', env.reward_range)
print('env.spec=', env.spec)
print('env.metadata=', env.metadata)

# 超參數
alpha = 0.8          # 學習速率
gamma = 0.95         # 折扣因子
num_episodes = 2000  # 迭代次數

# Q-table
Q = np.zeros([env.observation_space.n,env.action_space.n])

# 執行所有迭代
for i in range(num_episodes):
    # 初始化環境
    s, info = env.reset()
    # print('info=', info)
    rAll = 0
    d = False
    j = 0
    # Q-learning 更新規則
    while j < 99:
        j += 1
        # 選擇行動
        # print('s=', s)
        a = np.argmax(Q[s,:] + np.random.randn(1,env.action_space.n)*(1./(i+1)))
        # 取得新的狀態和報酬
        s1, reward, terminated, truncated, info = env.step(a)
        # 將新的知識累積到 Q-table 中
        Q[s,a] = Q[s,a] + alpha*(reward + gamma*np.max(Q[s1,:]) - Q[s,a])
        rAll += reward
        s = s1
        if terminated == True:
            break

print('Q=', Q)
print("完成迭代")

print('展示學習成果 ...')

env = gym.make('FrozenLake-v1', render_mode="human")
# env = gym.make('FrozenLake-v1', render_mode="ansi")
s, info = env.reset()
for i in range(100):
    env.render()
    a = np.argmax(Q[s,:])
    s, reward, terminated, truncated, info = env.step(a)
    if terminated == True:
        break
