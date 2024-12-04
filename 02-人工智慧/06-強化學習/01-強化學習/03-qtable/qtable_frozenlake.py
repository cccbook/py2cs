# import gym
import gymnasium as gym
import numpy as np

env = gym.make('FrozenLake-v1', render_mode="rgb_array")
print('env=', env)
print('env.observation_space=', env.observation_space)
print('env.action_space=', env.action_space)
# print('env.reward_range=', env.reward_range)
print('env.spec=', env.spec)
print('env.metadata=', env.metadata)

alpha = 0.8          # 學習速率
gamma = 0.95         # 折扣因子
num_episodes = 2000  # 迭代次數
Q = np.zeros([env.observation_space.n,env.action_space.n]) # Q-table 初始化

for i in range(num_episodes): # 學習循環
    s, info = env.reset() # 初始化環境
    for j in range(99): # 行動根據報酬調整 Q 表
        a = np.argmax(Q[s,:] + np.random.randn(1,env.action_space.n)*(1./(i+1))) # 選擇報酬最高的行動 (加上一點隨機，才有可能探索所有行動，但隨機性隨時間降低，逐漸消失)
        s1, reward, terminated, truncated, info = env.step(a) # 執行動作，取得回饋
        Q[s,a] += alpha*(reward + gamma*np.max(Q[s1,:]) - Q[s,a]) # Q-Learning 公式
        #       學習速率*(真實報酬+ 折扣因子*下一步最佳報酬 -預測報酬)
        s = s1
        if terminated == True:
            break

print('Q=', Q)

print('完成迭代，展示學習成果 ...')

env = gym.make('FrozenLake-v1', render_mode="human")
s, info = env.reset()
for i in range(100):
    env.render()
    a = np.argmax(Q[s,:]) # 永遠取 Q table 中的最佳行動
    s, reward, terminated, truncated, info = env.step(a)
    if terminated == True:
        break
