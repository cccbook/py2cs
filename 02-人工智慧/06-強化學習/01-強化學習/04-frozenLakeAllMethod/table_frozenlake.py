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

# method = "Q"
# method = "SARSA"
method = "TD_LAMBDA"
if method == "TD_LAMBDA":
    # lambda_ = 0.8         # 慢慢衰減 (會掉下去)
    # lambda_ = 1.0         # 不衰減 (會掉下去)
    # lambda_ = 0.0         # 完全衰減（會過，退化回 SARSA)
    lambda_ = 0.2           # 快速衰減（會過)

for i in range(num_episodes): # 學習循環
    s, info = env.reset() # 初始化環境

    if method == "TD_LAMBDA":
        E = np.zeros([env.observation_space.n, env.action_space.n])  # 初始化資格跡矩陣

    for j in range(99): # 行動根據報酬調整 Q 表
        a = np.argmax(Q[s,:] + np.random.randn(1,env.action_space.n)*(1./(i+1))) # 選擇報酬最高的行動 (加上一點隨機，才有可能探索所有行動，但隨機性隨時間降低，逐漸消失)
        s1, reward, terminated, truncated, info = env.step(a) # 執行動作，取得回饋
        # [Q-learning / SARSA 與 TD(lambda)](https://chatgpt.com/c/672adadf-1214-8012-9eef-16985976a352)
        if method == "Q": # Q-Learning: 不用考慮這次的動作 a1
            Q[s,a] += alpha*(reward + gamma*np.max(Q[s1,:]) - Q[s,a]) # Q-Learning 公式
            #       學習速率*(真實報酬+ 折扣因子*下一步最佳報酬 -預測報酬)
        elif method == "SARSA":
            # SARSA 公式: 考慮這次的動作 a1
            a1 = np.argmax(Q[s1, :] + np.random.randn(1, env.action_space.n) * (1. / (i + 1)))  # 選擇下一步行動 a'
            Q[s, a] += alpha * (reward + gamma * Q[s1, a1] - Q[s, a])  # 使用 SARSA 更新公式
        elif method == "TD_LAMBDA":  # TD(λ) : 引入 λ 權衡短期與長期回報的影響
            a1 = np.argmax(Q[s1, :] + np.random.randn(1, env.action_space.n) * (1. / (i + 1)))
            delta = reward + gamma * Q[s1, a1] - Q[s, a]  # TD誤差計算
            
            # 更新資格跡
            for s2 in range(env.observation_space.n):
                for a2 in range(env.action_space.n):
                    E[s2, a2] *= gamma * lambda_ # 資格跡衰減
            E[s,a] += 1

            Q[s, a] += alpha * delta * E[s, a]
            # Q[s, a] += alpha * delta # 不使用 E[s,a], 退化為 SARSA
        else:
            raise Exception(f"無法處理的 method={method}")

        s = s1
        if terminated == True:
            break

print('Q=', Q)

print('E=', E)
print('完成迭代，展示學習成果 ...')

env = gym.make('FrozenLake-v1', render_mode="human")
s, info = env.reset()
for i in range(100):
    env.render()
    a = np.argmax(Q[s,:]) # 永遠取 Q table 中的最佳行動
    s, reward, terminated, truncated, info = env.step(a)
    if terminated == True:
        break
