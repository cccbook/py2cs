# https://chatgpt.com/c/672825c8-41b0-8012-94d7-9315e8f2fe7a
# https://chatgpt.com/share/67282c61-dc08-8012-8fab-9224596b1b66
import sys
import logging
import itertools
import numpy as np
import gym

# 初始化隨機種子和環境
np.random.seed(0)
logging.basicConfig(level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        stream=sys.stdout, datefmt='%H:%M:%S')
env = gym.make('FrozenLake-v1')

# 紀錄環境的規格，顯示環境的主要屬性
for key in vars(env.spec):
    logging.info('%s: %s', key, vars(env.spec)[key])
for key in vars(env.unwrapped):
    logging.info('%s: %s', key, vars(env.unwrapped)[key])

# 定義一個ClosedFormAgent類別，用來計算與執行策略
class ClosedFormAgent:
    def __init__(self, env):
        # 獲取狀態數量和動作數量
        state_n, action_n = env.observation_space.n, env.action_space.n
        # 初始化價值矩陣v, q表以及策略表pi
        v = np.zeros((env.spec.max_episode_steps + 1, state_n))
        q = np.zeros((env.spec.max_episode_steps + 1, state_n, action_n))
        pi = np.zeros((env.spec.max_episode_steps + 1, state_n))

        # 根據策略進行倒序遍歷，計算最優策略
        for t in range(env.spec.max_episode_steps - 1, -1, -1):
            for s in range(state_n):
                for a in range(action_n):
                    # 計算Q值，根據轉移概率p、回報r和下一狀態的價值更新
                    for p, next_s, r, d in env.P[s][a]:
                        q[t, s, a] += p * (r + (1. - float(d)) * v[t + 1, next_s])
                # 取得狀態s的最優價值及動作
                v[t, s] = q[t, s].max()
                pi[t, s] = q[t, s].argmax()
        self.pi = pi  # 儲存最優策略表

    def reset(self, mode=None):
        # 重設時間步t
        self.t = 0

    def step(self, observation, reward, terminated):
        # 使用策略表pi決定動作，並更新時間步
        action = self.pi[self.t, observation]
        self.t += 1
        return action

    def close(self):
        # 結束代理（目前無需操作）
        pass

# 建立ClosedFormAgent實例
agent = ClosedFormAgent(env)

# 定義函式play_episode來模擬一個回合
def play_episode(env, agent, seed=None, mode=None, render=False):
    observation, _ = env.reset(seed=seed)  # 獲取初始觀察
    reward, terminated, truncated = 0., False, False  # 初始化回合參數
    agent.reset(mode=mode)  # 重設代理
    episode_reward, elapsed_steps = 0., 0  # 初始化獎勵和步數計數

    # 進行迴圈直到回合結束
    while True:
        action = agent.step(observation, reward, terminated)  # 決定動作
        if render:
            env.render()  # 若啟用渲染則顯示環境
        if terminated or truncated:
            break  # 若回合結束或截斷則跳出迴圈
        observation, reward, terminated, truncated, _ = env.step(action)  # 執行動作並更新狀態
        episode_reward += reward  # 累加獎勵
        elapsed_steps += 1  # 增加步數計數

    agent.close()  # 關閉代理
    return episode_reward, elapsed_steps  # 回傳回合獎勵及步數

# 進行測試迴圈，記錄和顯示多次回合結果
logging.info('==== test ====')
episode_rewards = []
for episode in range(100):
    episode_reward, elapsed_steps = play_episode(env, agent)  # 進行一回合
    episode_rewards.append(episode_reward)  # 記錄回合獎勵
    logging.info('test episode %d: reward = %.2f, steps = %d',
            episode, episode_reward, elapsed_steps)  # 紀錄每回合資訊

# 計算並顯示平均回合獎勵與標準差
logging.info('average episode reward = %.2f ± %.2f',
        np.mean(episode_rewards), np.std(episode_rewards))
