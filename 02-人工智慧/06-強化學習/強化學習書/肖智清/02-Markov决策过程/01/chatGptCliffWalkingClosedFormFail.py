import sys
import logging
import itertools

import numpy as np
np.random.seed(0)  # 設定隨機數種子，以便重現結果
import gym  # 匯入OpenAI的gym庫以便使用強化學習環境

# 設定日誌記錄的基本配置
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s [%(levelname)s] %(message)s',
                    stream=sys.stdout, datefmt='%H:%M:%S')

# 創建CliffWalking環境
env = gym.make('CliffWalking-v0')

# 記錄環境的屬性資訊
for key in vars(env.spec):
    logging.info('%s: %s', key, vars(env.spec)[key])  # 輸出環境規格
for key in vars(env.unwrapped):
    logging.info('%s: %s', key, vars(env.unwrapped)[key])  # 輸出環境未包裝的屬性

# 定義ClosedFormAgent類別
class ClosedFormAgent:
    def __init__(self, _):
        pass  # 這裡可以初始化代理的變數，當前不需要

    def reset(self, mode=None):
        pass  # 重置代理的狀態，根據需要進行初始化

    def step(self, observation, reward, terminated):
        # 根據觀察值決定行動
        if observation == 36:  # 如果在右下角（目標位置）
            action = 3  # 向左移動
        elif observation % 12 == 11:  # 如果在最右側的列（需要向下移動）
            action = 2  # 向下移動
        else:
            action = 1  # 否則，向右移動
        return action  # 返回選擇的行動

    def close(self):
        pass  # 可以在這裡處理代理的清理工作

# 創建一個ClosedFormAgent的實例
agent = ClosedFormAgent(env)

# 定義執行一集的函數
def play_episode(env, agent, seed=None, mode=None, render=False):
    observation, _ = env.reset(seed=seed)  # 重置環境並獲取初始觀察
    reward, terminated, truncated = 0., False, False  # 初始化獎勵和終止標誌
    agent.reset(mode=mode)  # 重置代理
    episode_reward, elapsed_steps = 0., 0  # 初始化總獎勵和步數
    while True:
        action = agent.step(observation, reward, terminated)  # 代理根據當前狀態決定行動
        if render:  # 如果需要渲染
            env.render()  # 顯示當前環境
        if terminated or truncated:  # 如果達到終止或截斷條件
            break  # 結束當前集
        # 執行行動並獲取新的觀察和獎勵
        observation, reward, terminated, truncated, _ = env.step(action)
        episode_reward += reward  # 更新總獎勵
        elapsed_steps += 1  # 更新步數
    agent.close()  # 關閉代理
    return episode_reward, elapsed_steps  # 返回總獎勵和步數

# 開始測試
logging.info('==== test ====')
episode_rewards = []  # 儲存每集的獎勵
for episode in range(100):  # 執行100集測試
    episode_reward, elapsed_steps = play_episode(env, agent)  # 執行一集
    episode_rewards.append(episode_reward)  # 儲存獎勵
    logging.info('test episode %d: reward = %.2f, steps = %d',
                 episode, episode_reward, elapsed_steps)  # 記錄測試結果

# 計算並記錄平均獎勵
logging.info('average episode reward = %.2f ± %.2f',
             np.mean(episode_rewards), np.std(episode_rewards))

env.close()  # 關閉環境
