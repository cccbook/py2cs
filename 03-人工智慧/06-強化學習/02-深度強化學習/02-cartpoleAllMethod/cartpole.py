# https://zhiqingxiao.github.io/rl-book/en2024/code/CartPole-v0_VPG_torch.html
import sys
import logging
import itertools
import numpy as np
np.random.seed(0)
import pandas as pd
import gym
import matplotlib.pyplot as plt

# 設定 logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s [%(levelname)s] %(message)s',
                    stream=sys.stdout, datefmt='%H:%M:%S')

# 創建 CartPole 環境
env = gym.make('CartPole-v0')
env._max_episode_steps = 1000 # CartPole-v0：預設情況下，環境會在達到 200 步時自動結束遊戲。 v1 是 500

for key in vars(env):
    logging.info('%s: %s', key, vars(env)[key])
for key in vars(env.spec):
    logging.info('%s: %s', key, vars(env.spec)[key])

from VPGAgent import VPGAgent
from VPGwBaselineAgent import VPGwBaselineAgent
from DQNAgent import DQNAgent
from SARSAAgent import SARSAAgent
from ActorCriticAgent import ActorCriticAgent
# from SARSALambdaAgent import SARSALambdaAgent

# 初始化代理
# agent = VPGAgent(env)
# agent = VPGwBaselineAgent(env)
# agent = DQNAgent(env)
# agent = SARSAAgent(env) # agent = SARSALambdaAgent(env) # fail
agent = ActorCriticAgent(env)

# 定義遊玩一個回合的函數
def play_episode(env, agent, seed=None, mode=None, render=False):
    observation, _ = env.reset(seed=seed)
    reward, terminated, truncated = 0., False, False
    agent.reset(mode=mode)
    episode_reward, elapsed_steps = 0., 0
    while True:
        action = agent.step(observation, reward, terminated)
        if render:
            env.render()
        if terminated or truncated:
            break
        observation, reward, terminated, truncated, _ = env.step(action)
        episode_reward += reward
        elapsed_steps += 1
    agent.close()
    return episode_reward, elapsed_steps

# 訓練代理
logging.info('==== train ====')
episode_rewards = []
for episode in itertools.count():
    episode_reward, elapsed_steps = play_episode(env, agent, seed=episode, mode='train')
    episode_rewards.append(episode_reward)
    logging.info('train episode %d: reward = %.2f, steps = %d',
                 episode, episode_reward, elapsed_steps)
    if np.mean(episode_rewards[-20:]) > env._max_episode_steps-10:
        break

# 繪製訓練獎勵
plt.plot(episode_rewards)
plt.xlabel('Episodes')
plt.ylabel('Episode Reward')
plt.title('Training Rewards')
plt.show()

# 測試代理
logging.info('==== test ====')
episode_rewards = []
for episode in range(100):
    episode_reward, elapsed_steps = play_episode(env, agent)
    episode_rewards.append(episode_reward)
    logging.info('test episode %d: reward = %.2f, steps = %d',
                 episode, episode_reward, elapsed_steps)

logging.info('average episode reward = %.2f ± %.2f',
             np.mean(episode_rewards), np.std(episode_rewards))

# 使用 render (for human) 動畫播放玩一次
env = gym.make('CartPole-v0', render_mode="human")
episode_reward, elapsed_steps = play_episode(env, agent, render=True)
env.close()