import sys
import logging
import itertools

import numpy as np
np.random.seed(0)
import gymnasium as gym  # 改用 gymnasium

logging.basicConfig(level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        stream=sys.stdout, datefmt='%H:%M:%S')

env = gym.make('CartPole-v1')  # 在 gymnasium 中使用 v1 版本，v0 已經過時
for key in vars(env.spec):
    logging.info('%s: %s', key, vars(env.spec)[key])
for key in vars(env.unwrapped):
    logging.info('%s: %s', key, vars(env.unwrapped)[key])

class ClosedFormAgent:
    def __init__(self, _):
        pass

    def reset(self, mode=None):
        pass

    def step(self, observation, reward, terminated):
        position, velocity, angle, angle_velocity = observation
        action = int(3. * angle + angle_velocity > 0.)
        return action

    def close(self):
        pass


agent = ClosedFormAgent(env)

def play_episode(env, agent, seed=None, mode=None, render=False):
    observation, info = env.reset(seed=seed)  # gymnasium 返回 (observation, info)
    reward, terminated, truncated = 0., False, False
    agent.reset(mode=mode)
    episode_reward, elapsed_steps = 0., 0
    while True:
        action = agent.step(observation, reward, terminated)
        if render:
            env.render()  # 在 gymnasium 中，render 方法已經在創建環境時設置
        if terminated or truncated:
            break
        observation, reward, terminated, truncated, info = env.step(action)  # gymnasium 返回 5 個值
        episode_reward += reward
        elapsed_steps += 1
    agent.close()
    return episode_reward, elapsed_steps


logging.info('==== test ====')
episode_rewards = []
for episode in range(100):
    episode_reward, elapsed_steps = play_episode(env, agent)
    episode_rewards.append(episode_reward)
    logging.info('test episode %d: reward = %.2f, steps = %d',
            episode, episode_reward, elapsed_steps)
logging.info('average episode reward = %.2f ± %.2f',
        np.mean(episode_rewards), np.std(episode_rewards))
env.close()

# 使用 render (for human) 動畫播放玩一次
env = gym.make('CartPole-v1', render_mode="human")  # 在 gymnasium 中使用 render_mode 參數
episode_reward, elapsed_steps = play_episode(env, agent, render=True)
env.close()