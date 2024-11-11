import sys
import logging
import itertools

import numpy as np
np.random.seed(0)
import gym

logging.basicConfig(level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        stream=sys.stdout, datefmt='%H:%M:%S')

env = gym.make('BipedalWalker-v3')
for key in vars(env):
    logging.info('%s: %s', key, vars(env)[key])
for key in vars(env.spec):
    logging.info('%s: %s', key, vars(env.spec)[key])

class ClosedFormAgent:
    def __init__(self, env):
        self.weights = np.array([
            [ 0.9, -0.7,  0.0, -1.4],
            [ 4.3, -1.6, -4.4, -2.0],
            [ 2.4, -4.2, -1.3, -0.1],
            [-3.1, -5.0, -2.0, -3.3],
            [-0.8,  1.4,  1.7,  0.2],
            [-0.7,  0.2, -0.2,  0.1],
            [-0.6, -1.5, -0.6,  0.3],
            [-0.5, -0.3,  0.2,  0.1],
            [ 0.0, -0.1, -0.1,  0.1],
            [ 0.4,  0.8, -1.6, -0.5],
            [-0.4,  0.5, -0.3, -0.4],
            [ 0.3,  2.0,  0.9, -1.6],
            [ 0.0, -0.2,  0.1, -0.3],
            [ 0.1,  0.2, -0.5, -0.3],
            [ 0.7,  0.3,  5.1, -2.4],
            [-0.4, -2.3,  0.3, -4.0],
            [ 0.1, -0.8,  0.3,  2.5],
            [ 0.4, -0.9, -1.8,  0.3],
            [-3.9, -3.5,  2.8,  0.8],
            [ 0.4, -2.8,  0.4,  1.4],
            [-2.2, -2.1, -2.2, -3.2],
            [-2.7, -2.6,  0.3,  0.6],
            [ 2.0,  2.8,  0.0, -0.9],
            [-2.2,  0.6,  4.7, -4.6],
            ])
        self.bias = np.array([3.2, 6.1, -4.0, 7.6])

    def reset(self, mode=None):
        pass

    def step(self, observation, reward, terminated):
        action = np.matmul(observation, self.weights) + self.bias
        return action

    def close(self):
        pass


agent = ClosedFormAgent(env)

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


logging.info('==== test ====')
episode_rewards = []
for episode in range(100):
    episode_reward, elapsed_steps = play_episode(env, agent)
    episode_rewards.append(episode_reward)
    logging.info('test episode %d: reward = %.2f, steps = %d',
            episode, episode_reward, elapsed_steps)
logging.info('average episode reward = %.2f ± %.2f',
        np.mean(episode_rewards), np.std(episode_rewards))

