import sys
import logging
import itertools
import numpy as np
import gym

# Initialize random seed and environment
np.random.seed(0)
logging.basicConfig(level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        stream=sys.stdout, datefmt='%H:%M:%S')
env = gym.make('FrozenLake-v1')

# Logging environment specifications
for key in vars(env.spec):
    logging.info('%s: %s', key, vars(env.spec)[key])
for key in vars(env.unwrapped):
    logging.info('%s: %s', key, vars(env.unwrapped)[key])

class ClosedFormAgent:
    def __init__(self, env):
        state_n, action_n = env.observation_space.n, env.action_space.n
        v = np.zeros((env.spec.max_episode_steps + 1, state_n))
        q = np.zeros((env.spec.max_episode_steps + 1, state_n, action_n))
        pi = np.zeros((env.spec.max_episode_steps + 1, state_n))

        # Compute optimal policy
        for t in range(env.spec.max_episode_steps - 1, -1, -1):
            for s in range(state_n):
                for a in range(action_n):
                    for p, next_s, r, d in env.P[s][a]:
                        q[t, s, a] += p * (r + (1. - float(d)) * v[t + 1, next_s])
                v[t, s] = q[t, s].max()
                pi[t, s] = q[t, s].argmax()
        self.pi = pi

    def reset(self, mode=None):
        self.t = 0

    def step(self, observation, reward, terminated):
        action = self.pi[self.t, observation]
        self.t += 1
        return action

    def close(self):
        pass

# Create agent instance
agent = ClosedFormAgent(env)

# Function to simulate one episode
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

# Main test loop
logging.info('==== test ====')
episode_rewards = []
for episode in range(100):
    episode_reward, elapsed_steps = play_episode(env, agent)
    episode_rewards.append(episode_reward)
    logging.info('test episode %d: reward = %.2f, steps = %d',
            episode, episode_reward, elapsed_steps)

logging.info('average episode reward = %.2f ± %.2f',
        np.mean(episode_rewards), np.std(episode_rewards))
