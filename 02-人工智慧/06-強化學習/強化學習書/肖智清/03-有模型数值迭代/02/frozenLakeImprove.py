import sys
import logging
import itertools

import numpy as np
np.random.seed(0)
import gym

logging.basicConfig(level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        stream=sys.stdout, datefmt='%H:%M:%S')

env = gym.make('FrozenLake-v1')
logging.info('observation space = %s', env.observation_space)
logging.info('action space = %s', env.action_space)
logging.info('number of states = %s', env.observation_space.n)
logging.info('number of actions = %s', env.action_space.n)
logging.info('P[14] = %s', env.P[14])
logging.info('P[14][2] = %s', env.P[14][2])
logging.info('reward threshold = %s', env.spec.reward_threshold)
logging.info('max episode steps = %s', env.spec.max_episode_steps)

def play_policy(env, policy, render=False):
    episode_reward = 0.
    observation, _ = env.reset()
    while True:
        if render:
            env.render()  # render the environment
        action = np.random.choice(env.action_space.n, p=policy[observation])
        observation, reward, terminated, truncated, _ = env.step(action)
        episode_reward += reward
        if terminated or truncated:
            break
    return episode_reward

logging.info('==== Random policy ====')
random_policy = np.ones((env.observation_space.n, env.action_space.n)) / \
        env.action_space.n

episode_rewards = [play_policy(env, random_policy)  for _ in range(100)]
logging.info('average episode reward = %.2f ± %.2f',
        np.mean(episode_rewards), np.std(episode_rewards))

def v2q(env, v, state=None, gamma=1.):  # calculate action value from state value
    if state is not None:  # solve for single state
        q = np.zeros(env.action_space.n)
        for action in range(env.action_space.n):
            for prob, next_state, reward, terminated in env.P[state][action]:
                q[action] += prob * \
                        (reward + gamma * v[next_state] * (1. - terminated))
    else:  # solve for all states
        q = np.zeros((env.observation_space.n, env.action_space.n))
        for state in range(env.observation_space.n):
            q[state] = v2q(env, v, state, gamma)
    return q

def evaluate_policy(env, policy, gamma=1., tolerant=1e-6):
    v = np.zeros(env.observation_space.n)  # initialize state values
    while True:
        delta = 0
        for state in range(env.observation_space.n):
            vs = sum(policy[state] * v2q(env, v, state, gamma))  # update state value
            delta = max(delta, abs(v[state]-vs))  # update max error
            v[state] = vs
        if delta < tolerant:  # check whether iterations can finish
            break
    return v

v_random = evaluate_policy(env, random_policy)
logging.info('state value:\n%s', v_random.reshape(4, 4))

q_random = v2q(env, v_random)
logging.info('action value:\n%s', q_random)

def improve_policy(env, v, policy, gamma=1.):
    optimal = True
    for state in range(env.observation_space.n):
        q = v2q(env, v, state, gamma)
        action = np.argmax(q)
        if policy[state][action] != 1.:
            optimal = False
            policy[state] = 0.
            policy[state][action] = 1.
    return optimal

policy = random_policy.copy()
optimal = improve_policy(env, v_random, policy)
if optimal:
    logging.info('No update. Optimal policy is:\n%s', policy)
else:
    logging.info('Updating completes. Updated policy is:\n%s', policy)

def iterate_policy(env, gamma=1., tolerant=1e-6):
    policy = np.ones((env.observation_space.n,
            env.action_space.n)) / env.action_space.n  # initialize
    while True:
        v = evaluate_policy(env, policy, gamma, tolerant)
        if improve_policy(env, v, policy):
            break
    return policy, v

policy_pi, v_pi = iterate_policy(env)
logging.info('optimal state value =\n%s', v_pi.reshape(4, 4))
logging.info('optimal policy =\n%s', np.argmax(policy_pi, axis=1).reshape(4, 4))

episode_rewards = [play_policy(env, policy_pi)  for _ in range(100)]
logging.info('average episode reward = %.2f ± %.2f',
        np.mean(episode_rewards), np.std(episode_rewards))

def iterate_value(env, gamma=1, tolerant=1e-6):
    v = np.zeros(env.observation_space.n)  # initialization
    while True:
        delta = 0
        for state in range(env.observation_space.n):
            vmax = max(v2q(env, v, state, gamma))  # update state value
            delta = max(delta, abs(v[state]-vmax))
            v[state] = vmax
        if delta < tolerant:  # check whether iterations can finish
            break

    # calculate optimal policy
    policy = np.zeros((env.observation_space.n, env.action_space.n))
    for state in range(env.observation_space.n):
        action = np.argmax(v2q(env, v, state, gamma))
        policy[state][action] = 1.
    return policy, v

policy_vi, v_vi = iterate_value(env)
logging.info('optimal state value =\n%s', v_vi.reshape(4, 4))
logging.info('optimal policy = \n%s', np.argmax(policy_vi, axis=1).reshape(4, 4))

episode_rewards = [play_policy(env, policy_vi) for _ in range(100)]
logging.info('average episode reward = %.2f ± %.2f',
        np.mean(episode_rewards), np.std(episode_rewards))

