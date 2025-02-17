import sys
import logging
import itertools
import inspect

import numpy as np
np.random.seed(0)
import scipy.optimize
import gym

logging.basicConfig(level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        stream=sys.stdout, datefmt='%H:%M:%S')

env = gym.make('CliffWalking-v0')
for key in vars(env):
    logging.info('%s: %s', key, vars(env)[key])
logging.info('type = %s', inspect.getmro(type(env)))

def evaluate_bellman(env, policy, gamma=1., verbose=True):
    if verbose:
        logging.info('policy = %s', policy)
    a, b = np.eye(env.nS), np.zeros((env.nS))
    for state in range(env.nS - 1):
        for action in range(env.nA):
            pi = policy[state][action]
            for p, next_state, reward, terminated in env.P[state][action]:
                a[state, next_state] -= (pi * gamma * p)
                b[state] += (pi * reward * p)
    v = np.linalg.solve(a, b)
    q = np.zeros((env.nS, env.nA))
    for state in range(env.nS - 1):
        for action in range(env.nA):
            for p, next_state, reward, terminated in env.P[state][action]:
                q[state][action] += ((reward + gamma * v[next_state]) * p)
    if verbose:
        logging.info('state values = %s', v)
        logging.info('action values = %s', q)
    return v, q

policy = np.ones((env.nS, env.nA)) / env.nA
state_values, action_values = evaluate_bellman(env, policy)

actions = np.ones(env.nS, dtype=int)
actions[36] = 0
actions[11::12] = 2
policy = np.eye(env.nA)[actions]
state_values, action_values = evaluate_bellman(env, policy)

policy = np.random.uniform(size=(env.nS, env.nA))
policy = policy / np.sum(policy, axis=1, keepdims=True)
state_values, action_values = evaluate_bellman(env, policy)

def optimal_bellman(env, gamma=1., verbose=True):
    p = np.zeros((env.nS, env.nA, env.nS))
    r = np.zeros((env.nS, env.nA))
    for state in range(env.nS - 1):
        for action in range(env.nA):
            for prob, next_state, reward, terminated in env.P[state][action]:
                p[state, action, next_state] += prob
                r[state, action] += (reward * prob)
    c = np.ones(env.nS)
    a_ub = gamma * p.reshape(-1, env.nS) - \
            np.repeat(np.eye(env.nS), env.nA, axis=0)
    b_ub = -r.reshape(-1)
    a_eq = np.zeros((0, env.nS))
    b_eq = np.zeros(0)
    bounds = [(None, None),] * env.nS
    res = scipy.optimize.linprog(c, a_ub, b_ub, bounds=bounds,
            method='interior-point')
    v = res.x
    q = r + gamma * np.dot(p, v)
    if verbose:
        logging.info('optimal state values = %s', v)
        logging.info('optimal action values = %s', q)
    return v, q

optimal_state_values, optimal_action_values = optimal_bellman(env)

optimal_actions = optimal_action_values.argmax(axis=1)
logging.info('optimal policy = %s', optimal_actions)
