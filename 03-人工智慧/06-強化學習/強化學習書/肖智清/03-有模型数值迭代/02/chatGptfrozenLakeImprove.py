# https://chatgpt.com/c/67282cb6-f8fc-8012-8dde-616029bcc0d2
# https://chatgpt.com/share/67282ef2-7760-8012-97bf-ab5a7f775a53

import sys
import logging
import itertools

import numpy as np
np.random.seed(0)  # 設定隨機種子，使結果可重現
import gym

# 設定日誌的基本配置
logging.basicConfig(level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        stream=sys.stdout, datefmt='%H:%M:%S')

# 建立 FrozenLake 環境
env = gym.make('FrozenLake-v1')
# 日誌顯示環境的觀測空間和動作空間資訊
logging.info('observation space = %s', env.observation_space)
logging.info('action space = %s', env.action_space)
logging.info('number of states = %s', env.observation_space.n)
logging.info('number of actions = %s', env.action_space.n)
logging.info('P[14] = %s', env.P[14])  # 顯示第 14 個狀態的動作結果
logging.info('P[14][2] = %s', env.P[14][2])  # 顯示第 14 個狀態下選擇動作 2 的結果
logging.info('reward threshold = %s', env.spec.reward_threshold)
logging.info('max episode steps = %s', env.spec.max_episode_steps)

# 定義執行政策的函式
def play_policy(env, policy, render=False):
    episode_reward = 0.
    observation, _ = env.reset()  # 重設環境
    while True:
        if render:
            env.render()  # 渲染環境
        # 根據當前狀態選擇動作
        action = np.random.choice(env.action_space.n, p=policy[observation])
        observation, reward, terminated, truncated, _ = env.step(action)
        episode_reward += reward  # 累加獎勵
        if terminated or truncated:  # 若遊戲結束則跳出
            break
    return episode_reward

# 測試隨機策略
logging.info('==== Random policy ====')
random_policy = np.ones((env.observation_space.n, env.action_space.n)) / \
        env.action_space.n  # 建立隨機策略

# 執行 100 次測試隨機策略的平均報酬
episode_rewards = [play_policy(env, random_policy)  for _ in range(100)]
logging.info('average episode reward = %.2f ± %.2f',
        np.mean(episode_rewards), np.std(episode_rewards))

# 計算給定狀態值的行為值
def v2q(env, v, state=None, gamma=1.):
    if state is not None:  # 若指定狀態，則只計算該狀態的行為值
        q = np.zeros(env.action_space.n)
        for action in range(env.action_space.n):
            for prob, next_state, reward, terminated in env.P[state][action]:
                q[action] += prob * (reward + gamma * v[next_state] * (1. - terminated))
    else:  # 若未指定狀態，則計算所有狀態的行為值
        q = np.zeros((env.observation_space.n, env.action_space.n))
        for state in range(env.observation_space.n):
            q[state] = v2q(env, v, state, gamma)
    return q

# 評估策略的狀態值
def evaluate_policy(env, policy, gamma=1., tolerant=1e-6):
    v = np.zeros(env.observation_space.n)  # 初始化狀態值
    while True:
        delta = 0
        for state in range(env.observation_space.n):
            # 計算當前策略在該狀態下的狀態值
            vs = sum(policy[state] * v2q(env, v, state, gamma))
            delta = max(delta, abs(v[state]-vs))  # 更新最大誤差
            v[state] = vs
        if delta < tolerant:  # 當變化小於容忍度時停止迭代
            break
    return v

# 評估隨機策略的狀態值
v_random = evaluate_policy(env, random_policy)
logging.info('state value:\n%s', v_random.reshape(4, 4))

# 計算隨機策略的行為值
q_random = v2q(env, v_random)
logging.info('action value:\n%s', q_random)

# 改善策略以取得最佳策略
def improve_policy(env, v, policy, gamma=1.):
    optimal = True
    for state in range(env.observation_space.n):
        q = v2q(env, v, state, gamma)
        action = np.argmax(q)  # 選擇最佳行為
        if policy[state][action] != 1.:  # 若策略不符合最佳行為則更新
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

# 執行策略迭代以求解最佳策略
def iterate_policy(env, gamma=1., tolerant=1e-6):
    policy = np.ones((env.observation_space.n,
            env.action_space.n)) / env.action_space.n  # 初始化策略
    while True:
        v = evaluate_policy(env, policy, gamma, tolerant)
        if improve_policy(env, v, policy):  # 若策略無法進一步提升則停止
            break
    return policy, v

policy_pi, v_pi = iterate_policy(env)
logging.info('optimal state value =\n%s', v_pi.reshape(4, 4))
logging.info('optimal policy =\n%s', np.argmax(policy_pi, axis=1).reshape(4, 4))

# 使用最佳策略測試遊戲的平均報酬
episode_rewards = [play_policy(env, policy_pi)  for _ in range(100)]
logging.info('average episode reward = %.2f ± %.2f',
        np.mean(episode_rewards), np.std(episode_rewards))

# 執行值迭代以求解最佳策略
def iterate_value(env, gamma=1, tolerant=1e-6):
    v = np.zeros(env.observation_space.n)  # 初始化狀態值
    while True:
        delta = 0
        for state in range(env.observation_space.n):
            vmax = max(v2q(env, v, state, gamma))  # 更新狀態值
            delta = max(delta, abs(v[state]-vmax))
            v[state] = vmax
        if delta < tolerant:  # 當變化小於容忍度時停止
            break

    # 計算最佳策略
    policy = np.zeros((env.observation_space.n, env.action_space.n))
    for state in range(env.observation_space.n):
        action = np.argmax(v2q(env, v, state, gamma))
        policy[state][action] = 1.
    return policy, v

policy_vi, v_vi = iterate_value(env)
logging.info('optimal state value =\n%s', v_vi.reshape(4, 4))
logging.info('optimal policy = \n%s', np.argmax(policy_vi, axis=1).reshape(4, 4))

# 使用最佳策略測試遊戲的平均報酬
episode_rewards = [play_policy(env, policy_vi) for _ in range(100)]
logging.info('average episode reward = %.2f ± %.2f',
        np.mean(episode_rewards), np.std(episode_rewards))
