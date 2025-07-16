import gym
import numpy as np

# 定义环境
env = gym.make("MountainCar-v0")

# 定义 Q 表
q_table = np.zeros((20, 20, 3))

# 定义超参数
alpha = 0.1
gamma = 0.9
epsilon = 0.1

# 定义训练循环
for episode in range(10000):
    # 初始化环境
    state = env.reset()
    done = False
    total_reward = 0

    # 开始 episode
    while not done:
        # 选择动作
        if np.random.uniform(0, 1) < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(q_table[state[0]][state[1]])
        
        # 执行动作，获得下一个状态、奖励和是否结束的信息
        next_state, reward, done, _ = env.step(action)
        total_reward += reward

        # 更新 Q 表
        q_table[state[0]][state[1]][action] = (1 - alpha) * q_table[state[0]][state[1]][action] + alpha * (reward + gamma * np.max(q_table[next_state[0]][next_state[1]]))

        # 更新状态
        state = next_state
    
    # 打印结果
    print("Episode {}: Total reward = {}".format(episode, total_reward))
