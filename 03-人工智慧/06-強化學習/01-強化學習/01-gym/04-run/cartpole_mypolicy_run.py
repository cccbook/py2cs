import gymnasium as gym
import random
env = gym.make("CartPole-v1", render_mode="human") # 若改用這個，會畫圖
# env = gym.make("CartPole-v1", render_mode="rgb_array")
observation, info = env.reset(seed=42)
position, velocity, angle, angle_velocity = observation
score = 0
for _ in range(1000):
    env.render()
    # action = env.action_space.sample()
    # action = 1 if angle > 0 else 0
    # action = 1 if angle > 0.1 else 0
    if angle > 0.01:
        action = 1
    elif angle < -0.01:
        action = 0
    else:
        action = 0 if random.uniform(-0.02,0.02) > angle else 1 # env.action_space.sample()
    observation, reward, terminated, truncated, info = env.step(action)
    # print('observation=', observation)
    position, velocity, angle, angle_velocity = observation
    score += reward
    if terminated or truncated:
        observation, info = env.reset()
        print('done, score=', score)
        score = 0
env.close()