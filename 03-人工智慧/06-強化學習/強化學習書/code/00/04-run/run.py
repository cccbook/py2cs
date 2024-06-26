import gymnasium as gym
# env = gym.make("CartPole-v1", render_mode="human") # 若改用這個，會畫圖
env = gym.make("CartPole-v1", render_mode="rgb_array")
observation, info = env.reset(seed=42)
for _ in range(100):
   action = env.action_space.sample()  # this is where you would insert your policy
   observation, reward, terminated, truncated, info = env.step(action)
   print('observation=', observation)
   if terminated or truncated:
      observation, info = env.reset()
      print('done')
env.close()