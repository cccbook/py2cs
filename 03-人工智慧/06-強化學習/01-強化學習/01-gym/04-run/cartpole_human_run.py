import gymnasium as gym
env = gym.make("CartPole-v1", render_mode="human") # 若改用這個，會畫圖
# env = gym.make("CartPole-v1", render_mode="rgb_array")
observation, info = env.reset(seed=42)
score = 0
for _ in range(1000):
   env.render()
   action = env.action_space.sample()  # this is where you would insert your policy
   observation, reward, terminated, truncated, info = env.step(action)
   # print('observation=', observation)
   score += reward
   if terminated or truncated:
      observation, info = env.reset()
      print('done, score=', score)
      score = 0
env.close()