import gymnasium as gym
import sys

env = gym.make(sys.argv[1], render_mode=sys.argv[2])
observation, info = env.reset(seed=42)
for _ in range(100):
   env.render()
   action = env.action_space.sample()  # this is where you would insert your policy
   observation, reward, terminated, truncated, info = env.step(action)
   print('observation=', observation)
   if terminated or truncated:
      observation, info = env.reset()
      print('done')
env.close()
