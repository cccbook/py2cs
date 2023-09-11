import gymnasium as gym
import sys

env = gym.make(sys.argv[1], render_mode="human")
print('env.action_space=', env.action_space)
print('env.observation_space=', env.observation_space)
observation, info = env.reset(seed=42)
print('observation=', observation, 'info=', info)
'''
for _ in range(10):
   action = env.action_space.sample()  # this is where you would insert your policy

   observation, reward, terminated, truncated, info = env.step(action)

   if terminated or truncated:
      observation, info = env.reset()
env.close()
'''
