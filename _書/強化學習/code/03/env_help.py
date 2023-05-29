import gymnasium as gym
env = gym.make('FrozenLake-v1', render_mode="rgb_array")
help(env.unwrapped)
