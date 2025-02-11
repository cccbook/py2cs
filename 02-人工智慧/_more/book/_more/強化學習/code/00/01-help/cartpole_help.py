import gymnasium as gym
env = gym.make('CartPole-v1')
help(env.unwrapped)
