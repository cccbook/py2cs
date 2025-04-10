import gymnasium as gym
env = gym.make("CartPole-v1")
print('observation_space=', env.observation_space)
print('  low, high=', env.observation_space.low, env.observation_space.high)
print('action_space=', env.action_space)
print('  n = ', env.action_space.n)
