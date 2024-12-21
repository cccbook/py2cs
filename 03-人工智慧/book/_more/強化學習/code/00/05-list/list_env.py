import gymnasium as gym
env_specs = gym.envs.registry
env_list = [env_spec for env_spec in env_specs]
print(env_list)