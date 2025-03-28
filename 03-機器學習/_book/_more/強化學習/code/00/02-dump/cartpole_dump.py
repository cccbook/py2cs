import gymnasium as gym
env = gym.make("CartPole-v1", render_mode="human")
print("env.action_space=", env.action_space)
print("env.observation_space=", env.observation_space)
observation, info = env.reset(seed=42)
print('''
 observation
 |  | Num | Observation           | Min                 | Max               |
 |  |-----|-----------------------|---------------------|-------------------|
 |  | 0   | Cart Position         | -4.8                | 4.8               |
 |  | 1   | Cart Velocity         | -Inf                | Inf               |
 |  | 2   | Pole Angle            | ~ -0.418 rad (-24°) | ~ 0.418 rad (24°) |
 |  | 3   | Pole Angular Velocity | -Inf                | Inf               |    
    ''')

for _ in range(2):
    action = env.action_space.sample()  # this is where you would insert your policy
    print('action=', action)
    # observation, reward, terminated, truncated, info = env.step(action)
    r = env.step(action)
    print('  r=', r)

