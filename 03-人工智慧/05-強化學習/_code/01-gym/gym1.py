import gymnasium as gym
env = gym.make("CartPole-v1")

observation, info = env.reset()
while True:
    action = env.action_space.sample()
    observation, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
    print(f'action={action} reward={reward}')
    if done:
        print('done')
        break
env.close()