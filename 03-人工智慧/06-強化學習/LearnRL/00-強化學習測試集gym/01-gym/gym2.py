import gymnasium as gym
env = gym.make("CartPole-v1")

episodes = 10

# run environment for 10 episodes
for ep in range(episodes):
    observation, info = env.reset()
    episode_reward = 0
    while True:
        # randomly chose an action from all available actions
        action = env.action_space.sample()
        # agent takes an action and interacts with the env, receiving state, reward, done and info
        state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        episode_reward += 1
        # if episode is over reset the env
        if done:
            print("Episode {} done with reward: {}".format(ep, episode_reward))
            break
