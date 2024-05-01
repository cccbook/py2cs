"""
Agent taking random actions.
Based on the example on the official website.
"""
#import gym
import gymnasium as gym


if __name__ == '__main__':
    # env = gym.make('CartPole-v1')
    env = gym.make('CartPole-v1', render_mode="human") # 改用這行會顯示動畫，但會變慢很多

    for i_episode in range(200):
        observation, info = env.reset() # reset environment to initial state for each episode
        # print('observation=', observation)
        rewards = 0 # accumulate rewards for each episode
        for t in range(250):
            env.render()

            action = env.action_space.sample() # choose a random action
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            rewards += reward

            if done:
                print('Episode finished after {} timesteps, total rewards {}'.format(t+1, rewards))
                break

    env.close() # need to close, or errors will be reported
