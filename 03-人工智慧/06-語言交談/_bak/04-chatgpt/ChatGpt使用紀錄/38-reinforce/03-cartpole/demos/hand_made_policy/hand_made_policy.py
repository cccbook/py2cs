"""
Agent taking reasonable actions based on hand-made rules,
i.e. go to right if leaning towards left, vice versa.

Observation space: [position of cart, velocity of cart, angle of pole, rotation rate of pole]
Action space: [moving cart to left, moving cart to right]

Note that good agents learn the policies by themselves,
and don't need to know the meaning of the observations.
"""
import gym


def choose_action(observation):
    pos, v, ang, rot = observation
    return 0 if ang < 0 else 1 # a simple rule based only on angles

if __name__ == '__main__':
    env = gym.make('CartPole-v1')

    for i_episode in range(200):
        observation, _ = env.reset() # reset environment to initial state for each episode
        rewards = 0 # accumulate rewards for each episode
        for t in range(250):
            env.render()

            action = choose_action(observation) # choose an action based on hand-made rule 
            observation, reward, terminated, truncated, info = env.step(action) # do the action, get the reward
            done = terminated or truncated
            rewards += reward

            if done:
                print('Episode finished after {} timesteps, total rewards {}'.format(t+1, rewards))
                break

    env.close() # need to close, or errors will be reported
