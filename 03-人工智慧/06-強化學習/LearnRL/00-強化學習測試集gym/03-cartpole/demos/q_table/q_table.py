"""
Agent learns the policy based on Q-learning with Q table.
Based on the example here: https://medium.com/@tuzzer/cart-pole-balancing-with-q-learning-b54c6068d947
"""
import math

import gym
import numpy as np


def choose_action(state, q_table, action_space, epsilon):
    if np.random.random_sample() < epsilon: # random action
        return action_space.sample() 
    else: # greddy action based on Q table
        return np.argmax(q_table[state]) 

def get_state(observation, n_buckets, state_bounds):
    # print('observation1=', observation1)
    # observation = observation1[0]
    state = [0] * len(observation) 
    # print('observation=', observation)
    for i, s in enumerate(observation):
        l, u = state_bounds[i][0], state_bounds[i][1] # lower- and upper-bounds for each feature in observation
        # print('l=', l, 'u=', u, 's=', s, 'i=', i)
        if s <= l:
            state[i] = 0
        elif s >= u:
            state[i] = n_buckets[i] - 1
        else:
            state[i] = int(((s - l) / (u - l)) * n_buckets[i])

    return tuple(state)

if __name__ == '__main__':
    env = gym.make('CartPole-v1')

    # Preparing Q table
    ## buckets for continuous state values to be assigned to
    n_buckets = (1, 1, 6, 3) # Observation space: [position of cart, velocity of cart, angle of pole, rotation rate of pole]
                             # Setting bucket size to 1 = ignoring the particular observation state 

    ## discrete actions
    n_actions = env.action_space.n

    ## state bounds
    state_bounds = list(zip(env.observation_space.low, env.observation_space.high))
    state_bounds[1] = [-0.5, 0.5]
    state_bounds[3] = [-math.radians(50), math.radians(50)]
    
    ## init Q table for each state-action pair
    q_table = np.zeros(n_buckets + (n_actions,))

    # Learning related constants; factors determined by trial-and-error
    get_epsilon = lambda i: max(0.01, min(1, 1.0 - math.log10((i+1)/25))) # epsilon-greedy, factor to explore randomly; discounted over time
    get_lr = lambda i: max(0.01, min(0.5, 1.0 - math.log10((i+1)/25))) # learning rate; discounted over time
    gamma = 0.99 # reward discount factor

    # Q-learning
    for i_episode in range(200):
        epsilon = get_epsilon(i_episode)
        lr = get_lr(i_episode)

        observation, _ = env.reset() # reset environment to initial state for each episode
        rewards = 0 # accumulate rewards for each episode
        state = get_state(observation, n_buckets, state_bounds) # turn observation into discrete state
        for t in range(250):
            env.render()

            # Agent takes action
            action = choose_action(state, q_table, env.action_space, epsilon) # choose an action based on q_table 
            # observation, reward, done, info = env.step(action) # do the action, get the reward
            observation, reward, terminated, truncated, info = env.step(action) # do the action, get the reward
            done = terminated or truncated
            rewards += reward
            next_state = get_state(observation, n_buckets, state_bounds)

            # Agent learns via Q-learning
            q_next_max = np.amax(q_table[next_state])
            q_table[state + (action,)] += lr * (reward + gamma * q_next_max - q_table[state + (action,)])

            # Transition to next state
            state = next_state

            if done:
                print('Episode finished after {} timesteps, total rewards {}'.format(t+1, rewards))
                break

    env.close() # need to close, or errors will be reported
