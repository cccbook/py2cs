import sys
import logging
import itertools
import copy

import numpy as np
np.random.seed(0)
import pandas as pd
import gym
import matplotlib.pyplot as plt
import torch
torch.manual_seed(0)
import torch.nn as nn
import torch.optim as optim

logging.basicConfig(level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        stream=sys.stdout, datefmt='%H:%M:%S')

env = gym.make('Pendulum-v1')
for key in vars(env.spec):
    logging.info('%s: %s', key, vars(env.spec)[key])
for key in vars(env.unwrapped):
    logging.info('%s: %s', key, vars(env.unwrapped)[key])

class DQNReplayer:
    def __init__(self, capacity):
        self.memory = pd.DataFrame(index=range(capacity),
                columns=['observation', 'action', 'reward',
                'next_observation', 'terminated'])
        self.i = 0
        self.count = 0
        self.capacity = capacity

    def store(self, *args):
        self.memory.loc[self.i] = np.asarray(args, dtype=object)
        self.i = (self.i + 1) % self.capacity
        self.count = min(self.count + 1, self.capacity)

    def sample(self, size):
        indices = np.random.choice(self.count, size=size)
        return (np.stack(self.memory.loc[indices, field]) for field in
                self.memory.columns)

class OrnsteinUhlenbeckProcess:
    def __init__(self, x0):
        self.x = x0

    def __call__(self, mu=0., sigma=1., theta=.15, dt=.01):
        n = np.random.normal(size=self.x.shape)
        self.x += (theta * (mu - self.x) * dt + sigma * np.sqrt(dt) * n)
        return self.x

class DDPGAgent:
    def __init__(self, env):
        state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.shape[0]
        self.action_low = env.action_space.low[0]
        self.action_high = env.action_space.high[0]
        self.gamma = 0.99

        self.replayer = DQNReplayer(20000)

        self.actor_evaluate_net = self.build_net(
                input_size=state_dim, hidden_sizes=[32, 64],
                output_size=self.action_dim)
        self.actor_optimizer = optim.Adam(self.actor_evaluate_net.parameters(),
                lr=0.0001)
        self.actor_target_net = copy.deepcopy(self.actor_evaluate_net)

        self.critic_evaluate_net = self.build_net(
                input_size=state_dim+self.action_dim, hidden_sizes=[64, 128])
        self.critic_optimizer = optim.Adam(self.critic_evaluate_net.parameters(),
                lr=0.001)
        self.critic_loss = nn.MSELoss()
        self.critic_target_net = copy.deepcopy(self.critic_evaluate_net)

    def build_net(self, input_size, hidden_sizes, output_size=1,
            output_activator=None):
        layers = []
        for input_size, output_size in zip(
                [input_size,] + hidden_sizes, hidden_sizes + [output_size,]):
            layers.append(nn.Linear(input_size, output_size))
            layers.append(nn.ReLU())
        layers = layers[:-1]
        if output_activator:
            layers.append(output_activator)
        net = nn.Sequential(*layers)
        return net

    def reset(self, mode=None):
        self.mode = mode
        if self.mode == 'train':
            self.trajectory = []
            self.noise = OrnsteinUhlenbeckProcess(np.zeros((self.action_dim,)))

    def step(self, observation, reward, terminated):
        if self.mode == 'train' and self.replayer.count < 3000:
            action = np.random.uniform(self.action_low, self.action_high)
        else:
            state_tensor = torch.as_tensor(observation,
                    dtype=torch.float).reshape(1, -1)
            action_tensor = self.actor_evaluate_net(state_tensor)
            action = action_tensor.detach().numpy()[0]
        if self.mode == 'train':
            # noisy action
            noise = self.noise(sigma=0.1)
            action = (action + noise).clip(self.action_low, self.action_high)

            self.trajectory += [observation, reward, terminated, action]
            if len(self.trajectory) >= 8:
                state, _, _, act, next_state, reward, terminated, _ = \
                        self.trajectory[-8:]
                self.replayer.store(state, act, reward, next_state, terminated)

            if self.replayer.count >= 3000:
                self.learn()
        return action

    def close(self):
        pass

    def update_net(self, target_net, evaluate_net, learning_rate=0.005):
        for target_param, evaluate_param in zip(
                target_net.parameters(), evaluate_net.parameters()):
            target_param.data.copy_(learning_rate * evaluate_param.data
                    + (1 - learning_rate) * target_param.data)

    def learn(self):
        # replay
        states, actions, rewards, next_states, terminateds = \
                self.replayer.sample(64)
        state_tensor = torch.as_tensor(states, dtype=torch.float)
        action_tensor = torch.as_tensor(actions, dtype=torch.long)
        reward_tensor = torch.as_tensor(rewards, dtype=torch.float)
        next_state_tensor = torch.as_tensor(next_states, dtype=torch.float)
        terminated_tensor = torch.as_tensor(terminateds, dtype=torch.float)

        # update critic
        next_action_tensor = self.actor_target_net(next_state_tensor)
        noise_tensor = (0.2 * torch.randn_like(action_tensor, dtype=torch.float))
        noisy_next_action_tensor = (next_action_tensor + noise_tensor).clamp(
                self.action_low, self.action_high)
        next_state_action_tensor = torch.cat([next_state_tensor,
                noisy_next_action_tensor], 1)
        next_q_tensor = self.critic_target_net(next_state_action_tensor).squeeze(1)
        critic_target_tensor = reward_tensor + (1. - terminated_tensor) * \
                self.gamma * next_q_tensor
        critic_target_tensor = critic_target_tensor.detach()

        state_action_tensor = torch.cat([state_tensor, action_tensor], 1)
        critic_pred_tensor = self.critic_evaluate_net(state_action_tensor
                ).squeeze(1)
        critic_loss_tensor = self.critic_loss(critic_pred_tensor,
                critic_target_tensor)
        self.critic_optimizer.zero_grad()
        critic_loss_tensor.backward()
        self.critic_optimizer.step()

        # update actor
        pred_action_tensor = self.actor_evaluate_net(state_tensor)
        pred_action_tensor = pred_action_tensor.clamp(self.action_low,
                self.action_high)
        pred_state_action_tensor = torch.cat([state_tensor, pred_action_tensor], 1)
        critic_pred_tensor = self.critic_evaluate_net(pred_state_action_tensor)
        actor_loss_tensor = -critic_pred_tensor.mean()
        self.actor_optimizer.zero_grad()
        actor_loss_tensor.backward()
        self.actor_optimizer.step()

        self.update_net(self.critic_target_net, self.critic_evaluate_net)
        self.update_net(self.actor_target_net, self.actor_evaluate_net)


agent = DDPGAgent(env)

def play_episode(env, agent, seed=None, mode=None, render=False):
    observation, _ = env.reset(seed=seed)
    reward, terminated, truncated = 0., False, False
    agent.reset(mode=mode)
    episode_reward, elapsed_steps = 0., 0
    while True:
        action = agent.step(observation, reward, terminated)
        if render:
            env.render()
        if terminated or truncated:
            break
        observation, reward, terminated, truncated, _ = env.step(action)
        episode_reward += reward
        elapsed_steps += 1
    agent.close()
    return episode_reward, elapsed_steps


logging.info('==== train ====')
episode_rewards = []
for episode in itertools.count():
    episode_reward, elapsed_steps = play_episode(env, agent, seed=episode,
            mode='train')
    episode_rewards.append(episode_reward)
    logging.info('train episode %d: reward = %.2f, steps = %d',
            episode, episode_reward, elapsed_steps)
    if np.mean(episode_rewards[-10:]) > -120:
        break
plt.plot(episode_rewards)


logging.info('==== test ====')
episode_rewards = []
for episode in range(100):
    episode_reward, elapsed_steps = play_episode(env, agent)
    episode_rewards.append(episode_reward)
    logging.info('test episode %d: reward = %.2f, steps = %d',
            episode, episode_reward, elapsed_steps)
logging.info('average episode reward = %.2f ± %.2f',
        np.mean(episode_rewards), np.std(episode_rewards))

env.close()

# 使用 render (for human) 動畫播放玩一次
env = gym.make('Pendulum-v1', render_mode="human")
episode_reward, elapsed_steps = play_episode(env, agent, render=True)
env.close()