import gymnasium as gym
import random
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

model_name = "cartpole-dqn.h5"

class DQNAgent:

    def __init__(self, state_size, action_size):
        self.state_size = state_size # 状态空间规模（即观测空间大小）
        self.action_size = action_size # 行动（动作）空间规模

        self.memory = deque(maxlen=2000)

        self.gamma = 0.95    # discount rate 长期回报的折扣因子

        self.epsilon = 1.0  # exploration rate 探索率
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995 # 探索率的衰减量

        self.learning_rate = 0.001 # 学习速率

        self.model = self._build_model()

    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        model = Sequential()
        model.add(Dense(24, input_dim=self.state_size, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse',
                      optimizer=Adam(learning_rate=self.learning_rate))
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])  # returns action

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = (reward + self.gamma *
                          np.amax(self.model.predict(next_state)[0]))
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)

num_episodes = 1000

def train(env):
    # size of states and actions
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    # initialize agent
    agent = DQNAgent(state_size, action_size)
    # training loop
    done = False
    batch_size = 32
    for e in range(num_episodes):
        state, info = env.reset()
        state = np.reshape(state, [1, state_size])
        for time in range(500):
            action = agent.act(state)
            next_state, reward, done, _, info = env.step(action)
            reward = reward if not done else -10
            next_state = np.reshape(next_state, [1, state_size])
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            if done:
                print("episode: {}/{}, score: {}, e: {:.2}"
                      .format(e, num_episodes, time, agent.epsilon))
                break
        if len(agent.memory) > batch_size:
            agent.replay(batch_size)
        if e % 5 == 0:
            agent.save(model_name)
            print('saved:episode(e)=', e)
    agent.save(model_name)
    return agent

def test(env):
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    agent = DQNAgent(state_size, action_size)
    agent.load(model_name)
    state, info = env.reset()
    for i in range(1000):
        env.render()
        a = agent.act(state)
        state, reward, terminated, truncated, info = env.step(a)
        if terminated == True:
            print('fail at i=', i)
            break

if __name__ == "__main__":
    env = gym.make('CartPole-v1', render_mode="human")
    import sys
    job = sys.argv[1]
    if job == 'train':
        print('training ....')
        train(env)
    else:
        print('testing ....')
        test(env)
    env.close()
