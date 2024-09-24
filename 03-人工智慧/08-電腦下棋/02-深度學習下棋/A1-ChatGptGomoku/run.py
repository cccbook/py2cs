from gomoku import *
from qlearning import *

def train_agent(agent, env, episodes=5000):
    for episode in range(episodes):
        state = env.reset()
        done = False
        while not done:
            state_key = agent.get_state_key(state)
            valid_actions = agent.get_valid_actions(state)
            action = agent.choose_action(state_key, valid_actions)
            
            next_state, reward, done, _ = env.step(action)
            next_state_key = agent.get_state_key(next_state)
            next_valid_actions = agent.get_valid_actions(next_state)
            
            # 更新 Q 值
            agent.update_q_value(state_key, action, reward, next_state_key, next_valid_actions)
            
            state = next_state

        if (episode + 1) % 500 == 0:
            print(f"Episode {episode + 1}/{episodes} complete.")

# 初始化环境和代理
env = GomokuEnv()
agent = QLearningAgent()

# 训练代理
train_agent(agent, env)

def test_agent(agent, env):
    state = env.reset()
    env.render()
    done = False
    while not done:
        state_key = agent.get_state_key(state)
        valid_actions = agent.get_valid_actions(state)
        action = agent.choose_action(state_key, valid_actions)
        state, reward, done, _ = env.step(action)
        env.render()

# 测试强化学习代理
test_agent(agent, env)
