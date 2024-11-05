import numpy as np
import gymnasium as gym
from collections import defaultdict

class TDLambda:
    def __init__(self, env, lambda_param=0.8, gamma=0.95, alpha=0.1, epsilon=0.2):
        self.env = env
        self.lambda_param = lambda_param
        self.gamma = gamma
        self.alpha = alpha
        self.epsilon = epsilon
        self.min_epsilon = 0.01  # 最小探索率
        self.epsilon_decay = 0.995  # epsilon 衰減率
        
        # 初始化 Q 值表為樂觀初始值
        self.q_table = defaultdict(lambda: np.ones(env.action_space.n))
        self.eligibility_traces = defaultdict(lambda: np.zeros(env.action_space.n))
    
    def get_action(self, state):
        """使用 ε-greedy 策略選擇動作"""
        if np.random.random() < self.epsilon:
            return self.env.action_space.sample()
        return int(np.argmax(self.q_table[state]))
    
    def decay_epsilon(self):
        """衰減探索率"""
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)
    
    def update(self, state, action, reward, next_state, done):
        """更新 Q 值和適用度追蹤"""
        # 修改獎勵機制
        modified_reward = reward
        if done and reward == 0:  # 如果遊戲結束但沒有得到獎勵（掉入洞中）
            modified_reward = -1
        elif not done:  # 如果還在遊戲中
            modified_reward = -0.01  # 小幅度懲罰以鼓勵儘快到達目標
        
        # 計算 TD 誤差
        if done:
            target = modified_reward
        else:
            next_q_values = self.q_table[next_state]
            next_action = np.argmax(next_q_values)
            target = modified_reward + self.gamma * next_q_values[next_action]
        
        td_error = target - self.q_table[state][action]
        
        # 更新適用度追蹤
        self.eligibility_traces[state][action] += 1
        
        # 更新所有狀態-動作對的 Q 值
        for s in self.eligibility_traces.keys():
            self.q_table[s] += self.alpha * td_error * self.eligibility_traces[s]
            self.eligibility_traces[s] *= self.gamma * self.lambda_param
    
    def train(self, episodes=5000):
        """訓練智能體"""
        rewards_history = []
        success_count = 0  # 記錄成功次數
        
        for episode in range(episodes):
            state, _ = self.env.reset()
            total_reward = 0
            done = False
            
            # 重置適用度追蹤
            self.eligibility_traces = defaultdict(lambda: np.zeros(self.env.action_space.n))
            
            while not done:
                action = self.get_action(state)
                next_state, reward, done, truncated, _ = self.env.step(action)
                done = done or truncated
                
                self.update(state, action, reward, next_state, done)
                total_reward += reward
                state = next_state
            
            # 如果這個回合成功到達目標
            if reward == 1:
                success_count += 1
            
            rewards_history.append(total_reward)
            self.decay_epsilon()  # 衰減探索率
            
            # 每100個回合輸出一次進度
            if (episode + 1) % 100 == 0:
                success_rate = success_count / 100
                print(f"Episode {episode + 1}, Success Rate: {success_rate:.2f}, Epsilon: {self.epsilon:.3f}")
                success_count = 0  # 重置計數器
        
        return rewards_history

def run_example():
    # 設置隨機種子以確保可重現性
    np.random.seed(42)
    env = gym.make('FrozenLake-v1', is_slippery=True)
    
    # 創建並訓練智能體
    agent = TDLambda(env)
    rewards = agent.train()
    
    # 測試階段
    print("\nTesting the trained agent:")
    success_count = 0
    test_episodes = 100
    
    for _ in range(test_episodes):
        state, _ = env.reset()
        done = False
        while not done:
            action = np.argmax(agent.q_table[state])  # 使用純貪婪策略
            state, reward, done, truncated, _ = env.step(action)
            if reward == 1:
                success_count += 1
    
    print(f"Test success rate: {success_count/test_episodes:.2f}")
    return agent, rewards

if __name__ == "__main__":
    agent, rewards = run_example()