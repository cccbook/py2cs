import numpy as np

class GeneralBoltzmannMachine:
    def __init__(self, num_neurons):
        self.num_neurons = num_neurons
        self.weights = np.random.randn(num_neurons, num_neurons) * 0.1
        np.fill_diagonal(self.weights, 0)
        self.weights = (self.weights + self.weights.T) / 2
        self.biases = np.zeros(num_neurons)
        
    def energy(self, state):
        return -0.5 * np.dot(np.dot(state, self.weights), state) - np.dot(self.biases, state)
    
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def sample_neuron(self, i, state):
        activation = np.dot(self.weights[i], state) + self.biases[i]
        prob = self.sigmoid(activation)
        return 1 if np.random.random() < prob else 0
    
    def gibbs_sampling(self, initial_state, num_steps):
        state = initial_state.copy()
        for _ in range(num_steps):
            for i in range(self.num_neurons):
                state[i] = self.sample_neuron(i, state)
        return state
    
    def contrastive_divergence(self, visible_data, learning_rate=0.1, k=1):
        pos_state = np.concatenate([visible_data, np.random.randint(2, size=self.num_neurons-len(visible_data))])
        pos_associations = np.outer(pos_state, pos_state)
        
        neg_state = self.gibbs_sampling(pos_state, k)
        neg_associations = np.outer(neg_state, neg_state)
        
        self.weights += learning_rate * (pos_associations - neg_associations)
        np.fill_diagonal(self.weights, 0)
        self.biases += learning_rate * (pos_state - neg_state)
    
    def train(self, data, epochs=1000, learning_rate=0.1, k=1):
        for epoch in range(epochs):
            for sample in data:
                self.contrastive_divergence(sample, learning_rate, k)
            if epoch % 100 == 0:
                print(f"Epoch {epoch}/{epochs}")
    
    def reconstruct(self, partial_data, visible_units, num_steps=1000):
        """根據部分數據重建完整狀態"""
        state = np.zeros(self.num_neurons)
        state[:len(partial_data)] = partial_data
        
        for _ in range(num_steps):
            for i in range(visible_units, self.num_neurons):  # 只更新隱藏單元
                state[i] = self.sample_neuron(i, state)
            for i in range(len(partial_data), visible_units):  # 更新未知的可見單元
                state[i] = self.sample_neuron(i, state)
        
        return state[:visible_units]  # 只返回可見單元的狀態

# 示範使用
num_neurons = 8  # 總神經元數量
visible_units = 6  # 可見單元數量
gbm = GeneralBoltzmannMachine(num_neurons)

# 創建一些示例數據（僅使用可見單元）
data = np.array([
    [1,1,1,0,0,0],
    [1,0,1,0,0,0],
    [1,1,1,0,0,0],
    [0,0,1,1,1,0],
    [0,0,1,1,0,1],
    [0,0,1,1,1,0]
])

# 訓練模型
gbm.train(data, epochs=5000, learning_rate=0.1, k=3)

# 記憶提取示例
# partial_data = np.array([1, 1, -1, -1, -1, 0])  # -1 表示未知
partial_data = np.array([1, 0, 1, 1, 0, 0])
reconstructed = gbm.reconstruct(partial_data, visible_units)

print("部分數據:", partial_data)
print("重建結果:", reconstructed)
print("最相似的訓練樣本:", data[np.argmin(np.sum((data - reconstructed)**2, axis=1))])