import numpy as np
import time

class BoltzmannMachine:
    def __init__(self, num_visible, num_hidden):
        self.num_visible = num_visible
        self.num_hidden = num_hidden

        # 權重初始化為小的隨機數
        self.weights = np.random.normal(loc=0.0, scale=0.01, size=(num_visible, num_hidden))
        # 偏置初始化
        self.visible_bias = np.zeros(num_visible)
        self.hidden_bias = np.zeros(num_hidden)

    def sigmoid(self, x):
        return 1.0 / (1.0 + np.exp(-x))

    def sample_prob(self, probs):
        # 根據概率生成二進制樣本
        return (probs > np.random.rand(len(probs))).astype(np.float32)

    def gibbs_sampling(self, visible, steps=1):
        # Gibbs Sampling 進行多步隨機更新，這裡的步數對收斂時間有很大影響
        for _ in range(steps):
            hidden_probs = self.sigmoid(np.dot(visible, self.weights) + self.hidden_bias)
            hidden_states = self.sample_prob(hidden_probs)

            visible_probs = self.sigmoid(np.dot(hidden_states, self.weights.T) + self.visible_bias)
            visible_states = self.sample_prob(visible_probs)

        return visible_states

    def train(self, data, learning_rate=0.1, epochs=1000, gibbs_steps=1):
        num_samples = data.shape[0]

        for epoch in range(epochs):
            start_time = time.time()

            for i in range(num_samples):
                visible = data[i]
                # Gibbs Sampling to estimate the negative phase
                visible_reconstructed = self.gibbs_sampling(visible, steps=gibbs_steps)

                # 計算正負梯度
                hidden_probs = self.sigmoid(np.dot(visible, self.weights) + self.hidden_bias)
                hidden_reconstructed_probs = self.sigmoid(np.dot(visible_reconstructed, self.weights) + self.hidden_bias)

                # 更新權重和偏置
                positive_gradient = np.outer(visible, hidden_probs)
                negative_gradient = np.outer(visible_reconstructed, hidden_reconstructed_probs)

                self.weights += learning_rate * (positive_gradient - negative_gradient)
                self.visible_bias += learning_rate * (visible - visible_reconstructed)
                self.hidden_bias += learning_rate * (hidden_probs - hidden_reconstructed_probs)

            end_time = time.time()
            if epoch % 10 == 0:
                # 計算重建誤差
                reconstruction_error = np.mean((data - visible_reconstructed) ** 2)
                print(f'Epoch {epoch}, Reconstruction Error: {reconstruction_error:.4f}, Time: {end_time - start_time:.4f}s')

# 創建 Boltzmann Machine
num_visible = 6  # 可見層單元數
num_hidden = 3   # 隱層單元數
bm = BoltzmannMachine(num_visible, num_hidden)

# 訓練數據：簡單的二進制向量
training_data = np.array([
    [1, 1, 1, 0, 0, 0],
    [1, 0, 1, 0, 0, 0],
    [0, 0, 1, 1, 1, 0],
    [0, 0, 1, 1, 0, 0]
])

# 訓練 Boltzmann Machine
bm.train(training_data, learning_rate=0.1, epochs=100, gibbs_steps=10)

# 測試：輸入 [1, 0, 1, 0, 0, 1]
test_input = np.array([1, 0, 1, 0, 0, 1])

# 經過 Gibbs Sampling 提取重建輸入
reconstructed_input = bm.gibbs_sampling(test_input, steps=10)

print("Original input:", test_input)
print("Reconstructed input:", reconstructed_input)
