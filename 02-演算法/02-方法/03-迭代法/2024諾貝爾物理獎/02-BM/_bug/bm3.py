import numpy as np

class BoltzmannMachine:
    def __init__(self, num_visible, num_hidden):
        self.num_visible = num_visible
        self.num_hidden = num_hidden

        # 隨機初始化權重
        self.weights = np.random.normal(0, 0.1, (num_visible, num_hidden))
        self.visible_bias = np.zeros(num_visible)
        self.hidden_bias = np.zeros(num_hidden)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sample_prob(self, probs):
        return (probs > np.random.rand(len(probs))).astype(np.float32)

    def gibbs_sampling(self, visible, steps=1):
        for _ in range(steps):
            hidden_probs = self.sigmoid(np.dot(visible, self.weights) + self.hidden_bias)
            hidden_states = self.sample_prob(hidden_probs)

            visible_probs = self.sigmoid(np.dot(hidden_states, self.weights.T) + self.visible_bias)
            visible = self.sample_prob(visible_probs)

        return visible

    def train(self, data, learning_rate=0.1, epochs=1000, gibbs_steps=5):
        num_samples = data.shape[0]

        for epoch in range(epochs):
            for i in range(num_samples):
                visible = data[i]
                # 正相和負相的計算
                hidden_probs = self.sigmoid(np.dot(visible, self.weights) + self.hidden_bias)
                hidden_states = self.sample_prob(hidden_probs)

                # Gibbs Sampling 提取重建輸入
                visible_reconstructed = self.gibbs_sampling(visible, steps=gibbs_steps)

                # 計算正負梯度
                positive_gradient = np.outer(visible, hidden_probs)
                hidden_reconstructed_probs = self.sigmoid(np.dot(visible_reconstructed, self.weights) + self.hidden_bias)
                negative_gradient = np.outer(visible_reconstructed, hidden_reconstructed_probs)

                # 更新權重和偏置
                self.weights += learning_rate * (positive_gradient - negative_gradient)
                self.visible_bias += learning_rate * (visible - visible_reconstructed)
                self.hidden_bias += learning_rate * (hidden_probs - hidden_reconstructed_probs)

            # 每 100 epochs 計算重建誤差
            if epoch % 100 == 0:
                reconstruction_error = np.mean((data - self.gibbs_sampling(data, steps=gibbs_steps)) ** 2)
                print(f'Epoch {epoch}, Reconstruction Error: {reconstruction_error:.4f}')

# 創建波茲曼機
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

# 訓練波茲曼機
bm.train(training_data, learning_rate=0.1, epochs=1000, gibbs_steps=5)

# 測試：輸入 [1, 0, 1, 0, 0, 1]
test_input = np.array([1, 0, 1, 0, 0, 1])

# 提取重建輸入
reconstructed_input = bm.gibbs_sampling(test_input, steps=10)

print("Original input:", test_input)
print("Reconstructed input:", reconstructed_input)
