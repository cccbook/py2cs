import numpy as np

class RBM:
    def __init__(self, num_visible, num_hidden, learning_rate=0.1):
        self.num_visible = num_visible
        self.num_hidden = num_hidden
        self.learning_rate = learning_rate
        self.weights = np.random.randn(num_visible, num_hidden) * 0.1
        self.visible_bias = np.zeros(num_visible)
        self.hidden_bias = np.zeros(num_hidden)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sample_h(self, v):
        """給定可見層 v，計算隱藏層的激活並取樣"""
        h_prob = self.sigmoid(np.dot(v, self.weights) + self.hidden_bias)
        h_sample = np.random.binomial(1, h_prob)
        return h_sample, h_prob

    def sample_v(self, h):
        """給定隱藏層 h，計算可見層的重建並取樣"""
        v_prob = self.sigmoid(np.dot(h, self.weights.T) + self.visible_bias)
        v_sample = np.random.binomial(1, v_prob)
        return v_sample, v_prob

    def contrastive_divergence(self, v0, k=1):
        """執行 CD-K 算法"""
        # 正向步驟
        h0, h0_prob = self.sample_h(v0)
        
        # K 次 Gibbs 取樣
        vk = v0
        for _ in range(k):
            vk, _ = self.sample_v(h0)

        # 最終的隱藏層取樣
        hk, _ = self.sample_h(vk)

        # 更新權重和偏差
        self.weights += self.learning_rate * (np.dot(v0.T, h0_prob) - np.dot(vk.T, hk)) / v0.shape[0]
        self.visible_bias += self.learning_rate * np.mean(v0 - vk, axis=0)
        self.hidden_bias += self.learning_rate * np.mean(h0_prob - hk, axis=0)

    def train(self, data, epochs=1000, k=1):
        for epoch in range(epochs):
            self.contrastive_divergence(data, k)
            if (epoch + 1) % 100 == 0:
                print(f'Epoch {epoch + 1}/{epochs} completed')

    def get_hidden_representation(self, v):
        """獲取可見層 v 的隱藏層表示"""
        h_prob = self.sigmoid(np.dot(v, self.weights) + self.hidden_bias)
        return h_prob

    def get_visible_representation(self, h):
        """根據隱藏層 h 的表示還原可見層"""
        v_prob = self.sigmoid(np.dot(h, self.weights.T) + self.visible_bias)
        return v_prob

# 固定的訓練樣本
data = np.array([
    [1, 0, 0, 1, 0, 1],
    [0, 1, 1, 0, 1, 0],
    [1, 1, 0, 0, 0, 1],
    [0, 0, 1, 1, 1, 0],
    [1, 0, 1, 0, 1, 1]
])  # 5 個樣本，6 個可見單元

# 初始化 RBM
rbm = RBM(num_visible=6, num_hidden=2)

# 訓練 RBM
rbm.train(data, epochs=1000, k=1)

# 輸出最終權重
print("Learned weights:\n", rbm.weights)

# 測試用不在樣本中的輸入
test_input = np.array([[1, 1, 1, 0, 0, 0]])  # 新的輸入，不在訓練樣本中

# 獲取隱藏層表示
hidden_representation = rbm.get_hidden_representation(test_input)
print("Hidden representation for test input:\n", hidden_representation)

# 根據隱藏層表示還原可見層
visible_representation = rbm.get_visible_representation(hidden_representation)
print("Reconstructed visible representation from hidden representation:\n", visible_representation)
