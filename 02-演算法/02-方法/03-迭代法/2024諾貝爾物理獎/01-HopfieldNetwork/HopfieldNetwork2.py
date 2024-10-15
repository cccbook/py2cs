import numpy as np

class HopfieldNetwork:
    def __init__(self, size):
        """初始化網絡大小，建立權重矩陣。"""
        self.size = size
        self.weights = np.zeros((size, size))

    def train(self, patterns):
        """使用 Hebbian 規則訓練網絡來儲存模式。"""
        for pattern in patterns:
            pattern = np.reshape(pattern, (self.size, 1))  # 確保模式是列向量
            self.weights += pattern @ pattern.T  # 更新權重矩陣
        np.fill_diagonal(self.weights, 0)  # 切斷神經元自身連接

    def sign(self, x):
        """符號函數，用於激活。"""
        return np.where(x >= 0, 1, -1)

    def recall(self, input_pattern, max_iterations=10):
        """回憶模式，根據輸入更新狀態直到穩定或達到最大迭代次數。"""
        pattern = np.array(input_pattern)
        for _ in range(max_iterations):
            previous_pattern = np.copy(pattern)
            # 隨機選擇順序更新每個神經元
            for i in range(self.size):
                net_input = np.dot(self.weights[i], pattern)
                pattern[i] = self.sign(net_input)  # 根據符號函數更新狀態
            if np.array_equal(pattern, previous_pattern):
                break  # 若狀態不變，則退出
        return pattern

if __name__ == "__main__":
    # 定義訓練模式 (使用 1 和 -1 表示)
    patterns = [
        [1, 1, -1, -1],
        [-1, 1, 1, -1],
        [-1, -1, 1, 1]
    ]

    # 創建 Hopfield 網絡
    size = len(patterns[0])
    hopfield_net = HopfieldNetwork(size)

    # 訓練網絡
    for i in range(100):
        hopfield_net.train(patterns)

    # 測試網絡的回憶功能
    test_input = [-1, 1, -1, -1]  # 有噪音的輸入
    recalled_pattern = hopfield_net.recall(test_input)

    print("輸入模式:", test_input)
    print("回憶模式:", recalled_pattern)
