import numpy as np

class LinearRegression:

    # 初始化線性回歸模型
    def __init__(self, lr=0.001, n_iters=1000):
        """
        :param lr: 學習率（learning rate），控制每次權重更新的步伐大小，默認為 0.001
        :param n_iters: 訓練的迭代次數，默認為 1000
        """
        self.lr = lr  # 設定學習率
        self.n_iters = n_iters  # 設定迭代次數
        self.weights = None  # 權重向量，初始化為 None
        self.bias = None  # 偏置，初始化為 None

    # 訓練模型，根據輸入特徵 X 和標籤 y
    def fit(self, X, y):
        """
        :param X: 訓練數據集的特徵矩陣，形狀為 (n_samples, n_features)
        :param y: 訓練數據集的標籤，形狀為 (n_samples,)
        """
        # 取得樣本數量和特徵數量
        n_samples, n_features = X.shape
        # 初始化權重為 0 向量，長度與特徵數量相同
        self.weights = np.zeros(n_features)
        # 初始化偏置為 0
        self.bias = 0

        # 開始進行迭代訓練
        for _ in range(self.n_iters):
            # 預測值 y_pred = X 的線性組合 = X * weights + bias
            y_pred = np.dot(X, self.weights) + self.bias

            # 計算權重的梯度（dw）和偏置的梯度（db）
            dw = (1/n_samples) * np.dot(X.T, (y_pred - y))  # dw 為 X.T * (y_pred - y) 的平均值
            db = (1/n_samples) * np.sum(y_pred - y)  # db 為 (y_pred - y) 的平均值

            # 根據梯度下降算法更新權重和偏置
            self.weights = self.weights - self.lr * dw  # 新的權重 = 舊的權重 - 學習率 * dw
            self.bias = self.bias - self.lr * db  # 新的偏置 = 舊的偏置 - 學習率 * db

    # 根據訓練好的模型預測輸入 X 的結果
    def predict(self, X):
        """
        :param X: 測試數據集的特徵矩陣，形狀為 (n_samples, n_features)
        :return: 預測的標籤，形狀為 (n_samples,)
        """
        # 使用學習到的權重和偏置進行預測
        y_pred = np.dot(X, self.weights) + self.bias
        return y_pred
