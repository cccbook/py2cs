import numpy as np  # 引入NumPy庫，用於數值計算

def sigmoid(x):
    # 計算Sigmoid函數的值
    return 1/(1+np.exp(-x))  # Sigmoid函數公式

class LogisticRegression():
    def __init__(self, lr=0.001, n_iters=1000):
        # 初始化Logistic回歸模型的參數
        self.lr = lr  # 學習率
        self.n_iters = n_iters  # 迭代次數
        self.weights = None  # 權重，初始為None
        self.bias = None  # 偏置，初始為None

    def fit(self, X, y):
        # 訓練模型，根據特徵X和標籤y調整權重和偏置
        n_samples, n_features = X.shape  # 獲取樣本數和特徵數
        self.weights = np.zeros(n_features)  # 初始化權重為0
        self.bias = 0  # 初始化偏置為0

        for _ in range(self.n_iters):  # 遍歷迭代次數
            linear_pred = np.dot(X, self.weights) + self.bias  # 計算線性預測值
            predictions = sigmoid(linear_pred)  # 使用Sigmoid函數獲得預測的概率

            # 計算權重的梯度
            dw = (1/n_samples) * np.dot(X.T, (predictions - y))  # 計算權重的偏導數
            db = (1/n_samples) * np.sum(predictions - y)  # 計算偏置的偏導數

            # 更新權重和偏置
            self.weights = self.weights - self.lr * dw  # 根據學習率和梯度更新權重
            self.bias = self.bias - self.lr * db  # 根據學習率和梯度更新偏置

    def predict(self, X):
        # 根據訓練好的模型進行預測
        linear_pred = np.dot(X, self.weights) + self.bias  # 計算線性預測值
        y_pred = sigmoid(linear_pred)  # 使用Sigmoid函數獲得預測的概率
        # 根據預測的概率判斷類別
        class_pred = [0 if y <= 0.5 else 1 for y in y_pred]  # 將概率轉換為類別標籤
        return class_pred  # 返回預測的類別標籤
