import numpy as np

# 定義單位階躍函數，輸入為 x，若 x 大於 0 返回 1，否則返回 0
def unit_step_func(x):
    return np.where(x > 0, 1, 0)

# 定義感知器類別
class Perceptron:
    # 初始化感知器的學習率和迭代次數
    def __init__(self, learning_rate=0.01, n_iters=1000):
        self.lr = learning_rate  # 學習率
        self.n_iters = n_iters  # 迭代次數
        self.activation_func = unit_step_func  # 激活函數
        self.weights = None  # 權重初始化為 None
        self.bias = None  # 偏置初始化為 None

    # 訓練感知器的函數
    def fit(self, X, y):
        n_samples, n_features = X.shape  # 獲取樣本數和特徵數

        # 初始化權重和偏置
        self.weights = np.zeros(n_features)  # 權重設為 0
        self.bias = 0  # 偏置設為 0

        # 將目標值轉換為二進制格式
        y_ = np.where(y > 0, 1, 0)

        # 開始學習權重
        for _ in range(self.n_iters):  # 進行 n_iters 次迭代
            for idx, x_i in enumerate(X):  # 遍歷每一個樣本
                linear_output = np.dot(x_i, self.weights) + self.bias  # 計算線性輸出
                y_predicted = self.activation_func(linear_output)  # 計算預測值

                # 根據感知器更新規則更新權重和偏置
                update = self.lr * (y_[idx] - y_predicted)  # 計算更新量
                self.weights += update * x_i  # 更新權重
                self.bias += update  # 更新偏置

    # 預測函數，根據訓練的權重和偏置進行預測
    def predict(self, X):
        linear_output = np.dot(X, self.weights) + self.bias  # 計算線性輸出
        y_predicted = self.activation_func(linear_output)  # 計算預測值
        return y_predicted  # 返回預測結果

# 測試程式
if __name__ == "__main__":
    # 導入必要的庫
    import matplotlib.pyplot as plt
    from sklearn.model_selection import train_test_split
    from sklearn import datasets

    # 計算準確率的函數
    def accuracy(y_true, y_pred):
        accuracy = np.sum(y_true == y_pred) / len(y_true)  # 計算正確率
        return accuracy

    # 生成二維數據集，包含兩個中心的二元分類問題
    X, y = datasets.make_blobs(
        n_samples=150, n_features=2, centers=2, cluster_std=1.05, random_state=2
    )
    # 切分數據集為訓練集和測試集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=123
    )

    # 創建感知器實例並訓練
    p = Perceptron(learning_rate=0.01, n_iters=1000)
    p.fit(X_train, y_train)  # 訓練感知器
    predictions = p.predict(X_test)  # 進行預測

    # 輸出分類準確率
    print("Perceptron classification accuracy", accuracy(y_test, predictions))

    # 繪製訓練數據點
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    plt.scatter(X_train[:, 0], X_train[:, 1], marker="o", c=y_train)

    # 計算邊界線的坐標
    x0_1 = np.amin(X_train[:, 0])  # X 軸最小值
    x0_2 = np.amax(X_train[:, 0])  # X 軸最大值

    # 根據權重和偏置計算對應的 Y 軸坐標
    x1_1 = (-p.weights[0] * x0_1 - p.bias) / p.weights[1]
    x1_2 = (-p.weights[0] * x0_2 - p.bias) / p.weights[1]

    # 繪製邊界線
    ax.plot([x0_1, x0_2], [x1_1, x1_2], "k")

    # 設定 Y 軸範圍
    ymin = np.amin(X_train[:, 1])
    ymax = np.amax(X_train[:, 1])
    ax.set_ylim([ymin - 3, ymax + 3])

    plt.show()  # 顯示圖形
