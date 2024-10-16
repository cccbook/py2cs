import numpy as np

class SVM:
    def __init__(self, learning_rate=0.001, lambda_param=0.01, n_iters=1000):
        # 初始化 SVM 的超參數
        self.lr = learning_rate         # 學習率
        self.lambda_param = lambda_param # 正則化參數
        self.n_iters = n_iters           # 迭代次數
        self.w = None                    # 權重向量，初始為 None
        self.b = None                    # 偏置項，初始為 None

    def fit(self, X, y):
        n_samples, n_features = X.shape  # 獲取樣本數和特徵數

        # 將標籤轉換為 -1 和 1
        y_ = np.where(y <= 0, -1, 1)

        # 初始化權重和偏置
        self.w = np.zeros(n_features)    # 權重向量初始化為零
        self.b = 0                        # 偏置初始化為零

        # 進行 n_iters 次迭代以優化權重和偏置
        for _ in range(self.n_iters):
            for idx, x_i in enumerate(X):  # 遍歷每一個樣本
                # 判斷該樣本是否滿足分類條件
                condition = y_[idx] * (np.dot(x_i, self.w) - self.b) >= 1
                if condition:
                    # 如果條件成立，執行正則化步驟
                    self.w -= self.lr * (2 * self.lambda_param * self.w)
                else:
                    # 如果條件不成立，進行權重和偏置的更新
                    self.w -= self.lr * (2 * self.lambda_param * self.w - np.dot(x_i, y_[idx]))
                    self.b -= self.lr * y_[idx]  # 更新偏置項

    def predict(self, X):
        # 根據學習到的權重和偏置進行預測
        approx = np.dot(X, self.w) - self.b  # 計算預測值
        return np.sign(approx)                # 返回預測的類別（-1 或 +1）

# 測試 SVM 類
if __name__ == "__main__":
    # 導入必要的庫
    from sklearn.model_selection import train_test_split
    from sklearn import datasets
    import matplotlib.pyplot as plt

    # 生成樣本數據
    X, y = datasets.make_blobs(
        n_samples=50, n_features=2, centers=2, cluster_std=1.05, random_state=40
    )
    y = np.where(y == 0, -1, 1)  # 將標籤轉換為 -1 和 1

    # 將數據集分為訓練集和測試集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=123
    )

    # 創建 SVM 對象並訓練模型
    clf = SVM()
    clf.fit(X_train, y_train)
    predictions = clf.predict(X_test)  # 進行預測

    # 定義準確率計算函數
    def accuracy(y_true, y_pred):
        accuracy = np.sum(y_true == y_pred) / len(y_true)  # 計算準確率
        return accuracy

    # 輸出 SVM 分類準確率
    print("SVM classification accuracy", accuracy(y_test, predictions))

    # 定義可視化 SVM 決策邊界的函數
    def visualize_svm():
        def get_hyperplane_value(x, w, b, offset):
            # 根據給定的 x 值計算超平面上的 y 值
            return (-w[0] * x + b + offset) / w[1]

        fig = plt.figure()  # 創建圖形
        ax = fig.add_subplot(1, 1, 1)  # 添加子圖
        plt.scatter(X[:, 0], X[:, 1], marker="o", c=y)  # 繪製數據點

        # 計算超平面的兩個端點
        x0_1 = np.amin(X[:, 0])
        x0_2 = np.amax(X[:, 0])

        # 計算超平面及其邊界
        x1_1 = get_hyperplane_value(x0_1, clf.w, clf.b, 0)
        x1_2 = get_hyperplane_value(x0_2, clf.w, clf.b, 0)

        x1_1_m = get_hyperplane_value(x0_1, clf.w, clf.b, -1)  # 負邊界
        x1_2_m = get_hyperplane_value(x0_2, clf.w, clf.b, -1)

        x1_1_p = get_hyperplane_value(x0_1, clf.w, clf.b, 1)   # 正邊界
        x1_2_p = get_hyperplane_value(x0_2, clf.w, clf.b, 1)

        # 繪製超平面和邊界
        ax.plot([x0_1, x0_2], [x1_1, x1_2], "y--")  # 超平面
        ax.plot([x0_1, x0_2], [x1_1_m, x1_2_m], "k")  # 負邊界
        ax.plot([x0_1, x0_2], [x1_1_p, x1_2_p], "k")   # 正邊界

        # 設置 y 軸的範圍
        x1_min = np.amin(X[:, 1])
        x1_max = np.amax(X[:, 1])
        ax.set_ylim([x1_min - 3, x1_max + 3])

        plt.show()  # 顯示圖形

    visualize_svm()  # 調用可視化函數
