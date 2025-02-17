import numpy as np
from collections import Counter

# 定義歐幾里得距離計算函數
def euclidean_distance(x1, x2):
    # 計算兩個點之間的歐幾里得距離
    distance = np.sqrt(np.sum((x1 - x2) ** 2))
    return distance

# 定義 KNN 類別
class KNN:
    # 初始化方法，k 代表選擇的最近鄰個數
    def __init__(self, k=3):
        self.k = k  # 將 k 的值存儲到實例中

    # 訓練模型，存儲訓練數據和標籤
    def fit(self, X, y):
        self.X_train = X  # 儲存訓練數據
        self.y_train = y  # 儲存訓練標籤

    # 對輸入數據進行預測
    def predict(self, X):
        # 對每個測試樣本進行預測，使用 _predict 方法
        predictions = [self._predict(x) for x in X]
        return predictions

    # 預測單個數據點的類別
    def _predict(self, x):
        # 計算該數據點與所有訓練數據的歐幾里得距離
        distances = [euclidean_distance(x, x_train) for x_train in self.X_train]
    
        # 根據距離由小到大排序，選取距離最近的 k 個點的索引
        k_indices = np.argsort(distances)[:self.k]
        # 根據最近的 k 個點的索引，獲取對應的標籤
        k_nearest_labels = [self.y_train[i] for i in k_indices]

        # 多數表決，統計 k 個最近鄰中出現最多的標籤
        most_common = Counter(k_nearest_labels).most_common()
        # 返回出現次數最多的標籤
        return most_common[0][0]
