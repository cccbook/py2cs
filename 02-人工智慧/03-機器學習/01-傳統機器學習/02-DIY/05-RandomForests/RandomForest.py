from DecisionTree import DecisionTree  # 引入決策樹類
import numpy as np  # 引入 NumPy 庫，用於數值運算
from collections import Counter  # 引入 Counter，用於計算頻率

class RandomForest:
    def __init__(self, n_trees=10, max_depth=10, min_samples_split=2, n_feature=None):
        # 初始化隨機森林的參數
        self.n_trees = n_trees  # 樹的數量
        self.max_depth = max_depth  # 每棵樹的最大深度
        self.min_samples_split = min_samples_split  # 最小分裂樣本數
        self.n_features = n_feature  # 每次分裂考慮的特徵數
        self.trees = []  # 存儲訓練好的樹

    def fit(self, X, y):
        # 訓練隨機森林模型
        self.trees = []  # 重置樹列表
        for _ in range(self.n_trees):
            # 為每棵樹創建一個新的決策樹
            tree = DecisionTree(max_depth=self.max_depth,
                                min_samples_split=self.min_samples_split,
                                n_features=self.n_features)
            # 使用自助法抽樣生成樣本
            X_sample, y_sample = self._bootstrap_samples(X, y)
            # 使用生成的樣本訓練決策樹
            tree.fit(X_sample, y_sample)
            # 將訓練好的樹添加到樹列表中
            self.trees.append(tree)

    def _bootstrap_samples(self, X, y):
        # 生成自助法樣本
        n_samples = X.shape[0]  # 獲取樣本數
        # 隨機選擇樣本索引，進行有放回抽樣
        idxs = np.random.choice(n_samples, n_samples, replace=True)
        # 返回抽樣的特徵和標籤
        return X[idxs], y[idxs]

    def _most_common_label(self, y):
        # 計算最常見的標籤
        counter = Counter(y)  # 計數每個標籤出現的次數
        most_common = counter.most_common(1)[0][0]  # 獲取出現次數最多的標籤
        return most_common  # 返回最常見的標籤

    def predict(self, X):
        # 對輸入數據進行預測
        predictions = np.array([tree.predict(X) for tree in self.trees])  # 獲取每棵樹的預測結果
        tree_preds = np.swapaxes(predictions, 0, 1)  # 轉置預測結果，使其形狀為 (樣本數, 樹的數量)
        # 對每個樣本進行投票，選擇最常見的預測結果
        predictions = np.array([self._most_common_label(pred) for pred in tree_preds])
        return predictions  # 返回最終預測結果
