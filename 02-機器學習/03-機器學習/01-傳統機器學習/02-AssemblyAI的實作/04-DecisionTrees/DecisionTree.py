import numpy as np  # 引入NumPy庫以進行數值計算
from collections import Counter  # 引入Counter以計算標籤的頻率

class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, *, value=None):
        # 初始化節點屬性
        self.feature = feature  # 用於分割的特徵
        self.threshold = threshold  # 分割的閾值
        self.left = left  # 左子樹
        self.right = right  # 右子樹
        self.value = value  # 如果是葉子節點，則為類別值
        
    def is_leaf_node(self):
        # 判斷當前節點是否為葉子節點
        return self.value is not None


class DecisionTree:
    def __init__(self, min_samples_split=2, max_depth=100, n_features=None):
        # 初始化決策樹的參數
        self.min_samples_split = min_samples_split  # 最小樣本數量以進行分割
        self.max_depth = max_depth  # 最大樹深度
        self.n_features = n_features  # 用於分割的特徵數量
        self.root = None  # 樹的根節點初始化為None

    def fit(self, X, y):
        # 擬合模型，根據訓練數據構建決策樹
        self.n_features = X.shape[1] if not self.n_features else min(X.shape[1], self.n_features)
        self.root = self._grow_tree(X, y)  # 從根開始生成樹

    def _grow_tree(self, X, y, depth=0):
        # 遞歸生成決策樹
        n_samples, n_feats = X.shape  # 獲取樣本數和特徵數
        n_labels = len(np.unique(y))  # 獲取標籤的唯一數量

        # 檢查停止條件
        if (depth >= self.max_depth or n_labels == 1 or n_samples < self.min_samples_split):
            # 如果達到最大深度或樣本數小於最小分割數，創建葉子節點
            leaf_value = self._most_common_label(y)  # 獲取最常見的標籤
            return Node(value=leaf_value)

        feat_idxs = np.random.choice(n_feats, self.n_features, replace=False)  # 隨機選擇特徵

        # 找到最佳分割
        best_feature, best_thresh = self._best_split(X, y, feat_idxs)

        # 創建子節點
        left_idxs, right_idxs = self._split(X[:, best_feature], best_thresh)  # 根據最佳特徵和閾值分割數據
        left = self._grow_tree(X[left_idxs, :], y[left_idxs], depth + 1)  # 遞歸生成左子樹
        right = self._grow_tree(X[right_idxs, :], y[right_idxs], depth + 1)  # 遞歸生成右子樹
        return Node(best_feature, best_thresh, left, right)  # 返回包含子樹的節點


    def _best_split(self, X, y, feat_idxs):
        # 找到最佳特徵和閾值進行分割
        best_gain = -1  # 初始化最佳增益
        split_idx, split_threshold = None, None  # 初始化分割索引和閾值

        for feat_idx in feat_idxs:  # 對每個特徵進行迭代
            X_column = X[:, feat_idx]  # 獲取特徵列
            thresholds = np.unique(X_column)  # 獲取特徵值的唯一值

            for thr in thresholds:  # 對每個閾值進行迭代
                # 計算信息增益
                gain = self._information_gain(y, X_column, thr)

                if gain > best_gain:  # 如果信息增益更好，更新最佳增益
                    best_gain = gain
                    split_idx = feat_idx
                    split_threshold = thr

        return split_idx, split_threshold  # 返回最佳特徵和閾值


    def _information_gain(self, y, X_column, threshold):
        # 計算信息增益
        # 父節點的熵
        parent_entropy = self._entropy(y)

        # 創建子節點
        left_idxs, right_idxs = self._split(X_column, threshold)

        if len(left_idxs) == 0 or len(right_idxs) == 0:
            return 0  # 如果其中一個子集為空，返回增益為0
        
        # 計算子節點的加權平均熵
        n = len(y)
        n_l, n_r = len(left_idxs), len(right_idxs)  # 獲取左子樹和右子樹的樣本數
        e_l, e_r = self._entropy(y[left_idxs]), self._entropy(y[right_idxs])  # 計算子樹的熵
        child_entropy = (n_l / n) * e_l + (n_r / n) * e_r  # 加權平均熵

        # 計算信息增益
        information_gain = parent_entropy - child_entropy
        return information_gain

    def _split(self, X_column, split_thresh):
        # 根據閾值將數據分割為左右子集
        left_idxs = np.argwhere(X_column <= split_thresh).flatten()  # 獲取左子集索引
        right_idxs = np.argwhere(X_column > split_thresh).flatten()  # 獲取右子集索引
        return left_idxs, right_idxs

    def _entropy(self, y):
        # 計算熵
        hist = np.bincount(y)  # 計算標籤的頻率
        ps = hist / len(y)  # 計算每個標籤的概率
        return -np.sum([p * np.log(p) for p in ps if p > 0])  # 計算熵

    def _most_common_label(self, y):
        # 獲取最常見的標籤
        counter = Counter(y)  # 使用Counter計算標籤頻率
        value = counter.most_common(1)[0][0]  # 獲取最常見的標籤
        return value

    def predict(self, X):
        # 根據決策樹預測新樣本的類別
        return np.array([self._traverse_tree(x, self.root) for x in X])  # 遍歷樹獲取預測結果

    def _traverse_tree(self, x, node):
        # 遞歸遍歷決策樹
        if node.is_leaf_node():  # 如果是葉子節點，返回節點值
            return node.value

        # 根據特徵值進行分支
        if x[node.feature] <= node.threshold:
            return self._traverse_tree(x, node.left)  # 遞歸左子樹
        return self._traverse_tree(x, node.right)  # 遞歸右子樹
