from sklearn import datasets  # 從scikit-learn引入datasets模塊以獲取內建數據集
from sklearn.model_selection import train_test_split  # 從scikit-learn引入train_test_split以分割數據集
import numpy as np  # 引入NumPy庫以進行數值計算
from DecisionTree import DecisionTree  # 從自定義的DecisionTree模組引入DecisionTree類別

# 加載乳腺癌數據集
data = datasets.load_breast_cancer()
X, y = data.data, data.target  # 獲取特徵和標籤

# 將數據集分割為訓練集和測試集，測試集佔20%
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=1234  # 隨機狀態用於可重複性
)

# 創建DecisionTree分類器，設定最大樹深度為10
clf = DecisionTree(max_depth=10)

# 使用訓練數據擬合決策樹模型
clf.fit(X_train, y_train)

# 使用測試數據進行預測
predictions = clf.predict(X_test)

def accuracy(y_test, y_pred):
    # 計算模型預測的準確率
    return np.sum(y_test == y_pred) / len(y_test)  # 正確預測的比例

# 計算準確率
acc = accuracy(y_test, predictions)

# 輸出準確率
print(acc)
