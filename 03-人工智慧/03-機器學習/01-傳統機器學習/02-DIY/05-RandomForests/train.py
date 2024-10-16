from random import random  # 引入隨機數生成函數（在此程式中未使用）
from sklearn import datasets  # 引入 sklearn 數據集模組，以載入內建數據集
from sklearn.model_selection import train_test_split  # 引入函數，用於分割數據集為訓練集和測試集
import numpy as np  # 引入 NumPy 庫，用於數值運算
from RandomForest import RandomForest  # 引入自定義的 RandomForest 類

# 載入乳腺癌數據集
data = datasets.load_breast_cancer()  # 使用 sklearn 的數據集模組載入乳腺癌數據
X = data.data  # 獲取特徵數據（特徵矩陣）
y = data.target  # 獲取標籤數據（目標向量）

# 將數據集分為訓練集和測試集，80% 用於訓練，20% 用於測試
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=1234  # 設定隨機種子以便重現結果
)

def accuracy(y_true, y_pred):
    # 計算預測準確率
    accuracy = np.sum(y_true == y_pred) / len(y_true)  # 計算正確預測的比例
    return accuracy  # 返回準確率

# 初始化隨機森林分類器，設定樹的數量為 20
clf = RandomForest(n_trees=20)
# 使用訓練數據進行隨機森林模型的訓練
clf.fit(X_train, y_train)
# 對測試集進行預測，得到預測結果
predictions = clf.predict(X_test)

# 計算預測的準確率
acc = accuracy(y_test, predictions)
# 輸出準確率
print(acc)
