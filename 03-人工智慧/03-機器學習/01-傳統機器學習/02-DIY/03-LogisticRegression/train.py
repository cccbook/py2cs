import numpy as np  # 引入NumPy庫，用於數值計算
from sklearn.model_selection import train_test_split  # 從sklearn庫中導入訓練測試數據劃分函數
from sklearn import datasets  # 從sklearn庫中導入數據集模組
import matplotlib.pyplot as plt  # 引入Matplotlib庫，用於數據可視化
from LogisticRegression import LogisticRegression  # 導入自定義的LogisticRegression模型

# 載入乳腺癌數據集
bc = datasets.load_breast_cancer()  
X, y = bc.data, bc.target  # 提取特徵X和標籤y

# 將數據集劃分為訓練集和測試集，80%用於訓練，20%用於測試
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

# 初始化邏輯回歸模型，設置學習率為0.01
clf = LogisticRegression(lr=0.01)  
# 使用訓練數據進行模型訓練
clf.fit(X_train, y_train)  
# 使用測試數據進行預測
y_pred = clf.predict(X_test)  

def accuracy(y_pred, y_test):
    # 計算預測準確率
    return np.sum(y_pred == y_test) / len(y_test)  # 返回正確預測的比例

# 計算模型的準確率
acc = accuracy(y_pred, y_test)  
# 輸出準確率
print(acc)  
