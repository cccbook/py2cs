import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from KNN import KNN  # 引入之前定義的 KNN 類別

# 定義顏色映射，用來在圖中區分不同類別的顏色
cmap = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])

# 加載 Iris 鳶尾花數據集，X 是特徵，y 是標籤
iris = datasets.load_iris()
X, y = iris.data, iris.target

# 將數據集分為訓練集和測試集，測試集占 20%，設置隨機種子 random_state 保持結果可重現
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

# 繪製特徵 X 的散點圖，這裡選擇 X 的第3列和第4列（花瓣的長度和寬度）進行可視化
plt.figure()
plt.scatter(X[:, 2], X[:, 3], c=y, cmap=cmap, edgecolor='k', s=20)
plt.show()

# 創建一個 KNN 模型實例，設置 k = 5，即選取 5 個最近鄰居進行預測
clf = KNN(k=5)

# 將訓練數據集餵入 KNN 模型進行訓練
clf.fit(X_train, y_train)

# 使用訓練好的模型對測試集進行預測，返回預測結果
predictions = clf.predict(X_test)

# 輸出預測結果
print(predictions)

# 計算預測的準確率，方法是計算預測正確的數據點佔測試集的比例
acc = np.sum(predictions == y_test) / len(y_test)

# 輸出準確率
print(acc)
