from sklearn.naive_bayes import GaussianNB
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# 載入 Iris 數據集
data = load_iris()
X = data.data  # 特徵
y = data.target  # 標籤

# 分割訓練集和測試集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 初始化朴素貝葉斯分類器
model = GaussianNB()

# 訓練模型
model.fit(X_train, y_train)

# 預測測試集
y_pred = model.predict(X_test)

# 輸出預測結果
print("Predictions:", y_pred)
