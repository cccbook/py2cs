from sklearn.ensemble import GradientBoostingClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# 載入資料集
data = load_iris()
X = data.data
y = data.target

# 分割資料集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 建立 GradientBoostingClassifier 模型
gb = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3)

# 訓練模型
gb.fit(X_train, y_train)

# 預測
y_pred = gb.predict(X_test)

# 評估模型
print(f"GradientBoostingClassifier Accuracy: {accuracy_score(y_test, y_pred)}")
