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

# 建立 GBDT 模型
gbdt = GradientBoostingClassifier(n_estimators=50, learning_rate=0.1, max_depth=3)

# 訓練模型
gbdt.fit(X_train, y_train)

# 預測
y_pred = gbdt.predict(X_test)

# 評估模型
print(f"GBDT Accuracy: {accuracy_score(y_test, y_pred)}")
