from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# 載入資料集
data = load_iris()
X = data.data
y = data.target

# 分割資料集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 建立基學習器（決策樹）
base_estimator = DecisionTreeClassifier(max_depth=1)

# 建立 AdaBoost 模型
ada_boost = AdaBoostClassifier(base_estimator=base_estimator, n_estimators=50)

# 訓練模型
ada_boost.fit(X_train, y_train)

# 預測
y_pred = ada_boost.predict(X_test)

# 評估模型
print(f"AdaBoost Accuracy: {accuracy_score(y_test, y_pred)}")
