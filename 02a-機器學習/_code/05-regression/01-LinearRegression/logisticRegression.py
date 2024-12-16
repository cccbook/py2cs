from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# 生成分類數據
X, y = make_classification(n_samples=100, n_features=2, n_classes=2, random_state=42)

# 分割數據集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 創建邏輯回歸模型
log_reg = LogisticRegression()

# 訓練模型
log_reg.fit(X_train, y_train)

# 預測結果
y_pred = log_reg.predict(X_test)

# 計算準確率
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
