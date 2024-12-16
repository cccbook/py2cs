import xgboost as xgb
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier

# 載入資料集
data = load_iris()
X = data.data
y = data.target

# 分割資料集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 建立 XGBClassifier 模型
xgb_model = XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=3)

# 訓練模型
xgb_model.fit(X_train, y_train)

# 預測
y_pred = xgb_model.predict(X_test)

# 評估模型
print(f"XGBoost Accuracy: {accuracy_score(y_test, y_pred)}")

