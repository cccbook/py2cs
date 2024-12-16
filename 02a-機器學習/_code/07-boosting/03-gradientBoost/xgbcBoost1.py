from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

# 載入資料集
data = load_iris()
X = data.data
y = data.target

# 分割資料集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 建立 Pipeline，包括特徵縮放與 XGBoost
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('xgb', XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=3))
])

# 訓練模型
pipeline.fit(X_train, y_train)

# 預測
y_pred = pipeline.predict(X_test)

# 評估模型
print(f"XGBoost with Pipeline Accuracy: {accuracy_score(y_test, y_pred)}")
