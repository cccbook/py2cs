import mlflow
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_iris

# 讀取數據集
data = load_iris()
X = data.data
y = data.target

# 拆分數據集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 開始 MLflow 實驗
with mlflow.start_run():
    # 訓練模型
    model = RandomForestClassifier(n_estimators=100, max_depth=3)
    model.fit(X_train, y_train)

    # 預測
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    # 記錄參數和指標
    mlflow.log_param("n_estimators", 100)
    mlflow.log_param("max_depth", 3)
    mlflow.log_metric("accuracy", accuracy)

    # 儲存模型
    mlflow.sklearn.log_model(model, "model")

    print(f"Model Accuracy: {accuracy:.4f}")


# 儲存模型
mlflow.sklearn.log_model(model, "rf_model")

# 加載模型
loaded_model = mlflow.sklearn.load_model("runs:/<run-id>/rf_model")

# 使用加載的模型進行預測
y_pred_loaded = loaded_model.predict(X_test)
