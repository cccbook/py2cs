from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier

# 創建一個包含數據預處理和模型訓練的 Pipeline
pipeline = Pipeline([
    ('scaler', StandardScaler()),  # 數據標準化
    ('mlp', MLPClassifier(hidden_layer_sizes=(100,), activation='relu', solver='adam', max_iter=500))
])

# 訓練模型
pipeline.fit(X_train, y_train)

# 預測測試集
y_pred = pipeline.predict(X_test)

# 評估模型
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")
