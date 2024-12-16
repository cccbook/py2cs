from sklearn.ensemble import GradientBoostingClassifier

# 定義梯度提升模型
model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3)

# 訓練模型
model.fit(X_train, y_train)

# 預測
y_pred = model.predict(X_test)
