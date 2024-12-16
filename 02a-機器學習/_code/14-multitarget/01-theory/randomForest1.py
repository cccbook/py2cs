from sklearn.ensemble import RandomForestClassifier

# 定義隨機森林模型
model = RandomForestClassifier(n_estimators=100, random_state=42)

# 訓練模型
model.fit(X_train, y_train)

# 預測
y_pred = model.predict(X_test)
