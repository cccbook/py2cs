# 使用多項式核的 SVC 模型
model = SVC(kernel='poly', degree=3, gamma='scale')
model.fit(X_train, y_train)

# 預測並評估模型
y_pred = model.predict(X_test)
print(f'Accuracy: {accuracy_score(y_test, y_pred):.4f}')
