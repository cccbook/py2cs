from sklearn.datasets import make_circles

# 生成非線性可分的數據（圓形分佈）
X, y = make_circles(n_samples=100, noise=0.1, factor=0.4, random_state=42)

# 切分數據集為訓練集和測試集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 使用 RBF 核的 SVC 模型
model = SVC(kernel='rbf', gamma='scale')  # 'scale' 是 gamma 的一個選項
model.fit(X_train, y_train)

# 預測並評估模型
y_pred = model.predict(X_test)
print(f'Accuracy: {accuracy_score(y_test, y_pred):.4f}')
