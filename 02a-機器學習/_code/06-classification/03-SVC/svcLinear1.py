import numpy as np
from sklearn.svm import SVC
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# 生成線性可分的數據
X, y = make_classification(n_samples=100, n_features=2, n_classes=2, random_state=42)

# 切分數據集為訓練集和測試集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 使用線性核的 SVC 模型
model = SVC(kernel='linear')
model.fit(X_train, y_train)

# 預測並評估模型
y_pred = model.predict(X_test)
print(f'Accuracy: {accuracy_score(y_test, y_pred):.4f}')
