from sklearn.svm import SVC
from sklearn.datasets import make_classification
import numpy as np

# 創建數據集
X, y = make_classification(n_samples=100, n_features=2, n_classes=2, random_state=42)

# 建立支持向量機模型，使用 RBF 核
model = SVC(kernel='rbf', gamma='scale')
model.fit(X, y)

# 預測與評估
y_pred = model.predict(X)
print("預測結果:", y_pred)
