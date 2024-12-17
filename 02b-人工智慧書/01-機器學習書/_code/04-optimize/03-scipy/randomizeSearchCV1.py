from sklearn.model_selection import RandomizedSearchCV
from sklearn.svm import SVC
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np

# 生成分類數據
X, y = make_classification(n_samples=100, n_features=2, n_classes=2, random_state=42)

# 分割數據集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 定義模型
svc = SVC()

# 定義超參數分佈
param_dist = {
    'C': np.logspace(-3, 3, 7),
    'kernel': ['linear', 'rbf'],
    'gamma': ['scale', 'auto']
}

# 定義 RandomizedSearchCV
random_search = RandomizedSearchCV(svc, param_distributions=param_dist, n_iter=10, cv=5, scoring='accuracy', random_state=42)

# 執行隨機搜索
random_search.fit(X_train, y_train)

# 顯示最佳參數
print("Best parameters:", random_search.best_params_)

# 預測並評估模型
y_pred = random_search.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Test accuracy:", accuracy)
