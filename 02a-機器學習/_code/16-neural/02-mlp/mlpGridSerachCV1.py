
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier

# 設置超參數搜索空間
param_grid = {
    'hidden_layer_sizes': [(50,), (100,), (50, 50)],
    'activation': ['relu', 'tanh'],
    'solver': ['adam', 'sgd'],
    'max_iter': [200, 500]
}

# 初始化 MLPClassifier
mlp = MLPClassifier()

# 使用 GridSearchCV 進行超參數搜索
grid_search = GridSearchCV(mlp, param_grid, cv=5, n_jobs=-1, verbose=1)

# 訓練模型
grid_search.fit(X_train, y_train)

# 輸出最佳參數
print(f"Best Parameters: {grid_search.best_params_}")

