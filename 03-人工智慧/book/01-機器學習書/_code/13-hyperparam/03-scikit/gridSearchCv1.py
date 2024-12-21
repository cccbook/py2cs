from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier

# 定義模型和超參數範圍
model = RandomForestClassifier()
param_grid = {
    'n_estimators': [10, 50, 100],
    'max_depth': [5, 10, 20]
}

# 使用 GridSearchCV 進行超參數調優
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5)
grid_search.fit(X_train, y_train)

# 輸出最佳參數和對應的分數
print("最佳參數：", grid_search.best_params_)
print("最佳得分：", grid_search.best_score_)
