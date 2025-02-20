from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from scipy.stats import randint

# 定義模型和超參數範圍
model = RandomForestClassifier()
param_dist = {
    'n_estimators': randint(10, 200),
    'max_depth': randint(5, 20)
}

# 使用 RandomizedSearchCV 進行隨機超參數調優
random_search = RandomizedSearchCV(estimator=model, param_distributions=param_dist, n_iter=100, cv=5)
random_search.fit(X_train, y_train)

# 輸出最佳參數和對應的分數
print("最佳參數：", random_search.best_params_)
print("最佳得分：", random_search.best_score_)
