from sklearn.model_selection import cross_val_score, KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris

# 加載數據集
data = load_iris()
X = data.data
y = data.target

# 創建分類器
model = RandomForestClassifier(n_estimators=100)

# 使用 K-fold 交叉驗證評估模型
cv = KFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')

# 顯示結果
print(f"Cross-validation scores: {scores}")
print(f"Mean accuracy: {scores.mean():.3f}")
