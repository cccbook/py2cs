from sklearn.model_selection import KFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris

# 加載數據集
data = load_iris()
X = data.data
y = data.target

# 創建分類器
model = RandomForestClassifier(n_estimators=100)

# 使用 K 折交叉驗證
kf = KFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(model, X, y, cv=kf, scoring='accuracy')

# 顯示結果
print(f"K-Fold Cross-validation Scores: {scores}")
print(f"Mean accuracy: {scores.mean():.3f}")
