from sklearn.model_selection import LeaveOneOut, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris

# 加載數據集
data = load_iris()
X = data.data
y = data.target

# 創建分類器
model = RandomForestClassifier(n_estimators=100)

# 使用留一法交叉驗證
loocv = LeaveOneOut()
scores = cross_val_score(model, X, y, cv=loocv, scoring='accuracy')

# 顯示結果
print(f"LOOCV Scores: {scores}")
print(f"Mean accuracy: {scores.mean():.3f}")
