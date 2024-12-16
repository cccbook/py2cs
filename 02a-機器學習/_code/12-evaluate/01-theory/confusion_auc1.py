from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve
import matplotlib.pyplot as plt

# 加載數據集
data = load_iris()
X = data.data
y = data.target

# 只選擇二分類的數據
X = X[y != 0]
y = y[y != 0]

# 分割數據集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 創建隨機森林分類器
model = RandomForestClassifier()

# 訓練模型
model.fit(X_train, y_train)

# 預測結果
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]  # 取得正類的預測概率

# 計算混淆矩陣
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)

# 計算 AUC
auc = roc_auc_score(y_test, y_prob)
print(f"AUC: {auc:.3f}")

# 計算 ROC 曲線
fpr, tpr, thresholds = roc_curve(y_test, y_prob)

# 畫出 ROC 曲線
plt.plot(fpr, tpr, label=f'AUC = {auc:.3f}')
plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.show()
